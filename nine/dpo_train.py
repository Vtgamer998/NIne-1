"""
DPO (Direct Preference Optimization) para NINE-1.

Alinha o modelo com preferencias humanas usando a loss function:
    L_DPO = -E[ log sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)))) ]

Uso:
    python -m nine.dpo_train --base nine/data/nine1-base.pt \\
        --data nine/data/preferences.jsonl --tok nine/data/nine1-tok.json \\
        --out nine/data/nine1-dpo.pt --beta 0.1 --epochs 3

Formato do dataset (JSONL):
    {"prompt": "escreva fibonacci", "chosen": "def fib...", "rejected": "print('erro')"}
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model import NINE1, NINEConfig, validate_checkpoint_state

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MAX_EXAMPLES = 10_000
MAX_SEQ_LEN = 2048


# ---------------------------------------------------------------------------
# DPO Dataset
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    """Dataset de preferencias (prompt, chosen, rejected) para DPO.

    Cada exemplo contem:
    - prompt: a instrucao/contexto
    - chosen: resposta preferida
    - rejected: resposta rejeitada

    O dataset codifica cada par (prompt + resposta) e prepara as mascaras
    para calcular log-probabilidades apenas na parte da resposta.
    """

    def __init__(self, path: str, block_size: int, tokenizer=None):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.examples: List[Dict[str, str]] = []
        errors = 0

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                if not all(k in ex for k in ("prompt", "chosen", "rejected")):
                    errors += 1
                    continue

                self.examples.append(ex)
                if len(self.examples) >= MAX_EXAMPLES:
                    break

        if errors:
            print(f"  [dpo] {errors} linhas ignoradas")
        if not self.examples:
            raise ValueError(f"Nenhum exemplo valido em {path}")
        print(f"  [dpo] Dataset: {len(self.examples)} pares de preferencia")

    def __len__(self):
        return len(self.examples)

    def _encode_pair(self, prompt: str, response: str):
        """Codifica prompt + resposta, retorna (ids, mask_resposta).

        Codifica prompt e response SEPARADAMENTE para evitar que o BPE
        context-dependente atrapalhe a identificacao do boundary.
        Depois concatena as listas de IDs.
        """
        prompt_text = f"# tarefa: {prompt}\n# solucao:\n"

        if self.tokenizer is not None:
            # Codifica separadamente (evita BPE context-dependent)
            prompt_ids = self.tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
            response_ids = self.tokenizer.encode(response, add_bos=False, add_eos=True)
        else:
            prompt_ids = [ord(c) % 65536 for c in prompt_text]
            response_ids = [ord(c) % 65536 for c in response]

        # Concatena
        ids = (prompt_ids + response_ids)[:self.block_size]
        prompt_len = min(len(prompt_ids), self.block_size)

        # Mascara: 1 para tokens da resposta, 0 para prompt
        mask = [0] * prompt_len + [1] * max(0, len(ids) - prompt_len)

        return ids, mask

    def __getitem__(self, idx):
        ex = self.examples[idx]

        chosen_ids, chosen_mask = self._encode_pair(ex["prompt"], ex["chosen"])
        rejected_ids, rejected_mask = self._encode_pair(ex["prompt"], ex["rejected"])

        # Pad ao mesmo tamanho
        max_len = max(len(chosen_ids), len(rejected_ids), 2)

        def pad(ids, mask, length):
            if len(ids) < length:
                ids = ids + [0] * (length - len(ids))
                mask = mask + [0] * (length - len(mask))
            return ids[:length], mask[:length]

        chosen_ids, chosen_mask = pad(chosen_ids, chosen_mask, max_len)
        rejected_ids, rejected_mask = pad(rejected_ids, rejected_mask, max_len)

        return {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_mask": torch.tensor(chosen_mask, dtype=torch.bool),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_mask": torch.tensor(rejected_mask, dtype=torch.bool),
        }


def collate_preference(batch):
    """Agrupa batch de preferencia com padding."""
    chosen_ids = torch.stack([b["chosen_ids"] for b in batch])
    chosen_mask = torch.stack([b["chosen_mask"] for b in batch])
    rejected_ids = torch.stack([b["rejected_ids"] for b in batch])
    rejected_mask = torch.stack([b["rejected_mask"] for b in batch])
    return chosen_ids, chosen_mask, rejected_ids, rejected_mask


# ---------------------------------------------------------------------------
# DPO Loss
# ---------------------------------------------------------------------------

def compute_log_probs(model, input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Calcula log-probabilidades dos tokens da resposta.

    Args:
        model: Modelo NINE-1 (em modo train ou eval).
        input_ids: (B, T) tokens completos (prompt + response).
        response_mask: (B, T) bool, True onde sao tokens da resposta.

    Returns:
        log_probs: (B,) soma das log-probabilidades por exemplo.
    """
    B, T = input_ids.shape

    logits, _, _ = model(input_ids)
    # logits: (B, T, V)

    # Log-softmax ao longo do vocab
    log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # Pega log_prob do token alvo
    target_ids = input_ids[:, 1:].contiguous()  # (B, T-1)
    log_probs_target = log_probs_all[:, :-1, :].gather(
        2, target_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)

    # Aplica mascara: so conta tokens da resposta
    response_mask_target = response_mask[:, 1:].contiguous()  # (B, T-1)
    log_probs_masked = log_probs_target * response_mask_target.float()

    # Soma por exemplo
    log_probs = log_probs_masked.sum(dim=1)  # (B,)

    return log_probs


def dpo_loss(
    policy_logps_chosen: torch.Tensor,
    policy_logps_rejected: torch.Tensor,
    ref_logps_chosen: torch.Tensor,
    ref_logps_rejected: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """DPO loss function.

    Args:
        policy_logps_chosen: (B,) log-probs do modelo atual para chosen.
        policy_logps_rejected: (B,) log-probs do modelo atual para rejected.
        ref_logps_chosen: (B,) log-probs do modelo de referencia para chosen.
        ref_logps_rejected: (B,) log-probs do modelo de referencia para rejected.
        beta: Hiperparametro de regularizacao (quanto maior, mais proximo do ref).

    Returns:
        loss: (,) tensor escalar.
        metrics: Dict com metricas de debug.
    """
    # Diferenca de log-probs: chosen - rejected
    log_ratio_chosen = policy_logps_chosen - ref_logps_chosen
    log_ratio_rejected = policy_logps_rejected - ref_logps_rejected
    log_ratio_diff = log_ratio_chosen - log_ratio_rejected

    # DPO loss: -log(sigmoid(beta * diff))
    loss = -F.logsigmoid(beta * log_ratio_diff).mean()

    # Metricas
    with torch.no_grad():
        accuracy = (log_ratio_diff > 0).float().mean().item()
        margin = log_ratio_diff.mean().item()
        chosen_reward = (beta * log_ratio_chosen).mean().item()
        rejected_reward = (beta * log_ratio_rejected).mean().item()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy,
        "margin": margin,
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
        "reward_diff": chosen_reward - rejected_reward,
    }

    return loss, metrics


# ---------------------------------------------------------------------------
# Treino DPO
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DPO fine-tuning for NINE-1")
    p.add_argument("--base", type=str, required=True, help="Modelo base .pt")
    p.add_argument("--data", type=str, required=True, help="Dataset preferencias .jsonl")
    p.add_argument("--tok", type=str, default=None, help="Tokenizer BPE .json")
    p.add_argument("--out", type=str, default="nine/data/nine1-dpo.pt")
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta (regularizacao)")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_iters", type=int, default=200)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print("=== NINE-1 DPO Training ===")
    print(f"Base: {args.base}")
    print(f"Data: {args.data}")
    print(f"Beta: {args.beta}, Device: {args.device}")

    # Carrega modelo base
    print(f"\nCarregando modelo base: {args.base}")
    ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg", ckpt.get("config", {}))
    cfg = NINEConfig.from_dict(cfg_dict)

    policy = NINE1(cfg)
    policy.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    policy.to(args.device)

    # Cria modelo de referencia (congelado)
    ref_model = NINE1(cfg)
    ref_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    ref_model.to(args.device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print(f"Modelo: {policy.num_params()/1e6:.2f}M parametros")

    # Carrega tokenizer
    tokenizer = None
    if args.tok:
        from .tokenizer import BPETokenizer
        try:
            tokenizer = BPETokenizer.load(args.tok)
            print(f"Tokenizer: {len(tokenizer)} tokens")
        except Exception as e:
            print(f"  Aviso: {e}")

    # Aplica LoRA (opcional, recomendado para DPO)
    if args.lora_r > 0:
        from .finetune import add_lora, count_trainable
        policy = add_lora(policy, r=args.lora_r, alpha=args.lora_alpha,
                          dropout=0.1, target="qkv")
        policy.to(args.device)
        n_train = count_trainable(policy)
        total = policy.num_params()
        print(f"LoRA: {n_train/1e6:.3f}M treinaveis de {total/1e6:.2f}M")

    # Dataset
    dataset = PreferenceDataset(args.data, block_size=cfg.block_size, tokenizer=tokenizer)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_preference,
    )

    # Otimizador
    optim_params = [p for p in policy.parameters() if p.requires_grad]
    if not optim_params:
        raise RuntimeError("Nenhum parametro treinavel! Use LoRA.")
    optim = torch.optim.AdamW(optim_params, lr=args.lr)

    # Loop de treino DPO
    print(f"\nIniciando treino DPO (beta={args.beta})...")
    policy.train()
    iter_total = 0
    t0 = time.time()

    for ep in range(args.epochs):
        epoch_losses = []
        epoch_accs = []

        for batch in loader:
            if iter_total >= args.max_iters:
                break

            chosen_ids, chosen_mask, rejected_ids, rejected_mask = [
                t.to(args.device) for t in batch
            ]

            # Forward: policy (com grad)
            policy_logps_chosen = compute_log_probs(policy, chosen_ids, chosen_mask)
            policy_logps_rejected = compute_log_probs(policy, rejected_ids, rejected_mask)

            # Forward: reference (sem grad)
            with torch.no_grad():
                ref_logps_chosen = compute_log_probs(ref_model, chosen_ids, chosen_mask)
                ref_logps_rejected = compute_log_probs(ref_model, rejected_ids, rejected_mask)

            # DPO loss
            loss, metrics = dpo_loss(
                policy_logps_chosen, policy_logps_rejected,
                ref_logps_chosen, ref_logps_rejected,
                beta=args.beta,
            )

            # Backward
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
            optim.step()

            epoch_losses.append(metrics["loss"])
            epoch_accs.append(metrics["accuracy"])

            if iter_total % 20 == 0:
                dt = time.time() - t0
                print(f"  ep {ep} iter {iter_total}: loss={metrics['loss']:.4f} "
                      f"acc={metrics['accuracy']:.2%} margin={metrics['margin']:.3f} "
                      f"reward_diff={metrics['reward_diff']:.3f} [{dt:.1f}s]")

            iter_total += 1

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            avg_acc = np.mean(epoch_accs)
            print(f"  >> Epoca {ep}: loss={avg_loss:.4f} acc={avg_acc:.2%}")

        if iter_total >= args.max_iters:
            break

    # Salva
    save_dict = {"cfg": cfg.__dict__, "args": vars(args)}

    if args.lora_r > 0:
        # Salva apenas LoRA
        lora_state = {k: v.cpu() for k, v in policy.state_dict().items() if "lora_" in k}
        save_dict["lora"] = lora_state
        save_dict["base"] = args.base
        save_dict["r"] = args.lora_r
        save_dict["alpha"] = args.lora_alpha
    else:
        # Salva modelo completo (full fine-tune)
        save_dict["model"] = {k: v.cpu() for k, v in policy.state_dict().items()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(save_dict, args.out)

    dt = time.time() - t0
    print(f"\nDPO finalizado em {dt:.1f}s")
    print(f"Salvo em: {args.out}")


if __name__ == "__main__":
    main()
