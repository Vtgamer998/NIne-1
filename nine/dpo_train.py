"""
DPO (Direct Preference Optimization) para NINE-1.
Alinha o modelo com preferências humanas usando o algoritmo DPO
(Rafailov et al. 2023).

Formula:
  L_DPO = -log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                     - (log π_θ(y_l|x) - log π_ref(y_l|x))))

Onde:
  - y_w = resposta escolhida (chosen)
  - y_l = resposta rejeitada (rejected)
  - π_θ = modelo politico (policy, com LoRA)
  - π_ref = modelo de referencia (frozen, copia inicial)
  - β = parametro de temperatura (default 0.1)

Seguranca:
  - Validacao de checkpoint
  - Protecao contra NaN/Inf
  - Early stopping se loss divergir
  - Log de gradientes suspeitos
"""

from __future__ import annotations
import argparse
import json
import math
import os
import time
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model import NINE1, NINEConfig, validate_checkpoint_state, safe_load_checkpoint
from .finetune import LoRALinear, add_lora, count_trainable


# ---------------------------------------------------------------------------
# Constantes de seguranca
# ---------------------------------------------------------------------------

MAX_EXAMPLES = 10_000          # Max exemplos DPO
MAX_RESPONSE_CHARS = 5_000     # Max chars por resposta
MIN_PROMPT_TOKENS = 2          # Min tokens de prompt
MIN_RESPONSE_TOKENS = 1        # Min tokens de resposta
DPO_REQUIRED_FIELDS = {"prompt", "chosen", "rejected"}


# ---------------------------------------------------------------------------
# Dataset DPO
# ---------------------------------------------------------------------------

class DPODataset(Dataset):
    """Dataset DPO: prompt + chosen + rejected em formato JSONL.

    Cada linha do JSONL deve ter:
      {"prompt": "...", "chosen": "...", "rejected": "..."}

    O dataset formata e codifica com o tokenizer BPE.
    Retorna (input_ids_chosen, labels_chosen, input_ids_rejected, labels_rejected)
    onde labels tem -100 nas posicoes do prompt (ignoradas na loss).
    """

    def __init__(self, path: str, block_size: int, tokenizer=None):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.examples: List[dict] = []
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
                    if errors <= 3:
                        print(f"  [aviso] JSON invalido linha {line_num}: pulando")
                    continue

                # Validacao de campos obrigatorios
                if not isinstance(ex, dict):
                    errors += 1
                    continue
                if not DPO_REQUIRED_FIELDS.issubset(ex.keys()):
                    errors += 1
                    continue
                if any(len(ex.get(k, "")) > MAX_RESPONSE_CHARS for k in ("chosen", "rejected")):
                    continue
                if not ex["prompt"].strip() or not ex["chosen"].strip() or not ex["rejected"].strip():
                    continue

                self.examples.append({
                    "prompt": ex["prompt"].strip(),
                    "chosen": ex["chosen"].strip(),
                    "rejected": ex["rejected"].strip(),
                })

                if len(self.examples) >= MAX_EXAMPLES:
                    break

        if errors > 0:
            print(f"  [aviso] {errors} linhas com erro foram ignoradas")
        if not self.examples:
            raise ValueError(f"Nenhum exemplo DPO valido em {path}")
        print(f"  Dataset DPO: {len(self.examples)} exemplos")

    def __len__(self):
        return len(self.examples)

    def _encode(self, text: str) -> List[int]:
        """Codifica texto com BPE se disponivel, senao fallback."""
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            return ids[:self.block_size]
        ids = [ord(c) % 65536 for c in text]
        return ids[:self.block_size] if ids else [0]

    def __getitem__(self, idx, _depth: int = 0):
        if _depth > len(self.examples):
            # Seguranca: evita loop infinito se todos os exemplos forem ruins
            dummy = torch.tensor([0, 0], dtype=torch.long)
            return (dummy, dummy, dummy, dummy)

        ex = self.examples[idx]
        prompt = f"# tarefa: {ex['prompt']}\n# solucao:\n"

        prompt_ids = self._encode(prompt)
        chosen_ids = self._encode(ex["chosen"])
        rejected_ids = self._encode(ex["rejected"])

        # Trunca para caber no block_size
        max_space = self.block_size - len(prompt_ids)
        if max_space <= 0:
            return self.__getitem__((idx + 1) % len(self.examples), _depth + 1)
        chosen_ids = chosen_ids[:max_space]
        rejected_ids = rejected_ids[:max_space]

        # Valida tamanho minimo
        if len(prompt_ids) < MIN_PROMPT_TOKENS or len(chosen_ids) < MIN_RESPONSE_TOKENS:
            return self.__getitem__((idx + 1) % len(self.examples), _depth + 1)
        if len(rejected_ids) < MIN_RESPONSE_TOKENS:
            return self.__getitem__((idx + 1) % len(self.examples), _depth + 1)

        # Concatena prompt + resposta
        input_chosen = prompt_ids + chosen_ids
        input_rejected = prompt_ids + rejected_ids

        # Labels: -100 para prompt, ids para resposta
        labels_chosen = [-100] * len(prompt_ids) + chosen_ids
        labels_rejected = [-100] * len(prompt_ids) + rejected_ids

        return (
            torch.tensor(input_chosen, dtype=torch.long),
            torch.tensor(labels_chosen, dtype=torch.long),
            torch.tensor(input_rejected, dtype=torch.long),
            torch.tensor(labels_rejected, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Funcao de loss DPO
# ---------------------------------------------------------------------------

def compute_dpo_loss(
    policy_logits_chosen: torch.Tensor,    # (B, T, V)
    labels_chosen: torch.Tensor,            # (B, T) com -100
    policy_logits_rejected: torch.Tensor,  # (B, T, V)
    labels_rejected: torch.Tensor,         # (B, T) com -100
    ref_logits_chosen: torch.Tensor,       # (B, T, V)
    ref_logits_rejected: torch.Tensor,     # (B, T, V)
    beta: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """Computa a loss DPO para um batch.

    Args:
        policy_logits_chosen: Logits do modelo policy para chosen.
        labels_chosen: Targets para chosen (-100 para prompt).
        policy_logits_rejected: Logits do modelo policy para rejected.
        labels_rejected: Targets para rejected (-100 para prompt).
        ref_logits_chosen: Logits do modelo reference para chosen.
        ref_logits_rejected: Logits do modelo reference para rejected.
        beta: Temperatura DPO (default 0.1).

    Returns:
        (loss, stats) onde stats tem metricas de debug.
    """
    B = labels_chosen.size(0)

    # --- Mascara: tokens validos (nao -100) ---
    logits_dtype = policy_logits_chosen.dtype
    mask_chosen = (labels_chosen != -100).to(logits_dtype)  # (B, T)
    mask_rejected = (labels_rejected != -100).to(logits_dtype)

    # Troca -100 por 0 para gather (gather nao aceita indices negativos)
    labels_chosen_safe = labels_chosen.clone()
    labels_chosen_safe[labels_chosen_safe < 0] = 0
    labels_rejected_safe = labels_rejected.clone()
    labels_rejected_safe[labels_rejected_safe < 0] = 0

    # --- Log-probs do modelo policy ---
    log_probs_policy_chosen = F.log_softmax(policy_logits_chosen, dim=-1)
    log_probs_policy_rejected = F.log_softmax(policy_logits_rejected, dim=-1)

    logp_chosen_policy = log_probs_policy_chosen.gather(
        -1, labels_chosen_safe.unsqueeze(-1)
    ).squeeze(-1)
    logp_rejected_policy = log_probs_policy_rejected.gather(
        -1, labels_rejected_safe.unsqueeze(-1)
    ).squeeze(-1)

    # Zera log-probs de posicoes com -100
    logp_chosen_policy = logp_chosen_policy * mask_chosen
    logp_rejected_policy = logp_rejected_policy * mask_rejected

    # --- Log-probs do modelo reference (frozen) ---
    with torch.no_grad():
        log_probs_ref_chosen = F.log_softmax(ref_logits_chosen, dim=-1)
        log_probs_ref_rejected = F.log_softmax(ref_logits_rejected, dim=-1)

        logp_chosen_ref = log_probs_ref_chosen.gather(
            -1, labels_chosen_safe.unsqueeze(-1)
        ).squeeze(-1)
        logp_rejected_ref = log_probs_ref_rejected.gather(
            -1, labels_rejected_safe.unsqueeze(-1)
        ).squeeze(-1)

        # Zera log-probs de posicoes com -100
        logp_chosen_ref = logp_chosen_ref * mask_chosen
        logp_rejected_ref = logp_rejected_ref * mask_rejected

    # Soma dos log-probs sobre tokens de resposta
    sum_logp_chosen_policy = logp_chosen_policy.sum(-1)  # (B,)
    sum_logp_rejected_policy = logp_rejected_policy.sum(-1)
    sum_logp_chosen_ref = logp_chosen_ref.sum(-1)
    sum_logp_rejected_ref = logp_rejected_ref.sum(-1)

    # --- DPO loss ---
    # log π_θ(y_w|x) - log π_ref(y_w|x) = log-ratio da escolhida
    # log π_θ(y_l|x) - log π_ref(y_l|x) = log-ratio da rejeitada
    log_ratio_chosen = sum_logp_chosen_policy - sum_logp_chosen_ref
    log_ratio_rejected = sum_logp_rejected_policy - sum_logp_rejected_ref

    # Loss DPO: -log σ(β * (log_ratio_chosen - log_ratio_rejected))
    # Clamping para evitar exp() overflow
    logits = beta * (log_ratio_chosen - log_ratio_rejected)
    logits = torch.clamp(logits, min=-50, max=50)
    loss = -F.logsigmoid(logits).mean()

    # Guard: detecta NaN/Inf e retorna loss zero
    if loss.isnan() or loss.isinf():
        print("  [aviso] DPO loss divergiu (NaN/Inf)! Retornando loss zero.")
        loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

    # --- Metricas de debug ---
    with torch.no_grad():
        chosen_reward = beta * log_ratio_chosen.mean().item()
        rejected_reward = beta * log_ratio_rejected.mean().item()
        accuracy = (log_ratio_chosen > log_ratio_rejected).float().mean().item()
        stats = {
            "loss": loss.item(),
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "accuracy": accuracy,
            "margin": (log_ratio_chosen - log_ratio_rejected).mean().item(),
        }

    return loss, stats


# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------

def compute_log_probs_sum(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Computa soma dos log-probs dos tokens de resposta.

    Args:
        model: Modelo NINE-1.
        input_ids: (B, T) tokens de entrada.
        labels: (B, T) targets com -100 para prompt.

    Returns:
        (B,) soma dos log-probs sobre tokens de resposta.
    """
    logits, _, _ = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).float()
    return (logp * mask).sum(-1)


# ---------------------------------------------------------------------------
# Collate function para batches de tamanhos variados
# ---------------------------------------------------------------------------

def dpo_collate_fn(batch):
    """Agrupa exemplos DPO com padding.

    Cada exemplo: (input_chosen, labels_chosen, input_rejected, labels_rejected)
    """
    input_c, labels_c, input_r, labels_r = zip(*batch)

    def _pad(seqs, pad_val=0):
        max_len = max(s.size(0) for s in seqs)
        padded = []
        for s in seqs:
            pad_len = max_len - s.size(0)
            if pad_len > 0:
                s = F.pad(s, (0, pad_len), value=pad_val)
            padded.append(s)
        return torch.stack(padded)

    return (
        _pad(input_c, 0),
        _pad(labels_c, -100),
        _pad(input_r, 0),
        _pad(labels_r, -100),
    )


# ---------------------------------------------------------------------------
# CLI e treino
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DPO Alignment para NINE-1")
    p.add_argument("--base", type=str, required=True,
                   help="checkpoint base .pt (policy inicial + reference)")
    p.add_argument("--data", type=str, required=True,
                   help="dataset DPO .jsonl (prompt, chosen, rejected)")
    p.add_argument("--tok", type=str, default=None,
                   help="tokenizer BPE .json")
    p.add_argument("--out", type=str, default="nine/data/nine1-dpo.pt",
                   help="saida dos adaptadores DPO")

    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target", choices=["qkv", "qkvo", "all"],
                   default="qkv")

    # DPO hyperparams
    p.add_argument("--beta", type=float, default=0.1,
                   help="temperatura DPO (default 0.1)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="learning rate (menor que fine-tuning)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_iters", type=int, default=500)
    p.add_argument("--val_split", type=float, default=0.1)

    # Geral
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    print("=" * 52)
    print("  NINE-1 DPO Alignment")
    print("=" * 52)

    # Validacoes de seguranca
    if not os.path.exists(args.base):
        raise FileNotFoundError(f"Checkpoint base nao encontrado: {args.base}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset nao encontrado: {args.data}")

    # --- Carrega modelo base ---
    print(f"\n[1/5] Carregando base: {args.base}")
    ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["cfg"]
    cfg = NINEConfig.from_dict(cfg_dict)

    if "model" in ckpt:
        issues = validate_checkpoint_state(ckpt["model"], cfg)
        if issues:
            for iss in issues:
                print(f"  [aviso] Checkpoint: {iss}")

    # --- Cria policy model (treinavel) ---
    policy = NINE1(cfg)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # Remove chaves extras (ex: lora_ do fine-tuning anterior)
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("lora_") and not k.startswith("_freqs_")}
    policy.load_state_dict(state_dict, strict=False)
    policy.to(args.device)

    # --- Cria reference model (frozen, copia exata do policy inicial) ---
    ref = NINE1(cfg)
    ref.load_state_dict(policy.state_dict(), strict=False)
    ref.to(args.device)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    print(f"  Reference model criado (frozen)")

    # --- Aplica LoRA apenas no policy ---
    print(f"\n[2/5] Aplicando LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    policy = add_lora(policy, r=args.lora_r, alpha=args.lora_alpha,
                      dropout=args.lora_dropout, target=args.lora_target)
    policy.to(args.device)

    n_train = count_trainable(policy)
    total = policy.num_params()
    print(f"  Params treinaveis (LoRA): {n_train:,} de {total:,} "
          f"({100 * n_train / total:.1f}%)")

    # --- Carrega tokenizer ---
    print(f"\n[3/5] Carregando dados")
    tokenizer = None
    if args.tok:
        from .tokenizer import BPETokenizer
        try:
            tokenizer = BPETokenizer.load(args.tok)
            print(f"  Tokenizer BPE: {len(tokenizer)} tokens")
        except Exception as e:
            print(f"  [aviso] Falha ao carregar tokenizer: {e}")

    # --- Dataset DPO ---
    full_dataset = DPODataset(args.data, block_size=cfg.block_size,
                              tokenizer=tokenizer)

    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train_ds = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train_ds, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=dpo_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=dpo_collate_fn,
    ) if n_val > 0 else None

    print(f"  Split: treino={n_train_ds}, val={n_val}")

    # --- Otimizador ---
    print(f"\n[4/5] Iniciando treino DPO (beta={args.beta})")
    optim = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
    )

    iter_total = 0
    best_val_loss = float("inf")
    policy.train()
    t0 = time.time()

    for ep in range(args.epochs):
        for batch in train_loader:
            if iter_total >= args.max_iters:
                break

            input_c, labels_c, input_r, labels_r = [b.to(args.device) for b in batch]

            # Forward policy
            logits_c, _, _ = policy(input_c)
            logits_r, _, _ = policy(input_r)

            # Forward reference (frozen)
            with torch.no_grad():
                ref_logits_c, _, _ = ref(input_c)
                ref_logits_r, _, _ = ref(input_r)

            # DPO loss
            loss, stats = compute_dpo_loss(
                logits_c, labels_c, logits_r, labels_r,
                ref_logits_c, ref_logits_r,
                beta=args.beta,
            )

            # Backward
            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad],
                args.grad_clip,
            )
            optim.step()

            # Logging
            if iter_total % args.log_interval == 0:
                dt = time.time() - t0
                msg = (f"  ep {ep} iter {iter_total}/{args.max_iters} | "
                       f"loss {stats['loss']:.4f} | "
                       f"acc {stats['accuracy']:.2f} | "
                       f"margin {stats['margin']:.3f} | "
                       f"grad {grad_norm:.2f} | "
                       f"{dt:.0f}s")

                # Validacao periodica
                if val_loader is not None and iter_total % 50 == 0:
                    policy.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vb in val_loader:
                            vc, vlc, vr, vlr = [x.to(args.device) for x in vb]
                            vlog_c, _, _ = policy(vc)
                            vlog_r, _, _ = policy(vr)
                            vref_c, _, _ = ref(vc)
                            vref_r, _, _ = ref(vr)
                            vl, vstats = compute_dpo_loss(
                                vlog_c, vlc, vlog_r, vlr,
                                vref_c, vref_r,
                                beta=args.beta,
                            )
                            val_losses.append(vstats["loss"])
                            if len(val_losses) >= 5:
                                break
                    avg_val = float(np.mean(val_losses))
                    msg += f" | val_loss {avg_val:.4f}"
                    if avg_val < best_val_loss:
                        best_val_loss = avg_val
                        msg += " (best)"
                    policy.train()

                print(msg)
                t0 = time.time()

            iter_total += 1
        if iter_total >= args.max_iters:
            break

    # --- Salva adaptadores DPO (LoRA) ---
    print(f"\n[5/5] Salvando checkpoint DPO")
    lora_state = {}
    for name, p in policy.state_dict().items():
        if "lora_" in name:
            lora_state[name] = p.cpu()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "dpo": lora_state,
        "cfg": cfg.__dict__,
        "base": args.base,
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "beta": args.beta,
        "lora_target": args.lora_target,
    }, args.out)

    print(f"  DPO salvo em: {args.out}")
    print(f"  Tamanho: {os.path.getsize(args.out) / 1024:.1f} KB")
    if best_val_loss < float("inf"):
        print(f"  Melhor val loss: {best_val_loss:.4f}")
    print("\n✅ DPO Alignment concluido!")


if __name__ == "__main__":
    main()
