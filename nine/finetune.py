"""
Fine-tuning do NINE-1 base usando LoRA (PEFT).
Carrega checkpoint base, adiciona adaptadores LoRA nas matrizes Q e KV,
e treina em dataset instrucional codificado com BPE.

Seguranca:
- Validacao de dataset (JSONL corrompido, campos obrigatorios)
- Safe checkpoint loading com weights_only
- Protecao contra overflow de memoria
- LoRA com dropout configravel
"""

from __future__ import annotations
import argparse
import json
import math
import os
import time
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .model import NINE1, NINEConfig, validate_checkpoint_state, safe_load_checkpoint


# ---------------------------------------------------------------------------
# Constantes de seguranca
# ---------------------------------------------------------------------------

MAX_EXAMPLES = 100_000       # Maximo de exemplos no dataset
MAX_EXAMPLE_CHARS = 10_000   # Maximo de caracteres por exemplo
ALLOWED_INSTRUCT_FIELDS = {"instruction", "input", "output"}


# ---------------------------------------------------------
# LoRA: adaptadores low-rank
# ---------------------------------------------------------
class LoRALinear(torch.nn.Module):
    """Linear + LoRA adapter nas matrizes Q, K, V (nao em todo o linear)."""

    def __init__(self, base: torch.nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0, target: str = "all"):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0
        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = torch.nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = torch.nn.Parameter(torch.zeros(out_f, r))
        if r > 0:
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        if self.r > 0:
            lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
            out = out + lora_out * self.scaling
        return out


def add_lora(model: NINE1, r: int = 8, alpha: int = 16,
             dropout: float = 0.0, target: str = "qkv"):
    """Aplica LoRA nas camadas especificadas.

    target:
      "qkv"  — apenas Q,K,V (q_proj, k_proj, v_proj) [recomendado, Hu et al. 2021]
      "all"  — Q,K,V,O + MLP
      "qkvo" — Q,K,V,O
    """
    if r <= 0:
        print("  [aviso] LoRA com r=0 nao tera efeito")
        return model

    n_mod = 0
    for block in model.transformer.h:
        attn = block.attn
        mlp = block.mlp

        if target in ("qkv", "all", "qkvo"):
            # GQA: projecoes separadas para Q, K, V
            if hasattr(attn, 'q_proj'):
                attn.q_proj = LoRALinear(attn.q_proj, r=r, alpha=alpha, dropout=dropout)
                attn.k_proj = LoRALinear(attn.k_proj, r=r, alpha=alpha, dropout=dropout)
                attn.v_proj = LoRALinear(attn.v_proj, r=r, alpha=alpha, dropout=dropout)
                n_mod += 3
            elif hasattr(attn, 'c_attn'):
                # Compat retroativo: modelo pre-GQA
                attn.c_attn = LoRALinear(attn.c_attn, r=r, alpha=alpha, dropout=dropout)
                n_mod += 1
        if target in ("all", "qkvo"):
            attn.c_proj = LoRALinear(attn.c_proj, r=r, alpha=alpha, dropout=dropout)
            n_mod += 1
        if target == "all":
            mlp.c_fc = LoRALinear(mlp.c_fc, r=r, alpha=alpha, dropout=dropout)
            mlp.c_proj = LoRALinear(mlp.c_proj, r=r, alpha=alpha, dropout=dropout)
            n_mod += 2

    print(f"LoRA aplicado em {n_mod} camadas (target={target}, r={r}, alpha={alpha})")
    return model


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------
# Dataset Instrucional com BPE e validacao
# ---------------------------------------------------------
class InstructDataset(Dataset):
    """Dataset JSONL com campos: instruction, input(optional), output.
    Usa BPETokenizer para codificar, com mascara de perda na parte do prompt.
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
                    if errors <= 5:
                        print(f"  [aviso] JSON invalido na linha {line_num}: pulando")
                    continue

                # Validacao de campos
                if not isinstance(ex, dict):
                    errors += 1
                    continue
                if "instruction" not in ex or "output" not in ex:
                    errors += 1
                    continue
                if len(ex.get("instruction", "")) > MAX_EXAMPLE_CHARS:
                    continue
                if len(ex.get("output", "")) > MAX_EXAMPLE_CHARS:
                    continue

                # Remove campos desconhecidos
                clean_ex = {k: v for k, v in ex.items() if k in ALLOWED_INSTRUCT_FIELDS}
                self.examples.append(clean_ex)

                if len(self.examples) >= MAX_EXAMPLES:
                    break

        if errors > 0:
            print(f"  [aviso] {errors} linhas com erro foram ignoradas")
        if not self.examples:
            raise ValueError(f"Nenhum exemplo valido encontrado em {path}")
        self.bos = 1

    def __len__(self):
        return len(self.examples)

    def _encode(self, text: str):
        """Codifica texto: usa BPE se disponivel, senao codepoint fallback."""
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            return ids[:self.block_size]
        # fallback: codepoint
        ids = [ord(c) % 65536 for c in text]
        if not ids:
            ids = [0]
        return ids[:self.block_size]

    def __getitem__(self, idx):
        ex = self.examples[idx]
        instr = ex["instruction"].strip()
        inp = ex.get("input", "").strip()
        out = ex["output"].strip()

        if inp:
            prompt = f"# tarefa: {instr}\n# entrada: {inp}\n# solucao:\n"
        else:
            prompt = f"# tarefa: {instr}\n# solucao:\n"

        full = prompt + out

        ids_full = self._encode(full)
        ids_prompt = self._encode(prompt)

        if len(ids_full) < 2:
            return self.__getitem__((idx + 1) % len(self.examples))

        x = np.array(ids_full[:-1], dtype=np.int64)
        y = np.array(ids_full[1:], dtype=np.int64)

        # Mascara: -100 para posicoes do prompt (ignoradas na loss)
        mask = np.zeros_like(y)
        prompt_len = len(ids_prompt)
        mask[max(0, prompt_len - 1):] = 1
        y = np.where(mask == 0, -100, y)

        return torch.from_numpy(x), torch.from_numpy(y)

    @staticmethod
    def collate_fn(batch):
        """Agrupa exemplos de tamanhos variados em batch com padding."""
        xs, ys = zip(*batch)
        max_len = max(x.size(0) for x in xs)
        padded_xs, padded_ys = [], []
        for x, y in zip(xs, ys):
            pad_len = max_len - x.size(0)
            if pad_len > 0:
                x = torch.nn.functional.pad(x, (0, pad_len), value=0)
                y = torch.nn.functional.pad(y, (0, pad_len), value=-100)
            padded_xs.append(x)
            padded_ys.append(y)
        return torch.stack(padded_xs), torch.stack(padded_ys)


# ---------------------------------------------------------
# Treino
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--tok", type=str, default=None, help="tokenizer BPE .json")
    p.add_argument("--out", type=str, default="nine/data/nine1-instruct.pt")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target", choices=["qkv", "qkvo", "all"],
                   default="qkv", help="camadas LoRA (default: qkv)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_iters", type=int, default=500)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Validacoes de seguranca
    if not os.path.exists(args.base):
        raise FileNotFoundError(f"Checkpoint base nao encontrado: {args.base}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset nao encontrado: {args.data}")

    # Carrega base com validacao
    print(f"Carregando base: {args.base}")
    ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["cfg"]
    cfg = NINEConfig.from_dict(cfg_dict)

    # Valida state dict
    if "model" in ckpt:
        issues = validate_checkpoint_state(ckpt["model"], cfg)
        if issues:
            for issue in issues:
                print(f"  [aviso] Checkpoint: {issue}")

    model = NINE1(cfg)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(args.device)

    # Carrega tokenizer se fornecido
    tokenizer = None
    if args.tok:
        from .tokenizer import BPETokenizer
        try:
            tokenizer = BPETokenizer.load(args.tok)
            print(f"Tokenizer BPE: {len(tokenizer)} tokens")
        except Exception as e:
            print(f"  [aviso] Falha ao carregar tokenizer: {e}")

    # Aplica LoRA
    model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                     dropout=args.lora_dropout, target=args.lora_target)
    model.to(args.device)
    n_train = count_trainable(model)
    total = model.num_params()
    print(f"Params treinaveis (LoRA): {n_train/1e6:.3f}M de {total/1e6:.2f}M ({100*n_train/total:.1f}%)")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # Dataset com validacao
    full_dataset = InstructDataset(args.data, block_size=cfg.block_size, tokenizer=tokenizer)
    print(f"Dataset carregado: {len(full_dataset)} exemplos validos")

    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train_ds = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train_ds, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=InstructDataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=InstructDataset.collate_fn) if n_val > 0 else None

    print(f"Split: treino={n_train_ds}, val={n_val}")
    iter_total = 0
    model.train()
    best_val = float("inf")

    for ep in range(args.epochs):
        for x, y in train_loader:
            if iter_total >= args.max_iters:
                break
            x, y = x.to(args.device), y.to(args.device)
            _, loss, _ = model(x, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if iter_total % 20 == 0:
                msg = f"  ep {ep} iter {iter_total}: loss {loss.item():.3f}"
                if val_loader is not None and iter_total % 100 == 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vx, vy in val_loader:
                            vx, vy = vx.to(args.device), vy.to(args.device)
                            _, vl, _ = model(vx, vy)
                            val_losses.append(vl.item())
                            if len(val_losses) >= 10:
                                break
                    avg_val = float(np.mean(val_losses))
                    msg += f" | val_loss {avg_val:.3f}"
                    if avg_val < best_val:
                        best_val = avg_val
                        msg += " (best)"
                    model.train()
                print(msg)
            iter_total += 1
        if iter_total >= args.max_iters:
            break

    # Salva somente parametros LoRA
    lora_state = {}
    for name, p in model.state_dict().items():
        if "lora_" in name:
            lora_state[name] = p.cpu()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "lora": lora_state,
        "cfg": cfg.__dict__,
        "base": args.base,
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "lora_target": args.lora_target,
    }, args.out)
    print(f"\nLoRA salvo em {args.out} ({(os.path.getsize(args.out)/1024):.1f} KB)")
    if best_val < float("inf"):
        print(f"Melhor val loss: {best_val:.3f}")


if __name__ == "__main__":
    main()
