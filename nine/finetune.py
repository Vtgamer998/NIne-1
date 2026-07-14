"""
Fine-tuning do NINE-1 base (pre-treinado) usando LoRA (PEFT).
Carrega um checkpoint base do NINE-1, adiciona adaptadores LoRA nas matrizes Q,V,
e treina em um dataset instrucional.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .model import NINE1, NINEConfig


# ---------------------------------------------------------
# LoRA minimal: implementacao propria para nao depender de peft
# ---------------------------------------------------------
class LoRALinear(torch.nn.Module):
    """Linear + LoRA adapter (low-rank)."""

    def __init__(self, base: torch.nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        # Base congelada
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        in_f = base.in_features
        out_f = base.out_features
        # Matrizes A e B (low-rank)
        self.lora_A = torch.nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = torch.nn.Parameter(torch.zeros(out_f, r))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + lora_out * self.scaling


def add_lora(model: NINE1, r: int = 8, alpha: int = 16, dropout: float = 0.0):
    """Substitui as projecoes Q,K,V,O e projecoes MLP por LoRALinear."""
    for block in model.transformer.h:
        attn = block.attn
        attn.c_attn = LoRALinear(attn.c_attn, r=r, alpha=alpha, dropout=dropout)
        attn.c_proj = LoRALinear(attn.c_proj, r=r, alpha=alpha, dropout=dropout)
        mlp = block.mlp
        mlp.c_fc = LoRALinear(mlp.c_fc, r=r, alpha=alpha, dropout=dropout)
        mlp.c_proj = LoRALinear(mlp.c_proj, r=r, alpha=alpha, dropout=dropout)
    return model


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------
# Dataset Instrucional
# ---------------------------------------------------------
class InstructDataset(Dataset):
    """Dataset JSONL com campos: instruction, input(optional), output."""

    def __init__(self, path: str, block_size: int, sentinel_token: int = 3):
        self.block_size = block_size
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))
        self.bos = 1
        self.eos = 2
        self.sentinel_token = sentinel_token

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        instr = ex["instruction"].strip()
        inp = ex.get("input", "").strip()
        out = ex["output"].strip()
        # Formato simples supervisionado
        if inp:
            prompt = f"# tarefa: {instr}\n# entrada: {inp}\n# solucao:\n"
        else:
            prompt = f"# tarefa: {instr}\n# solucao:\n"
        full = prompt + out

        # Tokenizacao naive (id = codepoint % 65536) — mesmo formato do corpus
        ids_full = [ord(c) % 65536 for c in full][:self.block_size]
        ids_prompt = [ord(c) % 65536 for c in prompt][:self.block_size]
        x = np.array(ids_full[:-1], dtype=np.int64)
        y = np.array(ids_full[1:], dtype=np.int64)
        # Mascara loss apenas para a parte de saida
        mask = np.zeros_like(y)
        mask[len(ids_prompt) - 1 :] = 1
        y = np.where(mask == 0, -100, y)
        return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------
# Treino
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, default="nine/data/nine1-instruct.pt")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_iters", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Carrega base
    print(f"Carregando base: {args.base}")
    ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = NINEConfig(**ckpt["cfg"])
    model = NINE1(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)

    # Aplica LoRA
    model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    model.to(args.device)
    n_train = count_trainable(model)
    print(f"Parametros treinaveis (LoRA): {n_train/1e6:.3f}M de {model.num_params()/1e6:.2f}M")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    dataset = InstructDataset(args.data, block_size=cfg.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Dataset instrucional: {len(dataset)} exemplos")
    iter_total = 0
    model.train()
    for ep in range(args.epochs):
        for x, y in loader:
            if iter_total >= args.max_iters:
                break
            x, y = x.to(args.device), y.to(args.device)
            _, loss = model(x, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if iter_total % 20 == 0:
                print(f"  ep {ep} iter {iter_total}: loss {loss.item():.3f}")
            iter_total += 1
        if iter_total >= args.max_iters:
            break

    # Salva somente os parametros LoRA (light-weight)
    lora_state = {}
    for name, p in model.state_dict().items():
        if "lora_" in name:
            lora_state[name] = p.cpu()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "lora": lora_state,
        "cfg": cfg.__dict__,
        "base": args.base,
    }, args.out)
    print(f"\nLoRA salvo em {args.out}")


if __name__ == "__main__":
    main()
