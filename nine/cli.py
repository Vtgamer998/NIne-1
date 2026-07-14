"""
CLI em PT-BR para o NINE-1.
Uso:
    nine "escreva uma funcao que retorna fatorial"
    nine --instruct "como faco um loop em python"
"""

from __future__ import annotations
import argparse
import sys

import torch

from .model import NINE1, NINEConfig


PROMPT_TEMPLATES = {
    "base": "{prompt}\n",
    "instruct": (
        "# tarefa: {prompt}\n"
        "# solucao:\n"
    ),
}


def load_model(checkpoint: str, device: str = "cpu") -> NINE1:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = NINEConfig(**ckpt["cfg"])
    model = NINE1(cfg)
    state = ckpt["model"] if "model" in ckpt else ckpt
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        pass
    model.to(device)
    model.eval()
    return model


def encode(text: str, block_size: int) -> torch.Tensor:
    ids = [ord(c) % 65536 for c in text[:block_size]]
    return torch.tensor([ids], dtype=torch.long)


def decode(ids) -> str:
    out = []
    for i in ids[0].tolist():
        if i <= 0 or i > 0x10FFFF:
            continue
        try:
            out.append(chr(i))
        except Exception:
            pass
    return "".join(out)


def main():
    p = argparse.ArgumentParser(
        prog="nine",
        description="NINE-1 - IA de programacao em PT-BR",
    )
    p.add_argument("prompt", type=str, nargs="?", help="prompt em linguagem natural")
    p.add_argument("--ckpt", type=str, default="nine/data/nine1-base.pt")
    p.add_argument("--mode", choices=["base", "instruct"], default="base")
    p.add_argument("--tokens", type=int, default=120, help="tokens a gerar")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    prompt = args.prompt or "def fala_oi():\n    "
    template = PROMPT_TEMPLATES[args.mode].format(prompt=prompt)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    model = load_model(args.ckpt, device=device)
    ids = encode(template, model.cfg.block_size)
    ids = ids.to(device)
    out = model.generate(ids, max_new_tokens=args.tokens, temperature=args.temp, top_k=args.top_k)
    text = decode(out)
    sys.stdout.write("\n--- NINE-1 ---\n")
    sys.stdout.write(text[len(template):])
    sys.stdout.write("\n--------------\n")


if __name__ == "__main__":
    main()
