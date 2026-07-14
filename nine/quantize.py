"""
Quantizacao/exportacao do NINE-1.

V1 (neste arquivo): workflow simples de quantizacao INT8 (QAT nao exigido; aqui
salvamos checkpoint em float16 e tambem em int8 via torch.ao ou numpy puro),
para reduzir tamanho do arquivo final.  Para rodar com llama.cpp seria necessario
um conversor especifico para 'GGUF' (fora do escopo deste MVP: ver README).

Implementacoes:
- to_half(): checkpoint .pt -> .pth (float16), tamanho ~2x menor.
- to_int8(): checkpoint .pt -> .q8.bin (int8 por linha), tamanho ~4x menor.
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import torch


def to_half(in_path: str, out_path: str):
    state = torch.load(in_path, map_location="cpu", weights_only=False)
    if "model" in state:
        sd = state["model"]
    else:
        sd = state
    new = {k: v.half() for k, v in sd.items()}
    torch.save({**state, "model": new} if "model" in state else new, out_path)
    print(f"  half salvo em {out_path}")


def to_int8(in_path: str, out_path: str, cfg: dict):
    """Quantizacao int8 por linha (linear, sem outlier handling)."""
    state = torch.load(in_path, map_location="cpu", weights_only=False)
    sd = state["model"] if "model" in state else state
    out = []
    meta = []
    for name, p in sd.items():
        arr = p.float().numpy()
        if arr.ndim < 2:
            # nao quantiza 1D (bias/Norm)
            out.append(arr.tobytes())
            meta.append((name, arr.shape, "raw"))
            continue
        # quant simetrico por linha
        per_row_max = np.max(np.abs(arr), axis=1, keepdims=True).clip(min=1e-8)
        q = np.clip(np.round(arr / per_row_max * 127), -127, 127).astype(np.int8)
        out.append(q.tobytes())
        out.append(per_row_max.astype(np.float32).tobytes())
        meta.append((name, q.shape, "int8"))
        meta.append((name + ".scale", per_row_max.shape, "f32"))

    blob = b"".join(out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(blob)
    with open(out_path + ".meta", "w") as f:
        import json
        json.dump({"meta": [(n, list(s), t) for n, s, t in meta], "cfg": cfg}, f)
    print(f"  int8 salvo em {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("in_path", type=str)
    p.add_argument("--mode", type=str, choices=["half", "int8"], default="half")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.out is None:
        out = args.in_path.replace(".pt", ".p" + ("th" if args.mode == "half" else "q8.bin"))
    else:
        out = args.out
    ckpt = torch.load(args.in_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    if args.mode == "half":
        to_half(args.in_path, out)
    else:
        to_int8(args.in_path, out, cfg)


if __name__ == "__main__":
    main()
