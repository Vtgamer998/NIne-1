"""
Pre-processamento de corpus para NINE-1.
- Le varios arquivos de codigo (preferencialmente Python)
- Junta, treina o tokenizer BPE do zero ou usa o pre-treinado
- Salva em .bin (uint16) para treino rapido em GPU
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
from typing import List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .tokenizer import BPETokenizer


def collect_files(paths: List[str], exts=(".py", ".txt", ".md", ".js", ".ts", ".java", ".cpp", ".c", ".cc", ".go")) -> List[str]:
    out = []
    for p in paths:
        if os.path.isfile(p):
            out.append(p)
        else:
            for ext in exts:
                out.extend(glob.glob(os.path.join(p, "**", f"*{ext}"), recursive=True))
    seen, final = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            final.append(p)
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths", nargs="+", required=True, help="arquivos ou diretorios com codigo")
    p.add_argument("--out", type=str, default="nine/data/corpus.txt")
    p.add_argument("--max_chars", type=int, default=50_000_000, help="teto de chars")
    p.add_argument("--train_bpe", action="store_true")
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--tok_out", type=str, default="nine/data/nine1-tok.json")
    p.add_argument("--bin_out", type=str, default="nine/data/corpus.bin")
    args = p.parse_args()

    files = collect_files(args.paths)
    print(f"Encontrados {len(files)} arquivos")

    # Junta tudo
    text_parts = []
    total = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            text_parts.append(txt)
            total += len(txt)
            if total >= args.max_chars:
                break
        except Exception as e:
            print(f"  pulou {fp}: {e}")
    text = "\n".join(text_parts)
    print(f"Total caracteres: {len(text)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    # Treina BPE se pedido
    if args.train_bpe:
        print("Treinando tokenizer BPE (do zero)...")
        bt = BPETokenizer(vocab_size=args.vocab)
        bt.train(text, verbose=True)
        bt.save(args.tok_out)
        print(f"Tokenizer salvo em {args.tok_out} ({len(bt)} tokens)")

    # Tokeniza para .bin (uint16)
    print("Codificando corpus em .bin...")
    if args.train_bpe:
        bt = BPETokenizer.load(args.tok_out)
        ids = bt.encode(text)
        if HAS_NUMPY:
            arr = np.array(ids, dtype=np.uint16)
        else:
            arr = ids  # list of int
    else:
        # Encoding naive (fallback): cada codepoint % 65536
        ids = [ord(c) % 65536 for c in text]
        arr = ids

    if HAS_NUMPY and isinstance(arr, np.ndarray):
        arr.tofile(args.bin_out)
    else:
        import struct
        with open(args.bin_out, "wb") as f:
            f.write(struct.pack(f"<{len(arr)}H", *arr))
    print(f"Salvo {args.bin_out} com {len(arr) if hasattr(arr,'__len__') else 0} tokens")


if __name__ == "__main__":
    main()
