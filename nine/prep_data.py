"""
Pre-processamento de corpus para NINE-1.
- Le varios arquivos de codigo (preferencialmente Python)
- Junta, treina o tokenizer BPE do zero ou usa o pre-treinado
- Salva em .bin (uint16) para treino rapido em GPU

Seguranca:
- Path traversal protection
- Validacao de extensoes de arquivo
- Limite de tamanho do corpus
- Filtragem de binarios
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


# ---------------------------------------------------------------------------
# Constantes de seguranca
# ---------------------------------------------------------------------------

# Extensoes de arquivo permitidas para coleta
ALLOWED_EXTENSIONS = frozenset({".py", ".txt", ".md", ".js", ".ts", ".java",
                                ".cpp", ".c", ".cc", ".h", ".hpp", ".rs",
                                ".go", ".rb", ".php", ".swift", ".kt", ".scala"})

# Tamanho maximo de arquivo (50 MB)
MAX_FILE_BYTES = 50 * 1024 * 1024
# Tamanho maximo do corpus final (500 MB)
MAX_TOTAL_BYTES = 500 * 1024 * 1024
# Caracteres suspeitos que indicam arquivo binario
BINARY_PATTERNS = [b"\\x00", b"\\xff\\xfe", b"\\xfe\\xff"]


# ---------------------------------------------------------------------------
# Funcoes seguras
# ---------------------------------------------------------------------------

def is_safe_path(path: str) -> bool:
    """Verifica se o caminho nao tem path traversal."""
    return ".." not in path.split(os.sep) and not os.path.isabs(path)


def is_text_content(content: bytes) -> bool:
    """Verifica heuristicamente se conteudo parece texto (primeiros 8KB).

    Args:
        content: Conteudo bytes a verificar.

    Returns:
        True se parece texto UTF-8 valido.
    """
    try:
        content[:8192].decode("utf-8")
        return True
    except (UnicodeDecodeError, IOError):
        return False


def collect_files(paths: List[str]) -> List[str]:
    """Coleta arquivos de codigo com validacao de seguranca.

    Args:
        paths: Lista de caminhos (arquivos ou diretorios).

    Returns:
        Lista de caminhos validados.
    """
    out: List[str] = []
    for p in paths:
        if not is_safe_path(p):
            print(f"  [aviso] Caminho ignorado (path traversal): {p}", file=sys.stderr)
            continue
        if os.path.isfile(p):
            ext = os.path.splitext(p)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                out.append(p)
            else:
                print(f"  [aviso] Extensao ignorada: {p}", file=sys.stderr)
        elif os.path.isdir(p):
            for ext in ALLOWED_EXTENSIONS:
                for found in glob.glob(os.path.join(p, "**", f"*{ext}"), recursive=True):
                    if is_safe_path(found):
                        out.append(found)
        else:
            print(f"  [aviso] Caminho nao encontrado: {p}", file=sys.stderr)

    # Remove duplicatas mantendo ordem
    seen: set = set()
    final: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            final.append(p)
    return final


def read_file_safe(path: str) -> str:
    """Le arquivo com validacoes de seguranca.

    Args:
        path: Caminho do arquivo.

    Returns:
        Conteudo do arquivo como string.

    Raises:
        ValueError: Se arquivo for muito grande ou binario.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    if not is_safe_path(path):
        raise ValueError(f"Path traversal detectado: {path}")

    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES:
        raise ValueError(f"Arquivo muito grande: {size/1024/1024:.1f} MB (max {MAX_FILE_BYTES/1024/1024:.0f} MB)")

    with open(path, "rb") as f:
        content = f.read()

    if not is_text_content(content):
        raise ValueError(f"Arquivo parece ser binario: {path}")

    return content.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # Coleta arquivos com seguranca
    files = collect_files(args.paths)
    print(f"Encontrados {len(files)} arquivos (apos filtros de seguranca)")

    # Le e valida cada arquivo
    text_parts: List[str] = []
    total_chars = 0
    for fp in files:
        try:
            txt = read_file_safe(fp)
            text_parts.append(txt)
            total_chars += len(txt)
            if total_chars >= args.max_chars:
                break
        except (ValueError, FileNotFoundError) as e:
            print(f"  pulou {fp}: {e}")
            continue

    if not text_parts:
        print("[erro] Nenhum arquivo valido encontrado!")
        sys.exit(1)

    text = "\n".join(text_parts)
    print(f"Total caracteres: {len(text):,}")

    # Salva corpus texto
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Corpus texto salvo em {args.out}")

    # Treina BPE se pedido
    if args.train_bpe:
        if len(text) < 1000:
            print("[aviso] Corpus muito pequeno para treinar BPE (< 1000 chars)")
        else:
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
            arr = ids
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
    bin_size = os.path.getsize(args.bin_out)
    print(f"Salvo {args.bin_out} ({bin_size:,} bytes, {len(ids):,} tokens)")


if __name__ == "__main__":
    main()
