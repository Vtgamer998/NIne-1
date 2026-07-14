"""
Tokenizer BPE (Byte Pair Encoding) implementado do zero, em Python puro.
Inspirado em: Sennrich et al. 2016, GPT-2 tokenizer, nanoGPT (Karpathy).
Suporta vocabulario base de 256 chars (latin-1) + merges aprendidos.
"""

from __future__ import annotations
import json
import os
import re
from collections import Counter
from typing import List, Tuple, Dict


PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
END_OF_WORD = "</w>"

PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-zÀ-ÿ]+| ?[0-9]+| ?[^\w\s]+|\s+(?!\S)|\s+"""


def get_pairs(word: Tuple[str, ...]) -> Counter:
    """Retorna os pares adjacentes de uma palavra."""
    pairs = Counter()
    for i in range(len(word) - 1):
        pairs[(word[i], word[i + 1])] += 1
    return pairs


class BPETokenizer:
    """
    Tokenizer BPE minimalista:
    - salva/carrega merges + vocab em JSON
    - encode/decode em PT (UTF-8 mapeado via chars latin-1)
    - tokens sao STRINGS (latin-1) -> JSON-serializavel
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        pat: str = PAT,
        special_tokens: List[str] = None,
    ):
        self.vocab_size = vocab_size
        self.pat = pat
        self.special_tokens = special_tokens or SPECIAL_TOKENS
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        self.regex = re.compile(pat)

    # ---------- Treino ----------
    def train(self, text: str, verbose: bool = False):
        if not text:
            raise ValueError("Texto vazio para treino.")

        vocab: Dict[int, str] = {i: chr(i) for i in range(256)}
        if END_OF_WORD not in [v for v in vocab.values()]:
            vocab_next = 256
        else:
            vocab_next = max(vocab.keys()) + 1
        vocab[vocab_next] = END_OF_WORD
        next_id = vocab_next + 1

        words_raw = self.regex.findall(text)
        words: List[Tuple[str, ...]] = []
        for w in words_raw:
            if not w:
                continue
            # itens sao chars latin-1 (chars 0-255); acentos vira multi-char
            chars = [c if ord(c) < 256 else "?" for c in w]
            stripped = w.rstrip() if w[-1].isspace() else w
            sp = w[-1].isspace() if w else False
            if w.strip() != w and sp:
                inner = [c if ord(c) < 256 else "?" for c in w.rstrip()]
                rest = [c if ord(c) < 256 else "?" for c in w[len(w.rstrip()):]]
                words.append(tuple(inner) + (END_OF_WORD,) + tuple(rest))
            elif w.strip() == w and w.strip():
                inner = [c if ord(c) < 256 else "?" for c in w]
                words.append(tuple(inner) + (END_OF_WORD,))
            else:
                inner = [c if ord(c) < 256 else "?" for c in w]
                if inner:
                    words.append(tuple(inner))

        target_merges = self.vocab_size - len(vocab) - len(self.special_tokens)
        if target_merges < 1:
            raise ValueError("vocab_size muito pequeno.")

        word_freq = Counter(words)

        for i in range(target_merges):
            pairs = Counter()
            for word, freq in word_freq.items():
                pairs.update(get_pairs(word))
            if not pairs:
                break
            pair = pairs.most_common(1)[0][0]
            tok_str = pair[0] + pair[1]
            vocab[next_id] = tok_str
            self.merges[pair] = tok_str
            next_id += 1
            new_word_freq = Counter()
            for word, freq in word_freq.items():
                new_word = self._apply_merge(word, pair, tok_str)
                new_word_freq[new_word] += freq
            word_freq = new_word_freq
            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i+1}/{target_merges}: {pair} -> {tok_str!r}")

        for st in self.special_tokens:
            if st not in vocab.values():
                vocab[next_id] = st
                next_id += 1

        self.vocab = {k: v for k, v in vocab.items() if k < self.vocab_size}
        self.token_to_id = {v: k for k, v in self.vocab.items()}
        return self

    @staticmethod
    def _apply_merge(word: Tuple[str, ...], pair: Tuple[str, str], new_token: str):
        out = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(new_token)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    # ---------- Encode ----------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if not self.vocab:
            raise RuntimeError("Tokenizer nao foi treinado ou carregado.")
        ids: List[int] = []
        if add_bos and BOS_TOKEN in self.token_to_id:
            ids.append(self.token_to_id[BOS_TOKEN])

        for chunk in self.regex.findall(text):
            if not chunk:
                continue
            tokens = self._tokenize_chunk(chunk)
            ids.extend(self.token_to_id[t] for t in tokens if t in self.token_to_id)

        if add_eos and EOS_TOKEN in self.token_to_id:
            ids.append(self.token_to_id[EOS_TOKEN])
        return ids

    def _tokenize_chunk(self, text: str) -> List[str]:
        # chars latin-1: substitui acentos por chars de byte ordinal > 255 por "?"
        # (para simplicacao: preservamos a estrtura mas ignoramos char-by-char)
        chars = [c if ord(c) < 256 else "?" for c in text]
        if "".join(chars).strip() != "".join(chars) and chars[-1].isspace():
            inner = list(chars)
            idx = len(inner)
            while idx > 0 and inner[idx - 1].isspace():
                idx -= 1
            word = tuple(inner[:idx]) + (END_OF_WORD,) + tuple(inner[idx:])
        elif "".join(chars).strip():
            word = tuple(chars) + (END_OF_WORD,)
        else:
            word = tuple(chars)

        for pair in sorted(self.merges.keys(), key=lambda p: self.token_to_id.get(self.merges[p], 0)):
            word = self._apply_merge(word, pair, self.merges[pair])
        return [tok for tok in word if tok]

    # ---------- Decode ----------
    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            tok = self.vocab.get(i)
            if tok is None:
                continue
            if tok in self.special_tokens:
                if tok == EOS_TOKEN:
                    break
                continue
            out.append(tok.replace(END_OF_WORD, ""))
        return "".join(out)

    # ---------- Persistencia ----------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        serial_merges = {f"{p[0]}\u241F{p[1]}": t for p, t in self.merges.items()}
        payload = {
            "vocab_size": self.vocab_size,
            "merges": serial_merges,
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
            "pat": self.pat,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls(vocab_size=payload["vocab_size"], pat=payload["pat"], special_tokens=payload["special_tokens"])
        tok.vocab = {int(k): v for k, v in payload["vocab"].items()}
        tok.token_to_id = {v: k for k, v in tok.vocab.items()}
        tok.merges = {}
        for key, token in payload["merges"].items():
            sep = "\u241F"
            assert sep in key, f"formato de merges invalido: {key!r}"
            a, b = key.split(sep, 1)
            tok.merges[(a, b)] = token
        return tok

    def __len__(self):
        return len(self.vocab)


if __name__ == "__main__":
    sample = "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"
    bt = BPETokenizer(vocab_size=512)
    bt.train(sample * 100, verbose=True)
    print(f"Vocab: {len(bt)} tokens")
    ids = bt.encode(sample)
    print("ids:", ids[:40])
    print("decode:", repr(bt.decode(ids)))
    bt.save("nine/data/_demo_tok.json")
    os.unlink("nine/data/_demo_tok.json")
