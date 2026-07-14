"""
NINE-1: Transformer decoder-only minimal, implementado do zero usando PyTorch.
Arquitetura:
- Embeddings de tokens + positional encodings (learned)
- N blocos: pre-norm Transformer
  - Multi-head causal self-attention
  - MLP (Linear -> GELU -> Linear) com expansao 4x
- Final: RMSNorm + Linear (tied weights com embedding)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NINEConfig:
    vocab_size: int = 512
    block_size: int = 256          # maximo de contexto (tokens)
    n_layer: int = 6               # blocos transformer
    n_head: int = 6                # cabecas de atencao
    n_embd: int = 384              # dimensao do embedding
    dropout: float = 0.0
    bias: bool = False             # sem bias em Linear (nanoGPT style)
    mlp_ratio: float = 4.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm (mais estavel que LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class CausalSelfAttention(nn.Module):
    """Atencao causal multi-cabeca com projecao QKV."""

    def __init__(self, cfg: NINEConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        # Projecao combinada Q,K,V
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # Projecao de saida
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        # Mascara causal: register_buffer para mover com o modelo
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape para (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Atencao com Flash-like (PyTorch >=2.0 tem scaled_dot_product_attention)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: NINEConfig):
        super().__init__()
        hidden = int(cfg.n_embd * cfg.mlp_ratio)
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: NINEConfig):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NINE1(nn.Module):
    """Modelo NINE-1: decoder Transformer pequeno."""

    def __init__(self, cfg: NINEConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),  # token embedding
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),   # positional
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=RMSNorm(cfg.n_embd),
        ))
        # LM head (tied com embedding -> economiza parametros)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Inicializacao
        self.apply(self._init_weights)
        # Pequeno scale na projecao de saida (mais estavel)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"contexto {T} > block_size {self.cfg.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)            # (B, T, C)
        pos_emb = self.transformer.wpe(pos)             # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss
        else:
            # Otimizacao: calcular logits apenas do ultimo token
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
        """Gera tokens um a um (com top-k e/ou nucleus sampling)."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = sorted_logits.softmax(-1).cumsum(-1)
                sorted_logits[cum > top_p] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = logits.softmax(-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def tiny_config(vocab_size: int = 512, block_size: int = 256) -> NINEConfig:
    """Configuracao 'tiny' para um modelo beem leve (~10M params)."""
    return NINEConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False,
    )


if __name__ == "__main__":
    cfg = tiny_config()
    m = NINE1(cfg)
    print(f"Parametros: {m.num_params()/1e6:.2f}M")
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = m(x, y)
    print(f"logits: {logits.shape}, loss: {loss.item():.3f}")
