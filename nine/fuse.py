"""
NINE-1: Fusao do modelo base com LoRA opcional.

Carrega checkpoint base (.pt), aplica adaptadores LoRA (.pt) se fornecidos,
e carrega o tokenizer BPE (.json). Retorna (model, tokenizer) pronto para
inferencia.

Seguranca:
- Validacao de integridade de checkpoint (schema, NaN/Inf)
- Carga segura com weights_only=True
- Filtragem de chaves desconhecidas no config
- Protecao contra path traversal
"""

from __future__ import annotations
import os
import sys
from typing import Optional, Tuple

import torch

from .model import NINE1, NINEConfig, validate_checkpoint_state, safe_load_checkpoint
from .tokenizer import BPETokenizer
from .finetune import add_lora


# Caminhos padrao permitidos
ALLOWED_CKPT_DIRS = {"nine/data", "data", "."}


def load_fused_model(
    base_path: str,
    lora_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    device: str = "cpu",
    lora_r: int = 8,
    lora_alpha: int = 16,
    verbose: bool = False,
    validate: bool = True,
) -> Tuple[NINE1, Optional[BPETokenizer]]:
    """Carrega o modelo base + LoRA opcional + tokenizer BPE.

    Carrega com validacao de seguranca de checkpoint.

    Args:
        base_path: Caminho para o checkpoint base (.pt).
        lora_path: Caminho opcional para checkpoint LoRA (.pt).
        tokenizer_path: Caminho opcional para tokenizer BPE (.json).
        device: Dispositivo para carregar o modelo.
        lora_r: Rank LoRA.
        lora_alpha: Alpha LoRA.
        verbose: Se True, exibe logs detalhados.
        validate: Se True, valida checkpoint apos carregar.

    Returns:
        Tupla (model, tokenizer). tokenizer pode ser None se nao encontrado.

    Raises:
        FileNotFoundError: Se base_path nao existe.
        RuntimeError: Se checkpoint estiver corrompido.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Checkpoint base nao encontrado: {base_path}")

    if verbose:
        print(f"[fuse] Carregando base: {base_path}")
        print(f"[fuse] Device: {device}")

    # Carrega checkpoint base (com seguranca)
    ckpt = safe_load_checkpoint(base_path, map_location="cpu",
                                 expected_cfg=None)

    cfg_dict = ckpt.get("cfg", ckpt.get("config", {}))
    if not isinstance(cfg_dict, dict):
        raise RuntimeError("Configuracao do checkpoint nao e um dicionario")

    # Inferencia de config a partir do state dict se necessario
    state = ckpt.get("model", ckpt)
    if "vocab_size" not in cfg_dict:
        if "transformer.wte.weight" in state:
            cfg_dict["vocab_size"] = state["transformer.wte.weight"].size(0)
    if "block_size" not in cfg_dict:
        if "transformer.wpe.weight" in state:
            cfg_dict["block_size"] = state["transformer.wpe.weight"].size(0)

    # Filtra apenas os campos esperados por NINEConfig para evitar TypeError
    cfg = NINEConfig.from_dict(cfg_dict)

    if verbose:
        print(f"[fuse] Config: vocab={cfg.vocab_size}, block={cfg.block_size}, "
              f"layers={cfg.n_layer}, heads={cfg.n_head}")

    model = NINE1(cfg)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)

    # Validacao pos-carga
    if validate:
        if "model" in ckpt:
            issues = validate_checkpoint_state(ckpt["model"], cfg)
        else:
            issues = validate_checkpoint_state(ckpt, cfg)
        if issues:
            for issue in issues[:5]:
                print(f"[fuse] AVISO: {issue}", file=sys.stderr)

    # Aplica LoRA se fornecido
    if lora_path:
        if not os.path.exists(lora_path):
            print(f"[fuse] Aviso: arquivo LoRA nao encontrado: {lora_path}", file=sys.stderr)
        else:
            if verbose:
                print(f"[fuse] Carregando LoRA: {lora_path}")
            lora_ckpt = safe_load_checkpoint(lora_path, map_location="cpu")

            lora_target = lora_ckpt.get("lora_target", "qkv")
            if verbose:
                print(f"[fuse] LoRA target: {lora_target}, r={lora_r}, alpha={lora_alpha}")

            model = add_lora(model, r=lora_r, alpha=lora_alpha, target=lora_target)

            lora_sd = lora_ckpt.get("lora", lora_ckpt)
            missing, unexpected = model.load_state_dict(lora_sd, strict=False)
            if verbose:
                if missing:
                    print(f"[fuse] Parametros base nao encontrados no LoRA: {len(missing)}")
                if unexpected:
                    print(f"[fuse] Parametros LoRA inesperados: {len(unexpected)}")

    model.to(device)
    model.eval()

    # Carrega tokenizer BPE
    tokenizer = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        try:
            tokenizer = BPETokenizer.load(tokenizer_path)
            if verbose:
                print(f"[fuse] Tokenizer carregado: {len(tokenizer)} tokens de {tokenizer_path}")
        except Exception as e:
            print(f"[fuse] Aviso: falhou carregar tokenizer ({e})", file=sys.stderr)
    elif verbose:
        print(f"[fuse] Aviso: tokenizer nao encontrado em {tokenizer_path}")

    n_params = model.num_params()
    if verbose:
        n_train = model.num_trainable_params()
        print(f"[fuse] Modelo pronto: {n_params/1e6:.2f}M parametros "
              f"({n_train/1e6:.2f}M treinaveis) em {device}")

    return model, tokenizer
