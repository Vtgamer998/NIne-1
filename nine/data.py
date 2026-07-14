"""
Dataset loader para o NINE-1.
Le arquivos de texto/codigo no formato binario (memmap) ou texto puro,
e cria batches aleatorios de (input, target) com deslocamento de 1 token.
"""

from __future__ import annotations
import os
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Memoria-map para arquivos binarios (.bin) gerados por prep_data.py,
    ou fallback para texto puro.
    """

    def __init__(self, data_path: str, block_size: int, vocab_size: Optional[int] = None):
        self.block_size = block_size
        if data_path.endswith(".bin"):
            self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                txt = f.read()
            self.data = np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        self.length = len(self.data) - block_size - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = random.randint(0, max(0, self.length))
        chunk = np.array(self.data[i : i + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: str = "cpu"):
    """Amostra um batch aleatorio do dataset binario."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix
    ])
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
