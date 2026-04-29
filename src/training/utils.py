"""Утилиты обучения: фиксация seed, выбор устройства, подсчёт параметров."""

from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Фиксирует ГПСЧ во всех источниках случайности, которые мы используем.

    Покрывает: ``random``, ``numpy``, ``torch`` (CPU и CUDA), а также
    ``transformers.set_seed`` — последний дополнительно фиксирует свой
    внутренний state (например, для DropoutAttention).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        # transformers — опциональная зависимость для тестов утилит.
        pass


def get_device() -> torch.device:
    """Возвращает наиболее производительное доступное устройство.

    Порядок предпочтения: CUDA → Apple MPS → CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Возвращает ``(total, trainable)`` число параметров модели."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
