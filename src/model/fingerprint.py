"""Стабильный отпечаток весов backbone — проверка происхождения артефактов.

Scope-голова (``models/scope_head_v3.pt``) обучается поверх **конкретного**
stage 1 backbone и валидна только с ним: подставишь другой чекпойнт — ``h_fused``
изменится, и предсказание Scope станет случайным. Совпадение **форм** весов это
не ловит (другой stage 1 той же архитектуры грузится без ошибок). Поэтому при
обучении в артефакт кладётся отпечаток весов backbone, а предиктор сверяет его
при загрузке.

Хешируются только модули, формирующие ``h_fused`` (вход Scope-головы):
``transformer`` + ``features_mlp`` + ``fusion``. Этого достаточно, чтобы
различить baseline- и dapt-чекпойнты — в обоих эти веса отличаются.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch

#: Модули, формирующие h_fused (а значит — вход Scope-головы).
BACKBONE_PREFIXES: tuple[str, ...] = ("transformer.", "features_mlp.", "fusion.")


def backbone_fingerprint(
    state: Mapping[str, torch.Tensor],
    prefixes: tuple[str, ...] = BACKBONE_PREFIXES,
) -> str:
    """SHA-256 по байтам весов backbone — детерминированно и независимо от device.

    Args:
        state: ``state_dict`` модели (или его эквивалент). Лишние ключи вне
            ``prefixes`` игнорируются.
        prefixes: префиксы имён параметров, попадающих в отпечаток. Пустой
            кортеж/``None`` — хешировать всё.

    Returns:
        Шестнадцатеричный SHA-256 дайджест (str).

    Note:
        Тензоры приводятся к CPU + contiguous, чтобы хешировались одни и те же
        биты независимо от того, на каком устройстве загружена модель (сумма на
        CUDA и CPU могла бы отличаться в младших разрядах — поэтому хешируем
        байты весов, а не их статистики).
    """
    digest = hashlib.sha256()
    for name in sorted(state):
        if prefixes and not name.startswith(tuple(prefixes)):
            continue
        tensor = state[name]
        if not torch.is_tensor(tensor):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        arr = tensor.detach().to("cpu").contiguous()
        try:
            digest.update(arr.numpy().tobytes())
        except (TypeError, RuntimeError):
            # dtype без поддержки numpy (например bfloat16) → через float32.
            digest.update(arr.to(torch.float32).numpy().tobytes())
    return digest.hexdigest()


__all__ = ["backbone_fingerprint", "BACKBONE_PREFIXES"]
