"""Кодирование CWE-идентификаторов в целочисленные индексы.

Зарезервированные индексы:
    0 — <PAD>  (используется в Embedding-слое как padding_idx);
    1 — <UNK>  (для CWE, не встречавшихся при обучении).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class CWEEncoder:
    """Словарь {CWE-ID → индекс} с поддержкой <PAD>/<UNK>."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_INDEX = 0
    UNK_INDEX = 1

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {self.PAD_TOKEN: self.PAD_INDEX, self.UNK_TOKEN: self.UNK_INDEX}

    # ----------------------------------------------------------------- public

    def fit(self, cwe_list: Iterable[str | None]) -> "CWEEncoder":
        """Строит словарь по уникальным значениям из обучающей выборки."""
        unique = sorted({cwe for cwe in cwe_list if cwe})
        self._vocab = {self.PAD_TOKEN: self.PAD_INDEX, self.UNK_TOKEN: self.UNK_INDEX}
        for i, cwe in enumerate(unique, start=2):
            self._vocab[cwe] = i
        return self

    def transform(self, cwe_id: str | None) -> int:
        if not cwe_id:
            return self.UNK_INDEX
        return self._vocab.get(cwe_id, self.UNK_INDEX)

    def transform_batch(self, cwe_ids: Iterable[str | None]) -> list[int]:
        return [self.transform(c) for c in cwe_ids]

    def __len__(self) -> int:
        return len(self._vocab)

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    # ------------------------------------------------------------------ I/O

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self._vocab, fh, ensure_ascii=False, indent=2)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "CWEEncoder":
        with Path(path).open("r", encoding="utf-8") as fh:
            vocab = json.load(fh)
        if cls.PAD_TOKEN not in vocab or cls.UNK_TOKEN not in vocab:
            raise ValueError(f"Файл словаря CWE не содержит спец-токены: {path}")
        encoder = cls()
        encoder._vocab = vocab
        return encoder


__all__ = ["CWEEncoder"]
