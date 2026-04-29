"""torch Dataset для двухэтапного обучения mBERT.

Каждый ``__getitem__`` возвращает dict вида::

    {
        "input_ids":      LongTensor[max_length],
        "attention_mask": LongTensor[max_length],
        "cwe_idx":        LongTensor scalar,
        "features":       FloatTensor[3]      ([epss, kev, exploit]),
        "labels":         dict[str, LongTensor scalar]   # по одной голове
    }

Структуру меток (``labels``) задаёт версия CVSS:
    - ``version='v3'`` → 8 stage 1 голов (V3_METRIC_ORDER);
    - ``version='v4'`` → 12 stage 2 голов (V4_METRIC_ORDER).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import pandas as pd
import torch
from torch.utils.data import Dataset

from .cvss_vector_parser import (
    V3_LABEL_MAPS,
    V3_METRIC_ORDER,
    V4_LABEL_MAPS,
    V4_METRIC_ORDER,
    parse_v3_vector,
    parse_v4_vector,
    vector_to_labels,
)
from .cwe_encoder import CWEEncoder
from .features_encoder import FeaturesEncoder
from .text_processor import TextProcessor

logger = logging.getLogger(__name__)


class _TokenizerProtocol(Protocol):
    def tokenize(self, text: str, max_length: int | None = ...) -> dict[str, list[int]]: ...


class CVSSDataset(Dataset):
    """torch.utils.data.Dataset для корпуса уязвимостей."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: _TokenizerProtocol,
        cwe_encoder: CWEEncoder,
        features_encoder: FeaturesEncoder,
        version: str = "v4",
        text_processor: TextProcessor | None = None,
        max_length: int = 512,
    ) -> None:
        if version not in ("v3", "v4"):
            raise ValueError(f"version must be 'v3' or 'v4', got {version!r}")

        self._df = dataframe.reset_index(drop=True)
        self._tokenizer = tokenizer
        self._cwe_encoder = cwe_encoder
        self._features_encoder = features_encoder
        self._text_processor = text_processor or TextProcessor()
        self._version = version
        self._max_length = max_length

        if version == "v4":
            self._parser = parse_v4_vector
            self._metric_order = V4_METRIC_ORDER
            self._label_maps = V4_LABEL_MAPS
            self._vector_column = "cvss_v4_vector"
        else:
            self._parser = parse_v3_vector
            self._metric_order = V3_METRIC_ORDER
            self._label_maps = V3_LABEL_MAPS
            self._vector_column = "cvss_v3_vector"

    # ----------------------------------------------------------------- public

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df.iloc[idx]

        # Текст: t = description + [SEP] + cwe_name → токенизация
        text = self._text_processor.prepare_text(
            row.get("d_ru"),
            row.get("d_en"),
            row.get("cwe_name"),
        )
        encoding = self._tokenizer.tokenize(text, max_length=self._max_length)
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)

        # CWE-индекс для embedding-слоя
        cwe_idx = torch.tensor(
            self._cwe_encoder.transform(row.get("cwe_id")),
            dtype=torch.long,
        )

        # Числовые признаки
        features_arr = self._features_encoder.encode(
            epss=row.get("epss"),
            kev=row.get("kev"),
            exploit=row.get("exploit"),
        )
        features = torch.from_numpy(features_arr)

        # Метки CVSS — по одной на каждую голову
        vector_str = row.get(self._vector_column)
        parsed = self._parser(vector_str)
        labels_array = vector_to_labels(parsed, self._metric_order, self._label_maps)
        labels: dict[str, torch.Tensor] = {
            metric: torch.tensor(int(labels_array[i]), dtype=torch.long)
            for i, metric in enumerate(self._metric_order)
        }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cwe_idx": cwe_idx,
            "features": features,
            "labels": labels,
        }

    # -------------------------------------------------------------- accessors

    @property
    def version(self) -> str:
        return self._version

    @property
    def metric_order(self) -> tuple[str, ...]:
        return self._metric_order

    @property
    def label_maps(self) -> dict[str, list[str]]:
        return self._label_maps

    @property
    def num_classes(self) -> dict[str, int]:
        return {m: len(self._label_maps[m]) for m in self._metric_order}


__all__ = ["CVSSDataset"]
