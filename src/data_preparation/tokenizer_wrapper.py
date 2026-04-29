"""Обёртка над HuggingFace AutoTokenizer для bert-base-multilingual-cased.

Гарантирует:
    - токен ``[CLS]`` в начале и ``[SEP]`` в конце;
    - truncation до ``max_length`` (по умолчанию 512);
    - padding до ``max_length`` (для стабильной формы тензоров).
"""

from __future__ import annotations

from typing import Sequence

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment]
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None


class CVSSTokenizer:
    """Тонкий слой над AutoTokenizer; всегда возвращает list[int]."""

    DEFAULT_MODEL = "bert-base-multilingual-cased"
    DEFAULT_MAX_LENGTH = 512

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        if AutoTokenizer is None:  # pragma: no cover
            raise ImportError(
                "transformers не установлен; установите requirements.txt"
            ) from _IMPORT_ERROR
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ----------------------------------------------------------------- public

    def tokenize(self, text: str, max_length: int | None = None) -> dict[str, list[int]]:
        ml = max_length or self._max_length
        encoding = self._tokenizer(
            text or "",
            max_length=ml,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors=None,
        )
        return {
            "input_ids": list(encoding["input_ids"]),
            "attention_mask": list(encoding["attention_mask"]),
        }

    def tokenize_batch(
        self,
        texts: Sequence[str],
        max_length: int | None = None,
    ) -> dict[str, list[list[int]]]:
        ml = max_length or self._max_length
        encoding = self._tokenizer(
            list(texts),
            max_length=ml,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors=None,
        )
        return {
            "input_ids": [list(x) for x in encoding["input_ids"]],
            "attention_mask": [list(x) for x in encoding["attention_mask"]],
        }

    # ------------------------------------------------------------------ meta

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def cls_token_id(self) -> int:
        return self._tokenizer.cls_token_id

    @property
    def sep_token_id(self) -> int:
        return self._tokenizer.sep_token_id

    @property
    def hf_tokenizer(self):
        """Доступ к нижележащему AutoTokenizer (для специализированных случаев)."""
        return self._tokenizer


__all__ = ["CVSSTokenizer"]
