"""Офлайн-справочник ``CWE-ID → человекочитаемое имя`` для инференса.

Модель обучалась на тексте ``description [SEP] cwe_name`` (см.
:class:`TextProcessor`), поэтому на инференсе имя CWE тоже надо подставлять —
иначе теряется ≈1 п.п. F1 на текстовых головах. Источник имён —
``data/raw/cwe_names.json`` (выгрузка MITRE, собирается
:class:`src.data_collection.cwe_names.CWENames`).

В отличие от :class:`CWENames`, этот загрузчик **офлайновый и безопасный**: не
тянет ``requests``/``tenacity`` и никогда не ходит в сеть. Если файла нет или он
повреждён — :meth:`get` возвращает ``None``, и пайплайн просто работает без
``cwe_name`` (как раньше), без падений.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CWE_NAMES_PATH = "data/raw/cwe_names.json"
_CWE_RE = re.compile(r"CWE-(\d+)", re.IGNORECASE)


class CWENameLookup:
    """Ленивая офлайн-загрузка словаря ``{"CWE-NNN": "имя"}``.

    Args:
        path: путь к JSON-словарю CWE-имён. Если файл отсутствует — лукап
            работает как пустой (всегда ``None``), без ошибок.
    """

    def __init__(self, path: str | Path = DEFAULT_CWE_NAMES_PATH) -> None:
        self._path = Path(path)
        self._mapping: dict[str, str] | None = None

    def get(self, cwe_id: str | None) -> str | None:
        """Имя CWE по идентификатору ``CWE-NNN`` (или ``NNN``); иначе ``None``."""
        if not cwe_id:
            return None
        match = _CWE_RE.search(str(cwe_id))
        if not match:
            return None
        return self._load().get(f"CWE-{match.group(1)}")

    def all(self) -> dict[str, str]:
        """Полный словарь ``{"CWE-NNN": имя}`` (копия)."""
        return dict(self._load())

    @property
    def available(self) -> bool:
        """``True``, если словарь успешно загружен и непуст."""
        return bool(self._load())

    def _load(self) -> dict[str, str]:
        if self._mapping is not None:
            return self._mapping
        if not self._path.exists():
            logger.warning(
                "Словарь CWE-имён не найден: %s — инференс пойдёт без cwe_name "
                "(текстовые головы потеряют ≈1 п.п. F1)",
                self._path,
            )
            self._mapping = {}
            return self._mapping
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._mapping = {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
            logger.info("CWE-имена загружены: %d записей из %s", len(self._mapping), self._path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Не удалось прочитать %s: %s — иду без cwe_name", self._path, exc)
            self._mapping = {}
        return self._mapping


__all__ = ["CWENameLookup", "DEFAULT_CWE_NAMES_PATH"]
