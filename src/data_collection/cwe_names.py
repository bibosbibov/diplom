"""Получение человекочитаемых имён CWE из MITRE.

Источник: https://cwe.mitre.org/data/xml/cwec_latest.xml.zip
"""

from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ._logging import get_logger

logger = get_logger("cwe")

_CWE_RE = re.compile(r"CWE-(\d+)", re.IGNORECASE)


class CWENames:
    """Соответствие CWE-ID → название (используется как cwe_name на входе модели)."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        cache_dir: str | Path = "data/raw",
    ) -> None:
        self._config = config or {}
        self._url = self._config.get("url", "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip")
        self._timeout = int(self._config.get("timeout_sec", 120))
        self._cache_path = Path(cache_dir) / "cwe_names.json"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._mapping: dict[str, str] | None = None

    # ----------------------------------------------------------------- public

    def get(self, cwe_id: str) -> str | None:
        """Возвращает имя CWE по идентификатору CWE-NNN."""
        if not cwe_id:
            return None
        match = _CWE_RE.search(cwe_id)
        if not match:
            return None
        normalized = f"CWE-{match.group(1)}"
        return self._ensure_loaded().get(normalized)

    def all(self) -> dict[str, str]:
        """Полный словарь CWE-ID → имя."""
        return dict(self._ensure_loaded())

    # ------------------------------------------------------------------ cache

    def _ensure_loaded(self) -> dict[str, str]:
        if self._mapping is not None:
            return self._mapping

        if self._cache_path.exists():
            try:
                with self._cache_path.open("r", encoding="utf-8") as fh:
                    self._mapping = json.load(fh)
                logger.info("CWE names загружены из кэша: %d записей", len(self._mapping))
                return self._mapping
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Кэш CWE names повреждён: %s", exc)

        raw = self._download()
        self._mapping = self._parse(raw)
        try:
            with self._cache_path.open("w", encoding="utf-8") as fh:
                json.dump(self._mapping, fh, ensure_ascii=False, indent=2)
        except OSError as exc:  # pragma: no cover - cache is best-effort
            logger.warning("Не удалось сохранить кэш CWE names: %s", exc)
        return self._mapping

    # ---------------------------------------------------------------- network

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _download(self) -> bytes:
        logger.info("Скачивание словаря CWE: %s", self._url)
        response = requests.get(self._url, timeout=self._timeout)
        response.raise_for_status()
        return response.content

    # ----------------------------------------------------------------- parser

    @staticmethod
    def _parse(zip_bytes: bytes) -> dict[str, str]:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            xml_name = next((n for n in zf.namelist() if n.lower().endswith(".xml")), None)
            if xml_name is None:
                raise RuntimeError("В архиве CWE отсутствует XML-файл")
            xml_bytes = zf.read(xml_name)

        root = ET.fromstring(xml_bytes)
        # Тег с пространством имён вида {http://cwe.mitre.org/cwe-7}Weakness
        mapping: dict[str, str] = {}
        for elem in root.iter():
            tag = elem.tag.split("}", 1)[-1]
            if tag in ("Weakness", "Category"):
                cwe_id = elem.attrib.get("ID")
                name = elem.attrib.get("Name")
                if cwe_id and name:
                    mapping[f"CWE-{cwe_id}"] = name
        return mapping


__all__ = ["CWENames"]
