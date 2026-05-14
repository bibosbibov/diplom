"""Коллектор уязвимостей из БДУ ФСТЭК России.

Поддерживает два формата выгрузки:
    - XLSX (vullist.xlsx) — pandas + openpyxl;
    - XML  (vullist.xml)  — потоковый парсинг ElementTree.iterparse.

Кэш парсенных записей хранится в data/raw/bdu_vullist.parquet (одна канонич.
схема для обоих форматов).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ._logging import get_logger

logger = get_logger("bdu")

# Возможные написания колонок в XLSX-выгрузке БДУ.
_COLUMN_ALIASES: dict[str, list[str]] = {
    "id": ["Идентификатор", "Идентификатор уязвимости", "ID"],
    "description": ["Описание уязвимости", "Описание"],
    "cwe": ["Идентификаторы CWE", "CWE", "Тип уязвимости"],
    "cve": ["Идентификаторы CVE", "CVE"],
    "cvss_v3": ["Вектор CVSS 3.x", "Вектор CVSS 3.0", "CVSS 3.x вектор"],
    "cvss_v4": ["Вектор CVSS 4.0", "CVSS 4.0 вектор"],
}

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)

# Канонические поля, в которых хранится кэш и которые возвращаются наружу.
_CANONICAL_FIELDS = (
    "id",
    "description_ru",
    "cwe_id",
    "cvss_v3_vector",
    "cvss_v4_vector",
    "cve_id",
)


class BDUCollector:
    """Загрузка и парсинг массовой выгрузки БДУ ФСТЭК (XLSX или XML)."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        cache_dir: str | Path = "data/raw",
        local_xml_path: str | Path | None = None,
    ) -> None:
        self._config = config or {}
        self._url_xlsx = self._config.get(
            "url_xlsx", "https://bdu.fstec.ru/files/documents/vullist.xlsx"
        )
        self._url_xml = self._config.get("url", "https://bdu.fstec.ru/files/documents/vullist.xml")
        self._timeout = int(self._config.get("timeout_sec", 60))
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / "bdu_vullist.parquet"
        self._local_xml_path = Path(local_xml_path) if local_xml_path else None
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ public

    def fetch_by_id(self, bdu_id: str) -> dict[str, Any] | None:
        """Возвращает каноническую запись БДУ по идентификатору."""
        df = self._ensure_loaded()
        mask = df["id"].astype(str).str.strip() == str(bdu_id).strip()
        rows = df[mask]
        if rows.empty:
            logger.warning("BDU id %s не найден в выгрузке", bdu_id)
            return None
        return self._row_to_record(rows.iloc[0])

    def fetch_bulk(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Возвращает все записи БДУ из XLSX-выгрузки (или из кэша)."""
        return self._collect(limit=limit, source="xlsx")

    def fetch_bulk_xml(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Возвращает все записи БДУ из XML-выгрузки vullist.xml.

        Парсинг потоковый (ElementTree.iterparse) — пиковая память не зависит
        от размера файла.
        """
        return self._collect(limit=limit, source="xml")

    # ----------------------------------------------------------------- internal

    def _collect(self, limit: int | None, source: str) -> list[dict[str, Any]]:
        df = self._ensure_loaded(source=source)
        if limit is not None:
            df = df.head(limit)
        return [self._row_to_record(row) for _, row in df.iterrows()]

    def _ensure_loaded(self, source: str = "xlsx") -> pd.DataFrame:
        if self._df is not None:
            return self._df

        if self._cache_file.exists():
            logger.info("Загрузка кэша БДУ из %s", self._cache_file)
            self._df = pd.read_parquet(self._cache_file)
            return self._df

        if source == "xml":
            records = list(self._fetch_via_xml())
        else:
            records = list(self._fetch_via_xlsx())

        self._df = pd.DataFrame(records, columns=list(_CANONICAL_FIELDS))
        try:
            self._df.to_parquet(self._cache_file, index=False)
            logger.info("Кэш БДУ сохранён: %s (%d строк)", self._cache_file, len(self._df))
        except Exception as exc:  # pragma: no cover - cache is best-effort
            logger.warning("Не удалось сохранить кэш БДУ: %s", exc)
        return self._df

    # ----------------------------------------------------------------- network

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _download_xlsx(self) -> bytes:
        logger.info("Скачивание БДУ XLSX: %s", self._url_xlsx)
        response = requests.get(self._url_xlsx, timeout=self._timeout)
        response.raise_for_status()
        return response.content

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _download_xml(self) -> bytes:
        logger.info("Скачивание БДУ XML: %s", self._url_xml)
        response = requests.get(self._url_xml, timeout=self._timeout)
        response.raise_for_status()
        return response.content

    # ------------------------------------------------------------------ XLSX

    def _fetch_via_xlsx(self) -> Iterator[dict[str, Any]]:
        raw = self._download_xlsx()
        df = pd.read_excel(BytesIO(raw), engine="openpyxl")
        df = self._normalize_columns(df)
        for _, row in df.iterrows():
            yield self._extract_canonical_xlsx(row)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename: dict[str, str] = {}
        lower_to_actual = {str(c).strip().lower(): c for c in df.columns}
        for canonical, aliases in _COLUMN_ALIASES.items():
            for alias in aliases:
                key = alias.lower()
                if key in lower_to_actual:
                    rename[lower_to_actual[key]] = canonical
                    break
        result = df.rename(columns=rename)
        for canonical in _COLUMN_ALIASES:
            if canonical not in result.columns:
                result[canonical] = None
        return result

    @staticmethod
    def _extract_canonical_xlsx(row: pd.Series) -> dict[str, Any]:
        cwe_match = _CWE_RE.search(str(row.get("cwe") or ""))
        cve_match = _CVE_RE.search(str(row.get("cve") or ""))
        return {
            "id": _clean(row.get("id")),
            "description_ru": _clean(row.get("description")),
            "cwe_id": cwe_match.group(0).upper() if cwe_match else None,
            "cvss_v3_vector": _clean(row.get("cvss_v3")),
            "cvss_v4_vector": _clean(row.get("cvss_v4")),
            "cve_id": cve_match.group(0).upper() if cve_match else None,
        }

    # ------------------------------------------------------------------- XML

    def _fetch_via_xml(self) -> Iterator[dict[str, Any]]:
        if self._local_xml_path is not None:
            logger.info("Чтение БДУ XML из локального файла: %s", self._local_xml_path)
            with self._local_xml_path.open("rb") as fh:
                yield from self._iter_xml(fh)
            return
        raw = self._download_xml()
        yield from self._iter_xml(BytesIO(raw))

    @staticmethod
    def _iter_xml(file_obj: BytesIO) -> Iterator[dict[str, Any]]:
        """Потоковый парсинг vullist.xml; yield канонических записей."""
        context = ET.iterparse(file_obj, events=("end",))
        for _event, elem in context:
            tag = _strip_ns(elem.tag)
            if tag != "vul":
                continue
            record = _parse_vul_element(elem)
            elem.clear()
            if record is None:
                continue
            yield record

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _row_to_record(row: pd.Series) -> dict[str, Any]:
        return {field: _clean(row.get(field)) for field in _CANONICAL_FIELDS}


# --------------------------------------------------------------- module helpers


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _child_text(parent: ET.Element, name: str) -> str | None:
    for child in parent:
        if _strip_ns(child.tag) == name:
            text = (child.text or "").strip()
            return text or None
    return None


def _children(parent: ET.Element, name: str) -> Iterator[ET.Element]:
    for child in parent:
        if _strip_ns(child.tag) == name:
            yield child


def _parse_vul_element(vul: ET.Element) -> dict[str, Any] | None:
    """Извлекает канонические поля из <vul>…</vul>.

    Реальная структура vullist.xml БДУ ФСТЭК:
        <vul>
          <identifier>BDU:YYYY-NNNNN</identifier>
          <description>...</description>
          <cwes>
            <cwe>
              <identifier>CWE-NN</identifier>
              <name>...</name>
            </cwe>
            ...
          </cwes>
          <identifiers>
            <identifier type="CVE" link="...">CVE-YYYY-NNNNN</identifier>
          </identifiers>
          <cvss>  <vector score="..">AV:N/AC:.../A:.</vector></cvss>      <!-- v2 -->
          <cvss3> <vector score="..">AV:N/AC:.../A:.</vector></cvss3>     <!-- v3.1 -->
          <cvss4> <vector score="..">AV:N/.../SA:.</vector></cvss4>       <!-- v4.0 -->
        </vul>
    """
    bdu_id = _child_text(vul, "identifier")
    if not bdu_id:
        return None

    description = _child_text(vul, "description")
    cwe_id = _extract_cwe_id(vul)
    cve_id = _extract_cve_id(vul)
    cvss_v3 = _extract_vector(vul, "cvss3")
    cvss_v4 = _extract_vector(vul, "cvss4") or _extract_vector(vul, "cvssv4")

    return {
        "id": bdu_id,
        "description_ru": description,
        "cwe_id": cwe_id,
        "cvss_v3_vector": cvss_v3,
        "cvss_v4_vector": cvss_v4,
        "cve_id": cve_id,
    }


def _extract_cwe_id(vul: ET.Element) -> str | None:
    """Возвращает первый встреченный CWE-NN из <cwes>/<cwe>/<identifier>."""
    candidates: list[ET.Element] = []
    for cwes in _children(vul, "cwes"):
        candidates.extend(_children(cwes, "cwe"))
    candidates.extend(_children(vul, "cwe"))  # fallback на старую вёрстку
    for cwe_elem in candidates:
        for ident in _children(cwe_elem, "identifier"):
            match = _CWE_RE.search((ident.text or "").strip())
            if match:
                return match.group(0).upper()
    return None


def _extract_cve_id(vul: ET.Element) -> str | None:
    """Возвращает первый CVE-идентификатор из <identifiers>."""
    for idents in _children(vul, "identifiers"):
        for ident in _children(idents, "identifier"):
            if (ident.attrib.get("type") or "").upper() != "CVE":
                continue
            match = _CVE_RE.search((ident.text or "").strip())
            if match:
                return match.group(0).upper()
    return None


def _extract_vector(vul: ET.Element, tag: str) -> str | None:
    """Извлекает <vector>...</vector> из дочернего <tag>; пустые placeholder'ы пропускает."""
    for cvss_elem in _children(vul, tag):
        for vec in _children(cvss_elem, "vector"):
            text = (vec.text or "").strip()
            if text:  # БДУ часто публикует <vector score="0"></vector> — это заглушка
                return text
    return None


__all__ = ["BDUCollector"]
