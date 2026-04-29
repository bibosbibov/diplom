"""Интегратор источников данных.

Реализует раздел 2.3.1: по списку идентификаторов BDU/CVE собирает запись
``R = {id, d_ru, d_en, cwe_id, cwe_name, epss, kev, exploit, cvss_vector}``.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm.auto import tqdm

from ._config import load_config
from ._logging import get_logger, setup_logger
from .bdu_collector import BDUCollector
from .cwe_names import CWENames
from .epss_collector import EPSSCollector
from .exploitdb_collector import ExploitDBCollector
from .kev_collector import KEVCollector
from .nvd_collector import NVDCollector

logger = get_logger("integrator")

_BDU_RE = re.compile(r"^BDU[:\-]?\d{4}-\d+$", re.IGNORECASE)
_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,7}$", re.IGNORECASE)


class DataIntegrator:
    """Главный оркестратор: вызывает все коллекторы и формирует датасет."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        max_workers: int = 5,
    ) -> None:
        setup_logger()
        self._config = config or load_config()
        self._max_workers = max_workers

        sources = self._config.get("data_sources", {})
        paths = self._config.get("paths", {})
        cache_dir = paths.get("data_raw", "data/raw")

        self.bdu = BDUCollector(sources.get("bdu_fstec"), cache_dir=cache_dir)
        self.nvd = NVDCollector(sources.get("nvd"))
        self.epss = EPSSCollector(sources.get("epss"), cache_dir=cache_dir)
        self.kev = KEVCollector(sources.get("cisa_kev"), cache_dir=cache_dir)
        self.exploit = ExploitDBCollector(sources.get("exploit_db"), cache_dir=cache_dir)
        self.cwe = CWENames(sources.get("cwe_mitre"), cache_dir=cache_dir)

        self._output_path = Path(cache_dir) / "dataset.parquet"

    # ----------------------------------------------------------------- public

    def collect_dataset(
        self,
        id_list: Iterable[str],
        save: bool = True,
    ) -> pd.DataFrame:
        """Собирает датасет по списку идентификаторов BDU и/или CVE."""
        ids = [s for s in dict.fromkeys(id_list) if s]
        if not ids:
            logger.warning("Список идентификаторов пуст")
            return pd.DataFrame()

        records: list[dict[str, Any]] = []
        logger.info("Сбор датасета: %d идентификаторов", len(ids))

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._collect_one, identifier): identifier for identifier in ids}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="collecting"):
                identifier = futures[fut]
                try:
                    record = fut.result()
                except Exception as exc:  # noqa: BLE001 - один битый id не должен ронять весь сбор
                    logger.exception("Ошибка для %s: %s", identifier, exc)
                    continue
                if record is not None:
                    records.append(record)

        df = pd.DataFrame(records)
        logger.info("Собрано %d записей из %d запрошенных", len(df), len(ids))

        if save and not df.empty:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self._output_path, index=False)
            logger.info("Датасет сохранён: %s", self._output_path)

        return df

    # ---------------------------------------------------------------- worker

    def _collect_one(self, identifier: str) -> dict[str, Any] | None:
        bdu_record: dict[str, Any] | None = None
        nvd_record: dict[str, Any] | None = None
        cve_id: str | None = None

        if _BDU_RE.match(identifier):
            bdu_record = self.bdu.fetch_by_id(identifier)
            if bdu_record is None:
                return None
            cve_id = bdu_record.get("cve_id")
        elif _CVE_RE.match(identifier):
            cve_id = identifier.upper()
        else:
            logger.warning("Неизвестный формат идентификатора: %s", identifier)
            return None

        if cve_id:
            try:
                nvd_record = self.nvd.fetch_by_cve(cve_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("NVD недоступен для %s: %s", cve_id, exc)
                nvd_record = None

        cwe_id = (bdu_record or {}).get("cwe_id") or (nvd_record or {}).get("cwe_id")
        cvss_v4_vector = (
            (bdu_record or {}).get("cvss_v4_vector")
            or (nvd_record or {}).get("cvss_v4_vector")
        )
        cvss_v3_vector = (
            (bdu_record or {}).get("cvss_v3_vector")
            or (nvd_record or {}).get("cvss_v3_vector")
        )

        epss_score = self.epss.fetch(cve_id) if cve_id else None
        kev_flag = self.kev.is_in_kev(cve_id) if cve_id else False
        exploit_flag = self.exploit.has_exploit(cve_id) if cve_id else False
        cwe_name = self.cwe.get(cwe_id) if cwe_id else None

        return {
            "id": bdu_record["id"] if bdu_record else cve_id,
            "cve_id": cve_id,
            "d_ru": (bdu_record or {}).get("description_ru"),
            "d_en": (nvd_record or {}).get("description_en"),
            "cwe_id": cwe_id,
            "cwe_name": cwe_name,
            "epss": epss_score,
            "kev": int(kev_flag) if kev_flag is not None else None,
            "exploit": int(exploit_flag) if exploit_flag is not None else None,
            "cvss_v3_vector": cvss_v3_vector,
            "cvss_v4_vector": cvss_v4_vector,
            "cvss_vector": cvss_v4_vector or cvss_v3_vector,
        }


__all__ = ["DataIntegrator"]
