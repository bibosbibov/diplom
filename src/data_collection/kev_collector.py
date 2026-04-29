"""Коллектор каталога CISA Known Exploited Vulnerabilities (KEV).

Источник: https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ._logging import get_logger

logger = get_logger("kev")

_DAY_SEC = 24 * 60 * 60


class KEVCollector:
    """Каталог KEV целиком; обновляется не чаще раза в сутки."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        cache_dir: str | Path = "data/raw",
        ttl_sec: int = _DAY_SEC,
    ) -> None:
        self._config = config or {}
        self._url = self._config.get(
            "url",
            "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        )
        self._timeout = int(self._config.get("timeout_sec", 30))
        self._cache_path = Path(cache_dir) / "kev_catalog.json"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_sec
        self._cve_set: set[str] | None = None

    # ----------------------------------------------------------------- public

    def is_in_kev(self, cve_id: str) -> bool:
        """True, если CVE присутствует в каталоге KEV."""
        return cve_id.upper() in self._ensure_loaded()

    def refresh(self) -> set[str]:
        """Принудительная перезагрузка каталога; возвращает множество CVE."""
        payload = self._download()
        self._save_cache(payload)
        self._cve_set = self._extract_cves(payload)
        return self._cve_set

    # ----------------------------------------------------------------- cache

    def _ensure_loaded(self) -> set[str]:
        if self._cve_set is not None:
            return self._cve_set

        if self._cache_is_fresh():
            try:
                with self._cache_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                self._cve_set = self._extract_cves(payload)
                logger.info("KEV загружен из кэша: %d CVE", len(self._cve_set))
                return self._cve_set
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Кэш KEV повреждён, перезагрузка: %s", exc)

        return self.refresh()

    def _cache_is_fresh(self) -> bool:
        if not self._cache_path.exists():
            return False
        age = time.time() - self._cache_path.stat().st_mtime
        return age < self._ttl

    def _save_cache(self, payload: dict[str, Any]) -> None:
        try:
            with self._cache_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
        except OSError as exc:  # pragma: no cover - cache is best-effort
            logger.warning("Не удалось сохранить кэш KEV: %s", exc)

    @staticmethod
    def _extract_cves(payload: dict[str, Any]) -> set[str]:
        return {
            (entry.get("cveID") or "").upper()
            for entry in payload.get("vulnerabilities", []) or []
            if entry.get("cveID")
        }

    # ---------------------------------------------------------------- network

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _download(self) -> dict[str, Any]:
        logger.info("Скачивание каталога KEV: %s", self._url)
        response = requests.get(self._url, timeout=self._timeout)
        response.raise_for_status()
        return response.json()


__all__ = ["KEVCollector"]
