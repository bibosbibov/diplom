"""Коллектор оценок EPSS (Exploit Prediction Scoring System).

Источник: https://api.first.org/data/v1/epss
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ._logging import get_logger

logger = get_logger("epss")

_BATCH_SIZE = 100  # API принимает до ~150 CVE через ?cve= в URL


class EPSSCollector:
    """Получение EPSS-оценок (вероятность эксплуатации в ближайшие 30 дней)."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        cache_dir: str | Path = "data/raw",
    ) -> None:
        self._config = config or {}
        self._base_url = self._config.get("base_url", "https://api.first.org/data/v1/epss")
        self._timeout = int(self._config.get("timeout_sec", 30))
        self._cache_path = Path(cache_dir) / "epss_cache.json"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, float | None] = self._load_cache()

    # ----------------------------------------------------------------- public

    def fetch(self, cve_id: str) -> float | None:
        """Возвращает EPSS-оценку CVE; None если запись не найдена."""
        if cve_id in self._cache:
            return self._cache[cve_id]
        result = self.fetch_batch([cve_id])
        return result.get(cve_id)

    def fetch_batch(self, cve_ids: Iterable[str]) -> dict[str, float | None]:
        """Запрашивает EPSS для пачки CVE; результат + кэширование."""
        ids = [cid for cid in dict.fromkeys(cve_ids) if cid]  # dedup, preserve order
        result: dict[str, float | None] = {}
        to_query: list[str] = []
        for cid in ids:
            if cid in self._cache:
                result[cid] = self._cache[cid]
            else:
                to_query.append(cid)

        for chunk in _chunked(to_query, _BATCH_SIZE):
            payload = self._request({"cve": ",".join(chunk)})
            returned: set[str] = set()
            for entry in payload.get("data", []) or []:
                cid = entry.get("cve")
                try:
                    score = float(entry["epss"])
                except (KeyError, TypeError, ValueError):
                    score = None
                self._cache[cid] = score
                result[cid] = score
                returned.add(cid)
            for missing in set(chunk) - returned:
                self._cache[missing] = None
                result[missing] = None

        self._save_cache()
        return result

    # ----------------------------------------------------------------- cache

    def _load_cache(self) -> dict[str, float | None]:
        if not self._cache_path.exists():
            return {}
        try:
            with self._cache_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Не удалось прочитать кэш EPSS: %s", exc)
            return {}

    def _save_cache(self) -> None:
        try:
            with self._cache_path.open("w", encoding="utf-8") as fh:
                json.dump(self._cache, fh, ensure_ascii=False)
        except OSError as exc:  # pragma: no cover - cache is best-effort
            logger.warning("Не удалось сохранить кэш EPSS: %s", exc)

    # ---------------------------------------------------------------- network

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        response = requests.get(self._base_url, params=params, timeout=self._timeout)
        response.raise_for_status()
        return response.json()


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


__all__ = ["EPSSCollector"]
