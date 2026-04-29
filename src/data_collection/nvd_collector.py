"""Коллектор уязвимостей из NVD API 2.0.

Документация: https://nvd.nist.gov/developers/vulnerabilities
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Iterator

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ._config import get_env
from ._logging import get_logger

logger = get_logger("nvd")


class _SlidingWindowRateLimiter:
    """Rate limiter скользящего окна: не более N событий за window секунд."""

    def __init__(self, max_calls: int, window_sec: float) -> None:
        self._max_calls = max_calls
        self._window = window_sec
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] > self._window:
                self._calls.popleft()
            if len(self._calls) >= self._max_calls:
                wait = self._window - (now - self._calls[0]) + 0.05
                logger.debug("Rate limit reached, sleeping %.2fs", wait)
                time.sleep(max(wait, 0))
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self._window:
                    self._calls.popleft()
            self._calls.append(now)


class NVDCollector:
    """Загрузка уязвимостей из NVD API 2.0 с учётом rate limits."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._base_url = self._config.get(
            "base_url", "https://services.nvd.nist.gov/rest/json/cves/2.0"
        )
        self._page_size = int(self._config.get("page_size", 2000))
        self._timeout = int(self._config.get("timeout_sec", 30))

        self._api_key = get_env("NVD_API_KEY") or None
        max_calls = (
            int(self._config.get("rate_limit_with_key", 50))
            if self._api_key
            else int(self._config.get("rate_limit_without_key", 5))
        )
        self._limiter = _SlidingWindowRateLimiter(max_calls=max_calls, window_sec=30.0)

        if not self._api_key:
            logger.warning("NVD_API_KEY не задан — rate limit 5 запросов / 30 сек")

    # -------------------------------------------------------------------- API

    def fetch_by_cve(self, cve_id: str) -> dict[str, Any] | None:
        """Возвращает стандартизированную запись по идентификатору CVE."""
        params = {"cveId": cve_id}
        payload = self._request(params)
        items = payload.get("vulnerabilities") or []
        if not items:
            logger.warning("CVE %s не найден в NVD", cve_id)
            return None
        return self._parse_vulnerability(items[0])

    def fetch_bulk(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Итератор записей за период.

        Даты — в ISO 8601 формата ``YYYY-MM-DDTHH:MM:SS.sssZ``. NVD требует
        указывать pubStartDate и pubEndDate в одном вызове.
        """
        start_index = 0
        total = None

        while True:
            params: dict[str, Any] = {
                "startIndex": start_index,
                "resultsPerPage": self._page_size,
            }
            if start_date:
                params["pubStartDate"] = start_date
            if end_date:
                params["pubEndDate"] = end_date

            payload = self._request(params)
            vulns = payload.get("vulnerabilities") or []
            for raw in vulns:
                yield self._parse_vulnerability(raw)

            if total is None:
                total = int(payload.get("totalResults", 0))
                logger.info("NVD bulk: всего %d записей", total)

            start_index += len(vulns)
            if not vulns or start_index >= total:
                break

    # ---------------------------------------------------------------- network

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        self._limiter.acquire()
        headers = {"User-Agent": "cvss-v4-mbert/0.1"}
        if self._api_key:
            headers["apiKey"] = self._api_key
        response = requests.get(
            self._base_url, params=params, headers=headers, timeout=self._timeout
        )
        response.raise_for_status()
        return response.json()

    # ----------------------------------------------------------------- parser

    @staticmethod
    def _parse_vulnerability(item: dict[str, Any]) -> dict[str, Any]:
        cve = item.get("cve", {})
        cve_id = cve.get("id")

        description_en = None
        for desc in cve.get("descriptions", []) or []:
            if desc.get("lang") == "en":
                description_en = desc.get("value")
                break

        cwe_id = None
        for weakness in cve.get("weaknesses", []) or []:
            for desc in weakness.get("description", []) or []:
                value = (desc.get("value") or "").strip()
                if value.upper().startswith("CWE-"):
                    cwe_id = value.upper()
                    break
            if cwe_id:
                break

        metrics = cve.get("metrics", {}) or {}
        cvss_v3_vector = _first_vector(
            metrics.get("cvssMetricV31") or metrics.get("cvssMetricV30")
        )
        cvss_v4_vector = _first_vector(metrics.get("cvssMetricV40"))

        cpe_list: list[str] = []
        for cfg in cve.get("configurations", []) or []:
            for node in cfg.get("nodes", []) or []:
                for match in node.get("cpeMatch", []) or []:
                    criteria = match.get("criteria")
                    if criteria:
                        cpe_list.append(criteria)

        return {
            "id": cve_id,
            "description_en": description_en,
            "cwe_id": cwe_id,
            "cvss_v3_vector": cvss_v3_vector,
            "cvss_v4_vector": cvss_v4_vector,
            "cpe_list": cpe_list,
        }


def _first_vector(metric_list: list[dict[str, Any]] | None) -> str | None:
    if not metric_list:
        return None
    cvss_data = metric_list[0].get("cvssData", {}) or {}
    vector = cvss_data.get("vectorString")
    return vector or None


__all__ = ["NVDCollector"]
