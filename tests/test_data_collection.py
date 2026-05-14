"""Юнит-тесты модуля src/data_collection/.

Сетевые вызовы изолированы через библиотеку `responses` (mock requests).
"""

from __future__ import annotations

import json
from io import BytesIO

import pandas as pd
import pytest
import responses

from src.data_collection.bdu_collector import BDUCollector
from src.data_collection.cwe_names import CWENames
from src.data_collection.epss_collector import EPSSCollector
from src.data_collection.exploitdb_collector import ExploitDBCollector
from src.data_collection.kev_collector import KEVCollector
from src.data_collection.nvd_collector import NVDCollector, _SlidingWindowRateLimiter
from src.data_collection.split_data import split_dataset

# --------------------------------------------------------------------------- NVD


def _nvd_payload(cve_id: str = "CVE-2021-44228") -> dict:
    return {
        "resultsPerPage": 1,
        "startIndex": 0,
        "totalResults": 1,
        "vulnerabilities": [
            {
                "cve": {
                    "id": cve_id,
                    "descriptions": [
                        {"lang": "en", "value": "Apache Log4j2 ... JNDI features..."},
                        {"lang": "es", "value": "..."},
                    ],
                    "metrics": {
                        "cvssMetricV31": [
                            {
                                "cvssData": {
                                    "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                                    "baseScore": 10.0,
                                }
                            }
                        ],
                        "cvssMetricV40": [
                            {
                                "cvssData": {
                                    "vectorString": "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H",
                                }
                            }
                        ],
                    },
                    "weaknesses": [{"description": [{"lang": "en", "value": "CWE-502"}]}],
                    "configurations": [
                        {
                            "nodes": [
                                {
                                    "cpeMatch": [
                                        {"criteria": "cpe:2.3:a:apache:log4j:2.0:*:*:*:*:*:*:*"}
                                    ]
                                }
                            ]
                        }
                    ],
                }
            }
        ],
    }


@responses.activate
def test_nvd_fetch_by_cve_parses_record(monkeypatch):
    monkeypatch.delenv("NVD_API_KEY", raising=False)
    responses.add(
        responses.GET,
        "https://services.nvd.nist.gov/rest/json/cves/2.0",
        json=_nvd_payload(),
        status=200,
    )

    collector = NVDCollector({"rate_limit_without_key": 100})
    record = collector.fetch_by_cve("CVE-2021-44228")

    assert record is not None
    assert record["id"] == "CVE-2021-44228"
    assert record["description_en"].startswith("Apache Log4j2")
    assert record["cwe_id"] == "CWE-502"
    assert record["cvss_v3_vector"].startswith("CVSS:3.1/")
    assert record["cvss_v4_vector"].startswith("CVSS:4.0/")
    assert "cpe:2.3:a:apache:log4j:2.0" in record["cpe_list"][0]


@responses.activate
def test_nvd_fetch_by_cve_returns_none_when_empty(monkeypatch):
    monkeypatch.delenv("NVD_API_KEY", raising=False)
    responses.add(
        responses.GET,
        "https://services.nvd.nist.gov/rest/json/cves/2.0",
        json={"vulnerabilities": []},
        status=200,
    )
    collector = NVDCollector({"rate_limit_without_key": 100})
    assert collector.fetch_by_cve("CVE-9999-0001") is None


def test_rate_limiter_blocks_above_threshold():
    """При max_calls=2 и window=10 третий вызов должен подождать."""
    import time

    limiter = _SlidingWindowRateLimiter(max_calls=2, window_sec=0.3)
    start = time.monotonic()
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()  # должен подождать ~0.3 сек
    elapsed = time.monotonic() - start
    assert elapsed >= 0.25


# --------------------------------------------------------------------------- EPSS


@responses.activate
def test_epss_fetch_batch_returns_scores_and_caches(tmp_path):
    responses.add(
        responses.GET,
        "https://api.first.org/data/v1/epss",
        json={
            "data": [
                {"cve": "CVE-2021-44228", "epss": "0.97412", "percentile": "0.99"},
                {"cve": "CVE-2020-0001", "epss": "0.00100", "percentile": "0.10"},
            ]
        },
        status=200,
    )
    collector = EPSSCollector(cache_dir=tmp_path)
    scores = collector.fetch_batch(["CVE-2021-44228", "CVE-2020-0001", "CVE-9999-9999"])
    assert scores["CVE-2021-44228"] == pytest.approx(0.97412)
    assert scores["CVE-2020-0001"] == pytest.approx(0.001)
    assert scores["CVE-9999-9999"] is None

    # Кэш используется при повторном вызове — без новых HTTP-вызовов.
    cached = collector.fetch("CVE-2021-44228")
    assert cached == pytest.approx(0.97412)
    assert (tmp_path / "epss_cache.json").exists()


@responses.activate
def test_epss_uses_cache_on_disk(tmp_path):
    cache = tmp_path / "epss_cache.json"
    cache.write_text(json.dumps({"CVE-2020-1": 0.5}), encoding="utf-8")
    collector = EPSSCollector(cache_dir=tmp_path)
    assert collector.fetch("CVE-2020-1") == pytest.approx(0.5)
    assert len(responses.calls) == 0  # сетевой вызов не выполнялся


# ---------------------------------------------------------------------------- KEV


@responses.activate
def test_kev_is_in_kev_loads_catalog(tmp_path):
    responses.add(
        responses.GET,
        "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        json={
            "vulnerabilities": [
                {"cveID": "CVE-2021-44228"},
                {"cveID": "CVE-2017-0144"},
            ]
        },
        status=200,
    )
    collector = KEVCollector(cache_dir=tmp_path)
    assert collector.is_in_kev("CVE-2021-44228") is True
    assert collector.is_in_kev("cve-2017-0144") is True
    assert collector.is_in_kev("CVE-9999-0000") is False


# ----------------------------------------------------------------------- ExploitDB


@responses.activate
def test_exploitdb_has_exploit_parses_csv(tmp_path):
    csv_text = (
        "id,file,description,date,author,type,platform,port,date_added,date_updated,"
        "verified,codes,tags,aliases,screenshot_url,application_url,source_url\n"
        "1,exploits/1.py,Sample,2024-01-01,me,remote,linux,,2024-01-01,2024-01-01,1,"
        "CVE-2021-44228;CVE-2020-0001,,,,,\n"
        "2,exploits/2.py,Other,2024-01-02,me,local,linux,,2024-01-02,2024-01-02,1,"
        "CVE-2019-1111,,,,,\n"
    )
    responses.add(
        responses.GET,
        "https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv",
        body=csv_text,
        status=200,
        content_type="text/csv",
    )
    collector = ExploitDBCollector(cache_dir=tmp_path)
    assert collector.has_exploit("CVE-2021-44228") is True
    assert collector.has_exploit("CVE-2019-1111") is True
    assert collector.has_exploit("CVE-9999-0000") is False


# --------------------------------------------------------------------------- BDU


def _make_bdu_xlsx(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows)
    buf = BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


@responses.activate
def test_bdu_collector_parses_xlsx(tmp_path):
    xlsx_bytes = _make_bdu_xlsx(
        [
            {
                "Идентификатор": "BDU:2024-00123",
                "Описание уязвимости": "Тестовое описание уязвимости",
                "Идентификаторы CWE": "CWE-79",
                "Идентификаторы CVE": "CVE-2024-0001",
                "Вектор CVSS 3.x": "AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:N",
                "Вектор CVSS 4.0": None,
            },
            {
                "Идентификатор": "BDU:2024-00124",
                "Описание уязвимости": "Другое описание",
                "Идентификаторы CWE": "CWE-89",
                "Идентификаторы CVE": "CVE-2024-0002",
                "Вектор CVSS 3.x": "AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H",
                "Вектор CVSS 4.0": None,
            },
        ]
    )
    responses.add(
        responses.GET,
        "https://bdu.fstec.ru/files/documents/vullist.xlsx",
        body=xlsx_bytes,
        status=200,
        content_type=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    )

    collector = BDUCollector(cache_dir=tmp_path)
    record = collector.fetch_by_id("BDU:2024-00123")
    assert record is not None
    assert record["description_ru"] == "Тестовое описание уязвимости"
    assert record["cwe_id"] == "CWE-79"
    assert record["cve_id"] == "CVE-2024-0001"

    bulk = collector.fetch_bulk(limit=1)
    assert len(bulk) == 1


@responses.activate
def test_bdu_collector_parses_xml(tmp_path):
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<vulnerabilities>"
        "  <vul>"
        "    <identifier>BDU:2024-00500</identifier>"
        "    <name>Уязвимость A</name>"
        "    <description>Уязвимость связана с переполнением буфера</description>"
        "    <cwes><cwe><identifier>CWE-119</identifier><name>Переполнение буфера</name></cwe></cwes>"
        "    <identifiers>"
        '      <identifier type="CVE" link="https://nvd.nist.gov/vuln/detail/CVE-2024-1234">CVE-2024-1234</identifier>'
        "    </identifiers>"
        '    <cvss><vector score="7.5">AV:N/AC:L/Au:N/C:P/I:P/A:P</vector></cvss>'
        '    <cvss3><vector score="9.8">AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H</vector></cvss3>'
        '    <cvss4><vector score="0"></vector></cvss4>'  # placeholder, должен пропуститься
        "  </vul>"
        "  <vul>"
        "    <identifier>BDU:2024-00501</identifier>"
        "    <description>Другая уязвимость</description>"
        "    <cwes><cwe><identifier>CWE-89</identifier></cwe></cwes>"
        '    <identifiers><identifier type="CVE">CVE-2024-9999</identifier></identifiers>'
        '    <cvss3><vector score="6.5">AV:N/AC:L/PR:L/UI:N/S:U/C:L/I:L/A:N</vector></cvss3>'
        '    <cvss4><vector score="8.7">AV:N/AC:L/AT:N/PR:L/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N</vector></cvss4>'
        "  </vul>"
        "</vulnerabilities>"
    )
    responses.add(
        responses.GET,
        "https://bdu.fstec.ru/files/documents/vullist.xml",
        body=xml.encode("utf-8"),
        status=200,
        content_type="application/xml",
    )
    collector = BDUCollector(cache_dir=tmp_path)
    records = collector.fetch_bulk_xml()
    assert len(records) == 2

    by_id = {r["id"]: r for r in records}
    rec = by_id["BDU:2024-00500"]
    assert rec["description_ru"].startswith("Уязвимость связана")
    assert rec["cwe_id"] == "CWE-119"
    assert rec["cve_id"] == "CVE-2024-1234"
    assert rec["cvss_v3_vector"].startswith("AV:N/AC:L/PR:N")
    # Пустой <cvss4> placeholder должен быть отфильтрован.
    assert rec["cvss_v4_vector"] is None

    # У второй записи реальный v4-вектор.
    rec2 = by_id["BDU:2024-00501"]
    assert rec2["cwe_id"] == "CWE-89"
    assert rec2["cvss_v4_vector"].startswith("AV:N/AC:L/AT:N")

    # Кэш заполнен; повторный fetch_by_id отвечает из parquet.
    assert (tmp_path / "bdu_vullist.parquet").exists()
    cached = collector.fetch_by_id("BDU:2024-00501")
    assert cached["cve_id"] == "CVE-2024-9999"


# --------------------------------------------------------------------------- CWE


@responses.activate
def test_cwe_names_parses_zip(tmp_path):
    import zipfile

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Weakness_Catalog xmlns="http://cwe.mitre.org/cwe-7">'
        "  <Weaknesses>"
        '    <Weakness ID="79" Name="Improper Neutralization of Input During Web Page Generation"/>'
        '    <Weakness ID="89" Name="SQL Injection"/>'
        "  </Weaknesses>"
        "</Weakness_Catalog>"
    )
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("cwec_v4.13.xml", xml)
    responses.add(
        responses.GET,
        "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip",
        body=zip_buf.getvalue(),
        status=200,
        content_type="application/zip",
    )
    collector = CWENames(cache_dir=tmp_path)
    assert collector.get("CWE-79").startswith("Improper Neutralization")
    assert collector.get("CWE-89") == "SQL Injection"
    assert collector.get("CWE-99999") is None


# --------------------------------------------------------------------------- split


def test_split_dataset_no_leakage_and_proportions():
    rows = []
    for i in range(200):
        av = ["N", "L", "A", "P"][i % 4]
        rows.append(
            {
                "id": f"CVE-2024-{i:04d}",
                "cve_id": f"CVE-2024-{i:04d}",
                "cvss_vector": f"CVSS:3.1/AV:{av}/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:L",
            }
        )
    df = pd.DataFrame(rows)
    parts = split_dataset(df, seed=42)
    total = sum(len(p) for p in parts.values())
    assert total == 200
    # Пропорции примерно 70/15/15.
    assert abs(len(parts["train"]) / total - 0.70) < 0.05
    assert abs(len(parts["val"]) / total - 0.15) < 0.05
    assert abs(len(parts["test"]) / total - 0.15) < 0.05
    # Нет утечки по cve_id.
    train_ids = set(parts["train"]["cve_id"])
    val_ids = set(parts["val"]["cve_id"])
    test_ids = set(parts["test"]["cve_id"])
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_split_dataset_deduplicates_by_cve_id():
    df = pd.DataFrame(
        [
            {"id": "a", "cve_id": "CVE-2024-0001", "cvss_vector": "AV:N/.."},
            {"id": "b", "cve_id": "CVE-2024-0001", "cvss_vector": "AV:N/.."},  # дубль
            {"id": "c", "cve_id": "CVE-2024-0002", "cvss_vector": "AV:L/.."},
            {"id": "d", "cve_id": "CVE-2024-0003", "cvss_vector": "AV:L/.."},
            {"id": "e", "cve_id": "CVE-2024-0004", "cvss_vector": "AV:N/.."},
            {"id": "f", "cve_id": "CVE-2024-0005", "cvss_vector": "AV:L/.."},
            {"id": "g", "cve_id": "CVE-2024-0006", "cvss_vector": "AV:N/.."},
            {"id": "h", "cve_id": "CVE-2024-0007", "cvss_vector": "AV:L/.."},
            {"id": "i", "cve_id": "CVE-2024-0008", "cvss_vector": "AV:N/.."},
            {"id": "j", "cve_id": "CVE-2024-0009", "cvss_vector": "AV:L/.."},
        ]
    )
    parts = split_dataset(df, seed=42)
    total = sum(len(p) for p in parts.values())
    assert total == 9  # один дубль удалён
