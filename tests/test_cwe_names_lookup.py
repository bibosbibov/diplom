"""Тесты офлайн-справочника CWE-имён для инференса."""

from __future__ import annotations

import json

from src.data_preparation import CWENameLookup


def _write(tmp_path, mapping):
    p = tmp_path / "cwe_names.json"
    p.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    return p


def test_get_by_full_id(tmp_path) -> None:
    lookup = CWENameLookup(_write(tmp_path, {"CWE-89": "SQL Injection"}))
    assert lookup.get("CWE-89") == "SQL Injection"
    assert lookup.available is True


def test_get_normalizes_bare_number(tmp_path) -> None:
    lookup = CWENameLookup(_write(tmp_path, {"CWE-79": "XSS"}))
    assert lookup.get("89") is None  # нет такого
    assert lookup.get("CWE-0079") is None  # ведущие нули не нормализуем
    assert lookup.get("cwe-79") == "XSS"  # регистронезависимо


def test_unknown_and_empty(tmp_path) -> None:
    lookup = CWENameLookup(_write(tmp_path, {"CWE-89": "SQL Injection"}))
    assert lookup.get("CWE-999999") is None
    assert lookup.get("") is None
    assert lookup.get(None) is None
    assert lookup.get("not-a-cwe") is None


def test_missing_file_is_graceful(tmp_path) -> None:
    lookup = CWENameLookup(tmp_path / "does_not_exist.json")
    assert lookup.get("CWE-89") is None
    assert lookup.available is False


def test_corrupt_file_is_graceful(tmp_path) -> None:
    p = tmp_path / "cwe_names.json"
    p.write_text("{ broken json", encoding="utf-8")
    lookup = CWENameLookup(p)
    assert lookup.get("CWE-89") is None
