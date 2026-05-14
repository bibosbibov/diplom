"""Юнит-тесты модуля src/data_preparation/."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_preparation.cvss_vector_parser import (
    IGNORE_INDEX,
    V3_LABEL_MAPS,
    V3_METRIC_ORDER,
    V4_LABEL_MAPS,
    V4_METRIC_ORDER,
    parse_v3_vector,
    parse_v4_vector,
    vector_to_labels,
)
from src.data_preparation.cwe_encoder import CWEEncoder
from src.data_preparation.features_encoder import FeaturesEncoder
from src.data_preparation.text_processor import TextProcessor

# ============================================================ CVSS v4.0 parser

# 5 реальных векторов: первые 3 из БДУ ФСТЭК, 2 из NVD-документации.
REAL_V4_VECTORS = [
    # CVE-2024-8937 / БДУ:2024-09683
    (
        "AV:N/AC:H/AT:P/PR:N/UI:N/VC:H/VI:L/VA:N/SC:N/SI:N/SA:N",
        {
            "AV": "N",
            "AC": "H",
            "AT": "P",
            "PR": "N",
            "UI": "N",
            "VC": "H",
            "VI": "L",
            "VA": "N",
            "SC": "N",
            "SI": "N",
            "SA": "N",
            "E": None,
        },
    ),
    # CVE-2024-11056 / БДУ:2024-09775
    (
        "AV:N/AC:L/AT:N/PR:L/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N",
        {
            "AV": "N",
            "AC": "L",
            "AT": "N",
            "PR": "L",
            "UI": "N",
            "VC": "H",
            "VI": "H",
            "VA": "H",
            "SC": "N",
            "SI": "N",
            "SA": "N",
            "E": None,
        },
    ),
    # CVE-2024-0012 / БДУ:2024-09796
    (
        "AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:L/SI:L/SA:L",
        {
            "AV": "N",
            "AC": "L",
            "AT": "N",
            "PR": "N",
            "UI": "N",
            "VC": "H",
            "VI": "H",
            "VA": "H",
            "SC": "L",
            "SI": "L",
            "SA": "L",
            "E": None,
        },
    ),
    # С префиксом и threat-метрикой E
    (
        "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H/E:A",
        {
            "AV": "N",
            "AC": "L",
            "AT": "N",
            "PR": "N",
            "UI": "N",
            "VC": "H",
            "VI": "H",
            "VA": "H",
            "SC": "H",
            "SI": "H",
            "SA": "H",
            "E": "A",
        },
    ),
    # БДУ:2024-09800 — с дополнительной (не базовой) AU:Y, должна игнорироваться
    (
        "AV:N/AC:L/AT:P/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N/AU:Y",
        {
            "AV": "N",
            "AC": "L",
            "AT": "P",
            "PR": "N",
            "UI": "N",
            "VC": "H",
            "VI": "H",
            "VA": "H",
            "SC": "N",
            "SI": "N",
            "SA": "N",
            "E": None,
        },
    ),
]


@pytest.mark.parametrize("vector,expected", REAL_V4_VECTORS)
def test_parse_v4_real_examples(vector, expected):
    parsed = parse_v4_vector(vector)
    assert parsed == expected


def test_parse_v4_returns_all_12_keys_even_when_missing():
    parsed = parse_v4_vector("AV:N")  # минимальный вектор
    assert set(parsed.keys()) == set(V4_METRIC_ORDER)
    assert parsed["AV"] == "N"
    assert all(parsed[m] is None for m in V4_METRIC_ORDER if m != "AV")


def test_parse_v4_empty_or_none():
    assert parse_v4_vector(None) == {m: None for m in V4_METRIC_ORDER}
    assert parse_v4_vector("") == {m: None for m in V4_METRIC_ORDER}


# ----- маркер X (Not Defined) в выгрузках NVD --------------------------------


class TestParseV4WithXMarker:
    """Обработка маркера ``X`` в CVSS v4.0 — спецификация FIRST §2.4.1."""

    def test_e_x_replaced_with_a(self):
        """E:X → A (Attacked) — значение по умолчанию для Exploit Maturity."""
        parsed = parse_v4_vector(
            "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N/E:X"
        )
        assert parsed["E"] == "A"

    @pytest.mark.parametrize(
        "metric",
        ["AV", "AC", "AT", "PR", "UI", "VC", "VI", "VA", "SC", "SI", "SA"],
    )
    def test_base_metric_x_raises(self, metric):
        """Любая из 11 базовых метрик со значением X → ValueError."""
        # Заведомо валидный базовый вектор и подмена одной метрики на X.
        base = {
            "AV": "N",
            "AC": "L",
            "AT": "N",
            "PR": "N",
            "UI": "N",
            "VC": "H",
            "VI": "H",
            "VA": "H",
            "SC": "N",
            "SI": "N",
            "SA": "N",
        }
        base[metric] = "X"
        vector = "CVSS:4.0/" + "/".join(f"{k}:{v}" for k, v in base.items())
        with pytest.raises(ValueError, match=metric):
            parse_v4_vector(vector)

    def test_full_nvd_vector(self):
        """Полный вектор из NVD со всеми X-маркерами в дополнительных метриках."""
        vector = (
            "CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:L/VI:L/VA:L/SC:N/SI:N/SA:N/"
            "E:X/CR:X/IR:X/AR:X/MAV:X/MAC:X/MAT:X/MPR:X/MUI:X/MVC:X/MVI:X/MVA:X/"
            "MSC:X/MSI:X/MSA:X/S:X/AU:X/R:X/V:X/RE:X/U:X"
        )
        parsed = parse_v4_vector(vector)
        assert parsed == {
            "AV": "N",
            "AC": "L",
            "AT": "N",
            "PR": "L",
            "UI": "N",
            "VC": "L",
            "VI": "L",
            "VA": "L",
            "SC": "N",
            "SI": "N",
            "SA": "N",
            "E": "A",
        }


# ============================================================ CVSS v3.x parser


def test_parse_v3_maps_C_I_A_to_VC_VI_VA():
    parsed = parse_v3_vector("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
    assert parsed["VC"] == "H"
    assert parsed["VI"] == "H"
    assert parsed["VA"] == "H"
    assert parsed["AV"] == "N"
    assert parsed["UI"] == "N"
    assert parsed["E"] is None
    # Scope (S) игнорируется и не появляется в результате.
    assert "S" not in parsed


def test_parse_v3_with_exploit_maturity():
    parsed = parse_v3_vector("AV:N/AC:L/PR:L/UI:N/S:C/C:L/I:L/A:N/E:F")
    assert parsed["E"] == "F"


# ================================================================ label_maps


def test_v4_label_maps_no_duplicates_and_match_order():
    assert len(V4_METRIC_ORDER) == 12
    assert set(V4_METRIC_ORDER) == set(V4_LABEL_MAPS.keys())
    for metric, classes in V4_LABEL_MAPS.items():
        assert len(set(classes)) == len(classes), f"дубликаты в {metric}"


def test_v3_label_maps_no_duplicates_and_match_order():
    assert len(V3_METRIC_ORDER) == 8
    assert set(V3_METRIC_ORDER) == set(V3_LABEL_MAPS.keys())
    for metric, classes in V3_LABEL_MAPS.items():
        assert len(set(classes)) == len(classes), f"дубликаты в {metric}"


def test_v4_label_maps_match_clauded_md_spec():
    """Спецификация из CLAUDE.md."""
    assert V4_LABEL_MAPS["AV"] == ["N", "A", "L", "P"]
    assert V4_LABEL_MAPS["AC"] == ["L", "H"]
    assert V4_LABEL_MAPS["AT"] == ["N", "P"]
    assert V4_LABEL_MAPS["PR"] == ["N", "L", "H"]
    assert V4_LABEL_MAPS["UI"] == ["N", "P", "A"]
    for impact in ("VC", "VI", "VA", "SC", "SI", "SA"):
        assert V4_LABEL_MAPS[impact] == ["H", "L", "N"]
    assert V4_LABEL_MAPS["E"] == ["A", "P", "U"]


# ============================================================ vector_to_labels


def test_vector_to_labels_v4_indices():
    parsed = parse_v4_vector("AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H")
    labels = vector_to_labels(parsed, V4_METRIC_ORDER, V4_LABEL_MAPS)
    assert labels.shape == (12,)
    assert labels.dtype == np.int64
    # AV: "N" → 0 ([N, A, L, P])
    assert labels[V4_METRIC_ORDER.index("AV")] == 0
    # AC: "L" → 0 ([L, H])
    assert labels[V4_METRIC_ORDER.index("AC")] == 0
    # AT: "N" → 0 ([N, P])
    assert labels[V4_METRIC_ORDER.index("AT")] == 0
    # PR: "N" → 0 ([N, L, H])
    assert labels[V4_METRIC_ORDER.index("PR")] == 0
    # VC: "H" → 0 ([H, L, N])
    assert labels[V4_METRIC_ORDER.index("VC")] == 0
    # E: не задано → IGNORE_INDEX
    assert labels[V4_METRIC_ORDER.index("E")] == IGNORE_INDEX


def test_vector_to_labels_unknown_value_is_ignored():
    # Z — несуществующий класс для AV
    parsed = {m: None for m in V4_METRIC_ORDER}
    parsed["AV"] = "Z"
    parsed["AC"] = "L"
    labels = vector_to_labels(parsed, V4_METRIC_ORDER, V4_LABEL_MAPS)
    assert labels[V4_METRIC_ORDER.index("AV")] == IGNORE_INDEX
    assert labels[V4_METRIC_ORDER.index("AC")] == 0


def test_vector_to_labels_all_missing():
    parsed = {m: None for m in V4_METRIC_ORDER}
    labels = vector_to_labels(parsed, V4_METRIC_ORDER, V4_LABEL_MAPS)
    assert (labels == IGNORE_INDEX).all()


# ============================================================== TextProcessor


def test_text_processor_prefers_russian():
    tp = TextProcessor()
    out = tp.prepare_text("Русское описание уязвимости", "English description", "XSS")
    assert out.startswith("Русское описание")
    assert "[SEP]" in out
    assert out.endswith("XSS")


def test_text_processor_falls_back_to_english():
    tp = TextProcessor()
    out = tp.prepare_text(None, "Buffer overflow in foo", "Buffer overflow")
    assert out.startswith("Buffer overflow")
    assert "[SEP]" in out


def test_text_processor_handles_empty_russian():
    tp = TextProcessor()
    out = tp.prepare_text("", "English text", None)
    assert out == "English text"


def test_text_processor_strips_html_and_entities():
    tp = TextProcessor()
    out = tp.prepare_text("<p>Vuln &amp; bug   </p>\n\nMore", None, None)
    assert "<p>" not in out
    assert "&amp;" not in out
    assert "Vuln & bug More" == out


def test_text_processor_returns_empty_for_no_descriptions():
    tp = TextProcessor()
    assert tp.prepare_text(None, None, None) == ""


# ================================================================ CWEEncoder


def test_cwe_encoder_fit_creates_unique_indices():
    enc = CWEEncoder().fit(["CWE-79", "CWE-89", "CWE-79", None, ""])
    assert enc.transform("CWE-79") != enc.transform("CWE-89")
    assert enc.transform("CWE-79") >= 2
    assert enc.transform("CWE-89") >= 2


def test_cwe_encoder_unknown_returns_unk():
    enc = CWEEncoder().fit(["CWE-79"])
    assert enc.transform("CWE-99999") == CWEEncoder.UNK_INDEX
    assert enc.transform(None) == CWEEncoder.UNK_INDEX
    assert enc.transform("") == CWEEncoder.UNK_INDEX


def test_cwe_encoder_save_load(tmp_path):
    enc = CWEEncoder().fit(["CWE-79", "CWE-89", "CWE-22"])
    path = tmp_path / "cwe_vocab.json"
    enc.save(path)
    assert path.exists()
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["<PAD>"] == 0
    assert data["<UNK>"] == 1

    restored = CWEEncoder.load(path)
    assert restored.transform("CWE-79") == enc.transform("CWE-79")
    assert len(restored) == len(enc)


# ============================================================ FeaturesEncoder


def test_features_encoder_normal_values():
    fe = FeaturesEncoder()
    arr = fe.encode(epss=0.5, kev=1, exploit=0)
    assert arr.shape == (3,)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, [0.5, 1.0, 0.0])


def test_features_encoder_missing_to_minus_one():
    fe = FeaturesEncoder()
    arr = fe.encode(epss=None, kev=None, exploit=None)
    np.testing.assert_allclose(arr, [-1.0, -1.0, -1.0])


def test_features_encoder_nan_treated_as_missing():
    fe = FeaturesEncoder()
    arr = fe.encode(epss=float("nan"), kev=1, exploit=0)
    assert arr[0] == -1.0
    assert arr[1] == 1.0
    assert arr[2] == 0.0


def test_features_encoder_out_of_range_epss():
    fe = FeaturesEncoder()
    assert fe.encode(epss=2.0, kev=0, exploit=0)[0] == -1.0
    assert fe.encode(epss=-0.1, kev=0, exploit=0)[0] == -1.0


def test_features_encoder_invalid_flag():
    fe = FeaturesEncoder()
    # любое не-1 целое → 0.0
    arr = fe.encode(epss=0.1, kev=2, exploit="abc")
    assert arr[1] == 0.0
    assert arr[2] == -1.0


# ================================================================ CVSSDataset


def _fake_tokenizer_factory(max_length: int):
    """Заглушка под CVSSTokenizer без обращения в сеть/HuggingFace."""

    class FakeTokenizer:
        def tokenize(self, text, max_length=max_length):
            ids = [101] + [42] * (max_length - 2) + [102]
            mask = [1] * max_length
            return {"input_ids": ids, "attention_mask": mask}

    return FakeTokenizer()


def test_dataset_v4_shapes_and_labels():
    pytest.importorskip("torch")
    import torch

    from src.data_preparation.dataset import CVSSDataset

    df = pd.DataFrame(
        [
            {
                "cve_id": "CVE-2024-1",
                "d_ru": None,
                "d_en": "Test vulnerability",
                "cwe_id": "CWE-79",
                "cwe_name": "Cross-site Scripting",
                "epss": 0.5,
                "kev": 0,
                "exploit": 1,
                "cvss_v4_vector": "AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H",
                "cvss_v3_vector": None,
            }
        ]
    )
    ds = CVSSDataset(
        df,
        tokenizer=_fake_tokenizer_factory(max_length=64),
        cwe_encoder=CWEEncoder().fit(["CWE-79"]),
        features_encoder=FeaturesEncoder(),
        version="v4",
        max_length=64,
    )
    assert len(ds) == 1
    item = ds[0]
    assert item["input_ids"].shape == (64,)
    assert item["input_ids"].dtype == torch.long
    assert item["attention_mask"].shape == (64,)
    assert item["features"].shape == (3,)
    assert item["features"].dtype == torch.float32
    np.testing.assert_allclose(item["features"].numpy(), [0.5, 0.0, 1.0])
    assert item["cwe_idx"].dtype == torch.long
    assert item["cwe_idx"].item() >= 2  # не PAD/UNK

    assert isinstance(item["labels"], dict)
    assert set(item["labels"].keys()) == set(V4_METRIC_ORDER)
    for metric, label in item["labels"].items():
        assert label.dtype == torch.long
        assert label.dim() == 0
    # E нет в векторе → IGNORE_INDEX
    assert item["labels"]["E"].item() == IGNORE_INDEX
    # AV:N → 0
    assert item["labels"]["AV"].item() == 0


def test_dataset_v3_uses_v3_metric_order():
    pytest.importorskip("torch")

    from src.data_preparation.dataset import CVSSDataset

    df = pd.DataFrame(
        [
            {
                "cve_id": "CVE-2024-1",
                "d_ru": None,
                "d_en": "X",
                "cwe_id": None,
                "cwe_name": None,
                "epss": None,
                "kev": None,
                "exploit": None,
                "cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                "cvss_v4_vector": None,
            }
        ]
    )
    ds = CVSSDataset(
        df,
        tokenizer=_fake_tokenizer_factory(max_length=32),
        cwe_encoder=CWEEncoder().fit([]),
        features_encoder=FeaturesEncoder(),
        version="v3",
        max_length=32,
    )
    item = ds[0]
    assert set(item["labels"].keys()) == set(V3_METRIC_ORDER)
    assert len(item["labels"]) == 8
    # VC получено из C:H → индекс 0
    assert item["labels"]["VC"].item() == 0
    # CWE отсутствует → UNK_INDEX (1)
    assert item["cwe_idx"].item() == CWEEncoder.UNK_INDEX
    # Все числовые признаки отсутствуют → -1
    np.testing.assert_allclose(item["features"].numpy(), [-1.0, -1.0, -1.0])


def test_dataset_dataloader_batches_correctly():
    pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from src.data_preparation.dataset import CVSSDataset

    rows = []
    for i in range(4):
        rows.append(
            {
                "cve_id": f"CVE-2024-{i:04d}",
                "d_ru": None,
                "d_en": f"vuln {i}",
                "cwe_id": "CWE-79",
                "cwe_name": "XSS",
                "epss": 0.1 * i,
                "kev": 0,
                "exploit": 0,
                "cvss_v4_vector": "AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H",
                "cvss_v3_vector": None,
            }
        )
    df = pd.DataFrame(rows)
    ds = CVSSDataset(
        df,
        tokenizer=_fake_tokenizer_factory(max_length=16),
        cwe_encoder=CWEEncoder().fit(["CWE-79"]),
        features_encoder=FeaturesEncoder(),
        version="v4",
        max_length=16,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 16)
    assert batch["features"].shape == (2, 3)
    # labels — dict; default_collate соберёт скалярные тензоры в (B,)
    assert batch["labels"]["AV"].shape == (2,)
