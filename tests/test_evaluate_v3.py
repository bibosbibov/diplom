"""Smoke-тесты модуля src/evaluation/evaluate_v3.py.

Реальную stage 1 модель здесь не грузим (нужны веса mBERT и обученный
чекпоинт). Вместо этого подсовываем фейковую :class:`CVSSModel` через
``monkeypatch.setattr`` на ``V3Evaluator._load_model``: фейк ничего не
делает с входами и возвращает контролируемые логиты, чтобы проверить
только пайплайн декодирования и формирования отчёта.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from torch import nn

from src.data_preparation import V3_LABEL_MAPS, V3_METRIC_ORDER
from src.data_preparation.cwe_encoder import CWEEncoder
from src.evaluation.evaluate_v3 import V3Evaluator, _render_markdown


# --------------------------------------------------------------- фейк-модель


class _FakeStage1Model(nn.Module):
    """Заглушка для :class:`CVSSModel`: ``forward`` возвращает фиксированные логиты.

    Для каждой метрики выдаётся одинаковый аргмакс — это даёт строго
    предсказуемые ответы при любом батче. Логиты делаем 4-классовыми для
    VC/VI/VA и 5-классовыми для E — чтобы заодно проверить слайс логитов
    до длины :data:`V3_LABEL_MAPS`.
    """

    def __init__(self) -> None:
        super().__init__()
        # nn.Module требует хотя бы один параметр, иначе .to(device) — no-op,
        # но это не критично; добавим dummy для общего тонуса.
        self._dummy = nn.Parameter(torch.zeros(1))
        self._head_sizes = {"AV": 4, "AC": 2, "PR": 3, "UI": 2,
                            "VC": 4, "VI": 4, "VA": 4, "E": 5}
        # «Идеальные» argmax-индексы — соответствуют декодировке ниже.
        self._preferred_idx = {
            "AV": 0,  # → "N"
            "AC": 0,  # → "L"
            "PR": 0,  # → "N"
            "UI": 0,  # → "N"
            "VC": 0,  # → "H"  (индекс 3 «X» не должен пройти — проверим слайсом)
            "VI": 0,  # → "H"
            "VA": 0,  # → "H"
            "E": 0,   # → "X"  (V3_LABEL_MAPS["E"][0] = "X")
        }

    def eval(self):  # type: ignore[override]
        return self

    def to(self, *args, **kwargs):  # type: ignore[override]
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cwe_idx: torch.Tensor,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(input_ids.shape[0])
        out: dict[str, torch.Tensor] = {}
        for metric, n_cls in self._head_sizes.items():
            logits = torch.full((batch_size, n_cls), -10.0)
            logits[:, self._preferred_idx[metric]] = 10.0
            out[metric] = logits
        return out


# ------------------------------------------------------------- фикстуры


@pytest.fixture()
def configs(tmp_path: Path) -> tuple[Path, Path]:
    """Минимальные train.yaml + config.yaml в tmp_path."""
    train_cfg = {
        "seed": 42,
        "stage1": {
            "metric_classes": {
                "AV": 4, "AC": 2, "PR": 3, "UI": 2,
                "VC": 4, "VI": 4, "VA": 4, "E": 5,
            },
        },
    }
    cfg = {
        "project": {"seed": 42},
        "paths": {
            "cwe_vocab": str(tmp_path / "cwe_vocab.json"),
            "train_data": str(tmp_path / "train.parquet"),
        },
        "data_preparation": {
            "pretrained_tokenizer": "bert-base-multilingual-cased",
            "max_length": 16,
        },
        "model": {"pretrained_name": "bert-base-multilingual-cased"},
    }
    train_yaml = tmp_path / "train.yaml"
    cfg_yaml = tmp_path / "config.yaml"
    train_yaml.write_text(yaml.safe_dump(train_cfg), encoding="utf-8")
    cfg_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return train_yaml, cfg_yaml


@pytest.fixture()
def cwe_vocab(tmp_path: Path) -> Path:
    """Готовый словарь CWE — без обращения к train.parquet."""
    encoder = CWEEncoder().fit(["CWE-79", "CWE-89"])
    path = tmp_path / "cwe_vocab.json"
    encoder.save(path)
    return path


@pytest.fixture()
def fake_checkpoint(tmp_path: Path) -> Path:
    """``.pt`` с минимальным state_dict — нужен только ``cwe_embedding.weight``.

    Реальная загрузка модели обходится monkeypatch'ем
    :meth:`V3Evaluator._load_model`; но :meth:`_infer_checkpoint_cwe_size`
    читает форму CWE-эмбеддинга, чтобы решить, выравнивать ли энкодер.
    Фейковый CWE-вокаб ``cwe_vocab`` фикстуры имеет 4 записи (PAD + UNK + 2 CWE).
    """
    path = tmp_path / "fake_stage1.pt"
    torch.save(
        {"model_state": {"features_mlp.cwe_embedding.weight": torch.zeros(4, 64)}},
        path,
    )
    return path


@pytest.fixture()
def synthetic_v3_df() -> pd.DataFrame:
    """5 синтетических строк с валидными CVSS:3.1 векторами."""
    base_vec = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    rows = []
    for i in range(5):
        rows.append(
            {
                "cve_id": f"CVE-2024-{1000 + i}",
                "d_ru": None,
                "d_en": f"SQL injection in login form, variant {i}",
                "cwe_id": "CWE-89" if i % 2 else "CWE-79",
                "cwe_name": "SQL Injection",
                "epss": 0.5,
                "kev": 0,
                "exploit": 0,
                "cvss_v3_vector": base_vec,
                "cvss_v4_vector": None,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- tests


def test_evaluate_v3_smoke(
    monkeypatch: pytest.MonkeyPatch,
    configs: tuple[Path, Path],
    cwe_vocab: Path,
    fake_checkpoint: Path,
    synthetic_v3_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Прогон V3Evaluator на синтетике с замоканной моделью.

    Проверяем: пайплайн считает метрики, сохраняет .md и .json без падений,
    в отчёте — все 8 голов и непустой macro_f1. Реальный mBERT не грузим.
    """
    train_yaml, cfg_yaml = configs

    # Подменяем тяжёлый _load_model и токенизатор до создания V3Evaluator.
    monkeypatch.setattr(V3Evaluator, "_load_model", lambda self, _: _FakeStage1Model())

    class _FakeTokenizer:
        def __init__(self, *args, **kwargs):
            self.max_length = kwargs.get("max_length", 16)

        def tokenize(self, text: str, max_length: int | None = None) -> dict[str, list[int]]:
            n = int(max_length or self.max_length)
            return {"input_ids": [0] * n, "attention_mask": [1] * n}

    monkeypatch.setattr("src.evaluation.evaluate_v3.CVSSTokenizer", _FakeTokenizer)

    evaluator = V3Evaluator(
        model_path=fake_checkpoint,
        train_config_path=train_yaml,
        config_path=cfg_yaml,
        device=torch.device("cpu"),
        batch_size=2,
        cwe_vocab_path=cwe_vocab,
    )

    results = evaluator.evaluate(synthetic_v3_df)

    assert set(results["per_metric"].keys()) == set(V3_METRIC_ORDER)
    # 7 базовых метрик имеют support; E в синтетических векторах отсутствует
    # (это нормально и для реального датасета — Exploit Maturity редко
    # указывается). Все support'ы ≥ 0, метрики в [0, 1].
    for metric in V3_METRIC_ORDER:
        scores = results["per_metric"][metric]
        assert scores["support"] >= 0
        assert 0.0 <= scores["f1_macro"] <= 1.0
        assert 0.0 <= scores["accuracy"] <= 1.0
    # Хотя бы 7 голов должны быть «живыми» (исключая E).
    base_metrics = [m for m in V3_METRIC_ORDER if m != "E"]
    assert all(results["per_metric"][m]["support"] >= 1 for m in base_metrics)

    agg = results["aggregated"]
    assert agg["samples_evaluated"] == len(synthetic_v3_df)
    assert 0.0 <= agg["macro_f1"] <= 1.0

    out_md = tmp_path / "v3_results.md"
    out_json = tmp_path / "v3_results.json"
    paths = evaluator.save_results(results, output_md=out_md, output_json=out_json)
    assert Path(paths["md"]).exists() and Path(paths["json"]).exists()

    md_text = out_md.read_text(encoding="utf-8")
    assert "# Результаты оценки этапа 1 (CVSS v3.1)" in md_text
    # В отчёте должны быть все 8 метрик.
    for metric in V3_METRIC_ORDER:
        assert f"| {metric:<9} |" in md_text


def test_render_markdown_structure() -> None:
    """``_render_markdown`` собирает таблицу нужной формы без сетевых вызовов."""
    results = {
        "per_metric": {
            m: {"f1_macro": 0.5, "accuracy": 0.6, "support": 100}
            for m in V3_METRIC_ORDER
        },
        "aggregated": {"macro_f1": 0.55, "samples_evaluated": 100},
    }
    md = _render_markdown(results)
    # Заголовок Markdown-таблицы должен содержать ожидаемые колонки.
    assert "F1 (macro)" in md and "Accuracy" in md and "Support" in md
    # Macro-F1 интегральной таблицы.
    assert "0.5500" in md
    # Все 8 голов в таблице — по одной строке.
    for metric in V3_METRIC_ORDER:
        assert f"| {metric:<9} |" in md


def test_filter_valid_v3_drops_empty(
    monkeypatch: pytest.MonkeyPatch,
    configs: tuple[Path, Path],
    cwe_vocab: Path,
    fake_checkpoint: Path,
) -> None:
    """Строки с пустым или неразборчивым cvss_v3_vector отбрасываются."""
    train_yaml, cfg_yaml = configs

    monkeypatch.setattr(V3Evaluator, "_load_model", lambda self, _: _FakeStage1Model())

    class _FakeTokenizer:
        def __init__(self, *args, **kwargs):
            self.max_length = kwargs.get("max_length", 16)

        def tokenize(self, text, max_length=None):
            n = int(max_length or self.max_length)
            return {"input_ids": [0] * n, "attention_mask": [1] * n}

    monkeypatch.setattr("src.evaluation.evaluate_v3.CVSSTokenizer", _FakeTokenizer)

    evaluator = V3Evaluator(
        model_path=fake_checkpoint,
        train_config_path=train_yaml,
        config_path=cfg_yaml,
        device=torch.device("cpu"),
        batch_size=2,
        cwe_vocab_path=cwe_vocab,
    )

    df = pd.DataFrame(
        [
            {"cve_id": "x", "d_ru": None, "d_en": "ok", "cwe_id": "CWE-79",
             "cwe_name": "X", "epss": 0.1, "kev": 0, "exploit": 0,
             "cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"},
            {"cve_id": "y", "d_ru": None, "d_en": "empty", "cwe_id": "CWE-79",
             "cwe_name": "X", "epss": 0.1, "kev": 0, "exploit": 0,
             "cvss_v3_vector": None},
            {"cve_id": "z", "d_ru": None, "d_en": "garbage", "cwe_id": "CWE-79",
             "cwe_name": "X", "epss": 0.1, "kev": 0, "exploit": 0,
             "cvss_v3_vector": "not-a-vector"},
        ]
    )
    filtered = evaluator._filter_valid_v3(df)
    assert len(filtered) == 1
    assert filtered.iloc[0]["cve_id"] == "x"


def test_logit_slicing_keeps_predictions_in_v3_label_map(
    monkeypatch: pytest.MonkeyPatch,
    configs: tuple[Path, Path],
    cwe_vocab: Path,
    fake_checkpoint: Path,
    synthetic_v3_df: pd.DataFrame,
) -> None:
    """Лишний 4-й класс VC/VI/VA не должен ломать декодирование.

    Подсовываем модель, которая на VC/VI/VA для лишнего класса (индекс 3)
    выставляет максимальный логит — без слайса argmax вернёт 3 и
    ``V3_LABEL_MAPS["VC"][3]`` упадёт IndexError. Со слайсом — индекс
    обрезается до 0..2.
    """
    train_yaml, cfg_yaml = configs

    class _XPreferringModel(_FakeStage1Model):
        def __init__(self) -> None:
            super().__init__()
            # Просим max на индексе 3 для V-метрик (тот самый «X» в конфиге).
            self._preferred_idx.update({"VC": 3, "VI": 3, "VA": 3})

    monkeypatch.setattr(V3Evaluator, "_load_model", lambda self, _: _XPreferringModel())

    class _FakeTokenizer:
        def __init__(self, *args, **kwargs):
            self.max_length = kwargs.get("max_length", 16)

        def tokenize(self, text, max_length=None):
            n = int(max_length or self.max_length)
            return {"input_ids": [0] * n, "attention_mask": [1] * n}

    monkeypatch.setattr("src.evaluation.evaluate_v3.CVSSTokenizer", _FakeTokenizer)

    evaluator = V3Evaluator(
        model_path=fake_checkpoint,
        train_config_path=train_yaml,
        config_path=cfg_yaml,
        device=torch.device("cpu"),
        batch_size=2,
        cwe_vocab_path=cwe_vocab,
    )

    # Не падает — а значит индексы после слайса валидные для V3_LABEL_MAPS.
    results = evaluator.evaluate(synthetic_v3_df)
    assert results["aggregated"]["samples_evaluated"] == len(synthetic_v3_df)
    # И декодированные классы — внутри допустимого множества.
    for metric in ("VC", "VI", "VA"):
        # support > 0 → метрика реально оценивалась.
        assert results["per_metric"][metric]["support"] == len(synthetic_v3_df)
        # f1_per_class содержит ключи только из V3_LABEL_MAPS — без «X».
        assert set(results["per_metric"][metric]["f1_per_class"].keys()) <= set(
            V3_LABEL_MAPS[metric]
        )
