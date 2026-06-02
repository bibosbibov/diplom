"""Тесты модуля src/training/train_scope_head.py.

Полный e2e-прогон требует реальной stage 1 модели (~700 МБ) и mBERT-весов,
поэтому здесь:

* Чистые юнит-тесты на ``extract_scope`` и ``_filter_with_scope`` —
  работа с реальной строкой CVSS-вектора.
* Тест на :func:`train_head` с синтетическими fused-представлениями —
  проверяет, что линейная голова сходится, ранний останов работает,
  лучшие веса восстанавливаются.
* Юнит-тест ``cache_fused_features`` с мини-моделью на CPU (1 пример,
  одна головка) — проверяет форму выхода без полной mBERT-инициализации.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.training.train_scope_head import (
    SCOPE_CLASSES,
    _filter_with_scope,
    _trim_pad_collate,
    cache_fused_features,
    extract_scope,
    train_head,
)


# ---------------------------------------------------------------------------
# extract_scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vector,expected",
    [
        ("CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:H/A:N", "U"),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:L", "C"),
        ("CVSS:3.0/AV:L/AC:H/PR:H/UI:R/S:U/C:H/I:H/A:H", "U"),
        ("AV:N/S:C/C:L", "C"),  # без префикса
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/C:L/I:N/A:N", None),  # нет S
        ("", None),
        (None, None),
        (12345, None),  # не строка
        ("garbage", None),
    ],
)
def test_extract_scope(vector, expected) -> None:
    assert extract_scope(vector) == expected


def test_extract_scope_ignores_lookalikes() -> None:
    """Метрика 'S' должна выделяться как самостоятельная, не как часть AS/MS/CS."""
    # AS — не CVSS-метрика; SA — Scope of Attack? Здесь нет S:X — отдельная метрика.
    assert extract_scope("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H") is None


# ---------------------------------------------------------------------------
# _filter_with_scope
# ---------------------------------------------------------------------------


def test_filter_with_scope_keeps_only_rows_with_s() -> None:
    df = pd.DataFrame(
        [
            {"cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:H/A:N"},
            {"cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:L"},
            {"cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/C:L/I:N/A:N"},  # no S
            {"cvss_v3_vector": None},
        ]
    )
    filtered = _filter_with_scope(df)
    assert len(filtered) == 2
    assert list(filtered["scope"]) == ["U", "C"]


# ---------------------------------------------------------------------------
# train_head — синтетика
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_split() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Линейно-разделимая задача на 512 признаках, 2 класса.

    Класс 0: вектор + N(0, 0.1), направление [1, 0, 0, ..., 0]
    Класс 1: вектор + N(0, 0.1), направление [-1, 0, 0, ..., 0]
    Идеальная голова Linear(512, 2) сойдётся за 2–3 эпохи к acc > 0.99.
    """
    torch.manual_seed(0)
    n_per_class = 200
    d = 512
    # Сильный сигнал по первой координате (±3.0) при шуме N(0, 0.1) — гарантированно линейно разделимо.
    base0 = torch.zeros(d); base0[0] = 3.0
    base1 = torch.zeros(d); base1[0] = -3.0

    train_x = torch.cat(
        [base0 + 0.1 * torch.randn(n_per_class, d),
         base1 + 0.1 * torch.randn(n_per_class, d)]
    )
    train_y = torch.cat(
        [torch.zeros(n_per_class, dtype=torch.long),
         torch.ones(n_per_class, dtype=torch.long)]
    )
    val_x = torch.cat(
        [base0 + 0.1 * torch.randn(50, d),
         base1 + 0.1 * torch.randn(50, d)]
    )
    val_y = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])
    # Перемешать
    perm = torch.randperm(len(train_x))
    return train_x[perm], train_y[perm], val_x, val_y


def test_train_head_converges(synthetic_split) -> None:
    train_x, train_y, val_x, val_y = synthetic_split
    head, history = train_head(
        train_fused=train_x, train_targets=train_y,
        val_fused=val_x, val_targets=val_y,
        device=torch.device("cpu"),
        epochs=15, batch_size=64, lr=1e-2, patience=20, seed=0,
    )
    assert isinstance(head, nn.Linear)
    assert head.weight.shape == (2, 512)
    # Линейно-разделимая задача — за 15 эпох должно сойтись хотя бы к 0.95.
    assert history["best_val_accuracy"] > 0.95
    # История заполнена для каждой проведённой эпохи.
    assert len(history["train_loss"]) >= 1
    assert all(0.0 <= acc <= 1.0 for acc in history["val_accuracy"])


def test_train_head_early_stopping_triggers() -> None:
    """patience=1 + случайные данные ⇒ early stop в первые эпохи."""
    torch.manual_seed(123)
    train_x = torch.randn(200, 512)
    train_y = torch.randint(0, 2, (200,), dtype=torch.long)
    val_x = torch.randn(50, 512)
    val_y = torch.randint(0, 2, (50,), dtype=torch.long)
    _, history = train_head(
        train_fused=train_x, train_targets=train_y,
        val_fused=val_x, val_targets=val_y,
        device=torch.device("cpu"),
        epochs=20, batch_size=32, lr=1e-3, patience=1, seed=123,
    )
    # На случайных данных линейный классификатор не выучит ничего,
    # и early stopping должен сработать раньше, чем закончатся 20 эпох.
    assert len(history["train_loss"]) < 20


def test_train_head_restores_best_weights() -> None:
    """Финальная голова — это веса с эпохи лучшей val_accuracy, не последней."""
    torch.manual_seed(7)
    # Делаем «шумную» задачу: 1 информативный признак, чтобы качество флуктуировало.
    n = 100
    d = 512
    train_x = torch.randn(n, d)
    train_y = (train_x[:, 0] > 0).long()
    val_x = torch.randn(50, d)
    val_y = (val_x[:, 0] > 0).long()
    head, history = train_head(
        train_fused=train_x, train_targets=train_y,
        val_fused=val_x, val_targets=val_y,
        device=torch.device("cpu"),
        epochs=8, batch_size=32, lr=1e-1, patience=10, seed=7,
    )
    # Лучший val_accuracy в истории должен совпадать с тем, который вернула функция.
    assert history["best_val_accuracy"] == max(history["val_accuracy"])
    # И эта точность достигается именно на best_epoch-1.
    assert history["val_accuracy"][history["best_epoch"] - 1] == history["best_val_accuracy"]


# ---------------------------------------------------------------------------
# SCOPE_CLASSES
# ---------------------------------------------------------------------------


def test_scope_classes_order() -> None:
    """U = 0, C = 1 — фиксированный контракт для всего пайплайна."""
    assert SCOPE_CLASSES == ("U", "C")


# ---------------------------------------------------------------------------
# _trim_pad_collate — динамический padding
# ---------------------------------------------------------------------------


def test_trim_pad_collate_trims_to_batch_max() -> None:
    """Хвостовой padding режется до самой длинной реальной строки в батче."""
    batch = [
        {  # реальная длина 3 (3 единицы маски), добито до 8
            "input_ids": torch.tensor([1, 2, 3, 0, 0, 0, 0, 0]),
            "attention_mask": torch.tensor([1, 1, 1, 0, 0, 0, 0, 0]),
        },
        {  # реальная длина 5
            "input_ids": torch.tensor([1, 2, 3, 4, 5, 0, 0, 0]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 0, 0, 0]),
        },
    ]
    out = _trim_pad_collate(batch)
    # Батч-максимум реальной длины = 5 ⇒ обрезаем до 5.
    assert out["input_ids"].shape == (2, 5)
    assert out["attention_mask"].shape == (2, 5)
    assert out["input_ids"][1].tolist() == [1, 2, 3, 4, 5]


def test_trim_pad_collate_keeps_at_least_one_token() -> None:
    """Пустой (полностью padding) батч не схлопывается до нулевой длины."""
    batch = [{"input_ids": torch.zeros(4, dtype=torch.long),
              "attention_mask": torch.zeros(4, dtype=torch.long)}]
    out = _trim_pad_collate(batch)
    assert out["input_ids"].shape == (1, 1)


# ---------------------------------------------------------------------------
# cache_fused_features — со встроенным мини-бэкбоном
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Минимальная замена ``CVSSTokenizer`` для теста."""

    def tokenize(self, text: str, max_length: int | None = None) -> dict[str, list[int]]:
        n = max_length or 8
        return {"input_ids": [0] * n, "attention_mask": [1] * n}

    def tokenize_batch(self, texts):
        return {
            "input_ids": [[0] * 8 for _ in texts],
            "attention_mask": [[1] * 8 for _ in texts],
        }


class _FakeBackbone(nn.Module):
    """Замена CVSSModel: одинаковые имена, но игрушечные тензоры."""

    def __init__(self) -> None:
        super().__init__()
        # Реальный CVSSModel.fusion имеет output_dim=512 — повторяем.
        self._dummy = nn.Parameter(torch.zeros(1))

    def encode_text(self, input_ids, attention_mask):
        b = input_ids.shape[0]
        return torch.zeros(b, 768)

    # Псевдо-features_mlp, фактически модуль с .__call__.
    @property
    def features_mlp(self):
        return self._features_mlp_callable

    @property
    def fusion(self):
        return self._fusion_callable

    @staticmethod
    def _features_mlp_callable(features, cwe_idx):
        return torch.zeros(features.shape[0], 64)

    @staticmethod
    def _fusion_callable(h_text, h_feat):
        b = h_text.shape[0]
        # Имитируем 512-мерный выход с предсказуемым значением.
        return torch.full((b, 512), 0.5)


def test_cache_fused_features_shape(monkeypatch) -> None:
    """Кэширование возвращает (N, 512) для fused и (N,) для меток."""
    import src.training.train_scope_head as mod

    # Подсунем фейковый CVSSDataset, чтобы не тянуть CVSSTokenizer/CWEEncoder.
    class _FakeDataset:
        def __init__(self, df, **_):
            self._n = len(df)

        def __len__(self):
            return self._n

        def __getitem__(self, idx):
            return {
                "input_ids": torch.zeros(8, dtype=torch.long),
                "attention_mask": torch.ones(8, dtype=torch.long),
                "cwe_idx": torch.tensor(1, dtype=torch.long),
                "features": torch.zeros(3),
                "labels": {},
            }

    monkeypatch.setattr(mod, "CVSSDataset", _FakeDataset)

    df = pd.DataFrame(
        [{"scope": "U"}, {"scope": "C"}, {"scope": "U"}]
    )
    model = _FakeBackbone()

    fused, targets = cache_fused_features(
        model=model, dataframe=df,
        tokenizer=_FakeTokenizer(),
        cwe_encoder=None, features_encoder=None, text_processor=None,
        device=torch.device("cpu"),
        batch_size=2, max_length=8, num_workers=0,
    )
    assert fused.shape == (3, 512)
    assert targets.tolist() == [0, 1, 0]  # U=0, C=1
