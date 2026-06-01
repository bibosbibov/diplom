"""Тесты модуля src/training/dapt.py.

End-to-end DAPT-прогон с реальным mBERT требует загрузки ~700 МБ весов
и десятки секунд на CPU — это слишком медленно для unit-слоя. Поэтому:

* Логику отбора текста и токенизации проверяем на синтетическом
  ``DAPTTextDataset`` с фейк-токенайзером.
* Конфиг-парсинг (``load_dapt_config``, debug-overrides) — чистые юнит-тесты.
* End-to-end проверка ``run()`` с replace HF-объектов на минимальные
  заглушки, чтобы убедиться, что цепочка собирается, обрабатывает
  parquet, передаёт всё в ``Trainer`` и пишет чекпоинт.
* Параллельно тест на ``train.py --pretrained-name`` — что значение
  действительно прокидывается в конфиг.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
import pytest
import yaml

from src.training.dapt import (
    DAPTTextDataset,
    DEFAULT_DAPT_CONFIG,
    _apply_debug_overrides,
    load_dapt_config,
    run,
)


# ---------------------------------------------------------------------------
# Фейк-токенайзер (вместо реального mBERT — даёт детерминированный вывод).
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Минимальная замена ``transformers.AutoTokenizer``.

    Поддерживает только то, что нужно :class:`DAPTTextDataset`:
    ``__call__(text, truncation, max_length, padding,
    return_special_tokens_mask) → dict``.
    Возвращает фиксированную короткую последовательность из ascii-кодов
    текста, обрезанную до ``max_length``. ``special_tokens_mask`` —
    первый/последний токены помечены как спец.
    """

    pad_token_id = 0
    mask_token_id = 103
    eos_token_id = 102
    cls_token_id = 101

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 512,
        padding: bool = False,
        return_special_tokens_mask: bool = True,
    ) -> dict[str, list[int]]:
        # Простой токенизатор: байты UTF-8 как «токены».
        body = list(text.encode("utf-8"))[: max_length - 2]
        ids = [self.cls_token_id, *body, self.eos_token_id]
        mask = [1] * len(ids)
        special = [1] + [0] * len(body) + [1]
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "special_tokens_mask": special,
        }


# ---------------------------------------------------------------------------
# Фикстуры.
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_train(tmp_path: Path) -> Path:
    """Маленький train.parquet со смесью русских и английских описаний."""
    rows = [
        {
            "d_ru": "Уязвимость удалённого выполнения кода в Apache HTTP Server",
            "d_en": "Remote code execution in Apache HTTP Server",
            "cwe_id": "CWE-78", "cwe_name": "OS Command Injection",
            "epss": 0.5, "kev": 1, "exploit": 1,
            "cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "cvss_v4_vector": None,
        },
        {
            "d_ru": None,
            "d_en": "SQL injection in login form allows authentication bypass",
            "cwe_id": "CWE-89", "cwe_name": "SQL Injection",
            "epss": 0.3, "kev": 0, "exploit": 1,
            "cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N",
            "cvss_v4_vector": None,
        },
        {
            "d_ru": "Локальная эскалация привилегий через setuid-бинарь",
            "d_en": None,
            "cwe_id": "CWE-269", "cwe_name": "Improper Privilege Management",
            "epss": 0.1, "kev": 0, "exploit": 0,
            "cvss_v3_vector": "CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H",
            "cvss_v4_vector": None,
        },
        # Запись без описаний — должна быть отфильтрована датасетом.
        {
            "d_ru": "   ", "d_en": None,
            "cwe_id": "CWE-79", "cwe_name": "XSS",
            "epss": 0.2, "kev": 0, "exploit": 0,
            "cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:L/I:L/A:N",
            "cvss_v4_vector": None,
        },
    ]
    path = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


@pytest.fixture()
def synthetic_config(tmp_path: Path, synthetic_train: Path) -> Path:
    """train.yaml с минимальным `dapt`-блоком и путём к фейк-train."""
    cfg = {
        "seed": 7,
        "dapt": {
            "output_dir": str(tmp_path / "dapt_out"),
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 5.0e-5,
            "warmup_ratio": 0.0,
            "max_length": 32,
            "mlm_probability": 0.15,
            "log_every_n_steps": 1,
            "save_total_limit": 1,
        },
        "paths": {
            "train_data": str(synthetic_train),
        },
        "model": {
            "pretrained_name": "bert-base-multilingual-cased",
        },
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# DAPTTextDataset.
# ---------------------------------------------------------------------------


def test_dataset_picks_ru_then_en_and_filters_empty(synthetic_train: Path) -> None:
    df = pd.read_parquet(synthetic_train)
    ds = DAPTTextDataset(df, tokenizer=_FakeTokenizer(), max_length=64)

    # 3 строки с непустым описанием — 4-я (только пробелы) отброшена.
    assert len(ds) == 3

    sample = ds[0]
    assert isinstance(sample["input_ids"], list)
    assert isinstance(sample["attention_mask"], list)
    assert "special_tokens_mask" in sample
    # Длина = body + 2 спец-токена, обрезано до max_length.
    assert len(sample["input_ids"]) <= 64
    assert len(sample["input_ids"]) == len(sample["attention_mask"])


def test_dataset_include_cwe_name_concatenates(synthetic_train: Path) -> None:
    """С ``include_cwe_name=True`` к описанию подклеивается CWE-имя через [SEP]."""
    df = pd.read_parquet(synthetic_train)
    ds_plain = DAPTTextDataset(df, tokenizer=_FakeTokenizer(), max_length=512)
    ds_cwe = DAPTTextDataset(
        df, tokenizer=_FakeTokenizer(), max_length=512, include_cwe_name=True
    )
    # У записи 0 cwe_name = "OS Command Injection" — должно добавиться в текст.
    assert len(ds_cwe[0]["input_ids"]) > len(ds_plain[0]["input_ids"])


def test_dataset_truncates_to_max_length() -> None:
    """``max_length`` соблюдается даже на очень длинной строке."""
    df = pd.DataFrame(
        [{"d_ru": "А" * 2000, "d_en": None}]  # очень длинная кириллица
    )
    ds = DAPTTextDataset(df, tokenizer=_FakeTokenizer(), max_length=16)
    assert len(ds) == 1
    assert len(ds[0]["input_ids"]) == 16


def test_dataset_empty_raises_only_on_explicit_check() -> None:
    """Пустой корпус — это валидное состояние датасета, не падает само по себе."""
    df = pd.DataFrame([{"d_ru": None, "d_en": None}])
    ds = DAPTTextDataset(df, tokenizer=_FakeTokenizer())
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# Config / overrides.
# ---------------------------------------------------------------------------


def test_load_dapt_config_uses_defaults_when_section_missing(tmp_path: Path) -> None:
    """Если в yaml нет ``dapt:``, всё равно возвращается полный конфиг."""
    raw = {"seed": 13, "paths": {"train_data": "x.parquet"}}
    path = tmp_path / "no_dapt.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = load_dapt_config(path)
    for key, value in DEFAULT_DAPT_CONFIG.items():
        assert cfg[key] == value
    assert cfg["_seed"] == 13
    assert cfg["_paths"]["train_data"] == "x.parquet"
    assert cfg["_pretrained_name"] == "bert-base-multilingual-cased"


def test_load_dapt_config_overrides_defaults(synthetic_config: Path) -> None:
    cfg = load_dapt_config(synthetic_config)
    assert cfg["epochs"] == 1
    assert cfg["batch_size"] == 2
    assert cfg["max_length"] == 32
    assert cfg["_seed"] == 7


def test_apply_debug_overrides_shrinks_run() -> None:
    cfg = {
        "epochs": 5,
        "batch_size": 32,
        "gradient_accumulation_steps": 4,
        "log_every_n_steps": 100,
    }
    out = _apply_debug_overrides(cfg)
    # Исходник не меняем.
    assert cfg["epochs"] == 5
    # Дебаг — короче и логирует каждый шаг.
    assert out["epochs"] == 1
    assert out["batch_size"] == 2
    assert out["gradient_accumulation_steps"] == 1
    assert out["log_every_n_steps"] == 1


# ---------------------------------------------------------------------------
# run() end-to-end (с замоканными HF-классами).
# ---------------------------------------------------------------------------


class _FakeTrainOutput:
    training_loss = 1.234


class _FakeTrainer:
    """Замена ``transformers.Trainer``: фиксирует вход, имитирует .train()/.save_model()."""

    captured: dict[str, Any] = {}

    def __init__(self, model, args, train_dataset, data_collator, tokenizer) -> None:
        _FakeTrainer.captured = {
            "model": model,
            "args": args,
            "train_dataset_len": len(train_dataset),
            "data_collator": data_collator,
            "tokenizer": tokenizer,
        }

    def train(self):
        return _FakeTrainOutput()

    def save_model(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "pytorch_model.bin").write_bytes(b"FAKE")
        _FakeTrainer.captured["save_model_called_with"] = output_dir


class _FakeModel:
    @classmethod
    def from_pretrained(cls, name: str) -> "_FakeModel":
        obj = cls()
        obj.name = name
        return obj


class _FakeTokenizerCls(_FakeTokenizer):
    @classmethod
    def from_pretrained(cls, name: str) -> "_FakeTokenizerCls":
        return cls()

    def save_pretrained(self, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class _FakeCollator:
    def __init__(self, tokenizer, mlm, mlm_probability) -> None:
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability


class _FakeTrainingArgs:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _patch_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Меняет тяжёлые ``transformers``-импорты внутри ``dapt.run`` на фейки."""
    fake_module = mock.MagicMock()
    fake_module.AutoModelForMaskedLM = _FakeModel
    fake_module.AutoTokenizer = _FakeTokenizerCls
    fake_module.DataCollatorForLanguageModeling = _FakeCollator
    fake_module.Trainer = _FakeTrainer
    fake_module.TrainingArguments = _FakeTrainingArgs

    import sys
    monkeypatch.setitem(sys.modules, "transformers", fake_module)


def test_run_calls_trainer_with_correct_config(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_config: Path,
    tmp_path: Path,
) -> None:
    """run() собирает датасет, инициирует Trainer и сохраняет модель."""
    _patch_hf(monkeypatch)
    _FakeTrainer.captured.clear()

    metrics = run(synthetic_config)

    cap = _FakeTrainer.captured
    assert cap["train_dataset_len"] == 3  # 3 непустых описания
    assert isinstance(cap["data_collator"], _FakeCollator)
    assert cap["data_collator"].mlm is True
    assert cap["data_collator"].mlm_probability == pytest.approx(0.15)
    # save_model вызвался с output_dir из конфига.
    assert cap["save_model_called_with"] == str(tmp_path / "dapt_out")
    assert (tmp_path / "dapt_out" / "pytorch_model.bin").exists()
    assert (tmp_path / "dapt_out" / "tokenizer_config.json").exists()

    assert metrics["samples_used"] == 3
    assert metrics["epochs"] == 1
    assert metrics["train_loss"] == pytest.approx(1.234)
    assert metrics["output_dir"] == str(tmp_path / "dapt_out")


def test_run_debug_mode_trims_dataset(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_config: Path,
) -> None:
    """``debug=True`` режет корпус до 20 строк (а здесь их 4)."""
    _patch_hf(monkeypatch)
    _FakeTrainer.captured.clear()

    metrics = run(synthetic_config, debug=True)
    # Все 3 непустых описания (debug не успевает срезать ниже, корпус мал).
    assert metrics["samples_used"] == 3
    cap = _FakeTrainer.captured
    # batch_size и epochs принудительно ужаты до debug-значений.
    assert cap["args"].num_train_epochs == 1
    assert cap["args"].per_device_train_batch_size == 2


def test_run_raises_on_missing_train(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Если train.parquet не существует — понятная FileNotFoundError."""
    _patch_hf(monkeypatch)
    cfg = {
        "dapt": {"output_dir": str(tmp_path / "out")},
        "paths": {"train_data": str(tmp_path / "no-such-file.parquet")},
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="train parquet"):
        run(path)


# ---------------------------------------------------------------------------
# train.py --pretrained-name пробрасывание.
# ---------------------------------------------------------------------------


def test_train_run_overrides_pretrained_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """train.run(pretrained_name=...) кладёт имя в config["model"], а не игнорит."""
    from src.training import train as train_mod

    cfg = {
        "seed": 42,
        "stage1": {
            "epochs": 1, "batch_size": 2, "learning_rate": 1.0e-5, "warmup_ratio": 0.0,
            "metrics": ["AV"], "metric_classes": {"AV": 4},
        },
        "stage2": {
            "epochs": 1, "batch_size": 2, "learning_rate": 1.0e-5, "warmup_ratio": 0.0,
            "metrics": ["AV"], "metric_classes": {"AV": 4}, "reinit_heads": [],
        },
        "common": {},
        "paths": {
            "train_data": str(tmp_path / "train.parquet"),
            "val_data": str(tmp_path / "val.parquet"),
            "models_dir": str(tmp_path / "models"),
            "checkpoints_dir": str(tmp_path / "ckpt"),
            "tensorboard_dir": str(tmp_path / "tb"),
        },
        "model": {"pretrained_name": "bert-base-multilingual-cased"},
    }
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Замоканная цепочка: load_config возвращает наш cfg; всё тяжёлое
    # (parquet, токенайзер, Trainer) — отрезается до того как стартует stage 1.
    monkeypatch.setattr(train_mod, "load_config", lambda _: cfg)
    monkeypatch.setattr(train_mod, "set_seed", lambda *_: None)

    captured: dict[str, Any] = {}

    def fake_pd_read_parquet(path):
        captured["read_path"] = path
        # Останавливаем выполнение здесь — нам важно лишь увидеть, что
        # cfg["model"]["pretrained_name"] был переопределён до запуска
        # тренировки.
        raise StopIteration("stop before training")

    monkeypatch.setattr(train_mod.pd, "read_parquet", fake_pd_read_parquet)

    with pytest.raises(StopIteration):
        train_mod.run(
            stage=1,
            config_path=cfg_path,
            pretrained_name="models/mbert_dapt",
        )
    assert cfg["model"]["pretrained_name"] == "models/mbert_dapt"
