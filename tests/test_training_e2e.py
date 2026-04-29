"""Интеграционные тесты двухэтапного обучения CVSSModel.

Используют tiny-BERT с рандомной инициализацией (без HuggingFace download)
и synthetic dataframe из 10 строк, чтобы прогон укладывался в десятки секунд
на CPU.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel

from src.data_preparation import CVSSDataset, CWEEncoder, FeaturesEncoder
from src.model import CVSSModel
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer
from src.training.utils import set_seed


# ---------------------------------------------------------------------------
# Tiny-BERT и фейковый токенизатор: без download HuggingFace.
# ---------------------------------------------------------------------------

_TINY_VOCAB = 200
_TINY_HIDDEN = 32
_TINY_MAXLEN = 16


def _tiny_transformer() -> BertModel:
    """Случайно инициализированный BERT минимального размера для тестов."""
    cfg = BertConfig(
        vocab_size=_TINY_VOCAB,
        hidden_size=_TINY_HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=_TINY_MAXLEN,
        type_vocab_size=2,
        pad_token_id=0,
    )
    return BertModel(cfg)


class _DummyTokenizer:
    """Детерминированный 'токенизатор': один и тот же текст → один и тот же id."""

    def __init__(self, max_length: int = _TINY_MAXLEN, vocab_size: int = _TINY_VOCAB) -> None:
        self.max_length = max_length
        self.vocab_size = vocab_size

    def tokenize(self, text: str, max_length: int | None = None) -> Dict[str, list[int]]:
        ml = int(max_length or self.max_length)
        text = text or ""
        # CLS + детерминированные id из символов + SEP, паддим нулями.
        ids: list[int] = [101]
        for i, ch in enumerate(text[: ml - 2]):
            ids.append(2 + (ord(ch) + i) % (self.vocab_size - 4))
        ids.append(102)
        attn = [1] * len(ids)
        # Паддинг до max_length.
        pad = ml - len(ids)
        if pad > 0:
            ids.extend([0] * pad)
            attn.extend([0] * pad)
        return {"input_ids": ids[:ml], "attention_mask": attn[:ml]}


# ---------------------------------------------------------------------------
# Синтетический датафрейм.
# ---------------------------------------------------------------------------

# Достаточно разнообразные CVSS-векторы, чтобы у каждой метрики были минимум
# 2 разных класса в небольшом датасете.
# Все векторы содержат явное значение E — иначе парсер ставит -100
# (IGNORE_INDEX), голова E пропускается в loss → у неё не будет градиента
# и тест на ненулевые градиенты падает.
_V3_VECTORS = [
    "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H/E:H",
    "CVSS:3.1/AV:L/AC:H/PR:H/UI:R/S:U/C:L/I:L/A:L/E:U",
    "CVSS:3.1/AV:N/AC:L/PR:L/UI:R/S:U/C:H/I:N/A:N/E:F",
    "CVSS:3.1/AV:A/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N/E:P",
    "CVSS:3.1/AV:P/AC:H/PR:H/UI:R/S:U/C:N/I:N/A:H/E:H",
]
_V4_VECTORS = [
    "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H/E:A",
    "CVSS:4.0/AV:L/AC:H/AT:P/PR:H/UI:A/VC:L/VI:L/VA:L/SC:N/SI:N/SA:N/E:U",
    "CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:P/VC:H/VI:N/VA:N/SC:N/SI:N/SA:N/E:P",
    "CVSS:4.0/AV:A/AC:L/AT:N/PR:N/UI:N/VC:N/VI:H/VA:N/SC:L/SI:N/SA:N/E:A",
    "CVSS:4.0/AV:P/AC:H/AT:P/PR:H/UI:A/VC:N/VI:N/VA:H/SC:N/SI:N/SA:H/E:U",
]


def _make_dummy_dataframe(n_rows: int = 10) -> pd.DataFrame:
    """Синтетический датафрейм со всеми колонками, нужными CVSSDataset."""
    rng = np.random.default_rng(42)
    rows: List[Dict[str, Any]] = []
    for i in range(n_rows):
        rows.append(
            {
                "id": f"CVE-2026-{1000 + i}",
                "d_ru": f"Описание уязвимости номер {i} с инъекцией",
                "d_en": f"Vulnerability description number {i} with injection",
                "cwe_id": f"CWE-{[79, 89, 22, 287][i % 4]}",
                "cwe_name": f"CWE-name-{i % 4}",
                "epss": float(rng.random()),
                "kev": int(i % 2),
                "exploit": int((i + 1) % 3 == 0),
                "cvss_v3_vector": _V3_VECTORS[i % len(_V3_VECTORS)],
                "cvss_v4_vector": _V4_VECTORS[i % len(_V4_VECTORS)],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Конфиг для тестов: эпохи и батчи минимальные.
# ---------------------------------------------------------------------------

def _make_test_config(checkpoints_dir: Path, models_dir: Path, tb_dir: Path) -> Dict[str, Any]:
    return {
        "seed": 42,
        "stage1": {
            "learning_rate": 5.0e-3,
            "batch_size": 2,
            "epochs": 3,
            "warmup_ratio": 0.0,
            "metrics": ["AV", "AC", "PR", "UI", "VC", "VI", "VA", "E"],
            "metric_classes": {
                "AV": 4, "AC": 2, "PR": 3, "UI": 2,
                "VC": 4, "VI": 4, "VA": 4, "E": 5,
            },
        },
        "stage2": {
            "learning_rate": 5.0e-3,
            "batch_size": 2,
            "epochs": 2,
            "warmup_ratio": 0.0,
            "reinit_heads": ["AT", "SC", "SI", "SA", "E"],
            "metrics": [
                "AV", "AC", "AT", "PR", "UI",
                "VC", "VI", "VA", "SC", "SI", "SA", "E",
            ],
            "metric_classes": {
                "AV": 4, "AC": 2, "AT": 2, "PR": 3, "UI": 3,
                "VC": 3, "VI": 3, "VA": 3, "SC": 3, "SI": 3, "SA": 3, "E": 3,
            },
        },
        "common": {
            "weight_decay": 0.01,
            "dropout": 0.1,
            "gradient_clip": 1.0,
            "early_stopping_patience": 3,
            "mixed_precision": False,
            "log_every_n_batches": 1,
            "checkpoint_every_epoch": False,
        },
        "paths": {
            "models_dir": str(models_dir),
            "checkpoints_dir": str(checkpoints_dir),
            "tensorboard_dir": str(tb_dir),
            "train_data": "data/processed/train.parquet",
            "val_data": "data/processed/val.parquet",
            "test_data": "data/processed/test.parquet",
        },
    }


# ---------------------------------------------------------------------------
# Фикстуры.
# ---------------------------------------------------------------------------

@pytest.fixture()
def setup(tmp_path):
    """Полный набор для интеграционного теста: данные, модель, trainer."""
    set_seed(42)
    df = _make_dummy_dataframe(10)

    cwe_encoder = CWEEncoder().fit(df["cwe_id"])
    features_encoder = FeaturesEncoder()
    tokenizer = _DummyTokenizer(max_length=_TINY_MAXLEN)

    config = _make_test_config(
        checkpoints_dir=tmp_path / "ckpt",
        models_dir=tmp_path / "models",
        tb_dir=tmp_path / "tb",
    )

    return {
        "df": df,
        "tmp_path": tmp_path,
        "config": config,
        "cwe_encoder": cwe_encoder,
        "features_encoder": features_encoder,
        "tokenizer": tokenizer,
    }


def _build_model(config: dict, stage_key: str, num_cwe: int) -> CVSSModel:
    return CVSSModel(
        num_cwe=num_cwe,
        metric_classes=config[stage_key]["metric_classes"],
        transformer=_tiny_transformer(),
        cwe_embedding_dim=8,
        feature_hidden_dim=16,
        feature_output_dim=8,
        fusion_output_dim=16,
        dropout=0.0,
    )


def _make_loaders(setup: dict, version: str) -> tuple[DataLoader, DataLoader]:
    df = setup["df"]
    train_df = df.iloc[:8].reset_index(drop=True)
    val_df = df.iloc[8:].reset_index(drop=True)
    common = dict(
        tokenizer=setup["tokenizer"],
        cwe_encoder=setup["cwe_encoder"],
        features_encoder=setup["features_encoder"],
        version=version,
        max_length=_TINY_MAXLEN,
    )
    train_ds = CVSSDataset(train_df, **common)
    val_ds = CVSSDataset(val_df, **common)
    bs = setup["config"][f"stage{1 if version == 'v3' else 2}"]["batch_size"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0),
    )


# ---------------------------------------------------------------------------
# Тесты.
# ---------------------------------------------------------------------------

def test_stage1_one_epoch(setup) -> None:
    """Stage1: loss падает, у активных голов есть ненулевые градиенты,
    голова AT (отсутствует на этапе 1) в модели не появилась."""
    config = setup["config"]
    model = _build_model(config, "stage1", num_cwe=len(setup["cwe_encoder"]))
    train_loader, val_loader = _make_loaders(setup, version="v3")

    trainer = Trainer(config, model, device=torch.device("cpu"))
    history = trainer.train_stage1(train_loader, val_loader, train_df=setup["df"])
    trainer.close()

    train_losses = history["train_loss"]
    assert len(train_losses) >= 1
    # Финальный loss строго меньше начального — модель учится.
    assert train_losses[-1] < train_losses[0], (
        f"loss не упал: {train_losses}"
    )

    # У всех 8 stage1-голов градиенты есть и не нулевые.
    stage1_metrics = config["stage1"]["metrics"]
    for metric in stage1_metrics:
        head = model.heads[metric]
        assert head.weight.grad is not None, f"head {metric}: grad is None"
        assert head.weight.grad.abs().sum().item() > 0, (
            f"head {metric}: grad — все нули"
        )

    # Голова AT (CVSS v4.0-only) в stage1-модели вообще отсутствует.
    assert "AT" not in model.heads


def test_stage2_reinit_heads(setup) -> None:
    """Stage2: AT/SC/SI/SA/E переинициализируются; у E теперь 3 класса вместо 5;
    после короткого обучения train_loss падает."""
    config = setup["config"]

    # 1) Создаём stage1-модель и сохраняем её state_dict как best_stage1.pt,
    #    чтобы trainer.train_stage2 мог его загрузить.
    set_seed(42)
    model_stage1 = _build_model(config, "stage1", num_cwe=len(setup["cwe_encoder"]))
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_stage1.state_dict(), models_dir / "best_stage1.pt")

    # E на этапе 1 — пятиклассовая голова.
    assert model_stage1.heads["E"].out_features == 5

    # 2) Теперь готовим модель для stage2. Запоминаем веса E ДО reinit
    #    (мы будем сравнивать с ними после обучения).
    set_seed(42)
    model_stage2 = _build_model(config, "stage1", num_cwe=len(setup["cwe_encoder"]))
    e_before = model_stage2.heads["E"].weight.detach().clone()

    trainer = Trainer(config, model_stage2, device=torch.device("cpu"))
    train_loader, val_loader = _make_loaders(setup, version="v4")

    history = trainer.train_stage2(train_loader, val_loader, train_df=setup["df"])
    trainer.close()

    # 3) Голова E должна теперь иметь 3 выхода (CVSS v4.0), а не 5.
    assert model_stage2.heads["E"].out_features == 3, (
        f"head E.out_features = {model_stage2.heads['E'].out_features}, ожидалось 3"
    )

    # 4) Все 12 v4-голов в модели присутствуют.
    expected_heads = {
        "AV", "AC", "AT", "PR", "UI",
        "VC", "VI", "VA", "SC", "SI", "SA", "E",
    }
    assert set(model_stage2.heads.keys()) == expected_heads

    # 5) Reinit-головы AT/SC/SI/SA точно не нулевые после Xavier init.
    #    (нулевые матрицы означали бы, что reinit не сработал.)
    for metric in ("AT", "SC", "SI", "SA"):
        head = model_stage2.heads[metric]
        assert head.weight.abs().sum().item() > 0, (
            f"head {metric}: weights нулевые после reinit"
        )

    # 6) E была [5, hidden] до, стала [3, hidden] — формы уже не совпадают,
    #    значит reinit точно произошёл (старые веса нельзя было перенести).
    assert model_stage2.heads["E"].weight.shape != e_before.shape

    # 7) Train loss упал за 2 эпохи stage2.
    train_losses = history["train_loss"]
    assert len(train_losses) >= 2
    assert train_losses[-1] < train_losses[0], (
        f"stage2 loss не упал: {train_losses}"
    )


def test_early_stopping_triggers(setup, monkeypatch) -> None:
    """С monkeypatch _evaluate возвращает константный F1 → ES срабатывает на patience+1."""
    config = deepcopy(setup["config"])
    # Достаточно эпох, чтобы patience успел закончиться (patience=3 → 5-я эпоха).
    config["stage1"]["epochs"] = 10

    model = _build_model(config, "stage1", num_cwe=len(setup["cwe_encoder"]))
    train_loader, val_loader = _make_loaders(setup, version="v3")

    trainer = Trainer(config, model, device=torch.device("cpu"))

    constant_eval = {
        "val_loss": 1.0,
        "macro_f1": 0.5,
        "per_metric": {},
    }

    def _fake_eval(self, *args, **kwargs):  # noqa: ARG001
        return dict(constant_eval)

    monkeypatch.setattr(Trainer, "_evaluate", _fake_eval, raising=True)

    history = trainer.train_stage1(train_loader, val_loader, train_df=setup["df"])
    trainer.close()

    # patience=3 + 1 = должны выполниться ровно 5 эпох (1 улучшение vs -inf,
    # затем 4 эпохи без улучшения; на 5-й counter становится 4 > 3 → стоп).
    assert len(history["train_loss"]) == 5, (
        f"ожидалось 5 эпох до останова, прошло {len(history['train_loss'])}"
    )
    assert history["best_macro_f1"] == pytest.approx(0.5)


def test_checkpoint_save_load(setup) -> None:
    """Сохранение чекпоинта и загрузка в новую модель восстанавливает state_dict."""
    config = setup["config"]
    model = _build_model(config, "stage1", num_cwe=len(setup["cwe_encoder"]))
    train_loader, val_loader = _make_loaders(setup, version="v3")

    trainer = Trainer(config, model, device=torch.device("cpu"))

    # Создаём фиктивные оптимизатор/scheduler — они нужны save_checkpoint.
    optimizer = trainer._build_optimizer(config["stage1"])
    scheduler = trainer._build_scheduler(optimizer, total_steps=10, warmup_ratio=0.0)
    # Делаем хотя бы один шаг, чтобы scheduler.state_dict не был тривиальным.
    optimizer.step()
    scheduler.step()

    ckpt_path = setup["tmp_path"] / "ckpt" / "saved.pt"
    trainer.save_checkpoint(ckpt_path, optimizer, scheduler, epoch=1, stage=1)
    trainer.close()
    assert ckpt_path.exists()

    # Создаём новую модель той же архитектуры и преднамеренно «портим» её веса,
    # чтобы убедиться, что load_checkpoint реально их переписывает.
    # Re-seed внутри Trainer.__init__ может выдать те же случайные веса —
    # принудительная модификация снимает зависимость от состояния RNG.
    new_model = _build_model(config, "stage1", num_cwe=len(setup["cwe_encoder"]))
    with torch.no_grad():
        for p in new_model.parameters():
            p.add_(1.0)
    assert not torch.allclose(
        new_model.heads["AV"].weight, model.heads["AV"].weight
    )

    new_trainer = Trainer(config, new_model, device=torch.device("cpu"))
    meta = new_trainer.load_checkpoint(ckpt_path)
    new_trainer.close()

    assert meta["epoch"] == 1
    assert meta["stage"] == 1

    # state_dict должен совпадать побайтово.
    sd_old = model.state_dict()
    sd_new = new_model.state_dict()
    assert set(sd_old.keys()) == set(sd_new.keys())
    for k in sd_old:
        assert torch.allclose(sd_old[k], sd_new[k]), f"расходится параметр {k}"
