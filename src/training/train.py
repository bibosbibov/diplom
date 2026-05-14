"""CLI-точка входа для двухэтапного обучения CVSSModel.

Использование::

    python -m src.training.train --stage 1               # только этап 1
    python -m src.training.train --stage 2               # только этап 2
    python -m src.training.train --stage 0               # этап 1, затем 2
    python -m src.training.train --stage 1 --debug       # smoke test (10 строк, 1 эпоха)
    python -m src.training.train --stage 2 --resume models/checkpoints/stage1_epoch3.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_preparation import (
    CVSSDataset,
    CVSSTokenizer,
    CWEEncoder,
    FeaturesEncoder,
)
from src.model import CVSSModel
from src.training.trainer import Trainer
from src.training.utils import get_device, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Подготовка датафрейма.
# ---------------------------------------------------------------------------

# Имена колонок, которые ожидает CVSSDataset, и их возможные алиасы во входных
# parquet-файлах. Сборщик данных может писать любые из них — нормализуем.
_COLUMN_ALIASES = {
    "d_ru": ("d_ru", "description_ru", "description_russian"),
    "d_en": ("d_en", "description_en", "description_english"),
    "cwe_id": ("cwe_id", "cwe"),
    "cwe_name": ("cwe_name",),
    "epss": ("epss", "epss_score"),
    "kev": ("kev", "in_kev", "is_kev"),
    "exploit": ("exploit", "has_exploit", "exploitdb"),
    "cvss_v3_vector": ("cvss_v3_vector", "cvss_v31_vector", "cvss3_vector"),
    "cvss_v4_vector": ("cvss_v4_vector", "cvss_v40_vector", "cvss4_vector"),
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит названия колонок к каноничным именам, ожидаемым CVSSDataset."""
    rename: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = canonical
                break
    return df.rename(columns=rename) if rename else df


# ---------------------------------------------------------------------------
# Конфиг.
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_debug_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """В debug-режиме урезаем эпохи и батч, чтобы прогон занимал минуты."""
    cfg = deepcopy(config)
    for stage_key in ("stage1", "stage2"):
        if stage_key in cfg:
            cfg[stage_key]["epochs"] = 1
            cfg[stage_key]["batch_size"] = 2
    cfg.setdefault("common", {})
    cfg["common"]["mixed_precision"] = False
    cfg["common"]["log_every_n_batches"] = 1
    cfg["common"]["checkpoint_every_epoch"] = False
    return cfg


# ---------------------------------------------------------------------------
# Сборка датасета и DataLoader-ов.
# ---------------------------------------------------------------------------


def _make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: CVSSTokenizer,
    cwe_encoder: CWEEncoder,
    features_encoder: FeaturesEncoder,
    version: str,
    batch_size: int,
    max_length: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = CVSSDataset(
        train_df,
        tokenizer,
        cwe_encoder,
        features_encoder,
        version=version,
        max_length=max_length,
    )
    val_ds = CVSSDataset(
        val_df,
        tokenizer,
        cwe_encoder,
        features_encoder,
        version=version,
        max_length=max_length,
    )
    # num_workers=0 чтобы избежать проблем с pickle/spawn на Windows.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def _build_model(config: Mapping[str, Any], stage_key: str, num_cwe: int) -> CVSSModel:
    """Создаёт CVSSModel с числом классов голов под нужный этап."""
    metric_classes = dict(config[stage_key]["metric_classes"])
    pretrained = config.get("model", {}).get("pretrained_name") or "bert-base-multilingual-cased"
    return CVSSModel(
        num_cwe=num_cwe,
        metric_classes=metric_classes,
        pretrained_name=pretrained,
    )


# ---------------------------------------------------------------------------
# Главный пайплайн.
# ---------------------------------------------------------------------------


def run(
    stage: int,
    config_path: Path,
    resume: Path | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Точка входа без argparse — удобно вызывать из тестов и ноутбуков."""
    config = load_config(config_path)
    if debug:
        config = _apply_debug_overrides(config)
    set_seed(int(config.get("seed", 42)))

    paths = config.get("paths", {})
    train_path = Path(paths.get("train_data", "data/processed/train.parquet"))
    val_path = Path(paths.get("val_data", "data/processed/val.parquet"))
    train_df = _normalize_columns(pd.read_parquet(train_path))
    val_df = _normalize_columns(pd.read_parquet(val_path))

    if debug:
        train_df = train_df.head(10).reset_index(drop=True)
        val_df = val_df.head(10).reset_index(drop=True)
        logger.info("debug: trimmed train→%d, val→%d rows", len(train_df), len(val_df))

    device = get_device()
    if device.type == "cpu" and not debug:
        print(
            "[WARN] Обучение на CPU займёт >24 часов. Используйте --debug для "
            "проверки или Google Colab для реального обучения.",
            file=sys.stderr,
        )

    # Кодировщики обучаются на train (CWE-словарь, маркеры и т.д.).
    # NaN/None отфильтровываются явно: CWEEncoder.fit сортирует уникальные
    # значения и падает на смешанных типах (NaN-float vs str).
    cwe_series = train_df.get("cwe_id", pd.Series(dtype=str)).dropna().astype(str)
    cwe_encoder = CWEEncoder().fit(cwe_series.tolist())
    features_encoder = FeaturesEncoder()

    pretrained = config.get("model", {}).get("pretrained_name") or "bert-base-multilingual-cased"
    tokenizer = CVSSTokenizer(model_name=pretrained, max_length=512)
    max_length = int(config.get("data_preparation", {}).get("max_length", 512))

    history: dict[str, Any] = {}

    if stage in (1, 0):
        logger.info("=== STAGE 1 ===")
        # Фильтруем строки без cvss_v3_vector — иначе все 8 голов получают
        # IGNORE_INDEX и MultiTaskLoss возвращает (logits*0).sum() с нулевым
        # градиентом, превращая батч в no-op.
        train_df_v3 = train_df[train_df["cvss_v3_vector"].notna()].reset_index(drop=True)
        val_df_v3 = val_df[val_df["cvss_v3_vector"].notna()].reset_index(drop=True)
        logger.info(
            "stage1: filtered to v3-only — train=%d, val=%d",
            len(train_df_v3),
            len(val_df_v3),
        )
        model = _build_model(config, "stage1", num_cwe=len(cwe_encoder))
        trainer = Trainer(config, model, device=device)
        if resume is not None:
            trainer.load_checkpoint(resume)
        train_loader, val_loader = _make_loaders(
            train_df_v3,
            val_df_v3,
            tokenizer,
            cwe_encoder,
            features_encoder,
            version="v3",
            batch_size=int(config["stage1"]["batch_size"]),
            max_length=max_length,
        )
        history["stage1"] = trainer.train_stage1(train_loader, val_loader, train_df=train_df_v3)
        trainer.close()

    if stage in (2, 0):
        logger.info("=== STAGE 2 ===")
        # Фильтруем строки без cvss_v4_vector: в датасете лишь ~3.8% строк
        # имеют v4-вектор, остальные дали бы пустой батч с нулевым градиентом.
        train_df_v4 = train_df[train_df["cvss_v4_vector"].notna()].reset_index(drop=True)
        val_df_v4 = val_df[val_df["cvss_v4_vector"].notna()].reset_index(drop=True)
        logger.info(
            "stage2: filtered to v4-only — train=%d, val=%d",
            len(train_df_v4),
            len(val_df_v4),
        )
        # Если идём stage 0 (1+2), модель надо построить под v4-головы заранее.
        # Trainer всё равно сделает reinit, но размеры и порядок голов должны
        # соответствовать stage2.metric_classes уже на этапе создания.
        model = _build_model(config, "stage2", num_cwe=len(cwe_encoder))
        trainer = Trainer(config, model, device=device)
        if resume is not None and stage == 2:
            trainer.load_checkpoint(resume)
        train_loader, val_loader = _make_loaders(
            train_df_v4,
            val_df_v4,
            tokenizer,
            cwe_encoder,
            features_encoder,
            version="v4",
            batch_size=int(config["stage2"]["batch_size"]),
            max_length=max_length,
        )
        history["stage2"] = trainer.train_stage2(train_loader, val_loader, train_df=train_df_v4)
        trainer.close()

        # Финальная модель сохраняется только когда выполнен stage 2.
        models_dir = Path(paths.get("models_dir", "models/"))
        models_dir.mkdir(parents=True, exist_ok=True)
        final_path = models_dir / "final_model.pt"
        torch.save(model.state_dict(), final_path)
        logger.info("Сохранена финальная модель: %s", final_path)

    return history


# ---------------------------------------------------------------------------
# argparse.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Двухэтапное обучение CVSSModel (CVSS v3.1 → CVSS v4.0).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(0, 1, 2),
        required=True,
        help="1 — только stage1, 2 — только stage2, 0 — оба последовательно",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="путь к YAML с гиперпараметрами",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="путь к чекпоинту для продолжения",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="ограничивает train/val до 10 записей и 1 эпохи (smoke test)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)
    run(stage=args.stage, config_path=args.config, resume=args.resume, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
