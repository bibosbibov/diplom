"""Стратифицированное разбиение датасета 70/15/15.

ВАЖНО: разбиение выполняется по уникальному CVE-идентификатору, чтобы исключить
утечку данных между train/val/test (одна CVE не может оказаться сразу в двух
наборах при последующем дообучении на CVSS v4.0).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ._config import load_config
from ._logging import get_logger, setup_logger

logger = get_logger("split")

_AV_RE = re.compile(r"AV:([NALP])", re.IGNORECASE)


def _stratify_key(vector: object) -> str:
    """Извлекает значение метрики AV для стратификации.

    Принимает строку, None или NaN; всё, что не строка, считается ``UNK``.
    """
    if not isinstance(vector, str) or not vector:
        return "UNK"
    match = _AV_RE.search(vector)
    return match.group(1).upper() if match else "UNK"


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    stratify_column: str = "cvss_vector",
    split_unit: str = "cve_id",
) -> dict[str, pd.DataFrame]:
    """Делит датасет на train/val/test со стратификацией по AV."""
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size должно равняться 1.0")

    df = df.copy()
    df = df.dropna(subset=[split_unit]).drop_duplicates(subset=[split_unit])

    df["_stratify"] = df[stratify_column].apply(_stratify_key)
    counts = df["_stratify"].value_counts()
    rare = counts[counts < 2].index
    if len(rare):
        logger.warning("Классы %s имеют <2 примеров — переводим в 'UNK'", list(rare))
        df.loc[df["_stratify"].isin(rare), "_stratify"] = "UNK"

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df["_stratify"],
    )

    rel_test = test_size / (val_size + test_size)
    temp_class_counts = temp_df["_stratify"].value_counts()
    can_stratify = temp_df["_stratify"].nunique() > 1 and temp_class_counts.min() >= 2
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        random_state=seed,
        stratify=temp_df["_stratify"] if can_stratify else None,
    )

    for part in (train_df, val_df, test_df):
        part.drop(columns="_stratify", inplace=True)

    logger.info(
        "Размер сплитов: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    _check_no_leakage(train_df, val_df, test_df, key=split_unit)

    return {"train": train_df, "val": val_df, "test": test_df}


def _check_no_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    key: str,
) -> None:
    train_ids = set(train[key].dropna())
    val_ids = set(val[key].dropna())
    test_ids = set(test[key].dropna())
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError(f"Обнаружена утечка по полю {key} между сплитами")


def split_and_save(
    config: dict[str, Any] | None = None,
    input_path: str | Path | None = None,
) -> dict[str, Path]:
    """Загружает dataset.parquet, делит и сохраняет train/val/test."""
    setup_logger()
    config = config or load_config()
    paths = config.get("paths", {})
    raw_dir = Path(paths.get("data_raw", "data/raw"))
    processed_dir = Path(paths.get("data_processed", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    src = Path(input_path) if input_path else raw_dir / "dataset.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Датасет не найден: {src}")
    logger.info("Загрузка датасета: %s", src)
    df = pd.read_parquet(src)

    seed = int(config.get("project", {}).get("seed", 42))
    parts = split_dataset(
        df, train_size=0.70, val_size=0.15, test_size=0.15, seed=seed
    )

    targets = {
        "train": Path(paths.get("train_split", processed_dir / "train.parquet")),
        "val": Path(paths.get("val_split", processed_dir / "val.parquet")),
        "test": Path(paths.get("test_split", processed_dir / "test.parquet")),
    }
    for name, frame in parts.items():
        target = targets[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        logger.info("Сохранён %s: %s", name, target)

    return targets


def main() -> None:  # pragma: no cover - CLI обёртка
    parser = argparse.ArgumentParser(description="Стратифицированный split 70/15/15")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", default=None, help="Путь к dataset.parquet")
    args = parser.parse_args()
    split_and_save(load_config(args.config), input_path=args.input)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["split_dataset", "split_and_save"]
