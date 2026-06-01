"""Domain-Adaptive Pretraining (DAPT) mBERT на корпусе уязвимостей.

Запускается ОДИН РАЗ перед :func:`src.training.train.run` (этапом 1).
Берёт все описания уязвимостей (``d_ru`` приоритетнее ``d_en``) из
обучающего parquet, маскирует 15% токенов и доучивает предобученный mBERT
1–2 эпохи на задаче Masked Language Modeling. Полученные веса
сохраняются в ``models/mbert_dapt/`` в формате HuggingFace
``save_pretrained`` — оттуда их подхватывает :class:`CVSSModel`, если
передать путь через ``--pretrained-name``.

Логика отбора описаний полностью повторяет :class:`TextProcessor`: если
есть русский текст — берём его, иначе английский. Это гарантирует, что
domain-adapted представление одинаково подходит для обоих сценариев
``stage 1`` и ``stage 2``.

Запуск из CLI::

    python -m src.training.dapt --config configs/train.yaml
    python -m src.training.dapt --config configs/train.yaml --debug

Архитектурное замечание: грузим mBERT через ``AutoModelForMaskedLM`` —
это та же модель, что и в :class:`CVSSModel.transformer`, но с прицепленной
MLM-головой ``cls.predictions`` (она лежит в чекпоинте mBERT, при
``AutoModel.from_pretrained`` отбрасывается). После DAPT эту MLM-голову
не сохраняем отдельно — ``save_pretrained`` положит и backbone, и
MLM-голову; :class:`CVSSModel` при загрузке возьмёт только backbone.
"""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from src.data_preparation import TextProcessor
from src.training.utils import get_device, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------


class DAPTTextDataset(Dataset):
    """Корпус описаний уязвимостей для MLM-предобучения.

    Каждый ``__getitem__`` возвращает токенизированный ``input_ids`` +
    ``attention_mask``. Маскирование выполняет
    :class:`transformers.DataCollatorForLanguageModeling`, поэтому здесь
    его не делаем.

    Args:
        dataframe: DataFrame с колонками ``d_ru`` и/или ``d_en``.
        tokenizer: HuggingFace-токенайзер (через
            ``AutoTokenizer.from_pretrained``).
        max_length: максимальная длина последовательности (по умолчанию 512).
        text_processor: ``TextProcessor`` для нормализации текста; если
            ``None`` — создаётся новый.
        include_cwe_name: если ``True``, к описанию подклеивается
            ``[SEP] cwe_name`` (как в ``CVSSDataset``); если ``False`` —
            только описание. Default ``False``: на этапе DAPT хотим
            адаптировать представление текста уязвимости как такового,
            без перекоса в сторону названий CWE.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Any,
        max_length: int = 512,
        text_processor: TextProcessor | None = None,
        include_cwe_name: bool = False,
    ) -> None:
        self._df = dataframe.reset_index(drop=True)
        self._tokenizer = tokenizer
        self._max_length = int(max_length)
        self._text_processor = text_processor or TextProcessor()
        self._include_cwe_name = include_cwe_name

        # Заранее посчитанные строки описаний — экономит CPU на тяжёлом цикле.
        self._texts: list[str] = self._extract_texts()

    def _extract_texts(self) -> list[str]:
        texts: list[str] = []
        for _, row in self._df.iterrows():
            if self._include_cwe_name:
                text = self._text_processor.prepare_text(
                    row.get("d_ru"), row.get("d_en"), row.get("cwe_name")
                )
            else:
                picked = TextProcessor._pick_description(row.get("d_ru"), row.get("d_en"))
                text = self._text_processor.clean(picked)
            if text:
                texts.append(text)
        return texts

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        encoding = self._tokenizer(
            self._texts[idx],
            truncation=True,
            max_length=self._max_length,
            padding=False,
            return_special_tokens_mask=True,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
        }


# ---------------------------------------------------------------------------
# DAPT-config helpers.
# ---------------------------------------------------------------------------

#: Дефолты, если в ``configs/train.yaml`` не задана секция ``dapt``.
DEFAULT_DAPT_CONFIG: dict[str, Any] = {
    "output_dir": "models/mbert_dapt",
    "epochs": 2,
    "batch_size": 16,
    "learning_rate": 5.0e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "mlm_probability": 0.15,
    "max_length": 512,
    "gradient_accumulation_steps": 1,
    "include_cwe_name": False,
    "log_every_n_steps": 50,
    "save_total_limit": 1,
}


def load_dapt_config(yaml_path: Path) -> dict[str, Any]:
    """Грузит секцию ``dapt`` из ``configs/train.yaml``, проставляя дефолты."""
    with yaml_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    cfg = deepcopy(DEFAULT_DAPT_CONFIG)
    cfg.update(raw.get("dapt", {}) or {})
    cfg["_paths"] = dict(raw.get("paths", {}))
    cfg["_seed"] = int(raw.get("seed", 42))
    cfg["_pretrained_name"] = (
        (raw.get("model", {}) or {}).get("pretrained_name")
        or "bert-base-multilingual-cased"
    )
    return cfg


# ---------------------------------------------------------------------------
# Main pipeline.
# ---------------------------------------------------------------------------


def _apply_debug_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """``--debug``: 1 эпоха, batch 2 — для smoke test'а за минуту."""
    cfg = deepcopy(config)
    cfg["epochs"] = 1
    cfg["batch_size"] = 2
    cfg["gradient_accumulation_steps"] = 1
    cfg["log_every_n_steps"] = 1
    return cfg


def run(config_path: Path, debug: bool = False) -> dict[str, Any]:
    """DAPT-цикл без argparse — удобен для тестов и ноутбуков.

    Args:
        config_path: путь к ``configs/train.yaml`` (читается секция ``dapt``).
        debug: если True, ограничивает корпус до 20 строк и 1 эпохи.

    Returns:
        dict с метаданными прогона: ``{"output_dir", "train_loss",
        "samples_used", "epochs"}``.
    """
    # Локальный импорт: transformers тяжёлый, не хотим тянуть его в
    # ``src.training`` модулях, которые могут импортироваться без него.
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    config_path = Path(config_path)
    cfg = load_dapt_config(config_path)
    if debug:
        cfg = _apply_debug_overrides(cfg)
    set_seed(cfg["_seed"])

    paths = cfg["_paths"]
    train_path = Path(paths.get("train_data", "data/processed/train.parquet"))
    if not train_path.exists():
        raise FileNotFoundError(f"train parquet не найден: {train_path}")

    pretrained_name = cfg["_pretrained_name"]
    logger.info(
        "DAPT старт: bckb=%s, train=%s, debug=%s, device=%s",
        pretrained_name,
        train_path,
        debug,
        get_device(),
    )

    df = pd.read_parquet(train_path, columns=_required_columns(cfg["include_cwe_name"]))
    if debug:
        df = df.head(20).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    dataset = DAPTTextDataset(
        df,
        tokenizer=tokenizer,
        max_length=int(cfg["max_length"]),
        include_cwe_name=bool(cfg["include_cwe_name"]),
    )
    if len(dataset) == 0:
        raise ValueError("после фильтрации пустых описаний корпус DAPT пуст")
    logger.info("DAPT корпус: %d описаний (из %d строк)", len(dataset), len(df))

    model = AutoModelForMaskedLM.from_pretrained(pretrained_name)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(cfg["mlm_probability"]),
    )

    output_dir = Path(cfg["output_dir"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=int(cfg["epochs"]),
        per_device_train_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        logging_steps=int(cfg["log_every_n_steps"]),
        save_strategy="epoch",
        save_total_limit=int(cfg["save_total_limit"]),
        fp16=use_fp16,
        report_to=[],  # без wandb / TB интеграции HF — TB у нас отдельный
        seed=cfg["_seed"],
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = {
        "output_dir": str(output_dir),
        "train_loss": float(train_result.training_loss),
        "samples_used": len(dataset),
        "epochs": int(cfg["epochs"]),
    }
    logger.info(
        "DAPT завершён: loss=%.4f, %d примеров, %d эпох, сохранено в %s",
        metrics["train_loss"],
        metrics["samples_used"],
        metrics["epochs"],
        metrics["output_dir"],
    )
    return metrics


def _required_columns(include_cwe_name: bool) -> list[str]:
    cols = ["d_ru", "d_en"]
    if include_cwe_name:
        cols.append("cwe_name")
    return cols


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DAPT — доменная адаптация mBERT на корпусе уязвимостей.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="путь к YAML с секцией dapt",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="ограничивает корпус до 20 описаний и 1 эпохи (smoke test)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)
    run(config_path=args.config, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DAPTTextDataset",
    "DEFAULT_DAPT_CONFIG",
    "load_dapt_config",
    "main",
    "run",
]
