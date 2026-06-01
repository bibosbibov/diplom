"""Дообучение голоВЫ Scope (CVSS v3.1 S-метрика) поверх замороженного stage 1.

Идея: stage 1-модель уже выучила хорошее представление текста уязвимости.
Для метрики **Scope** (отсутствует у текущей stage 1, есть только в CVSS 3.1)
достаточно прицепить одну линейную голову ``Linear(512, 2)`` к выходу
:class:`FusionLayer` и обучить её одну, не трогая остальные веса.

Это **классический transfer learning**: бэкбон заморожен, downstream-классификатор
тренируется поверх кэшированных представлений. На корпусе из ~122 тыс. v3-векторов
обучение занимает 5–10 минут на T4 (1 проход замороженного forward для кэша
+ N эпох обучения линейного слоя — последние занимают доли секунды).

Артефакт — :data:`OUTPUT_PATH` (``models/scope_head_v3.pt``) — словарь::

    {
        "state_dict": {"weight": tensor[2, 512], "bias": tensor[2]},
        "classes": ["U", "C"],
        "trained_from": "<путь к stage1 чекпойнту>",
        "val_accuracy": float,
        "val_f1_macro": float,
        "n_train": int,
        "n_val": int,
        "epochs": int,
        "config": {...},  # гиперпараметры
    }

Запуск из CLI::

    python -m src.training.train_scope_head                                     \\
        --stage1 models/dapt_mbert/best_stage1.pt                               \\
        --output models/scope_head_v3.pt                                        \\
        --epochs 10                                                             \\
        --batch-size 256                                                        \\
        --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.data_preparation import (
    CVSSDataset,
    CVSSTokenizer,
    CWEEncoder,
    FeaturesEncoder,
    TextProcessor,
)
from src.model import CVSSModel
from src.training.utils import get_device, set_seed

logger = logging.getLogger(__name__)

#: Классы Scope в порядке индексации (U = Unchanged по умолчанию ⇒ индекс 0).
SCOPE_CLASSES: tuple[str, ...] = ("U", "C")

#: Regex для извлечения значения метрики S из CVSS 3.x вектора.
_SCOPE_RE = re.compile(r"\bS:([UC])\b")

#: Дефолты CLI.
DEFAULT_STAGE1 = Path("models/dapt_mbert/best_stage1.pt")
DEFAULT_OUTPUT = Path("models/scope_head_v3.pt")
DEFAULT_TRAIN_DATA = Path("data/processed/train.parquet")
DEFAULT_VAL_DATA = Path("data/processed/val.parquet")
DEFAULT_CWE_VOCAB = Path("data/processed/cwe_vocab.json")
DEFAULT_CONFIG = Path("configs/train.yaml")


# ---------------------------------------------------------------------------
# Извлечение метки Scope.
# ---------------------------------------------------------------------------


def extract_scope(vector_str: str | None) -> str | None:
    """Возвращает значение метрики ``S`` из CVSS 3.x вектора или ``None``.

    ``parse_v3_vector`` намеренно игнорирует Scope (в v4 этой метрики нет),
    поэтому здесь отдельная регулярка вместо переиспользования.
    """
    if not isinstance(vector_str, str):
        return None
    match = _SCOPE_RE.search(vector_str)
    return match.group(1) if match else None


def _filter_with_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет строки с распарсенной меткой Scope. Добавляет колонку ``scope``."""
    df = df.copy()
    df["scope"] = df["cvss_v3_vector"].map(extract_scope)
    return df[df["scope"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Загрузка stage 1 модели.
# ---------------------------------------------------------------------------


def _load_metric_classes(config_path: Path, stage: str) -> dict[str, int]:
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    classes = cfg.get(stage, {}).get("metric_classes")
    if not classes:
        raise ValueError(f"В {config_path} нет {stage}.metric_classes")
    return dict(classes)


def load_frozen_backbone(
    stage1_path: Path,
    cwe_vocab_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[CVSSModel, CWEEncoder]:
    """Грузит stage 1 чекпойнт в режиме eval, отключает grad для всех весов."""
    cwe_encoder = CWEEncoder.load(cwe_vocab_path)
    metric_classes = _load_metric_classes(config_path, stage="stage1")

    model = CVSSModel(
        num_cwe=len(cwe_encoder),
        metric_classes=metric_classes,
    )
    state = torch.load(stage1_path, map_location="cpu", weights_only=False)
    if isinstance(state, Mapping) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Полная заморозка — никаких градиентов через backbone не пойдёт.
    for param in model.parameters():
        param.requires_grad = False
    logger.info("Stage 1 backbone загружен из %s (заморожен)", stage1_path)
    return model, cwe_encoder


# ---------------------------------------------------------------------------
# Кэширование h_fused.
# ---------------------------------------------------------------------------


@torch.no_grad()
def cache_fused_features(
    model: CVSSModel,
    dataframe: pd.DataFrame,
    tokenizer: CVSSTokenizer,
    cwe_encoder: CWEEncoder,
    features_encoder: FeaturesEncoder,
    text_processor: TextProcessor,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Один проход замороженного backbone, кэширует ``h_fused`` и метки Scope.

    Returns:
        (``fused`` ``FloatTensor[N, 512]``, ``labels`` ``LongTensor[N]``).
        Тензоры на CPU — для последующего быстрого обучения линейной головы.
    """
    dataset = CVSSDataset(
        dataframe,
        tokenizer=tokenizer,
        cwe_encoder=cwe_encoder,
        features_encoder=features_encoder,
        version="v3",  # для v3-токенизации; метки v3 не используются — берём свои.
        text_processor=text_processor,
        max_length=max_length,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    fused_chunks: list[torch.Tensor] = []
    scope_targets = torch.as_tensor(
        [SCOPE_CLASSES.index(s) for s in dataframe["scope"].tolist()],
        dtype=torch.long,
    )
    start = time.perf_counter()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cwe_idx = batch["cwe_idx"].to(device)
        features = batch["features"].to(device)

        h_text = model.encode_text(input_ids, attention_mask)
        h_feat = model.features_mlp(features, cwe_idx)
        h_fused = model.fusion(h_text, h_feat)  # [B, 512]
        fused_chunks.append(h_fused.detach().to("cpu"))

    fused = torch.cat(fused_chunks, dim=0)  # [N, 512]
    elapsed = time.perf_counter() - start
    logger.info(
        "Кэширование завершено: %d записей за %.1f сек (%.1f rec/sec)",
        fused.shape[0],
        elapsed,
        fused.shape[0] / max(elapsed, 1e-6),
    )
    return fused, scope_targets


# ---------------------------------------------------------------------------
# Обучение линейной головы.
# ---------------------------------------------------------------------------


def train_head(
    train_fused: torch.Tensor,
    train_targets: torch.Tensor,
    val_fused: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    patience: int = 3,
    seed: int = 42,
) -> tuple[nn.Linear, dict[str, Any]]:
    """Стандартный AdamW-цикл для одной :class:`nn.Linear` головы.

    Использует ранний останов по val-accuracy.

    Returns:
        ``(head, history)`` — лучшая голова (со снятыми ``state_dict()``
        весами) и словарь со списками потерь / метрик по эпохам и сводными
        полями ``best_epoch``, ``best_val_accuracy``, ``best_val_f1_macro``.
    """
    set_seed(seed)
    head = nn.Linear(512, len(SCOPE_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(train_fused, train_targets),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_fused_dev = val_fused.to(device)
    val_targets_dev = val_targets.to(device)

    history: dict[str, list[Any]] = {
        "train_loss": [], "val_loss": [],
        "val_accuracy": [], "val_f1_macro": [],
    }
    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] = {}
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        head.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        head.eval()
        with torch.no_grad():
            val_logits = head(val_fused_dev)
            val_loss = float(loss_fn(val_logits, val_targets_dev).item())
            val_preds = val_logits.argmax(dim=-1).cpu().numpy()
        val_acc = float(accuracy_score(val_targets.numpy(), val_preds))
        val_f1 = float(f1_score(val_targets.numpy(), val_preds, average="macro", zero_division=0))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_f1_macro"].append(val_f1)

        logger.info(
            "epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch, epochs, train_loss, val_loss, val_acc, val_f1,
        )
        print(
            f"[scope_head] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("early stopping на эпохе %d (best epoch=%d)", epoch, best_epoch)
                break

    # Восстанавливаем лучшие веса.
    head.load_state_dict(best_state)
    summary = {
        **history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_f1_macro": history["val_f1_macro"][best_epoch - 1]
            if best_epoch >= 1 else 0.0,
    }
    return head, summary


# ---------------------------------------------------------------------------
# Сборка артефакта.
# ---------------------------------------------------------------------------


def save_scope_head(
    head: nn.Linear,
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Сохраняет state_dict линейной головы + метаданные обучения."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "classes": list(SCOPE_CLASSES),
        **metadata,
    }
    torch.save(payload, output_path)
    logger.info("Scope-голова сохранена: %s", output_path)


# ---------------------------------------------------------------------------
# Основная процедура.
# ---------------------------------------------------------------------------


def run(
    stage1_path: Path = DEFAULT_STAGE1,
    output_path: Path = DEFAULT_OUTPUT,
    train_data: Path = DEFAULT_TRAIN_DATA,
    val_data: Path = DEFAULT_VAL_DATA,
    cwe_vocab_path: Path = DEFAULT_CWE_VOCAB,
    config_path: Path = DEFAULT_CONFIG,
    epochs: int = 10,
    batch_size_cache: int = 32,
    batch_size_train: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    patience: int = 3,
    seed: int = 42,
    max_length: int = 512,
    debug: bool = False,
) -> dict[str, Any]:
    """Обучение Scope-головы без argparse — удобно для ноутбуков и тестов."""

    device = get_device()
    set_seed(seed)
    logger.info("Устройство: %s", device)

    # ----- 1) Загрузка stage 1 и кодировщиков
    model, cwe_encoder = load_frozen_backbone(
        stage1_path=stage1_path,
        cwe_vocab_path=cwe_vocab_path,
        config_path=config_path,
        device=device,
    )

    pretrained_name = "bert-base-multilingual-cased"
    tokenizer = CVSSTokenizer(model_name=pretrained_name, max_length=max_length)
    features_encoder = FeaturesEncoder()
    text_processor = TextProcessor()

    # ----- 2) Подготовка данных
    train_df = _filter_with_scope(pd.read_parquet(train_data))
    val_df = _filter_with_scope(pd.read_parquet(val_data))
    if debug:
        train_df = train_df.head(20).reset_index(drop=True)
        val_df = val_df.head(10).reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("В train-наборе нет строк с распарсенным Scope.")
    if len(val_df) == 0:
        raise ValueError("В val-наборе нет строк с распарсенным Scope.")

    train_dist = train_df["scope"].value_counts().to_dict()
    val_dist = val_df["scope"].value_counts().to_dict()
    logger.info("train: %d записей, распределение Scope: %s", len(train_df), train_dist)
    logger.info("val: %d записей, распределение Scope: %s", len(val_df), val_dist)

    # ----- 3) Кэш представлений
    logger.info("Кэширую train fused...")
    train_fused, train_targets = cache_fused_features(
        model, train_df, tokenizer, cwe_encoder, features_encoder, text_processor,
        device=device, batch_size=batch_size_cache, max_length=max_length,
    )
    logger.info("Кэширую val fused...")
    val_fused, val_targets = cache_fused_features(
        model, val_df, tokenizer, cwe_encoder, features_encoder, text_processor,
        device=device, batch_size=batch_size_cache, max_length=max_length,
    )

    # ----- 4) Обучение линейной головы
    logger.info("Обучение Scope-головы: epochs=%d, batch=%d, lr=%.1e",
                epochs, batch_size_train, lr)
    head, history = train_head(
        train_fused, train_targets,
        val_fused, val_targets,
        device=device,
        epochs=epochs,
        batch_size=batch_size_train,
        lr=lr, weight_decay=weight_decay,
        patience=patience, seed=seed,
    )

    # ----- 5) Финальные метрики на лучшем чекпойнте
    head.eval()
    with torch.no_grad():
        val_logits = head(val_fused.to(device))
        val_preds = val_logits.argmax(dim=-1).cpu().numpy()
    final_acc = float(accuracy_score(val_targets.numpy(), val_preds))
    final_f1 = float(f1_score(val_targets.numpy(), val_preds, average="macro", zero_division=0))

    metadata = {
        "trained_from": str(stage1_path),
        "val_accuracy": final_acc,
        "val_f1_macro": final_f1,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "train_dist": {k: int(v) for k, v in train_dist.items()},
        "val_dist": {k: int(v) for k, v in val_dist.items()},
        "best_epoch": history["best_epoch"],
        "epochs_run": len(history["train_loss"]),
        "config": {
            "epochs": epochs, "batch_size_train": batch_size_train,
            "batch_size_cache": batch_size_cache, "lr": lr,
            "weight_decay": weight_decay, "patience": patience,
            "seed": seed, "max_length": max_length,
        },
        "history": {
            "train_loss": history["train_loss"], "val_loss": history["val_loss"],
            "val_accuracy": history["val_accuracy"], "val_f1_macro": history["val_f1_macro"],
        },
    }

    save_scope_head(head, output_path, metadata)
    print()
    print("=" * 60)
    print(f"Готово. val_accuracy={final_acc:.4f}, val_f1_macro={final_f1:.4f}")
    print(f"Артефакт: {output_path}")
    print("=" * 60)
    return metadata


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Дообучение Scope-головы (CVSS v3.1 S) поверх stage 1.",
    )
    parser.add_argument("--stage1", type=Path, default=DEFAULT_STAGE1,
                        help=f"чекпойнт stage 1 (по умолчанию {DEFAULT_STAGE1})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"куда сохранять веса (по умолчанию {DEFAULT_OUTPUT})")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument("--cwe-vocab", type=Path, default=DEFAULT_CWE_VOCAB)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size-cache", type=int, default=32,
                        help="batch для кэширования (зависит от VRAM)")
    parser.add_argument("--batch-size-train", type=int, default=256,
                        help="batch для обучения линейной головы")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--debug", action="store_true",
                        help="20 train + 10 val + 1 эпоха (smoke test)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)
    epochs = 1 if args.debug else args.epochs
    metadata = run(
        stage1_path=args.stage1,
        output_path=args.output,
        train_data=args.train_data,
        val_data=args.val_data,
        cwe_vocab_path=args.cwe_vocab,
        config_path=args.config,
        epochs=epochs,
        batch_size_cache=args.batch_size_cache,
        batch_size_train=args.batch_size_train,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        max_length=args.max_length,
        debug=args.debug,
    )
    # Дублируем сводку в JSON для удобства.
    summary_path = args.output.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    print(f"Метаданные: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCOPE_CLASSES",
    "cache_fused_features",
    "extract_scope",
    "load_frozen_backbone",
    "main",
    "run",
    "train_head",
]
