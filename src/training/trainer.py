"""Главный класс :class:`Trainer` — двухэтапное обучение CVSSModel.

Этап 1: предобучение на CVSS v3.1 (8 общих метрик).
Этап 2: дообучение на CVSS v4.0 (все 12 метрик; головы AT/SC/SI/SA/E
переинициализируются — у E в v3.1 было 5 классов, в v4.0 — 3).

Контракт даталоадера: каждый батч — dict со следующими ключами:
    * ``input_ids``      — LongTensor[B, L]
    * ``attention_mask`` — LongTensor[B, L]
    * ``cwe_idx``        — LongTensor[B]
    * ``features``       — FloatTensor[B, num_features]
    * ``labels``         — dict[str, LongTensor[B]] по именам метрик

Совместим с :class:`src.model.CVSSModel`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.model import CVSSModel

from .early_stopping import EarlyStopping
from .loss import IGNORE_INDEX, MultiTaskLoss, compute_class_weights
from .utils import get_device, set_seed

logger = logging.getLogger(__name__)


class Trainer:
    """Двухэтапный многозадачный тренер CVSSModel.

    Args:
        config: Конфиг, загруженный из ``configs/train.yaml``. Должен
            содержать секции ``stage1``, ``stage2``, ``common``, ``paths``
            и поле ``seed``.
        model: Инициализированная :class:`CVSSModel`.
        device: Целевое устройство; если ``None``, выбирается через
            :func:`utils.get_device` (CUDA → MPS → CPU).
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        model: CVSSModel,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config: Dict[str, Any] = dict(config)
        self.common: Dict[str, Any] = dict(self.config.get("common", {}))
        self.paths: Dict[str, str] = dict(self.config.get("paths", {}))

        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)

        # Mixed precision применима только на CUDA. На CPU/MPS падаем в обычный
        # fp32 — autocast на CPU работает медленнее, чем без него.
        cfg_amp = bool(self.common.get("mixed_precision", False))
        self.use_amp: bool = cfg_amp and torch.cuda.is_available()

        self.gradient_clip: float = float(self.common.get("gradient_clip", 1.0))
        self.log_every_n: int = int(self.common.get("log_every_n_batches", 50))
        self.checkpoint_every_epoch: bool = bool(
            self.common.get("checkpoint_every_epoch", True)
        )
        self.patience: int = int(self.common.get("early_stopping_patience", 3))

        seed = int(self.config.get("seed", 42))
        set_seed(seed)

        self.checkpoints_dir = Path(
            self.paths.get("checkpoints_dir", "models/checkpoints/")
        )
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path(self.paths.get("models_dir", "models/"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        tb_dir = Path(self.paths.get("tensorboard_dir", "logs/tensorboard/"))
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        self._global_step: int = 0

    # ------------------------------------------------------------------ utils

    def close(self) -> None:
        """Закрывает TensorBoard writer (вызывать после обучения)."""
        self.writer.flush()
        self.writer.close()

    # ------------------------------------------------------- optimizer/scheduler

    def _build_optimizer(self, stage_config: Mapping[str, Any]) -> AdamW:
        """AdamW со стандартной для BERT группировкой по weight decay.

        Bias и параметры LayerNorm выводятся из-под weight decay; всё остальное
        — обучается с decay из ``common.weight_decay``.
        """
        no_decay_keys = ("bias", "LayerNorm.weight", "LayerNorm.bias")
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in no_decay_keys):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        weight_decay = float(self.common.get("weight_decay", 0.01))
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        lr = float(stage_config["learning_rate"])
        return AdamW(param_groups, lr=lr)

    def _build_scheduler(
        self,
        optimizer: AdamW,
        total_steps: int,
        warmup_ratio: float,
    ) -> LambdaLR:
        """Linear warmup → linear decay (`transformers.get_linear_schedule_with_warmup`)."""
        num_warmup = int(warmup_ratio * total_steps)
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=total_steps,
        )

    # --------------------------------------------------------------- batch I/O

    def _move_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        """Переносит тензоры батча на ``self.device``.

        Поле ``labels`` — это dict[str, Tensor], его обрабатываем отдельно.
        """
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            if key == "labels":
                moved[key] = {k: v.to(self.device) for k, v in value.items()}
            elif isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    # ------------------------------------------------------------------ epoch

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler: LambdaLR,
        loss_fn: MultiTaskLoss,
        scaler: Optional[torch.amp.GradScaler],
        active_metrics: List[str],
        epoch_num: int,
        stage_num: int,
    ) -> float:
        """Один проход обучения. Возвращает средний loss за эпоху."""
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        progress = tqdm(
            loader,
            desc=f"stage{stage_num} epoch {epoch_num}",
            leave=False,
        )
        for batch in progress:
            batch = self._move_batch(batch)
            optimizer.zero_grad(set_to_none=True)

            if self.use_amp and scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    logits = self.model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["cwe_idx"],
                        batch["features"],
                    )
                    loss, per_metric = loss_fn(logits, batch["labels"])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = self.model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["cwe_idx"],
                    batch["features"],
                )
                loss, per_metric = loss_fn(logits, batch["labels"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                optimizer.step()

            scheduler.step()
            self._global_step += 1
            n_batches += 1
            running_loss += float(loss.detach().item())

            progress.set_postfix(loss=f"{loss.item():.4f}")

            if self._global_step % self.log_every_n == 0:
                tag_prefix = f"stage{stage_num}/train"
                self.writer.add_scalar(
                    f"{tag_prefix}/loss", loss.item(), self._global_step
                )
                self.writer.add_scalar(
                    f"{tag_prefix}/lr",
                    optimizer.param_groups[0]["lr"],
                    self._global_step,
                )
                if torch.cuda.is_available():
                    self.writer.add_scalar(
                        f"{tag_prefix}/gpu_memory_mb",
                        torch.cuda.memory_allocated() / (1024 ** 2),
                        self._global_step,
                    )
                for metric_name, metric_loss in per_metric.items():
                    self.writer.add_scalar(
                        f"{tag_prefix}/loss_{metric_name}",
                        metric_loss,
                        self._global_step,
                    )

        return running_loss / max(n_batches, 1)

    # ------------------------------------------------------------- evaluation

    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
        loss_fn: MultiTaskLoss,
        active_metrics: List[str],
    ) -> Dict[str, Any]:
        """Прогон по валидации; возвращает loss + per-metric F1/accuracy."""
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        y_true: Dict[str, List[int]] = defaultdict(list)
        y_pred: Dict[str, List[int]] = defaultdict(list)

        for batch in tqdm(loader, desc="eval", leave=False):
            batch = self._move_batch(batch)
            logits = self.model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["cwe_idx"],
                batch["features"],
            )
            loss, _ = loss_fn(logits, batch["labels"])
            running_loss += float(loss.item())
            n_batches += 1

            for metric in active_metrics:
                if metric not in logits or metric not in batch["labels"]:
                    continue
                preds = logits[metric].argmax(dim=-1)
                targets = batch["labels"][metric]
                # Отфильтровать ignore_index перед расчётом метрик.
                mask = targets != IGNORE_INDEX
                if mask.sum() == 0:
                    continue
                y_pred[metric].extend(preds[mask].cpu().tolist())
                y_true[metric].extend(targets[mask].cpu().tolist())

        per_metric: Dict[str, Dict[str, float]] = {}
        f1_values: List[float] = []
        for metric in active_metrics:
            if not y_true[metric]:
                continue
            f1 = float(
                f1_score(
                    y_true[metric],
                    y_pred[metric],
                    average="macro",
                    zero_division=0,
                )
            )
            acc = float(accuracy_score(y_true[metric], y_pred[metric]))
            per_metric[metric] = {"f1": f1, "accuracy": acc}
            f1_values.append(f1)

        return {
            "val_loss": running_loss / max(n_batches, 1),
            "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
            "per_metric": per_metric,
        }

    # ----------------------------------------------------------------- weights

    def _make_class_weights(
        self,
        train_df: Optional[pd.DataFrame],
        active_metrics: List[str],
        metric_classes: Mapping[str, int],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Считает веса классов по train-DataFrame для каждой активной метрики.

        Если ``train_df`` не передан или в нём нет колонки метрики — веса
        для этой метрики пропускаются.
        """
        if train_df is None:
            return None
        weights: Dict[str, torch.Tensor] = {}
        for metric in active_metrics:
            if metric not in train_df.columns:
                continue
            n_cls = int(metric_classes[metric])
            weights[metric] = compute_class_weights(train_df, metric, n_cls).to(
                self.device
            )
        return weights or None

    # ------------------------------------------------------ stage 2 head reinit

    def _reinit_heads_for_stage2(self) -> None:
        """Пересобирает ``model.heads`` под CVSS v4.0 со случайной инициализацией
        для метрик из ``stage2.reinit_heads``.

        Головы, не входящие в reinit-список и совпадающие по числу классов
        со стадией 1, сохраняют свои веса. Остальные — Xavier для weight,
        zeros для bias.
        """
        from src.model.classification_heads import ClassificationHeads

        stage2 = self.config["stage2"]
        new_classes: Dict[str, int] = dict(stage2["metric_classes"])
        reinit: set[str] = set(stage2.get("reinit_heads", []))

        old_heads = self.model.heads
        input_dim = old_heads.input_dim
        new_heads = ClassificationHeads(
            input_dim=input_dim,
            metric_classes=new_classes,
        ).to(self.device)

        kept: List[str] = []
        reinitialized: List[str] = []
        for metric, head in new_heads.items():
            if metric in reinit:
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)
                reinitialized.append(metric)
                continue
            old_head = old_heads[metric] if metric in old_heads else None
            if (
                old_head is not None
                and old_head.weight.shape == head.weight.shape
            ):
                head.weight.data.copy_(old_head.weight.data)
                head.bias.data.copy_(old_head.bias.data)
                kept.append(metric)
            else:
                # Размер класса изменился, а в reinit явно не указан —
                # всё равно нужна свежая инициализация.
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)
                reinitialized.append(metric)

        self.model.heads = new_heads
        logger.info(
            "stage2: reinit heads=%s; transferred from stage1=%s",
            reinitialized,
            kept,
        )

    # -------------------------------------------------------- stage1 weight load

    def _load_stage1_weights_if_present(self, path: Path) -> None:
        """Грузит совместимые веса из чекпоинта stage1 (strict=False + фильтр форм).

        Параметры с несовпадающими формами (например, головы с другим числом
        классов) автоматически пропускаются — они либо переинициализируются
        в :meth:`_reinit_heads_for_stage2`, либо сохраняют дефолтную инициализацию.
        """
        if not path.exists():
            logger.warning("stage1 checkpoint not found at %s — skipping load", path)
            return
        state = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]

        own = self.model.state_dict()
        compatible = {
            k: v
            for k, v in state.items()
            if k in own and own[k].shape == v.shape
        }
        skipped = sorted(set(state) - set(compatible))
        missing = sorted(set(own) - set(compatible))
        self.model.load_state_dict(compatible, strict=False)
        logger.info(
            "stage1 → stage2 weight transfer: loaded=%d, skipped=%d, missing=%d",
            len(compatible),
            len(skipped),
            len(missing),
        )

    # --------------------------------------------------------- training stages

    def _run_stage(
        self,
        stage_num: int,
        stage_config: Mapping[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_df: Optional[pd.DataFrame],
        best_save_path: Path,
    ) -> Dict[str, Any]:
        """Общий цикл обучения, параметризованный конфигом этапа."""
        active_metrics: List[str] = list(stage_config["metrics"])
        epochs: int = int(stage_config["epochs"])
        warmup_ratio: float = float(stage_config["warmup_ratio"])
        metric_classes: Mapping[str, int] = stage_config["metric_classes"]

        class_weights = self._make_class_weights(
            train_df, active_metrics, metric_classes
        )
        loss_fn = MultiTaskLoss(
            active_metrics=active_metrics,
            class_weights=class_weights,
        ).to(self.device)

        optimizer = self._build_optimizer(stage_config)
        total_steps = epochs * max(len(train_loader), 1)
        scheduler = self._build_scheduler(optimizer, total_steps, warmup_ratio)
        scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        early_stopping = EarlyStopping(
            patience=self.patience,
            mode="max",
            save_path=best_save_path,
        )

        history: Dict[str, List[Any]] = {
            "train_loss": [],
            "val_loss": [],
            "macro_f1": [],
            "per_metric": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(
                train_loader,
                optimizer,
                scheduler,
                loss_fn,
                scaler,
                active_metrics,
                epoch_num=epoch,
                stage_num=stage_num,
            )
            val = self._evaluate(val_loader, loss_fn, active_metrics)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val["val_loss"])
            history["macro_f1"].append(val["macro_f1"])
            history["per_metric"].append(val["per_metric"])

            tag = f"stage{stage_num}"
            self.writer.add_scalar(f"{tag}/epoch_train_loss", train_loss, epoch)
            self.writer.add_scalar(f"{tag}/epoch_val_loss", val["val_loss"], epoch)
            self.writer.add_scalar(f"{tag}/macro_f1", val["macro_f1"], epoch)
            for metric, scores in val["per_metric"].items():
                self.writer.add_scalar(
                    f"{tag}/val_f1/{metric}", scores["f1"], epoch
                )
                self.writer.add_scalar(
                    f"{tag}/val_acc/{metric}", scores["accuracy"], epoch
                )

            logger.info(
                "stage%d epoch %d: train_loss=%.4f val_loss=%.4f macro_f1=%.4f",
                stage_num,
                epoch,
                train_loss,
                val["val_loss"],
                val["macro_f1"],
            )
            print(
                f"[stage{stage_num}] epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val['val_loss']:.4f} "
                f"macro_f1={val['macro_f1']:.4f}"
            )
            for metric, scores in val["per_metric"].items():
                print(
                    f"    {metric}: f1={scores['f1']:.3f} "
                    f"acc={scores['accuracy']:.3f}"
                )

            if self.checkpoint_every_epoch:
                ckpt_path = (
                    self.checkpoints_dir
                    / f"stage{stage_num}_epoch{epoch}.pt"
                )
                self.save_checkpoint(
                    ckpt_path, optimizer, scheduler, epoch, stage_num
                )

            should_stop = early_stopping.step(val["macro_f1"], self.model)
            if should_stop:
                logger.info(
                    "stage%d: early stopping at epoch %d (best macro_f1=%.4f)",
                    stage_num,
                    epoch,
                    early_stopping.get_best_score(),
                )
                break

        early_stopping.restore_best_weights(self.model)
        history["best_macro_f1"] = early_stopping.get_best_score()
        history["best_checkpoint"] = str(best_save_path)
        return history

    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Этап 1: предобучение на 8 общих с CVSS v3.1 метриках."""
        stage_config = self.config["stage1"]
        best_path = self.models_dir / "best_stage1.pt"
        return self._run_stage(
            stage_num=1,
            stage_config=stage_config,
            train_loader=train_loader,
            val_loader=val_loader,
            train_df=train_df,
            best_save_path=best_path,
        )

    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Этап 2: дообучение на 12 метриках CVSS v4.0.

        Перед стартом:
            1. Грузим best_stage1.pt с фильтрацией по форме (strict=False).
            2. Переинициализируем головы из ``stage2.reinit_heads`` —
               AT/SC/SI/SA новые, E пересоздаётся, т.к. в v3.1 у неё было
               5 классов, а в v4.0 — 3.
        """
        self._load_stage1_weights_if_present(self.models_dir / "best_stage1.pt")
        self._reinit_heads_for_stage2()
        # Параметры новых голов уже на нужном устройстве; на всякий случай
        # перегоняем всю модель.
        self.model.to(self.device)

        stage_config = self.config["stage2"]
        best_path = self.models_dir / "best_stage2.pt"
        return self._run_stage(
            stage_num=2,
            stage_config=stage_config,
            train_loader=train_loader,
            val_loader=val_loader,
            train_df=train_df,
            best_save_path=best_path,
        )

    # --------------------------------------------------------------- checkpoint

    def save_checkpoint(
        self,
        path: Path,
        optimizer: AdamW,
        scheduler: LambdaLR,
        epoch: int,
        stage: int,
    ) -> None:
        """Полный чекпоинт для возобновления: модель + опт + sched + метаданные."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": int(epoch),
                "stage": int(stage),
                "config": self.config,
                "global_step": self._global_step,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Восстанавливает model/optimizer/scheduler из чекпоинта.

        Returns:
            Словарь ``{"epoch": int, "stage": int}`` для продолжения цикла.
            Optimizer/scheduler возвращаются через возвращаемый dict под
            ключами ``"optimizer_state"``/``"scheduler_state"`` — вызывающая
            сторона должна сама пересобрать их и применить эти state_dict
            (нужны актуальные lr/total_steps текущего этапа).
        """
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self._global_step = int(ckpt.get("global_step", 0))
        return {
            "epoch": int(ckpt["epoch"]),
            "stage": int(ckpt["stage"]),
            "optimizer_state": ckpt.get("optimizer_state"),
            "scheduler_state": ckpt.get("scheduler_state"),
            "config": ckpt.get("config"),
        }


__all__ = ["Trainer"]
