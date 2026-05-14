"""Парсинг TensorBoard-логов и построение кривых обучения.

Тренер пишет скаляры с тегами вида ``stage1/epoch_train_loss``,
``stage1/epoch_val_loss``, ``stage1/macro_f1`` и аналогично для ``stage2``.
В каталоге ``logs/tensorboard/`` может лежать несколько event-файлов от
разных запусков; здесь мы агрегируем их все и для каждого тега берём
последнюю записанную (epoch, value) пару — это даёт окончательную кривую
последнего успешного запуска без дублей.

Соответствует разделам 2.3.3 / 2.3.6 ВКР: визуализация процесса обучения.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logger = logging.getLogger(__name__)

#: Имя тега → ключ в выходной структуре. Берём три ключевых эпошных метрики;
#: остальные (per-метрика F1, lr, gpu_memory) для общей кривой избыточны.
_TAG_TO_KEY: dict[str, str] = {
    "epoch_train_loss": "epoch_train_loss",
    "epoch_val_loss": "epoch_val_loss",
    "macro_f1": "macro_f1",
}

_STAGES: tuple[str, ...] = ("stage1", "stage2")


def _iter_event_files(logdir: Path) -> list[Path]:
    """Возвращает event-файлы из ``logdir`` (рекурсивно), отсортированные по mtime."""
    if logdir.is_file():
        return [logdir]
    candidates = sorted(
        logdir.rglob("events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates


def parse_tensorboard_logs(
    logdir: str | Path,
    stages: Iterable[str] = _STAGES,
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Парсит все event-файлы из каталога и собирает эпошные скаляры по этапам.

    Args:
        logdir: каталог с tfevents (обычно ``logs/tensorboard``) или путь к
            одному event-файлу.
        stages: имена этапов, которые ищем (``"stage1"``, ``"stage2"``).

    Returns:
        Словарь вида::

            {
                "stage1": {
                    "epoch_train_loss": [(epoch, value), ...],
                    "epoch_val_loss":   [(epoch, value), ...],
                    "macro_f1":         [(epoch, value), ...],
                },
                "stage2": {...},
            }

        Если для какого-то этапа в логах ничего нет — словарь будет пуст
        (``{}``), а не отсутствовать целиком (это позволяет ноутбуку
        одинаково обрабатывать оба этапа).

        Если несколько event-файлов содержат один и тот же (stage, tag, epoch),
        побеждает значение из файла с бóльшим mtime (последний запуск).
    """
    logdir = Path(logdir)
    if not logdir.exists():
        raise FileNotFoundError(f"Каталог логов не найден: {logdir}")

    stages = tuple(stages)
    # для каждого (stage, key) храним {epoch: value}, перезаписываем по мере
    # появления более свежих файлов — так совпадающие эпохи разных запусков
    # дают одну итоговую кривую.
    accumulators: dict[str, dict[str, dict[int, float]]] = {
        stage: {key: {} for key in _TAG_TO_KEY.values()} for stage in stages
    }

    event_files = _iter_event_files(logdir)
    if not event_files:
        logger.warning("В %s нет event-файлов TensorBoard", logdir)

    for event_path in event_files:
        ea = EventAccumulator(
            str(event_path),
            size_guidance={"scalars": 0},  # 0 → загрузить все точки
        )
        try:
            ea.Reload()
        except Exception as exc:  # pragma: no cover - битый файл
            logger.warning("Не удалось прочитать %s: %s", event_path, exc)
            continue
        tags = ea.Tags().get("scalars", [])
        for tag in tags:
            stage, _, suffix = tag.partition("/")
            if stage not in stages or suffix not in _TAG_TO_KEY:
                continue
            key = _TAG_TO_KEY[suffix]
            for event in ea.Scalars(tag):
                accumulators[stage][key][int(event.step)] = float(event.value)

    result: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for stage in stages:
        stage_data: dict[str, list[tuple[int, float]]] = {}
        for key, points in accumulators[stage].items():
            if not points:
                continue
            stage_data[key] = sorted(points.items())
        result[stage] = stage_data
    return result


def _series_xy(points: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
    return [p[0] for p in points], [p[1] for p in points]


def plot_training_curves(
    scalars: dict[str, dict[str, list[tuple[int, float]]]],
    save_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Строит фигуру из 3 subplot'ов: train loss, val loss, macro F1.

    Stage 1 и Stage 2 рисуются разными линиями в каждом subplot. Подписи —
    на русском. Файл сохраняется в ``save_path`` (родитель создаётся при
    необходимости).

    Args:
        scalars: результат :func:`parse_tensorboard_logs`.
        save_path: куда сохранить PNG.
        dpi: разрешение (по умолчанию 300, как требует ВКР).

    Returns:
        Путь к сохранённому файлу.
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=dpi)
    panels = [
        ("epoch_train_loss", "Функция потерь (train)", "Эпоха", "Loss"),
        ("epoch_val_loss", "Функция потерь (val)", "Эпоха", "Loss"),
        ("macro_f1", "Macro-F1 (val)", "Эпоха", "F1"),
    ]
    stage_styles = {
        "stage1": {"label": "Этап 1 (CVSS v3.1)", "color": "#1f77b4", "marker": "o"},
        "stage2": {"label": "Этап 2 (CVSS v4.0)", "color": "#d62728", "marker": "s"},
    }

    for ax, (key, title, xlabel, ylabel) in zip(axes, panels):
        for stage, style in stage_styles.items():
            points = scalars.get(stage, {}).get(key)
            if not points:
                continue
            xs, ys = _series_xy(points)
            ax.plot(
                xs,
                ys,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=2,
                markersize=5,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        if ax.has_data():
            ax.legend(loc="best")

    fig.suptitle("Кривые обучения mBERT для оценки CVSS v4.0", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Сохранены кривые обучения: %s", save_path)
    return save_path


__all__ = [
    "parse_tensorboard_logs",
    "plot_training_curves",
]
