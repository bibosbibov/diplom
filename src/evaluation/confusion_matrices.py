"""Построение и визуализация confusion-матриц (раздел 2.3.6 ВКР).

``matplotlib`` / ``seaborn`` импортируются лениво внутри функций отрисовки —
расчётную часть (``build_confusion_matrix``) можно использовать и без них.
Все графики сохраняются в PNG 300 dpi с русскими подписями осей, как требуется
для иллюстраций ВКР.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .metrics import SEVERITY_ORDER

#: Размер одной фигуры (дюймы) — компромисс между читаемостью аннотаций и
#: компактностью на странице отчёта.
_FIGSIZE = (6, 5)
_DPI = 300


def build_confusion_matrix(y_true, y_pred, labels) -> pd.DataFrame:
    """Строит матрицу ошибок с подписанными строками/столбцами.

    Args:
        y_true: истинные классы.
        y_pred: предсказанные классы.
        labels: порядок классов (строки = истинные, столбцы = предсказанные).

    Returns:
        ``pd.DataFrame`` ``len(labels) x len(labels)``; ``index`` — истинный
        класс, ``columns`` — предсказанный.
    """
    labels = list(labels)
    matrix = confusion_matrix(list(y_true), list(y_pred), labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def _row_normalize(cm_df: pd.DataFrame) -> pd.DataFrame:
    """Делит каждую строку на её сумму; пустые строки → нули."""
    data = cm_df.astype(np.float64)
    row_sums = data.sum(axis=1)
    safe_sums = row_sums.replace(0.0, np.nan)
    return data.div(safe_sums, axis=0).fillna(0.0)


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    title: str,
    save_path,
    normalize: bool = True,
) -> Path:
    """Рисует heatmap матрицы ошибок и сохраняет в PNG.

    Args:
        cm_df: матрица из :func:`build_confusion_matrix`.
        title: заголовок графика.
        save_path: путь к выходному PNG (директории создаются автоматически).
        normalize: если ``True`` — нормировать по строкам (сумма строки = 1.0),
            аннотации с двумя знаками после запятой; иначе — абсолютные счётчики.

    Returns:
        Путь к сохранённому файлу.
    """
    import matplotlib

    matplotlib.use("Agg")  # без интерактивного бэкенда — пишем сразу в файл
    import matplotlib.pyplot as plt
    import seaborn as sns

    if normalize:
        data = _row_normalize(cm_df)
        fmt = ".2f"
        vmax: float | None = 1.0
    else:
        data = cm_df.astype(np.int64)
        fmt = "d"
        vmax = None

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        square=False,
        vmin=0.0,
        vmax=vmax,
        ax=ax,
    )
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_all_per_metric_matrices(
    per_metric_results: Mapping[str, Mapping],
    output_dir,
) -> dict[str, str]:
    """Сохраняет confusion-матрицы по всем 12 метрикам CVSS v4.0.

    Args:
        per_metric_results: ``{метрика: {... "confusion": DataFrame ...}}`` —
            структура из :meth:`Evaluator.evaluate`. ``confusion`` может быть
            как ``pd.DataFrame``, так и вложенным dict (после сериализации).
        output_dir: директория для PNG-файлов.

    Returns:
        ``{метрика: путь_к_png}`` для записанных файлов.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for metric, result in per_metric_results.items():
        cm = result.get("confusion") if isinstance(result, Mapping) else None
        if cm is None:
            continue
        cm_df = cm if isinstance(cm, pd.DataFrame) else pd.DataFrame(cm)
        path = output_dir / f"confusion_{metric}.png"
        plot_confusion_matrix(
            cm_df,
            title=f"Матрица ошибок — метрика {metric} (CVSS v4.0)",
            save_path=path,
            normalize=True,
        )
        saved[metric] = str(path)
    return saved


def plot_severity_confusion_matrix(true_sev, pred_sev, save_path) -> Path:
    """Агрегированная матрица 5×5 по уровням критичности (None…Critical)."""
    cm_df = build_confusion_matrix(true_sev, pred_sev, labels=SEVERITY_ORDER)
    return plot_confusion_matrix(
        cm_df,
        title="Матрица ошибок — уровень критичности CVSS v4.0",
        save_path=save_path,
        normalize=True,
    )


__all__ = [
    "build_confusion_matrix",
    "plot_confusion_matrix",
    "plot_all_per_metric_matrices",
    "plot_severity_confusion_matrix",
]
