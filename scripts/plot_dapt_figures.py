"""Сборщик графиков для главы 3 ВКР: сравнение baseline mBERT vs mBERT + DAPT.

Генерирует 4 PNG-файла в ``reports/dapt_experiment/figures/``:

1. ``training_curves_compare.png`` — кривые val_macro_f1 по эпохам, две стадии
   на одной фигуре, две модели на каждой подграфике (4 линии).
2. ``per_metric_compare_v4.png`` — барплот F1 по 12 v4-метрикам, baseline и
   DAPT рядом, с подписями Δ.
3. ``confusion_compare_AV_SI.png`` — side-by-side матрицы ошибок для AV
   (единственная регрессия) и SI (самый большой прирост), всего 4 субплота.
4. ``cumulative_gain_v4.png`` — три точки (Май, baseline, DAPT) с делением
   прироста на «очистка данных» и «DAPT» для macro-F1 и vector accuracy.

Источники данных:
- Кривые: ``logs/dapt_experiment/{baseline,dapt}_{stage1,stage2}.tfevents``
- Per-metric F1 и confusion matrix: ``reports/dapt_experiment/v4_{baseline,dapt}.json``
- Майские числа: захардкожены из ``reports/final_results.md`` (см. build_dapt_report.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Поддержка кириллицы.
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "dapt_experiment"
EXP_DIR = ROOT / "reports" / "dapt_experiment"
FIG_DIR = EXP_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

#: Соответствие имя_прогона → файл TB.
RUNS: dict[str, Path] = {
    "baseline_stage1": LOG_DIR / "baseline_stage1.tfevents",
    "dapt_stage1":     LOG_DIR / "dapt_stage1.tfevents",
    "baseline_stage2": LOG_DIR / "baseline_stage2.tfevents",
    "dapt_stage2":     LOG_DIR / "dapt_stage2.tfevents",
}

#: Старый майский baseline (из reports/final_results.md).
MAY_BASELINE_V4: dict[str, float | dict[str, float]] = {
    "macro_f1": 0.7090,
    "vector_accuracy": 0.3992,
    "per_metric_f1": {
        "AV": 0.5175, "AC": 0.7482, "AT": 0.7433, "PR": 0.6308, "UI": 0.6414,
        "VC": 0.7886, "VI": 0.8198, "VA": 0.7946, "SC": 0.6228, "SI": 0.6705,
        "SA": 0.656, "E": 0.8751,
    },
}


def read_tb_scalar(event_file: Path, tag: str) -> tuple[list[int], list[float]]:
    """Возвращает (steps, values) для скаляра ``tag`` из TB-файла."""
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return [], []
    evts = acc.Scalars(tag)
    return [e.step for e in evts], [e.value for e in evts]


# --------------------------------------------------------------- 1. CURVES


def plot_training_curves() -> Path:
    """Кривые macro-F1 по эпохам, 2 подграфика × 2 линии."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # Stage 1
    ax = axes[0]
    for run, label, color in [
        ("baseline_stage1", "baseline mBERT", "#1f77b4"),
        ("dapt_stage1",     "mBERT + DAPT", "#d62728"),
    ]:
        steps, vals = read_tb_scalar(RUNS[run], "stage1/macro_f1")
        ax.plot(steps, vals, marker="o", label=label, color=color, linewidth=2)
        ax.annotate(
            f"{vals[-1]:.4f}",
            xy=(steps[-1], vals[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9, color=color,
        )
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("val macro-F1")
    ax.set_title("Stage 1: предобучение на CVSS v3.1 (8 голов, val n=26 341)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xticks(range(1, 11))

    # Stage 2
    ax = axes[1]
    for run, label, color in [
        ("baseline_stage2", "baseline mBERT", "#1f77b4"),
        ("dapt_stage2",     "mBERT + DAPT", "#d62728"),
    ]:
        steps, vals = read_tb_scalar(RUNS[run], "stage2/macro_f1")
        ax.plot(steps, vals, marker="o", label=label, color=color, linewidth=2)
        # Метка best epoch.
        best_idx = int(np.argmax(vals))
        ax.annotate(
            f"best={vals[best_idx]:.4f}\n(epoch {steps[best_idx]})",
            xy=(steps[best_idx], vals[best_idx]),
            xytext=(8, -15),
            textcoords="offset points",
            fontsize=9, color=color,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8},
        )
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("val macro-F1")
    ax.set_title("Stage 2: дообучение на CVSS v4.0 (12 голов, val n=1 041)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xticks(range(1, 21, 2))

    fig.suptitle(
        "Кривые обучения: baseline mBERT vs mBERT + DAPT",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out = FIG_DIR / "training_curves_compare.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------- 2. PER-METRIC


def plot_per_metric_v4() -> Path:
    """Барплот F1 по 12 v4-метрикам: baseline vs DAPT рядом."""
    with (EXP_DIR / "v4_baseline.json").open(encoding="utf-8") as fh:
        v4_b = json.load(fh)
    with (EXP_DIR / "v4_dapt.json").open(encoding="utf-8") as fh:
        v4_d = json.load(fh)

    metrics = ["AV", "AC", "AT", "PR", "UI", "VC", "VI", "VA", "SC", "SI", "SA", "E"]
    b_vals = [v4_b["per_metric"][m]["f1_macro"] for m in metrics]
    d_vals = [v4_d["per_metric"][m]["f1_macro"] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.4

    fig, ax = plt.subplots(figsize=(13, 5.5))
    bars1 = ax.bar(x - w / 2, b_vals, w, label="baseline mBERT", color="#1f77b4")
    bars2 = ax.bar(x + w / 2, d_vals, w, label="mBERT + DAPT", color="#d62728")

    # Подписи Δ над парой столбцов.
    for i, (b, d) in enumerate(zip(b_vals, d_vals)):
        delta = d - b
        sign = "+" if delta >= 0 else "−"
        color = "#2ca02c" if delta >= 0.005 else "#d62728" if delta < -0.005 else "#7f7f7f"
        ax.annotate(
            f"{sign}{abs(delta):.3f}",
            xy=(i, max(b, d) + 0.015),
            ha="center",
            fontsize=9,
            color=color,
            fontweight="bold" if abs(delta) >= 0.025 else "normal",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("F1 (macro)")
    ax.set_title("Per-metric F1 на v4-тесте (972 строки): baseline vs DAPT")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    out = FIG_DIR / "per_metric_compare_v4.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------- 3. CONFUSION


def _confusion_to_array(cm_dict: dict, labels: list[str]) -> np.ndarray:
    """``{row_label: {col_label: count}}`` → ``np.ndarray[k, k]`` в порядке labels."""
    arr = np.zeros((len(labels), len(labels)), dtype=int)
    for i, row in enumerate(labels):
        for j, col in enumerate(labels):
            arr[i, j] = cm_dict.get(row, {}).get(col, 0)
    return arr


def _plot_confusion_ax(ax, cm: np.ndarray, labels: list[str], title: str) -> None:
    """Рисует нормированную CM на готовом ax."""
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истина")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(cm[i, j])
            ratio = norm[i, j]
            color = "white" if ratio > 0.5 else "black"
            ax.text(
                j, i, f"{count}\n({ratio:.2f})",
                ha="center", va="center", fontsize=9, color=color,
            )
    return im


def plot_confusion_av_si() -> Path:
    """Side-by-side CM для AV (регрессия) и SI (самый большой прирост)."""
    with (EXP_DIR / "v4_baseline.json").open(encoding="utf-8") as fh:
        v4_b = json.load(fh)
    with (EXP_DIR / "v4_dapt.json").open(encoding="utf-8") as fh:
        v4_d = json.load(fh)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # AV
    av_labels = ["N", "A", "L", "P"]
    cm_b = _confusion_to_array(v4_b["per_metric"]["AV"]["confusion"], av_labels)
    cm_d = _confusion_to_array(v4_d["per_metric"]["AV"]["confusion"], av_labels)
    f1_b = v4_b["per_metric"]["AV"]["f1_macro"]
    f1_d = v4_d["per_metric"]["AV"]["f1_macro"]
    _plot_confusion_ax(axes[0, 0], cm_b, av_labels, f"AV baseline (F1={f1_b:.4f})")
    _plot_confusion_ax(axes[0, 1], cm_d, av_labels, f"AV + DAPT (F1={f1_d:.4f}, регрессия {f1_d - f1_b:+.4f})")

    # SI
    si_labels = ["H", "L", "N"]
    cm_b = _confusion_to_array(v4_b["per_metric"]["SI"]["confusion"], si_labels)
    cm_d = _confusion_to_array(v4_d["per_metric"]["SI"]["confusion"], si_labels)
    f1_b = v4_b["per_metric"]["SI"]["f1_macro"]
    f1_d = v4_d["per_metric"]["SI"]["f1_macro"]
    _plot_confusion_ax(axes[1, 0], cm_b, si_labels, f"SI baseline (F1={f1_b:.4f})")
    _plot_confusion_ax(axes[1, 1], cm_d, si_labels, f"SI + DAPT (F1={f1_d:.4f}, прирост {f1_d - f1_b:+.4f})")

    fig.suptitle(
        "Матрицы ошибок для AV (регрессия) и SI (максимальный прирост)\n"
        "Нормировка по строке: значения = доля от истинной строки",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = FIG_DIR / "confusion_compare_AV_SI.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------- 4. CUMULATIVE


def plot_cumulative_gain() -> Path:
    """Три точки (Май, baseline, DAPT) для macro-F1 и vector accuracy."""
    with (EXP_DIR / "v4_baseline.json").open(encoding="utf-8") as fh:
        v4_b = json.load(fh)
    with (EXP_DIR / "v4_dapt.json").open(encoding="utf-8") as fh:
        v4_d = json.load(fh)

    points = ["Май\n(12-CWE)", "mBERT\n(baseline)", "mBERT\n+ DAPT"]
    macro = [
        MAY_BASELINE_V4["macro_f1"],
        v4_b["aggregated"]["macro_f1"],
        v4_d["aggregated"]["macro_f1"],
    ]
    vec = [
        MAY_BASELINE_V4["vector_accuracy"],
        v4_b["aggregated"]["vector_accuracy"],
        v4_d["aggregated"]["vector_accuracy"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    def _draw(ax, values, title, ylabel, color):
        x = np.arange(len(points))
        ax.plot(x, values, marker="o", markersize=12, color=color, linewidth=2.5)
        # Подписи значений.
        for i, v in enumerate(values):
            ax.annotate(
                f"{v:.4f}",
                xy=(i, v), xytext=(0, 12),
                textcoords="offset points", ha="center",
                fontsize=11, fontweight="bold",
            )
        # Подписи приростов между точками.
        for i in range(1, len(values)):
            delta = values[i] - values[i - 1]
            mid_x = (i - 1 + i) / 2
            mid_y = (values[i - 1] + values[i]) / 2
            label = f"+{delta:.4f}" if delta >= 0 else f"−{abs(delta):.4f}"
            note = "очистка данных" if i == 1 else "DAPT"
            ax.annotate(
                f"{label}\n({note})",
                xy=(mid_x, mid_y),
                xytext=(0, -32),
                textcoords="offset points",
                ha="center", fontsize=9, color="#2ca02c",
                fontstyle="italic",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(points)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        # Просторное место для подписей сверху.
        ymin = min(values) - 0.05
        ymax = max(values) + 0.05
        ax.set_ylim(ymin, ymax)

    _draw(axes[0], macro, "Macro-F1 (12 голов) на v4-тесте", "Macro-F1", "#1f77b4")
    _draw(axes[1], vec, "Vector accuracy (11 базовых) на v4-тесте", "Vector accuracy", "#d62728")

    fig.suptitle(
        f"Накопленный прирост качества на v4-тесте (n={v4_b['aggregated']['samples_evaluated']})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = FIG_DIR / "cumulative_gain_v4.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------- main

if __name__ == "__main__":
    print("1/4 Рисую кривые обучения...")
    p1 = plot_training_curves()
    print(f"  -> {p1}")

    print("2/4 Рисую per-metric барплот...")
    p2 = plot_per_metric_v4()
    print(f"  -> {p2}")

    print("3/4 Рисую confusion AV/SI...")
    p3 = plot_confusion_av_si()
    print(f"  -> {p3}")

    print("4/4 Рисую накопленный прирост...")
    p4 = plot_cumulative_gain()
    print(f"  -> {p4}")

    print("\nГотово.")
