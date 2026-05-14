"""Тесты на MultiTaskLoss, EarlyStopping и compute_class_weights."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.training import (
    IGNORE_INDEX,
    EarlyStopping,
    MultiTaskLoss,
    compute_class_weights,
)

# ---------------------------------------------------------------------------
# Хелперы.
# ---------------------------------------------------------------------------


def _dummy_logits_and_labels(
    metrics: dict[str, int],
    batch: int = 4,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Создаёт случайные logits/labels по словарю ``{metric: num_classes}``."""
    g = torch.Generator().manual_seed(seed)
    logits, labels = {}, {}
    for name, n_cls in metrics.items():
        logits[name] = torch.randn(batch, n_cls, generator=g, requires_grad=True)
        labels[name] = torch.randint(0, n_cls, (batch,), generator=g)
    return logits, labels


# ---------------------------------------------------------------------------
# MultiTaskLoss.
# ---------------------------------------------------------------------------


def test_loss_basic() -> None:
    """forward возвращает скаляр + dict с ключами активных метрик."""
    metric_classes = {"AV": 4, "AC": 2, "PR": 3}
    logits, labels = _dummy_logits_and_labels(metric_classes)
    loss_fn = MultiTaskLoss(active_metrics=list(metric_classes))

    total, per_metric = loss_fn(logits, labels)

    assert isinstance(total, torch.Tensor)
    assert total.dim() == 0  # scalar
    assert total.item() > 0
    assert set(per_metric.keys()) == {"AV", "AC", "PR"}
    for v in per_metric.values():
        assert isinstance(v, float)
        assert v > 0
    # Сумма пер-метричных значений ≈ total (с плавающей погрешностью).
    assert abs(sum(per_metric.values()) - total.item()) < 1e-5


def test_loss_only_active_metrics() -> None:
    """Метрики, отсутствующие в active_metrics, не дают вклада в total."""
    metric_classes = {"AV": 4, "AC": 2, "PR": 3, "UI": 2}
    logits, labels = _dummy_logits_and_labels(metric_classes)

    loss_all = MultiTaskLoss(active_metrics=["AV", "AC", "PR", "UI"])
    loss_three = MultiTaskLoss(active_metrics=["AV", "AC", "PR"])

    total_all, _ = loss_all(logits, labels)
    total_three, per_three = loss_three(logits, labels)

    # UI отсутствует в активных — он не должен попасть в per_metric.
    assert "UI" not in per_three
    assert set(per_three) == {"AV", "AC", "PR"}
    # И не должен входить в total — иначе total_three == total_all.
    assert total_three.item() < total_all.item()


def test_loss_with_weights() -> None:
    """Передача class_weights меняет значение лосса (веса применяются)."""
    metric_classes = {"AV": 4}
    logits, labels = _dummy_logits_and_labels(metric_classes, batch=8, seed=1)

    loss_unweighted = MultiTaskLoss(active_metrics=["AV"])
    weights = torch.tensor([0.1, 1.0, 5.0, 10.0])  # резко неравные
    loss_weighted = MultiTaskLoss(
        active_metrics=["AV"],
        class_weights={"AV": weights},
    )

    v1, _ = loss_unweighted(logits, labels)
    v2, _ = loss_weighted(logits, labels)
    assert v1.item() != pytest.approx(v2.item())


def test_loss_ignore_index() -> None:
    """Метки -100 не учитываются: лосс считается только по валидным примерам."""
    metric_classes = {"AV": 4}
    g = torch.Generator().manual_seed(2)
    logits = {"AV": torch.randn(4, 4, generator=g, requires_grad=True)}

    # Все метки валидные.
    labels_full = {"AV": torch.tensor([0, 1, 2, 3])}
    # Тот же набор, но половина помечена ignore_index.
    labels_partial = {"AV": torch.tensor([0, IGNORE_INDEX, 2, IGNORE_INDEX])}

    loss_fn = MultiTaskLoss(active_metrics=["AV"])
    loss_full, _ = loss_fn(logits, labels_full)
    loss_partial, _ = loss_fn(logits, labels_partial)

    # Лоссы должны различаться — иначе ignore_index фактически не работает.
    assert loss_full.item() != pytest.approx(loss_partial.item())

    # Если ВСЕ метки отмечены ignore_index — лосс должен быть 0 (метрика
    # пропущена), а не NaN.
    labels_all_ignored = {"AV": torch.full((4,), IGNORE_INDEX)}
    loss_zero, per = loss_fn(logits, labels_all_ignored)
    assert loss_zero.item() == 0.0
    assert per == {}


# ---------------------------------------------------------------------------
# EarlyStopping.
# ---------------------------------------------------------------------------


def test_early_stopping_improvement(tmp_path) -> None:
    """Метрика растёт каждый шаг → не останавливается, patience не накапливается."""
    model = nn.Linear(4, 2)
    es = EarlyStopping(patience=2, mode="max", save_path=tmp_path / "best.pt")

    for value in [0.50, 0.55, 0.60, 0.65, 0.70]:
        stop = es.step(value, model)
        assert stop is False
    assert es.get_best_score() == pytest.approx(0.70)


def test_early_stopping_no_improvement(tmp_path) -> None:
    """patience+1 эпох без улучшения → step возвращает True."""
    model = nn.Linear(4, 2)
    es = EarlyStopping(patience=3, mode="max", save_path=tmp_path / "best.pt")

    # Первый шаг — улучшение по сравнению с -inf, лучшее = 0.70.
    assert es.step(0.70, model) is False
    # Дальше метрика только падает.
    assert es.step(0.69, model) is False  # counter=1
    assert es.step(0.68, model) is False  # counter=2
    assert es.step(0.67, model) is False  # counter=3 == patience
    # Четвёртый шаг без улучшения превышает patience — останов.
    assert es.step(0.66, model) is True
    assert es.get_best_score() == pytest.approx(0.70)


def test_early_stopping_save_load(tmp_path) -> None:
    """Сохранение лучших весов и восстановление возвращают исходное состояние."""
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    es = EarlyStopping(patience=2, mode="max", save_path=tmp_path / "best.pt")

    # Зафиксировали "лучшую" точку — веса сохранены.
    es.step(0.80, model)
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    assert (tmp_path / "best.pt").exists()

    # Сильно изменили веса (имитация дальнейшего обучения с деградацией).
    with torch.no_grad():
        for p in model.parameters():
            p.add_(10.0)
    # Метрика упала, новые веса хуже сохранённых — патч не должен переписать.
    es.step(0.50, model)

    # Восстанавливаем — веса должны совпасть с зафиксированными.
    es.restore_best_weights(model)
    for k, v in model.state_dict().items():
        assert torch.allclose(v, best_state[k])


# ---------------------------------------------------------------------------
# compute_class_weights.
# ---------------------------------------------------------------------------


def test_compute_class_weights() -> None:
    """На несбалансированном наборе вес редкого класса больше веса частого."""
    # Класс 0: 90 примеров, класс 1: 10 — сильный дисбаланс.
    df = pd.DataFrame({"AV": [0] * 90 + [1] * 10})
    weights = compute_class_weights(df, "AV", num_classes=4)

    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (4,)
    assert weights.dtype == torch.float32
    # Редкий класс должен иметь больший вес, чем частый.
    assert weights[1] > weights[0]
    # Отсутствующие в данных классы → нейтральный вес 1.0.
    assert weights[2].item() == pytest.approx(1.0)
    assert weights[3].item() == pytest.approx(1.0)
