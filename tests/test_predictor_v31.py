"""Тесты декодирования v3.1-предиктора (без загрузки реальной модели).

Конструируем экземпляр через ``object.__new__`` и проверяем чистую логику
:meth:`VulnerabilityPredictorV31._decode`: маппинг голов VC/VI/VA → C/I/A,
обрезку нетренированного 4-го логита (X) и декодирование Scope. Загрузка
весов stage 1 + scope-головы здесь не нужна — это интеграционный путь.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.inference.predictor_v31 import VulnerabilityPredictorV31


def _decoder() -> VulnerabilityPredictorV31:
    obj = object.__new__(VulnerabilityPredictorV31)
    obj._scope_classes = ["U", "C"]
    return obj


def _peak(size: int, idx: int) -> torch.Tensor:
    """Логиты [1, size] с пиком в позиции idx."""
    t = torch.full((size,), -10.0)
    t[idx] = 10.0
    return t.unsqueeze(0)


def test_decode_maps_heads_and_scope() -> None:
    obj = _decoder()
    head_logits = {
        "AV": _peak(4, 0),  # N
        "AC": _peak(2, 1),  # H
        "PR": _peak(3, 0),  # N
        "UI": _peak(2, 1),  # R
        "VC": _peak(4, 0),  # H  (4-й логит X игнорируется)
        "VI": _peak(4, 2),  # N
        "VA": _peak(4, 1),  # L
        "E": _peak(5, 0),   # не используется в базовом v3.1
    }
    scope_logits = torch.tensor([[-5.0, 5.0]])  # → C

    metrics, confidence = obj._decode(head_logits, scope_logits)

    assert metrics == {
        "AV": "N", "AC": "H", "PR": "N", "UI": "R",
        "C": "H", "I": "N", "A": "L", "S": "C",
    }
    assert set(confidence) == set(metrics)
    assert all(0.0 <= c <= 1.0 for c in confidence.values())


def test_decode_slices_untrained_x_logit() -> None:
    """4-й логит (X) у VC/VI/VA доминирует, но должен обрезаться до argmax по H/L/N."""
    obj = _decoder()
    head_logits = {
        "AV": _peak(4, 0),
        "AC": _peak(2, 0),
        "PR": _peak(3, 0),
        "UI": _peak(2, 0),
        "VC": torch.tensor([[5.0, 0.0, 0.0, 100.0]]),  # X доминирует → срезается
        "VI": torch.tensor([[5.0, 0.0, 0.0, 100.0]]),
        "VA": torch.tensor([[5.0, 0.0, 0.0, 100.0]]),
        "E": _peak(5, 0),
    }
    scope_logits = torch.tensor([[5.0, -5.0]])  # → U

    metrics, _ = obj._decode(head_logits, scope_logits)

    assert metrics["C"] == "H"  # индекс 0 после слайса [:3]
    assert metrics["I"] == "H"
    assert metrics["A"] == "H"
    assert metrics["S"] == "U"
