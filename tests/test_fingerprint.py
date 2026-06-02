"""Тесты отпечатка весов backbone (provenance-чек для Scope-головы)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.model import backbone_fingerprint


def _state() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "transformer.encoder.weight": torch.randn(8, 8),
        "features_mlp.cwe_embedding.weight": torch.randn(4, 4),
        "fusion.proj.weight": torch.randn(6, 6),
        "heads.AV.weight": torch.randn(4, 6),  # вне backbone-префиксов
    }


def test_fingerprint_is_deterministic() -> None:
    state = _state()
    assert backbone_fingerprint(state) == backbone_fingerprint(state)


def test_fingerprint_changes_with_backbone_weights() -> None:
    state = _state()
    fp_before = backbone_fingerprint(state)
    state["fusion.proj.weight"] = state["fusion.proj.weight"] + 1.0
    assert backbone_fingerprint(state) != fp_before


def test_fingerprint_ignores_non_backbone_modules() -> None:
    """Изменение голов (heads.*) не меняет отпечаток — в него входит только
    то, что формирует h_fused."""
    state = _state()
    fp_before = backbone_fingerprint(state)
    state["heads.AV.weight"] = state["heads.AV.weight"] + 100.0
    assert backbone_fingerprint(state) == fp_before


def test_fingerprint_device_independent() -> None:
    """Отпечаток не должен зависеть от устройства (хешируются байты на CPU)."""
    state = _state()
    fp_cpu = backbone_fingerprint(state)
    if torch.cuda.is_available():
        state_cuda = {k: v.cuda() for k, v in state.items()}
        assert backbone_fingerprint(state_cuda) == fp_cpu
