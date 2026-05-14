"""Юнит-тесты модуля src/model/.

Тесты используют фейковый трансформер с тем же ``hidden_size=768``, что и
у mBERT, чтобы избежать скачивания 700 МБ весов и работать на CI без сети.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.model import (
    DEFAULT_METRIC_CLASSES,
    ClassificationHeads,
    CVSSModel,
    FeaturesMLP,
    FusionLayer,
)

# -------------------------------------------------------------- fake transformer


class FakeBERT(nn.Module):
    """Минимальная BERT-совместимая заглушка (последний hidden = эмбеддинг)."""

    def __init__(self, hidden_size: int = 768, vocab_size: int = 1000) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


# ---------------------------------------------------------------- FeaturesMLP


def test_features_mlp_output_shape():
    mlp = FeaturesMLP(num_cwe=100)
    features = torch.randn(4, 3)
    cwe_idx = torch.randint(0, 100, (4,))
    out = mlp(features, cwe_idx)
    assert out.shape == (4, 64)
    assert out.dtype == torch.float32


def test_features_mlp_handles_missing_marker():
    """Маркер -1 в признаках не должен ломать прямой проход."""
    mlp = FeaturesMLP(num_cwe=10)
    features = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float32)
    cwe_idx = torch.tensor([1], dtype=torch.long)
    out = mlp(features, cwe_idx)
    assert out.shape == (1, 64)
    assert torch.isfinite(out).all()


def test_features_mlp_padding_idx_zero_grad():
    mlp = FeaturesMLP(num_cwe=10, padding_idx=0)
    pad_emb = mlp.cwe_embedding(torch.tensor([0]))
    assert torch.allclose(pad_emb, torch.zeros_like(pad_emb))


def test_features_mlp_internal_layout():
    mlp = FeaturesMLP(
        num_cwe=50, num_features=3, cwe_embedding_dim=64, hidden_dim=128, output_dim=64
    )
    layers = [m for m in mlp.net if isinstance(m, nn.Linear)]
    assert layers[0].in_features == 67  # 3 + 64
    assert layers[0].out_features == 128
    assert layers[1].in_features == 128
    assert layers[1].out_features == 64


# ----------------------------------------------------------------- FusionLayer


def test_fusion_layer_output_shape():
    fusion = FusionLayer(text_dim=768, feature_dim=64, output_dim=512)
    h_text = torch.randn(2, 768)
    h_feat = torch.randn(2, 64)
    out = fusion(h_text, h_feat)
    assert out.shape == (2, 512)


def test_fusion_layer_internal_dims():
    fusion = FusionLayer()
    assert fusion.proj.in_features == 832  # 768 + 64
    assert fusion.proj.out_features == 512
    assert isinstance(fusion.dropout, nn.Dropout)
    assert fusion.dropout.p == pytest.approx(0.1)


# ---------------------------------------------------------- ClassificationHeads


def test_default_metric_classes_count():
    assert len(DEFAULT_METRIC_CLASSES) == 12
    assert DEFAULT_METRIC_CLASSES["AV"] == 4
    assert DEFAULT_METRIC_CLASSES["AC"] == 2
    assert DEFAULT_METRIC_CLASSES["AT"] == 2
    assert DEFAULT_METRIC_CLASSES["E"] == 3


def test_classification_heads_returns_all_metrics():
    heads = ClassificationHeads(input_dim=512)
    h = torch.randn(3, 512)
    logits = heads(h)
    assert set(logits.keys()) == set(DEFAULT_METRIC_CLASSES.keys())
    for metric, num_classes in DEFAULT_METRIC_CLASSES.items():
        assert logits[metric].shape == (3, num_classes), (
            f"метрика {metric} вернула shape {logits[metric].shape}, "
            f"ожидалось (3, {num_classes})"
        )


def test_classification_heads_metric_order_preserved():
    heads = ClassificationHeads()
    expected_order = ("AV", "AC", "AT", "PR", "UI", "VC", "VI", "VA", "SC", "SI", "SA", "E")
    assert heads.metric_order == expected_order


# -------------------------------------------------------------------- CVSSModel


def _build_model(num_cwe: int = 100, hidden_size: int = 768) -> CVSSModel:
    fake = FakeBERT(hidden_size=hidden_size, vocab_size=1000)
    return CVSSModel(num_cwe=num_cwe, transformer=fake)


def test_cvss_model_forward_shapes():
    batch = 4
    seq_len = 32
    model = _build_model()

    input_ids = torch.randint(0, 1000, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    cwe_idx = torch.randint(0, 100, (batch,))
    features = torch.randn(batch, 3)

    logits = model(input_ids, attention_mask, cwe_idx, features)

    assert isinstance(logits, dict)
    assert len(logits) == 12
    for metric, num_classes in DEFAULT_METRIC_CLASSES.items():
        assert logits[metric].shape == (batch, num_classes)


def test_cvss_model_inferred_text_dim():
    model = _build_model(hidden_size=768)
    assert model.text_dim == 768
    assert model.fusion.proj.in_features == 832


def test_cvss_model_predict_returns_label_and_confidence():
    batch = 2
    model = _build_model()
    input_ids = torch.randint(0, 1000, (batch, 16))
    attention_mask = torch.ones(batch, 16, dtype=torch.long)
    cwe_idx = torch.randint(0, 100, (batch,))
    features = torch.randn(batch, 3)

    preds = model.predict(input_ids, attention_mask, cwe_idx, features)

    for metric, num_classes in DEFAULT_METRIC_CLASSES.items():
        assert "label_idx" in preds[metric]
        assert "confidence" in preds[metric]
        assert "probs" in preds[metric]
        assert preds[metric]["label_idx"].shape == (batch,)
        assert preds[metric]["label_idx"].dtype == torch.long
        assert preds[metric]["confidence"].shape == (batch,)
        # вероятности — валидное распределение
        assert preds[metric]["probs"].shape == (batch, num_classes)
        prob_sums = preds[metric]["probs"].sum(dim=-1)
        torch.testing.assert_close(prob_sums, torch.ones(batch))
        # confidence ∈ [0,1]
        assert (preds[metric]["confidence"] >= 0).all()
        assert (preds[metric]["confidence"] <= 1).all()


def test_cvss_model_backward_runs_without_error():
    """Проверяем, что граф градиентов собирается по всем 12 головам."""
    model = _build_model()
    input_ids = torch.randint(0, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    cwe_idx = torch.randint(0, 100, (2,))
    features = torch.randn(2, 3)

    logits = model(input_ids, attention_mask, cwe_idx, features)
    loss = sum(F.cross_entropy(t, torch.zeros(2, dtype=torch.long)) for t in logits.values())
    loss.backward()

    # MLP-голова первой метрики должна получить градиент
    grad = model.heads["AV"].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    # Эмбеддинг CWE тоже должен иметь градиент
    assert model.features_mlp.cwe_embedding.weight.grad is not None


def test_cvss_model_eval_mode_predict_restores_train():
    model = _build_model()
    model.train()
    assert model.training
    model.predict(
        torch.randint(0, 1000, (1, 8)),
        torch.ones(1, 8, dtype=torch.long),
        torch.tensor([1]),
        torch.zeros(1, 3),
    )
    assert model.training, "predict должен восстановить training-режим"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA недоступна")
def test_cvss_model_cuda_forward():
    device = torch.device("cuda")
    model = _build_model().to(device)
    input_ids = torch.randint(0, 1000, (2, 8), device=device)
    attention_mask = torch.ones(2, 8, dtype=torch.long, device=device)
    cwe_idx = torch.randint(0, 100, (2,), device=device)
    features = torch.randn(2, 3, device=device)
    logits = model(input_ids, attention_mask, cwe_idx, features)
    for metric, t in logits.items():
        assert t.device.type == "cuda"
