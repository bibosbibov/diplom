"""Inference-pipeline для системы автоматической оценки CVSS v4.0."""

from .predictor import VulnerabilityPredictor
from .predictor_v31 import VulnerabilityPredictorV31

__all__ = ["VulnerabilityPredictor", "VulnerabilityPredictorV31"]
