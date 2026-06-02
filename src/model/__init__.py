"""Архитектура классификации (раздел 2.2.5 ВКР).

mBERT + FeaturesMLP + FusionLayer + 12 классификационных голов.
"""

from .classification_heads import DEFAULT_METRIC_CLASSES, ClassificationHeads
from .cvss_model import DEFAULT_PRETRAINED_NAME, CVSSModel
from .features_mlp import FeaturesMLP
from .fingerprint import BACKBONE_PREFIXES, backbone_fingerprint
from .fusion_layer import FusionLayer

__all__ = [
    "CVSSModel",
    "ClassificationHeads",
    "DEFAULT_METRIC_CLASSES",
    "DEFAULT_PRETRAINED_NAME",
    "FeaturesMLP",
    "FusionLayer",
    "backbone_fingerprint",
    "BACKBONE_PREFIXES",
]
