"""Модуль подготовки данных (раздел 2.3.2).

Преобразует строки сырого датасета в тензоры, ожидаемые архитектурой mBERT
+ MLP + Fusion (раздел 2.2.5 ВКР).
"""

from .cvss_vector_parser import (
    IGNORE_INDEX,
    V3_LABEL_MAPS,
    V3_METRIC_ORDER,
    V4_LABEL_MAPS,
    V4_METRIC_ORDER,
    parse_v3_vector,
    parse_v4_vector,
    vector_to_labels,
)
from .cwe_encoder import CWEEncoder
from .features_encoder import FeaturesEncoder
from .text_processor import TextProcessor

# torch / transformers подтягиваются лениво — модули можно импортировать,
# даже если ML-стек не установлен (например, на CI-узле для unit-тестов).
try:
    from .dataset import CVSSDataset  # noqa: F401
except ImportError:  # pragma: no cover
    CVSSDataset = None  # type: ignore[assignment]

try:
    from .tokenizer_wrapper import CVSSTokenizer  # noqa: F401
except ImportError:  # pragma: no cover
    CVSSTokenizer = None  # type: ignore[assignment]

__all__ = [
    "CVSSDataset",
    "CVSSTokenizer",
    "CWEEncoder",
    "FeaturesEncoder",
    "IGNORE_INDEX",
    "TextProcessor",
    "V3_LABEL_MAPS",
    "V3_METRIC_ORDER",
    "V4_LABEL_MAPS",
    "V4_METRIC_ORDER",
    "parse_v3_vector",
    "parse_v4_vector",
    "vector_to_labels",
]
