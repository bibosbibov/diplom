"""Парсеры CVSS v3.1 / v4.0 векторов и таблицы соответствия классов.

Соответствие версии этапа обучения:
    - этап 1 (предобучение)  → V3_METRIC_ORDER + V3_LABEL_MAPS;
    - этап 2 (дообучение)    → V4_METRIC_ORDER + V4_LABEL_MAPS.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

# -------------------------------------------------------------------- label maps

#: CVSS v4.0 — все 12 метрик базового вектора + Exploit Maturity (E).
V4_LABEL_MAPS: dict[str, list[str]] = {
    "AV": ["N", "A", "L", "P"],          # Network / Adjacent / Local / Physical
    "AC": ["L", "H"],                    # Low / High
    "AT": ["N", "P"],                    # None / Present
    "PR": ["N", "L", "H"],               # None / Low / High
    "UI": ["N", "P", "A"],               # None / Passive / Active
    "VC": ["H", "L", "N"],
    "VI": ["H", "L", "N"],
    "VA": ["H", "L", "N"],
    "SC": ["H", "L", "N"],
    "SI": ["H", "L", "N"],
    "SA": ["H", "L", "N"],
    "E":  ["A", "P", "U"],               # Attacked / POC / Unreported
}

V4_METRIC_ORDER: tuple[str, ...] = (
    "AV", "AC", "AT", "PR", "UI",
    "VC", "VI", "VA",
    "SC", "SI", "SA",
    "E",
)

#: CVSS v3.1 — 8 общих метрик этапа 1 (импакт-метрики C/I/A приводятся к
#: каноническим именам v4 — VC/VI/VA, чтобы названия голов модели совпадали).
V3_LABEL_MAPS: dict[str, list[str]] = {
    "AV": ["N", "A", "L", "P"],
    "AC": ["L", "H"],
    "PR": ["N", "L", "H"],
    "UI": ["N", "R"],                    # None / Required (v3)
    "VC": ["H", "L", "N"],               # parsed from C:
    "VI": ["H", "L", "N"],               # parsed from I:
    "VA": ["H", "L", "N"],               # parsed from A:
    "E":  ["X", "U", "P", "F", "H"],     # NotDefined / Unproven / POC / Functional / High
}

V3_METRIC_ORDER: tuple[str, ...] = (
    "AV", "AC", "PR", "UI", "VC", "VI", "VA", "E",
)

#: Sentinel-индекс для CrossEntropyLoss(ignore_index=-100).
IGNORE_INDEX: int = -100


# ----------------------------------------------------------------------- parsers

# Метрика CVSS — буквы; значение — буква или цифра. CVSS:3.1 / CVSS:4.0
# попадут под этот regex как (CVSS, '3' / '4'); такие пары мы просто отфильтруем.
_KV_RE = re.compile(r"\b([A-Za-z]+):([A-Za-z0-9])\b")


def _kv_pairs(vector_str: str | None) -> dict[str, str]:
    if not isinstance(vector_str, str):
        return {}
    return {m.group(1).upper(): m.group(2).upper() for m in _KV_RE.finditer(vector_str)}


def parse_v4_vector(vector_str: str | None) -> dict[str, str | None]:
    """Парсит CVSS:4.0 вектор в dict из 12 канонических метрик.

    Поддерживает формат с префиксом ``CVSS:4.0/`` и без; значения метрик
    из дополнительных групп (например ``AU:Y``) игнорируются.
    """
    pairs = _kv_pairs(vector_str)
    return {metric: pairs.get(metric) for metric in V4_METRIC_ORDER}


def parse_v3_vector(vector_str: str | None) -> dict[str, str | None]:
    """Парсит CVSS:3.x вектор в dict из 8 stage 1 метрик.

    C/I/A → VC/VI/VA, чтобы имена меток совпадали со схемой v4.
    Метрика S (Scope) игнорируется (в v4 её нет).
    """
    pairs = _kv_pairs(vector_str)
    return {
        "AV": pairs.get("AV"),
        "AC": pairs.get("AC"),
        "PR": pairs.get("PR"),
        "UI": pairs.get("UI"),
        "VC": pairs.get("C"),
        "VI": pairs.get("I"),
        "VA": pairs.get("A"),
        "E":  pairs.get("E"),
    }


def vector_to_labels(
    parsed: dict[str, str | None],
    metric_order: Sequence[str],
    label_maps: dict[str, list[str]],
) -> np.ndarray:
    """Преобразует распарсенный dict в массив целочисленных индексов классов.

    Длина результата = ``len(metric_order)``. Для отсутствующих или неизвестных
    значений в позицию ставится :data:`IGNORE_INDEX`, чтобы CrossEntropyLoss
    их игнорировал на стадии обучения.
    """
    out = np.full(len(metric_order), IGNORE_INDEX, dtype=np.int64)
    for i, metric in enumerate(metric_order):
        value = parsed.get(metric)
        if value is None:
            continue
        classes = label_maps.get(metric)
        if not classes:
            continue
        try:
            out[i] = classes.index(value.upper())
        except ValueError:
            continue
    return out


__all__ = [
    "IGNORE_INDEX",
    "V3_LABEL_MAPS",
    "V3_METRIC_ORDER",
    "V4_LABEL_MAPS",
    "V4_METRIC_ORDER",
    "parse_v3_vector",
    "parse_v4_vector",
    "vector_to_labels",
]
