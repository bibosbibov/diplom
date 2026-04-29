"""Высокоуровневая обёртка над CVSS4 для использования в инференс-пайплайне.

Принимает словарь метрик в виде, в котором его выдают 12 классификационных
голов модели mBERT (см. `src/model/`), формирует CVSS-вектор по
спецификации FIRST CVSS v4.0 и считает итоговый балл через
:class:`src.cvss_calculator.core.CVSS4`.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .core import CVSS4, CVSS4MalformedError, CVSS4MandatoryError

# Порядок обязательных метрик в строке вектора (CVSS v4.0, раздел 2 спецификации).
# 11 mandatory + опциональная E (Exploit Maturity) — суммарно 12 предсказываемых
# моделью полей.
_MANDATORY_ORDER = [
    "AV",
    "AC",
    "AT",
    "PR",
    "UI",
    "VC",
    "VI",
    "VA",
    "SC",
    "SI",
    "SA",
]
_OPTIONAL_ORDER = ["E"]
VECTOR_PREFIX = "CVSS:4.0"


class CVSSCalculator:
    """Обёртка над :class:`CVSS4` для расчёта итогового балла по словарю метрик."""

    def calculate(self, metrics: Dict[str, str]) -> Tuple[float, str, str]:
        """Считает базовый балл CVSS v4.0 по словарю метрик.

        Args:
            metrics: словарь вида ``{"AV": "N", "AC": "L", ..., "E": "A"}``,
                полученный из выходов классификационных голов модели mBERT.
                Ключ ``E`` опционален — если отсутствует, считается как
                ``Not Defined`` (worst case по спецификации FIRST).

        Returns:
            Кортеж ``(score, severity, vector)``:
                - ``score`` — базовый балл от 0.0 до 10.0;
                - ``severity`` — одна из строк ``None``/``Low``/``Medium``/
                  ``High``/``Critical``;
                - ``vector`` — каноническая CVSS-строка (например
                  ``CVSS:4.0/AV:N/AC:L/...``).
        """
        vector = self.build_vector_string(metrics)
        try:
            cvss = CVSS4(vector)
        except (CVSS4MalformedError, CVSS4MandatoryError) as exc:
            # Превращаем низкоуровневые исключения CVSS4 в идиоматический
            # ValueError для публичного API обёртки.
            raise ValueError(str(exc)) from exc
        score = float(cvss.base_score)
        severity = self._score_to_severity(score)
        return score, severity, vector

    def build_vector_string(self, metrics: Dict[str, str]) -> str:
        """Собирает каноническую CVSS-строку из словаря метрик.

        Поля упорядочиваются согласно спецификации (mandatory сначала, затем
        E). Отсутствующие mandatory-поля приведут к ошибке при разборе вектора
        в :class:`CVSS4` — здесь мы не валидируем их явно, чтобы сохранить
        единственное место валидации.
        """
        parts = [VECTOR_PREFIX]
        for key in _MANDATORY_ORDER:
            if key in metrics:
                parts.append(f"{key}:{metrics[key]}")
        for key in _OPTIONAL_ORDER:
            if key in metrics and metrics[key] != "X":
                parts.append(f"{key}:{metrics[key]}")
        return "/".join(parts)

    def parse_vector_string(self, vector: str) -> Dict[str, str]:
        """Разбирает CVSS-строку обратно в словарь ``{метрика: значение}``.

        Префикс ``CVSS:4.0`` отбрасывается. Поддерживается любой порядок
        полей, дублирование не проверяется (это задача :class:`CVSS4`).
        """
        result: Dict[str, str] = {}
        if not vector:
            return result
        tokens = vector.split("/")
        # Первый токен — префикс CVSS:4.0; пропускаем его, если он есть.
        if tokens and tokens[0].startswith("CVSS:"):
            tokens = tokens[1:]
        for token in tokens:
            if not token:
                continue
            if ":" not in token:
                continue
            key, value = token.split(":", 1)
            result[key] = value
        return result

    @staticmethod
    def _score_to_severity(score: float) -> str:
        """Конвертирует числовой балл в текстовый уровень критичности.

        Таблица соответствует разделу "Qualitative Severity Rating Scale"
        спецификации FIRST CVSS v4.0:
            * 0.0           → ``None``
            * 0.1  – 3.9    → ``Low``
            * 4.0  – 6.9    → ``Medium``
            * 7.0  – 8.9    → ``High``
            * 9.0  – 10.0   → ``Critical``
        """
        if score <= 0.0:
            return "None"
        if score < 4.0:
            return "Low"
        if score < 7.0:
            return "Medium"
        if score < 9.0:
            return "High"
        return "Critical"
