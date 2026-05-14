"""Базовые тесты для обёртки :class:`CVSSCalculator`.

Проверяют корректность сборки/разбора вектора и согласованность итогового
балла с эталонными значениями для типичных классов уязвимостей.
"""

from __future__ import annotations

import pytest

from src.cvss_calculator import CVSSCalculator


@pytest.fixture()
def calc() -> CVSSCalculator:
    return CVSSCalculator()


# ---------------------------------------------------------------------------
# Граничные случаи: максимальный и минимальный балл.
# ---------------------------------------------------------------------------


def test_worst_case(calc: CVSSCalculator) -> None:
    """Все impact-метрики High, AV:N, доступ без условий, активная эксплуатация → 10.0."""
    metrics = {
        "AV": "N",
        "AC": "L",
        "AT": "N",
        "PR": "N",
        "UI": "N",
        "VC": "H",
        "VI": "H",
        "VA": "H",
        "SC": "H",
        "SI": "H",
        "SA": "H",
        "E": "A",
    }
    score, severity, _ = calc.calculate(metrics)
    assert score == 10.0
    assert severity == "Critical"


def test_best_case(calc: CVSSCalculator) -> None:
    """Все impact-метрики None, тяжёлый доступ → 0.0."""
    metrics = {
        "AV": "P",
        "AC": "H",
        "AT": "P",
        "PR": "H",
        "UI": "A",
        "VC": "N",
        "VI": "N",
        "VA": "N",
        "SC": "N",
        "SI": "N",
        "SA": "N",
    }
    score, severity, _ = calc.calculate(metrics)
    assert score == 0.0
    assert severity == "None"


# ---------------------------------------------------------------------------
# Типичные классы уязвимостей.
# ---------------------------------------------------------------------------


def test_xss_typical(calc: CVSSCalculator) -> None:
    """Типичный reflected XSS: сетевая атака, требуется действие пользователя,
    низкий impact на C/I/A. Ожидается Medium (4.0–6.9)."""
    metrics = {
        "AV": "N",
        "AC": "L",
        "AT": "N",
        "PR": "N",
        "UI": "A",
        "VC": "L",
        "VI": "L",
        "VA": "N",
        "SC": "L",
        "SI": "L",
        "SA": "N",
    }
    score, severity, _ = calc.calculate(metrics)
    assert 4.0 <= score <= 7.0
    assert severity == "Medium"


def test_sqli_typical(calc: CVSSCalculator) -> None:
    """Типичный SQL injection: сетевая атака, нужны минимальные привилегии,
    высокий impact на уязвимую систему. Ожидается High (7.0–8.9)."""
    metrics = {
        "AV": "N",
        "AC": "L",
        "AT": "N",
        "PR": "L",
        "UI": "N",
        "VC": "H",
        "VI": "H",
        "VA": "H",
        "SC": "N",
        "SI": "N",
        "SA": "N",
    }
    score, severity, _ = calc.calculate(metrics)
    assert 7.0 <= score <= 9.0
    assert severity == "High"


# ---------------------------------------------------------------------------
# Влияние Exploit Maturity (модификатор k_E).
# ---------------------------------------------------------------------------

_BASE_FOR_E = {
    "AV": "N",
    "AC": "L",
    "AT": "N",
    "PR": "N",
    "UI": "N",
    "VC": "H",
    "VI": "H",
    "VA": "H",
    "SC": "L",
    "SI": "L",
    "SA": "L",
}


def test_e_modifier_attacked(calc: CVSSCalculator) -> None:
    """E:A не должен снижать балл — это худший случай и одновременно дефолт."""
    score_a, _, _ = calc.calculate({**_BASE_FOR_E, "E": "A"})
    score_default, _, _ = calc.calculate(_BASE_FOR_E)
    assert score_a == score_default
    assert score_a >= 9.0


def test_e_modifier_unreported(calc: CVSSCalculator) -> None:
    """E:U должен дать балл строго ниже, чем E:A (модификатор k_E=0.91 < 1.0)."""
    score_a, _, _ = calc.calculate({**_BASE_FOR_E, "E": "A"})
    score_u, _, _ = calc.calculate({**_BASE_FOR_E, "E": "U"})
    assert score_u < score_a
    # Разумный диапазон: разница не нулевая и не чрезмерная.
    assert 0.1 <= (score_a - score_u) <= 3.0


# ---------------------------------------------------------------------------
# Сериализация / десериализация вектора.
# ---------------------------------------------------------------------------


def test_build_vector_string(calc: CVSSCalculator) -> None:
    metrics = {
        "AV": "N",
        "AC": "L",
        "AT": "N",
        "PR": "N",
        "UI": "N",
        "VC": "H",
        "VI": "H",
        "VA": "H",
        "SC": "N",
        "SI": "N",
        "SA": "N",
        "E": "A",
    }
    expected = "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/" "VC:H/VI:H/VA:H/SC:N/SI:N/SA:N/E:A"
    assert calc.build_vector_string(metrics) == expected


def test_parse_vector_string(calc: CVSSCalculator) -> None:
    vector = "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/" "VC:H/VI:H/VA:H/SC:N/SI:N/SA:N/E:A"
    parsed = calc.parse_vector_string(vector)
    assert parsed == {
        "AV": "N",
        "AC": "L",
        "AT": "N",
        "PR": "N",
        "UI": "N",
        "VC": "H",
        "VI": "H",
        "VA": "H",
        "SC": "N",
        "SI": "N",
        "SA": "N",
        "E": "A",
    }


def test_round_trip(calc: CVSSCalculator) -> None:
    metrics = {
        "AV": "A",
        "AC": "H",
        "AT": "P",
        "PR": "L",
        "UI": "P",
        "VC": "L",
        "VI": "H",
        "VA": "N",
        "SC": "H",
        "SI": "L",
        "SA": "N",
        "E": "P",
    }
    assert calc.parse_vector_string(calc.build_vector_string(metrics)) == metrics


# ---------------------------------------------------------------------------
# Дополнительные инварианты.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, "None"),
        (0.1, "Low"),  # граница None/Low — частая ошибка с < vs <=
        (3.9, "Low"),
        (4.0, "Medium"),
        (6.9, "Medium"),
        (7.0, "High"),
        (8.9, "High"),
        (9.0, "Critical"),
        (10.0, "Critical"),
    ],
)
def test_severity_boundaries(calc: CVSSCalculator, score: float, expected: str) -> None:
    """Проверка границ qualitative severity rating scale CVSS v4.0."""
    assert calc._score_to_severity(score) == expected


def test_e_metric_ordering(calc: CVSSCalculator) -> None:
    """Монотонность модификатора Exploit Maturity: A ≥ P ≥ U.

    По спецификации FIRST k_E убывает в порядке Attacked → POC → Unreported
    (1.0 → 0.94 → 0.91), поэтому балл не должен расти при ослаблении E.
    Хотя бы одно из неравенств обязано быть строгим — иначе модификатор
    фактически не применяется."""
    s_a, _, _ = calc.calculate({**_BASE_FOR_E, "E": "A"})
    s_p, _, _ = calc.calculate({**_BASE_FOR_E, "E": "P"})
    s_u, _, _ = calc.calculate({**_BASE_FOR_E, "E": "U"})
    assert s_a >= s_p >= s_u
    assert s_a > s_u  # хотя бы одно строгое неравенство


def test_e_default_equals_attacked(calc: CVSSCalculator) -> None:
    """Отсутствие E в векторе ≡ E:A (умолчание по спецификации FIRST)."""
    s_default, _, _ = calc.calculate(_BASE_FOR_E)
    s_attacked, _, _ = calc.calculate({**_BASE_FOR_E, "E": "A"})
    assert s_default == s_attacked


# ---------------------------------------------------------------------------
# Эталонные примеры FIRST CVSS v4.0.
#
# Векторы взяты из официального каталога https://www.first.org/cvss/v4-0/examples
# и из канонических примеров спецификации (раздел "Examples").
# Эталонные баллы перепроверены через референсную реализацию Red Hat
# (pip-пакет ``cvss``), чтобы исключить опечатки источника.
# Допустимая погрешность — 0.0 (точное совпадение).
# ---------------------------------------------------------------------------

FIRST_EXAMPLES = [
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H", 10.0),
    ("CVSS:4.0/AV:A/AC:L/AT:N/PR:N/UI:N/VC:L/VI:L/VA:L/SC:N/SI:N/SA:N", 5.3),
    ("CVSS:4.0/AV:L/AC:L/AT:N/PR:L/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N", 8.5),
    ("CVSS:4.0/AV:P/AC:H/AT:P/PR:H/UI:A/VC:N/VI:N/VA:L/SC:N/SI:N/SA:N", 1.0),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N", 9.3),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:P/VC:L/VI:L/VA:L/SC:L/SI:L/SA:L", 5.3),
    ("CVSS:4.0/AV:N/AC:H/AT:N/PR:N/UI:N/VC:H/VI:N/VA:N/SC:N/SI:N/SA:N", 8.2),
    ("CVSS:4.0/AV:L/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H", 9.4),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:H/SC:N/SI:N/SA:N", 8.7),
    ("CVSS:4.0/AV:N/AC:L/AT:P/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N", 9.2),
    ("CVSS:4.0/AV:L/AC:L/AT:P/PR:L/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N", 7.3),
    ("CVSS:4.0/AV:N/AC:L/AT:P/PR:N/UI:P/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N", 7.7),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:A/VC:N/VI:N/VA:N/SC:L/SI:L/SA:N", 5.1),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:N/SC:L/SI:L/SA:N", 6.9),
    ("CVSS:4.0/AV:L/AC:L/AT:N/PR:H/UI:N/VC:N/VI:N/VA:N/SC:H/SI:N/SA:N", 5.9),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:H/VA:N/SC:H/SI:H/SA:H", 9.3),
    ("CVSS:4.0/AV:L/AC:L/AT:N/PR:L/UI:N/VC:H/VI:N/VA:N/SC:N/SI:N/SA:N/E:A", 6.8),
    ("CVSS:4.0/AV:L/AC:L/AT:N/PR:H/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N", 8.4),
    ("CVSS:4.0/AV:N/AC:H/AT:P/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N/E:P", 8.2),
    ("CVSS:4.0/AV:N/AC:L/AT:P/PR:N/UI:N/VC:H/VI:N/VA:N/SC:N/SI:N/SA:N", 8.2),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:H/SC:N/SI:N/SA:L", 8.7),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:A/VC:L/VI:L/VA:N/SC:L/SI:L/SA:N", 5.1),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:A/VC:H/VI:H/VA:H/SC:L/SI:L/SA:N", 8.5),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:A/VC:N/VI:L/VA:N/SC:N/SI:N/SA:N", 5.1),
    ("CVSS:4.0/AV:N/AC:L/AT:P/PR:L/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N", 7.7),
    ("CVSS:4.0/AV:L/AC:L/AT:N/PR:H/UI:N/VC:H/VI:H/VA:N/SC:N/SI:N/SA:N", 8.3),
    ("CVSS:4.0/AV:L/AC:L/AT:N/PR:N/UI:P/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N", 8.5),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:N/VA:N/SC:N/SI:N/SA:N", 8.7),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:H/UI:N/VC:H/VI:N/VA:N/SC:N/SI:N/SA:N", 6.9),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:N/SC:N/SI:L/SA:N", 6.9),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:L/VA:N/SC:L/SI:N/SA:H", 7.8),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:L/VA:N/SC:L/SI:L/SA:L", 6.9),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:N/VI:N/VA:N/SC:H/SI:L/SA:H", 6.4),
    ("CVSS:4.0/AV:P/AC:H/AT:P/PR:H/UI:A/VC:N/VI:N/VA:N/SC:N/SI:N/SA:N", 0.0),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N/E:U", 6.3),
]


@pytest.mark.parametrize("vector, expected_score", FIRST_EXAMPLES)
def test_first_official_examples(calc: CVSSCalculator, vector: str, expected_score: float) -> None:
    """Сравнение с эталонными баллами FIRST для 35 разнотипных векторов."""
    metrics = calc.parse_vector_string(vector)
    score, severity, _ = calc.calculate(metrics)
    assert score == expected_score, f"Vector {vector!r}: expected {expected_score}, got {score}"
    # Severity также должен соответствовать таблице FIRST.
    assert severity == calc._score_to_severity(expected_score)


# ---------------------------------------------------------------------------
# Контрольная сверка с референсной библиотекой Red Hat (pip cvss).
# 50 случайных валидных векторов CVSS v4.0; ожидается побитовое совпадение.
# ---------------------------------------------------------------------------

_RANDOM_VALUES = {
    "AV": ["N", "A", "L", "P"],
    "AC": ["L", "H"],
    "AT": ["N", "P"],
    "PR": ["N", "L", "H"],
    "UI": ["N", "P", "A"],
    "VC": ["H", "L", "N"],
    "VI": ["H", "L", "N"],
    "VA": ["H", "L", "N"],
    "SC": ["H", "L", "N"],
    "SI": ["H", "L", "N"],
    "SA": ["H", "L", "N"],
    "E": ["A", "P", "U", None],  # None → метрика не указана (≡ X)
}


def _random_metrics(rng) -> dict:
    """Генерирует случайный валидный набор метрик CVSS v4.0."""
    out = {}
    for key, values in _RANDOM_VALUES.items():
        v = rng.choice(values)
        if v is not None:
            out[key] = v
    return out


def test_random_50_match_reference(calc: CVSSCalculator) -> None:
    """50 случайных векторов: наш балл должен побитово совпасть с pip cvss.

    Если хотя бы один вектор расходится — выводится список расхождений
    с обоими баллами для отладки.
    """
    pytest.importorskip("cvss", reason="референсная библиотека cvss не установлена")
    import random

    from cvss import CVSS4 as RefCVSS4  # type: ignore[import-not-found]  # noqa: N811

    rng = random.Random(42)

    diffs = []
    for _ in range(50):
        metrics = _random_metrics(rng)
        vector = calc.build_vector_string(metrics)
        ours = calc.calculate(metrics)[0]
        reference = float(RefCVSS4(vector).base_score)
        if ours != reference:
            diffs.append((vector, ours, reference))

    assert not diffs, "Расхождения с референсной библиотекой:\n" + "\n".join(
        f"  {v}: ours={o} vs ref={r}" for v, o, r in diffs
    )


# ---------------------------------------------------------------------------
# Тесты на корректность парсинга / обработку ошибок.
# ---------------------------------------------------------------------------


def test_parse_invalid_vector(calc: CVSSCalculator) -> None:
    """Полностью невалидная строка → ValueError при попытке расчёта."""
    parsed = calc.parse_vector_string("totally not a vector")
    with pytest.raises(ValueError):
        calc.calculate(parsed)


def test_parse_missing_metric(calc: CVSSCalculator) -> None:
    """Отсутствие обязательной метрики → ValueError."""
    parsed = calc.parse_vector_string("CVSS:4.0/AV:N/AC:L")
    with pytest.raises(ValueError, match="(?i)mandatory|missing"):
        calc.calculate(parsed)


def test_lowercase_metrics(calc: CVSSCalculator) -> None:
    """Метрики в нижнем регистре не поддерживаются спецификацией FIRST.

    Допустимы оба исхода: либо явная ошибка, либо нормальная работа после
    нормализации. На сегодняшний день обёртка падает с ``ValueError`` —
    проверяем, что сообщение информативное."""
    parsed = calc.parse_vector_string(
        "cvss:4.0/av:n/ac:l/at:n/pr:n/ui:n/vc:h/vi:h/va:h/sc:n/si:n/sa:n"
    )
    with pytest.raises(ValueError):
        calc.calculate(parsed)


def test_calculate_returns_correct_types(calc: CVSSCalculator) -> None:
    """Контракт возвращаемого типа: (float, str ∈ {уровни}, str с префиксом CVSS:4.0/)."""
    metrics = {
        "AV": "N",
        "AC": "L",
        "AT": "N",
        "PR": "N",
        "UI": "N",
        "VC": "H",
        "VI": "H",
        "VA": "H",
        "SC": "N",
        "SI": "N",
        "SA": "N",
        "E": "A",
    }
    result = calc.calculate(metrics)
    assert isinstance(result, tuple)
    assert len(result) == 3
    score, severity, vector = result
    # Строго встроенный float — отлавливает случайные numpy.float64 и т.п.
    assert type(score) is float
    assert isinstance(severity, str)
    assert severity in {"None", "Low", "Medium", "High", "Critical"}
    assert isinstance(vector, str)
    assert vector.startswith("CVSS:4.0/")
