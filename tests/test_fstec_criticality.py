"""Тесты расчёта уровня критичности по Методике ФСТЭК (30.06.2025).

Эталонные кейсы — оба примера из приложения к Методике (BDU:2025-01611
и BDU:2025-03219). Проверяются промежуточные показатели (I_infr, I_at, I_imp),
итоговое V (с допуском ±0.01 на округление) и присвоенный уровень.
"""

from __future__ import annotations

import pytest

from src.cvss_calculator import FSTECCriticalityCalculator
from src.cvss_calculator.fstec_criticality import (
    E_WEIGHT,
    H_WEIGHT,
    K_WEIGHT,
    L_WEIGHT,
    P_WEIGHT,
    level_for,
    suggest_e,
    suggest_h,
)


@pytest.fixture
def calc() -> FSTECCriticalityCalculator:
    return FSTECCriticalityCalculator()


# ---------------------------------------------------------------------------
# Пример 1 — BDU:2025-01611 (приложение, стр. 12–13)
# ---------------------------------------------------------------------------


def test_example_1_bdu_2025_01611(calc) -> None:
    res = calc.calculate(
        i_cvss=8.8,
        k=["firewall"],          # межсетевые экраны (0.9)
        l=["from_10_to_50"],     # 10–50% (0.6)
        p="internet_accessible", # доступно из Интернета (1.1)
        e=["no_info"],           # отсутствуют сведения (0.1)
        h=["privilege_escalation"],  # Повышение привилегий (0.5)
    )
    # Промежуточные показатели
    assert res.k_value == 0.9
    assert res.l_value == 0.6
    assert res.p_value == 1.1
    assert res.i_infr == pytest.approx(0.9)   # 0.45 + 0.12 + 0.33
    assert res.i_at == pytest.approx(0.1)
    assert res.i_imp == pytest.approx(0.5)
    # Слагаемые I_infr
    assert res.k_term == pytest.approx(0.45)
    assert res.l_term == pytest.approx(0.12)
    assert res.p_term == pytest.approx(0.33)
    # Итоговое V и уровень
    assert res.v == pytest.approx(4.75, abs=0.01)
    assert res.level == "Средний"


# ---------------------------------------------------------------------------
# Пример 2 — BDU:2025-03219 (приложение, стр. 14–15)
# ---------------------------------------------------------------------------


def test_example_2_bdu_2025_03219(calc) -> None:
    res = calc.calculate(
        i_cvss=9.8,
        # K: критические процессы (1.1) + серверы (0.7) + АРМ (0.5) + СХД (0.4) → max 1.1
        k=["critical_process", "server", "workstation", "storage"],
        # L: >70% (1.0) + 50–70% (0.8) + 10–50% (0.6) + 10–50% (0.6) → max 1.0
        l=["gt_70", "from_50_to_70", "from_10_to_50"],
        p="internet_accessible",      # доступно из Интернета (1.1)
        e=["exploit_available"],      # имеется эксплойт (0.3)
        h=["arbitrary_code_execution"],  # Выполнение произвольного кода (0.5)
    )
    assert res.k_value == 1.1   # правило max (п.15)
    assert res.l_value == 1.0   # правило max
    assert res.p_value == 1.1
    assert res.i_infr == pytest.approx(1.08)  # 0.55 + 0.2 + 0.33
    assert res.i_at == pytest.approx(0.3)
    assert res.i_imp == pytest.approx(0.5)
    assert res.v == pytest.approx(8.47, abs=0.01)
    assert res.level == "Критический"


# ---------------------------------------------------------------------------
# Правило максимума и веса
# ---------------------------------------------------------------------------


def test_max_rule_picks_largest(calc) -> None:
    """K из {серверы 0.7, СХД 0.4} → берётся max 0.7."""
    res = calc.calculate(
        i_cvss=5.0, k=["server", "storage"], l=["lt_10"],
        p="internet_isolated", e=["no_info"], h=["cross_site_scripting"],
    )
    assert res.k_value == 0.7


def test_weights_match_methodology() -> None:
    assert (K_WEIGHT, L_WEIGHT, P_WEIGHT, E_WEIGHT, H_WEIGHT) == (0.5, 0.2, 0.3, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Уровни (Таблица 2) и валидация
# ---------------------------------------------------------------------------


def test_level_boundaries() -> None:
    assert level_for(8.01) == "Критический"
    assert level_for(8.0) == "Высокий"
    assert level_for(5.0) == "Высокий"
    assert level_for(4.99) == "Средний"
    assert level_for(2.0) == "Средний"
    assert level_for(1.99) == "Низкий"
    assert level_for(0.0) == "Низкий"


def test_empty_multiselect_raises(calc) -> None:
    with pytest.raises(ValueError, match="требует хотя бы одного"):
        calc.calculate(i_cvss=5.0, k=[], l=["lt_10"], p="internet_isolated",
                       e=["no_info"], h=["denial_of_service"])


def test_unknown_code_raises(calc) -> None:
    with pytest.raises(ValueError, match="неизвестный код"):
        calc.calculate(i_cvss=5.0, k=["nonexistent"], l=["lt_10"],
                       p="internet_isolated", e=["no_info"], h=["denial_of_service"])


def test_cvss_out_of_range_raises(calc) -> None:
    with pytest.raises(ValueError, match="0.0–10.0"):
        calc.calculate(i_cvss=11.0, k=["server"], l=["lt_10"],
                       p="internet_isolated", e=["no_info"], h=["denial_of_service"])


# ---------------------------------------------------------------------------
# Предзаполнение E и H (подсказки, не влияют на расчёт V)
# ---------------------------------------------------------------------------


def test_suggest_e_priority_kev_over_exploit() -> None:
    # KEV приоритетнее ExploitDB
    assert suggest_e(kev=True, exploit=True) == ["in_the_wild"]
    assert suggest_e(kev=True, exploit=False) == ["in_the_wild"]
    # только ExploitDB
    assert suggest_e(kev=False, exploit=True) == ["exploit_available"]
    # нет сведений → отсутствуют сведения (0.1), как в Примере 1 «данные уточняются»
    assert suggest_e(kev=False, exploit=False) == ["no_info"]
    assert suggest_e(kev=None, exploit=None) == ["no_info"]


def test_suggest_h_by_cwe() -> None:
    assert suggest_h("CWE-79") == ["cross_site_scripting"]
    assert suggest_h("CWE-269") == ["privilege_escalation"]
    assert suggest_h("CWE-89") == ["code_injection"]
    assert suggest_h("CWE-200") == ["obtain_sensitive_information"]
    assert suggest_h("CWE-94") == ["code_injection", "arbitrary_code_execution"]
    # нормализация и неизвестные/пустые
    assert suggest_h("cwe-79") == ["cross_site_scripting"]
    assert suggest_h("CWE-99999") == []
    assert suggest_h(None) == []
    assert suggest_h("not-a-cwe") == []


def test_suggested_h_codes_are_valid_options() -> None:
    """Все коды из CWE→H существуют среди опций показателя H."""
    from src.cvss_calculator.fstec_criticality import CWE_TO_H, H_OPTIONS
    valid = {o.code for o in H_OPTIONS}
    for cwe, codes in CWE_TO_H.items():
        for code in codes:
            assert code in valid, f"{cwe}: неизвестный код H {code!r}"


def test_suggestions_do_not_affect_v(calc) -> None:
    """Подсказки — лишь предзаполнение; формула V считает по переданным кодам."""
    # Берём предложенные E/H для Примера 1 и убеждаемся, что V тот же 4.75.
    e = suggest_e(kev=False, exploit=False)          # ["no_info"] = 0.1
    h = suggest_h("CWE-269")                          # ["privilege_escalation"] = 0.5
    res = calc.calculate(i_cvss=8.8, k=["firewall"], l=["from_10_to_50"],
                         p="internet_accessible", e=e, h=h)
    assert res.v == pytest.approx(4.75, abs=0.01)
    assert res.level == "Средний"
