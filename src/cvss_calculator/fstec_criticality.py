"""Расчёт уровня критичности уязвимости по Методике ФСТЭК России (30.06.2025).

Методика вычисляет уровень критичности уязвимости в **конкретной**
информационной системе (п.12):

    V = I_cvss × I_infr × (I_at + I_imp)

где (п.14, 16, 17):
    I_infr = k·K + l·L + p·P   — влияние на функционирование ИС;
    I_at   = e·E               — возможность эксплуатации;
    I_imp  = h·H               — последствия эксплуатации.

``I_cvss`` — базовый балл CVSS 3.1 (п.13); в этом проекте берётся из предсказания
модели v3.1, у пользователя не запрашивается. Контекстные показатели K/L/P/E/H
зависят от информационной системы и выбираются пользователем.

Правило максимума (п.15, 16, 17): показатели K, L, E, H могут принимать
несколько значений одновременно — итоговой оценке присваивается **наибольшее**.
P — одиночный выбор.

Все веса и значения показателей взяты строго из Таблицы 1 Методики, пороги
уровней — из Таблицы 2. Оба примера приложения воспроизведены в
``tests/test_fstec_criticality.py``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Весовые коэффициенты (Таблица 1: столбец «Весовой коэффициент»).
# ---------------------------------------------------------------------------

K_WEIGHT = 0.5  # k — тип компонента (п.14)
L_WEIGHT = 0.2  # l — количество уязвимых компонентов (п.14)
P_WEIGHT = 0.3  # p — влияние на защиту периметра (п.14)
E_WEIGHT = 1.0  # e — эксплуатация уязвимости (п.16)
H_WEIGHT = 1.0  # h — последствия воздействий (п.17)


@dataclass(frozen=True)
class Option:
    """Вариант значения показателя Таблицы 1.

    Attributes:
        code: машинный идентификатор (для API/фронта/тестов).
        label: человекочитаемое наименование из Таблицы 1.
        value: числовое значение показателя (Ki/Lj/Pm/En/Hk).
    """

    code: str
    label: str
    value: float


# ---------------------------------------------------------------------------
# Таблица 1 — значения показателей. Порядок сохранён как в Методике.
# ---------------------------------------------------------------------------

#: K — тип компонента ИС, подверженного уязвимости (Таблица 1, п/п 1).
K_OPTIONS: tuple[Option, ...] = (
    Option("critical_process", "Компоненты, реализующие важные процессы (бизнес-процессы), функции, полномочия", 1.1),
    Option("firewall", "Межсетевые экраны", 0.9),
    Option("network_device", "Сетевые устройства и шлюзы", 0.9),
    Option("telecom", "Телекоммуникационное оборудование, система управления сетью передачи данных", 0.8),
    Option("server", "Серверы (центральные вычислительные узлы)", 0.7),
    Option("workstation", "Пользовательские устройства (автоматизированные рабочие места)", 0.5),
    Option("storage", "Системы хранения данных", 0.4),
    Option("other", "Другие компоненты", 0.1),
)

#: L — количество уязвимых компонентов ИС (Таблица 1, п/п 2).
L_OPTIONS: tuple[Option, ...] = (
    Option("gt_70", "Более 70% компонентов от общего числа", 1.0),
    Option("from_50_to_70", "50–70% компонентов", 0.8),
    Option("from_10_to_50", "10–50% компонентов", 0.6),
    Option("lt_10", "Менее 10% компонентов", 0.5),
)

#: P — влияние на эффективность защиты периметра ИС (Таблица 1, п/п 3).
P_OPTIONS: tuple[Option, ...] = (
    Option("internet_accessible", "Уязвимое средство доступно из сети «Интернет»", 1.1),
    Option("internet_isolated", "Уязвимое средство недоступно из сети «Интернет»", 0.6),
)

#: E — эксплуатация уязвимости (Таблица 1, п/п 4).
E_OPTIONS: tuple[Option, ...] = (
    Option("in_the_wild", "Эксплуатируется в реальных атаках", 0.6),
    Option("exploit_available", "Имеются сведения о наличии средств эксплуатации (эксплойта)", 0.3),
    Option("no_info", "Отсутствуют сведения об эксплуатации в реальных атаках (наличии эксплойта)", 0.1),
)

#: H — последствия воздействий (Таблица 1, п/п 5).
H_OPTIONS: tuple[Option, ...] = (
    Option("arbitrary_code_execution", "Выполнение произвольного кода (Arbitrary Code Execution)", 0.5),
    Option("privilege_escalation", "Повышение привилегий (Privilege Escalation)", 0.5),
    Option("security_bypass", "Обход механизмов безопасности (Security Bypass)", 0.4),
    Option("code_injection", "Внедрение кода (Code Injection)", 0.34),
    Option("obtain_sensitive_information", "Получение конфиденциальной информации (Obtain Sensitive Information)", 0.3),
    Option("loss_of_integrity", "Нарушение целостности данных (Loss of Integrity)", 0.3),
    Option("denial_of_service", "Отказ в обслуживании (DoS)", 0.26),
    Option("overwrite_arbitrary_files", "Перезапись произвольных файлов (Overwrite Arbitrary Files)", 0.22),
    Option("write_local_files", "Запись локальных файлов (Write Local Files)", 0.2),
    Option("read_local_files", "Чтение локальных файлов (Read Local Files)", 0.18),
    Option("spoof_user_interface", "Поддельный пользовательский интерфейс (Spoof User Interface)", 0.12),
    Option("cross_site_scripting", "Межсайтовый скриптинг (Cross Site Scripting)", 0.1),
)

#: Полный каталог показателей — для отдачи на фронт/в API.
INDICATOR_CATALOG: dict[str, dict[str, object]] = {
    "K": {"weight": K_WEIGHT, "multiselect": True, "options": K_OPTIONS},
    "L": {"weight": L_WEIGHT, "multiselect": True, "options": L_OPTIONS},
    "P": {"weight": P_WEIGHT, "multiselect": False, "options": P_OPTIONS},
    "E": {"weight": E_WEIGHT, "multiselect": True, "options": E_OPTIONS},
    "H": {"weight": H_WEIGHT, "multiselect": True, "options": H_OPTIONS},
}

_BY_CODE: dict[str, dict[str, Option]] = {
    name: {opt.code: opt for opt in cfg["options"]}  # type: ignore[union-attr]
    for name, cfg in INDICATOR_CATALOG.items()
}


# ---------------------------------------------------------------------------
# Результат расчёта.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FSTECResult:
    """Результат расчёта уровня критичности по Методике ФСТЭК.

    Attributes:
        v: уровень критичности V, округлённый до 2 знаков (как в примерах).
        v_exact: точное (неокруглённое) значение V.
        level: наименование уровня (Таблица 2).
        i_cvss: базовый балл CVSS 3.1.
        i_infr / i_at / i_imp: промежуточные показатели.
        k_value / l_value / p_value / e_value / h_value: итоговые (с учётом max)
            значения показателей Ki/Lj/Pm/En/Hk.
        k_term / l_term / p_term: слагаемые I_infr (k·K, l·L, p·P).
    """

    v: float
    v_exact: float
    level: str
    i_cvss: float
    i_infr: float
    i_at: float
    i_imp: float
    k_value: float
    l_value: float
    p_value: float
    e_value: float
    h_value: float
    k_term: float
    l_term: float
    p_term: float
    selected: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для сериализации (API/отчёт)."""
        return {
            "v": self.v,
            "v_exact": self.v_exact,
            "level": self.level,
            "i_cvss": self.i_cvss,
            "i_infr": self.i_infr,
            "i_at": self.i_at,
            "i_imp": self.i_imp,
            "breakdown": {
                "k_value": self.k_value,
                "l_value": self.l_value,
                "p_value": self.p_value,
                "e_value": self.e_value,
                "h_value": self.h_value,
                "k_term": self.k_term,
                "l_term": self.l_term,
                "p_term": self.p_term,
            },
            "selected": self.selected,
        }


# ---------------------------------------------------------------------------
# Предзаполнение показателей E и H (вспомогательное, не влияет на формулу V).
#
# Методика предписывает, что итоговое решение принимает специалист (п.9), поэтому
# это лишь ПРЕДЛОЖЕНИЕ значений — пользователь правит вручную в интерфейсе.
# ---------------------------------------------------------------------------

#: Стартовый словарь соответствия CWE → коды последствий (H). Для частых CWE;
#: для отсутствующих в словаре H не предлагается (пользователь выбирает сам).
CWE_TO_H: dict[str, tuple[str, ...]] = {
    # Внедрение и выполнение кода
    "CWE-94": ("code_injection", "arbitrary_code_execution"),
    "CWE-95": ("code_injection", "arbitrary_code_execution"),
    "CWE-78": ("arbitrary_code_execution", "code_injection"),
    "CWE-77": ("arbitrary_code_execution", "code_injection"),
    "CWE-89": ("code_injection",),
    "CWE-91": ("code_injection",),
    "CWE-917": ("code_injection",),
    "CWE-502": ("arbitrary_code_execution", "code_injection"),
    "CWE-434": ("arbitrary_code_execution",),
    "CWE-98": ("arbitrary_code_execution", "code_injection"),
    # Нарушение работы с памятью → как правило произвольный код
    "CWE-787": ("arbitrary_code_execution",),
    "CWE-119": ("arbitrary_code_execution",),
    "CWE-120": ("arbitrary_code_execution",),
    "CWE-121": ("arbitrary_code_execution",),
    "CWE-122": ("arbitrary_code_execution",),
    "CWE-416": ("arbitrary_code_execution",),
    "CWE-415": ("arbitrary_code_execution",),
    "CWE-190": ("arbitrary_code_execution",),
    # Чтение данных / раскрытие информации
    "CWE-125": ("obtain_sensitive_information",),
    "CWE-200": ("obtain_sensitive_information",),
    "CWE-209": ("obtain_sensitive_information",),
    "CWE-532": ("obtain_sensitive_information",),
    "CWE-538": ("obtain_sensitive_information",),
    "CWE-359": ("obtain_sensitive_information",),
    "CWE-798": ("obtain_sensitive_information",),
    "CWE-522": ("obtain_sensitive_information",),
    "CWE-311": ("obtain_sensitive_information",),
    "CWE-918": ("obtain_sensitive_information",),  # SSRF
    # Повышение привилегий
    "CWE-269": ("privilege_escalation",),
    "CWE-264": ("privilege_escalation",),
    "CWE-266": ("privilege_escalation",),
    "CWE-268": ("privilege_escalation",),
    "CWE-250": ("privilege_escalation",),
    # Обход механизмов безопасности / аутентификация / авторизация
    "CWE-287": ("security_bypass",),
    "CWE-306": ("security_bypass",),
    "CWE-862": ("security_bypass",),
    "CWE-863": ("security_bypass",),
    "CWE-639": ("security_bypass",),
    "CWE-732": ("security_bypass",),
    "CWE-285": ("security_bypass",),
    "CWE-288": ("security_bypass",),
    "CWE-295": ("security_bypass",),
    # Межсайтовый скриптинг
    "CWE-79": ("cross_site_scripting",),
    "CWE-80": ("cross_site_scripting",),
    # Подделка интерфейса / открытые редиректы / clickjacking
    "CWE-601": ("spoof_user_interface",),
    "CWE-1021": ("spoof_user_interface",),
    # Чтение локальных файлов / обход пути / XXE
    "CWE-22": ("read_local_files",),
    "CWE-23": ("read_local_files",),
    "CWE-36": ("read_local_files",),
    "CWE-611": ("read_local_files", "obtain_sensitive_information"),
    # Запись/перезапись файлов
    "CWE-73": ("overwrite_arbitrary_files",),
    # Отказ в обслуживании
    "CWE-400": ("denial_of_service",),
    "CWE-770": ("denial_of_service",),
    "CWE-476": ("denial_of_service",),
    "CWE-835": ("denial_of_service",),
    # Нарушение целостности (CSRF — несанкционированные действия)
    "CWE-352": ("loss_of_integrity",),
}

_CWE_RE = re.compile(r"CWE-(\d+)", re.IGNORECASE)


def suggest_e(kev: bool | None, exploit: bool | None) -> list[str]:
    """Предлагаемое значение E по флагам эксплуатации (п.16).

    KEV → «эксплуатируется в реальных атаках» (0.6); иначе ExploitDB → «имеются
    сведения об эксплойте» (0.3); иначе → «отсутствуют сведения» (0.1). Последнее
    соответствует случаю «данные уточняются» из Примера 1 Методики.
    """
    if kev:
        return ["in_the_wild"]
    if exploit:
        return ["exploit_available"]
    return ["no_info"]


def suggest_h(cwe_id: str | None) -> list[str]:
    """Предлагаемые коды последствий H по типу CWE (см. :data:`CWE_TO_H`).

    Для CWE вне словаря возвращает пустой список — H выбирает пользователь.
    """
    if not cwe_id:
        return []
    match = _CWE_RE.search(str(cwe_id))
    if not match:
        return []
    return list(CWE_TO_H.get(f"CWE-{match.group(1)}", ()))


#: Уровни критичности (Таблица 2 Методики).
LEVEL_CRITICAL = "Критический"
LEVEL_HIGH = "Высокий"
LEVEL_MEDIUM = "Средний"
LEVEL_LOW = "Низкий"


def level_for(v: float) -> str:
    """Наименование уровня критичности по значению V (Таблица 2).

    ``V > 8.0`` → Критический; ``5.0 ≤ V ≤ 8.0`` → Высокий;
    ``2.0 ≤ V < 5.0`` → Средний; ``V < 2.0`` → Низкий.
    """
    if v > 8.0:
        return LEVEL_CRITICAL
    if v >= 5.0:
        return LEVEL_HIGH
    if v >= 2.0:
        return LEVEL_MEDIUM
    return LEVEL_LOW


class FSTECCriticalityCalculator:
    """Калькулятор уровня критичности уязвимости по Методике ФСТЭК (30.06.2025)."""

    def calculate(
        self,
        i_cvss: float,
        k: Sequence[str],
        l: Sequence[str],
        p: str,
        e: Sequence[str],
        h: Sequence[str],
    ) -> FSTECResult:
        """Считает V и уровень критичности.

        Args:
            i_cvss: базовый балл CVSS 3.1 (0.0–10.0).
            k: коды выбранных типов компонентов (мультивыбор, ≥1); берётся max.
            l: коды доли уязвимых компонентов (мультивыбор, ≥1); берётся max.
            p: код влияния на периметр (одиночный выбор).
            e: коды сведений об эксплуатации (мультивыбор, ≥1); берётся max.
            h: коды последствий воздействий (мультивыбор, ≥1); берётся max.

        Returns:
            :class:`FSTECResult` с V, уровнем и пошаговой разбивкой.

        Raises:
            ValueError: если ``i_cvss`` вне диапазона, какой-либо мультивыбор
                пуст или передан неизвестный код.
        """
        if not 0.0 <= i_cvss <= 10.0:
            raise ValueError(f"i_cvss должен быть в диапазоне 0.0–10.0, получено {i_cvss}")

        # Правило максимума (п.15/16/17) для K, L, E, H; P — одиночный.
        k_value = self._max_value("K", k)
        l_value = self._max_value("L", l)
        e_value = self._max_value("E", e)
        h_value = self._max_value("H", h)
        p_value = self._single_value("P", p)

        k_term = K_WEIGHT * k_value
        l_term = L_WEIGHT * l_value
        p_term = P_WEIGHT * p_value
        i_infr = k_term + l_term + p_term  # п.14
        i_at = E_WEIGHT * e_value          # п.16
        i_imp = H_WEIGHT * h_value         # п.17

        v_exact = i_cvss * i_infr * (i_at + i_imp)  # п.12
        v = round(v_exact, 2)

        return FSTECResult(
            v=v,
            v_exact=v_exact,
            level=level_for(v),
            i_cvss=float(i_cvss),
            i_infr=i_infr,
            i_at=i_at,
            i_imp=i_imp,
            k_value=k_value,
            l_value=l_value,
            p_value=p_value,
            e_value=e_value,
            h_value=h_value,
            k_term=k_term,
            l_term=l_term,
            p_term=p_term,
            selected={
                "K": list(k), "L": list(l), "P": [p], "E": list(e), "H": list(h)
            },
        )

    def validate(
        self,
        k: Sequence[str],
        l: Sequence[str],
        p: str,
        e: Sequence[str],
        h: Sequence[str],
    ) -> None:
        """Проверяет коды показателей без расчёта (fail-fast до тяжёлого инференса).

        Raises:
            ValueError: пустой мультивыбор или неизвестный код.
        """
        self._max_value("K", k)
        self._max_value("L", l)
        self._single_value("P", p)
        self._max_value("E", e)
        self._max_value("H", h)

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _max_value(indicator: str, codes: Sequence[str]) -> float:
        """Наибольшее значение показателя среди выбранных кодов (правило max)."""
        if not codes:
            raise ValueError(f"показатель {indicator} требует хотя бы одного значения")
        table = _BY_CODE[indicator]
        values: list[float] = []
        for code in codes:
            if code not in table:
                raise ValueError(f"неизвестный код {code!r} для показателя {indicator}")
            values.append(table[code].value)
        return max(values)

    @staticmethod
    def _single_value(indicator: str, code: str) -> float:
        """Значение показателя для одиночного выбора (P)."""
        table = _BY_CODE[indicator]
        if not code:
            raise ValueError(f"показатель {indicator} требует одного значения")
        if code not in table:
            raise ValueError(f"неизвестный код {code!r} для показателя {indicator}")
        return table[code].value


__all__ = [
    "FSTECCriticalityCalculator",
    "FSTECResult",
    "Option",
    "INDICATOR_CATALOG",
    "K_OPTIONS",
    "L_OPTIONS",
    "P_OPTIONS",
    "E_OPTIONS",
    "H_OPTIONS",
    "K_WEIGHT",
    "L_WEIGHT",
    "P_WEIGHT",
    "E_WEIGHT",
    "H_WEIGHT",
    "CWE_TO_H",
    "suggest_e",
    "suggest_h",
    "level_for",
    "LEVEL_CRITICAL",
    "LEVEL_HIGH",
    "LEVEL_MEDIUM",
    "LEVEL_LOW",
]
