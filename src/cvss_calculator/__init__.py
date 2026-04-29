"""CVSS v4.0 calculator.

Реализация перенесена из открытой Python-библиотеки cvss от Red Hat
(https://github.com/RedHatProductSecurity/cvss, лицензия LGPLv3+).
Алгоритм соответствует официальной спецификации FIRST CVSS v4.0
(https://www.first.org/cvss/v4.0/specification-document).

Текст лицензии: licenses/cvss-LGPL-LICENSE.
"""

from .calculator import CVSSCalculator
from .core import (
    CVSS4,
    CVSS4Error,
    CVSS4MalformedError,
    CVSS4MandatoryError,
    CVSS4RHMalformedError,
    CVSS4RHScoreDoesNotMatch,
    CVSSError,
    final_rounding,
)

__all__ = [
    "CVSSCalculator",
    "CVSS4",
    "CVSSError",
    "CVSS4Error",
    "CVSS4MalformedError",
    "CVSS4MandatoryError",
    "CVSS4RHMalformedError",
    "CVSS4RHScoreDoesNotMatch",
    "final_rounding",
]
