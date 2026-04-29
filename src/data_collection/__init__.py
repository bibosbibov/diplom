"""Модуль сбора и интеграции данных об уязвимостях.

Реализует раздел 2.3.1 «Алгоритм сбора и интеграции данных».
"""

from .bdu_collector import BDUCollector
from .cwe_names import CWENames
from .data_integrator import DataIntegrator
from .epss_collector import EPSSCollector
from .exploitdb_collector import ExploitDBCollector
from .kev_collector import KEVCollector
from .nvd_collector import NVDCollector

__all__ = [
    "BDUCollector",
    "CWENames",
    "DataIntegrator",
    "EPSSCollector",
    "ExploitDBCollector",
    "KEVCollector",
    "NVDCollector",
]
