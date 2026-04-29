"""Утилиты загрузки конфигурации проекта."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_DEFAULT_CONFIG_PATH = "configs/config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Загружает YAML-конфиг и подмешивает переменные окружения.

    Порядок выбора пути:
        1. явный аргумент path,
        2. переменная окружения CONFIG_PATH,
        3. configs/config.yaml.
    """
    load_dotenv(override=False)
    config_path = Path(path or os.getenv("CONFIG_PATH") or _DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    return config


def get_env(name: str, default: str | None = None) -> str | None:
    """Тонкая обёртка над os.getenv с предварительным load_dotenv."""
    load_dotenv(override=False)
    return os.getenv(name, default)
