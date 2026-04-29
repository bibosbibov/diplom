"""Настройка логирования модуля сбора данных."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def setup_logger(
    name: str = "data_collection",
    log_file: str | Path = "logs/collection.log",
    file_level: int = logging.INFO,
    console_level: int = logging.WARNING,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Возвращает настроенный логгер.

    Файловый обработчик с ротацией пишет в logs/collection.log на уровне INFO,
    консольный — на уровне WARNING. Повторный вызов не дублирует обработчики.
    """
    global _CONFIGURED
    logger = logging.getLogger(name)

    if _CONFIGURED:
        return logger

    logger.setLevel(min(file_level, console_level))
    logger.propagate = False

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    env_level = os.getenv("LOG_LEVEL")
    if env_level:
        try:
            logger.setLevel(getattr(logging, env_level.upper()))
        except AttributeError:
            pass

    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Возвращает дочерний логгер с уже настроенными обработчиками."""
    setup_logger()
    return logging.getLogger(f"data_collection.{name}")
