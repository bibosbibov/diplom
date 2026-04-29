"""Подготовка текстового входа модели.

Реализует начальную часть алгоритма раздела 2.3.2: выбор языка описания
(d_ru приоритетнее d_en), очистка HTML, нормализация пробелов, склейка
описания с человекочитаемым именем CWE через [SEP].
"""

from __future__ import annotations

import html
import re

_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


class TextProcessor:
    """Подготовка строки t = description + [SEP] + cwe_name."""

    SEP = "[SEP]"

    def prepare_text(
        self,
        d_ru: str | None,
        d_en: str | None,
        cwe_name: str | None = None,
    ) -> str:
        description = self._pick_description(d_ru, d_en)
        description = self.clean(description)
        cwe = self.clean(cwe_name) if cwe_name else ""
        if cwe:
            return f"{description} {self.SEP} {cwe}".strip()
        return description

    @staticmethod
    def _pick_description(d_ru: str | None, d_en: str | None) -> str:
        if isinstance(d_ru, str) and d_ru.strip():
            return d_ru
        if isinstance(d_en, str) and d_en.strip():
            return d_en
        return ""

    @staticmethod
    def clean(text: str | None) -> str:
        if not isinstance(text, str):
            return ""
        text = html.unescape(text)
        text = _HTML_RE.sub(" ", text)
        text = _WS_RE.sub(" ", text)
        return text.strip()


__all__ = ["TextProcessor"]
