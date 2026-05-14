"""End-to-end предсказание CVSS-вектора по описанию уязвимости.

Модуль связывает компоненты подготовки данных (токенизатор, CWE-encoder,
features-encoder), обученную модель :class:`CVSSModel` и собственный
калькулятор :class:`CVSSCalculator` в единый pipeline, удобный для
интерактивного и пакетного применения.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..cvss_calculator import CVSSCalculator
from ..data_preparation.cvss_vector_parser import V4_LABEL_MAPS, V4_METRIC_ORDER
from ..data_preparation.cwe_encoder import CWEEncoder
from ..data_preparation.features_encoder import FeaturesEncoder
from ..data_preparation.text_processor import TextProcessor
from ..data_preparation.tokenizer_wrapper import CVSSTokenizer
from ..model.cvss_model import CVSSModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/final_model.pt"
DEFAULT_CONFIG_PATH = "configs/train.yaml"
DEFAULT_CWE_VOCAB_PATH = "data/processed/cwe_vocab.json"


class VulnerabilityPredictor:
    """Высокоуровневый pipeline инференса CVSS v4.0.

    Args:
        model_path: Путь к сохранённому ``state_dict`` модели.
        config_path: YAML-конфиг с гиперпараметрами (используется секция
            ``stage2.metric_classes``).
        device: ``"auto"`` (cuda при доступности, иначе cpu), либо явное
            ``"cuda"`` / ``"cpu"``.
        confidence_threshold: Порог softmax-уверенности; всё, что ниже,
            попадает в ``low_confidence_metrics`` в результате.
        cwe_vocab_path: Путь к JSON со словарём CWE → индекс.
    """

    METRIC_ORDER: tuple[str, ...] = V4_METRIC_ORDER
    LABEL_MAPS: dict[str, list[str]] = V4_LABEL_MAPS

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
        device: str = "auto",
        confidence_threshold: float = 0.7,
        cwe_vocab_path: str = DEFAULT_CWE_VOCAB_PATH,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.device = self._resolve_device(device)

        self._metric_classes = self._load_metric_classes(config_path)
        self._cwe_encoder = CWEEncoder.load(cwe_vocab_path)
        self._features_encoder = FeaturesEncoder()
        self._text_processor = TextProcessor()
        self._tokenizer = CVSSTokenizer()
        self._calculator = CVSSCalculator()

        self.model = CVSSModel(
            num_cwe=len(self._cwe_encoder),
            metric_classes=self._metric_classes,
        )
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    # --------------------------------------------------------------- public API

    def predict(
        self,
        description: str | None = None,
        cwe_id: str | None = None,
        description_lang: str | None = None,
        description_ru: str | None = None,
        epss: float | None = None,
        kev: int | None = None,
        exploit: int | None = None,
    ) -> dict[str, Any]:
        """Предсказывает CVSS v4.0-вектор для одной уязвимости.

        Args:
            description: Текстовое описание уязвимости (русский или английский).
                Если язык не указан в ``description_lang``, определяется
                автоматически по доле кириллицы.
            cwe_id: Идентификатор CWE в формате ``"CWE-XXX"`` (например,
                ``"CWE-89"`` для SQL injection).
            description_lang: Явное указание языка ``description``: ``"ru"``,
                ``"en"`` или ``None`` (автоопределение).
            description_ru: Отдельное русскоязычное описание, если
                ``description`` английское и есть также русское.
            epss: Вероятность эксплуатации в ближайшие 30 дней (0..1).
                ``None`` — признак отсутствует.
            kev: 1 / 0 — присутствие в каталоге CISA KEV. ``None`` если
                неизвестно.
            exploit: 1 / 0 — наличие публичного эксплойта (ExploitDB).
                ``None`` если неизвестно.

        Returns:
            Словарь с предсказанием:

                - ``vector`` (str): полная строка CVSS v4.0 вектора;
                - ``score`` (float): итоговый балл 0,0–10,0;
                - ``severity`` (str): Critical / High / Medium / Low / None;
                - ``metrics`` (dict[str, str]): значения 12 предсказанных метрик;
                - ``confidence`` (dict[str, float]): softmax-уверенность по метрикам;
                - ``low_confidence_metrics`` (list[str]): метрики с уверенностью
                  ниже ``confidence_threshold`` — требуют ручной проверки;
                - ``input`` (dict): эхо использованных входных данных.

        Raises:
            ValueError: если ``description`` пустой или ``cwe_id`` имеет
                невалидный формат.

        Example:
            >>> predictor = VulnerabilityPredictor()
            >>> result = predictor.predict(
            ...     description="SQL injection in login form",
            ...     cwe_id="CWE-89",
            ... )
            >>> print(result["score"], result["severity"])
            6.9 Medium
        """
        d_ru, d_en = self._split_descriptions(description, description_lang, description_ru)
        cwe_name = ""  # cwe_name по идентификатору не подгружаем — лишняя зависимость
        text = self._text_processor.prepare_text(d_ru, d_en, cwe_name)
        encoding = self._tokenizer.tokenize(text)

        input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            [encoding["attention_mask"]], dtype=torch.long, device=self.device
        )
        cwe_idx = torch.tensor(
            [self._cwe_encoder.transform(cwe_id)], dtype=torch.long, device=self.device
        )
        features_np = self._features_encoder.encode(epss=epss, kev=kev, exploit=exploit)
        features = torch.from_numpy(features_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model.predict(input_ids, attention_mask, cwe_idx, features)

        metrics, confidence = self._decode_single(preds)
        score, severity, vector = self._calculator.calculate(metrics)
        low_conf = [m for m, c in confidence.items() if c < self.confidence_threshold]

        return {
            "vector": vector,
            "metrics": metrics,
            "confidence": confidence,
            "score": score,
            "severity": severity,
            "low_confidence_metrics": low_conf,
            "input": {
                "description_used": text,
                "cwe_id": cwe_id,
                "epss": epss,
                "kev": kev,
                "exploit": exploit,
            },
        }

    def predict_batch(
        self,
        items: Sequence[Mapping[str, Any]],
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """Эффективная пакетная обработка списка уязвимостей.

        Использует batch-инференс mBERT — в ~3 раза быстрее, чем
        последовательные вызовы :meth:`predict`.

        Args:
            items: Список словарей, принимающих те же ключи, что и
                аргументы :meth:`predict` (``description``, ``cwe_id``,
                ``epss``, ``kev``, ``exploit``, ``description_ru``,
                ``description_lang``).
            batch_size: Размер батча для одной forward-итерации. Дефолт 16
                достаточен для CPU; на GPU можно ставить 32–64.

        Returns:
            Список словарей того же формата, что у :meth:`predict`, в порядке
            ``items``.
        """
        if not items:
            return []

        results: list[dict[str, Any]] = []
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            results.extend(self._predict_chunk(batch))
        return results

    # ------------------------------------------------------------ internal

    def _predict_chunk(self, batch: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        texts: list[str] = []
        cwe_idxs: list[int] = []
        features_rows: list[np.ndarray] = []
        echoes: list[dict[str, Any]] = []

        for item in batch:
            description = item.get("description")
            description_lang = item.get("description_lang")
            description_ru = item.get("description_ru")
            cwe_id = item.get("cwe_id")
            epss = item.get("epss")
            kev = item.get("kev")
            exploit = item.get("exploit")

            d_ru, d_en = self._split_descriptions(description, description_lang, description_ru)
            text = self._text_processor.prepare_text(d_ru, d_en, "")
            texts.append(text)
            cwe_idxs.append(self._cwe_encoder.transform(cwe_id))
            features_rows.append(self._features_encoder.encode(epss=epss, kev=kev, exploit=exploit))
            echoes.append(
                {
                    "description_used": text,
                    "cwe_id": cwe_id,
                    "epss": epss,
                    "kev": kev,
                    "exploit": exploit,
                }
            )

        encoding = self._tokenizer.tokenize_batch(texts)
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            encoding["attention_mask"], dtype=torch.long, device=self.device
        )
        cwe_idx = torch.tensor(cwe_idxs, dtype=torch.long, device=self.device)
        features = torch.from_numpy(np.stack(features_rows)).to(self.device)

        with torch.no_grad():
            preds = self.model.predict(input_ids, attention_mask, cwe_idx, features)

        results: list[dict[str, Any]] = []
        for i in range(len(batch)):
            metrics, confidence = self._decode_row(preds, i)
            score, severity, vector = self._calculator.calculate(metrics)
            low_conf = [m for m, c in confidence.items() if c < self.confidence_threshold]
            results.append(
                {
                    "vector": vector,
                    "metrics": metrics,
                    "confidence": confidence,
                    "score": score,
                    "severity": severity,
                    "low_confidence_metrics": low_conf,
                    "input": echoes[i],
                }
            )
        return results

    def _decode_single(
        self, preds: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> tuple[dict[str, str], dict[str, float]]:
        return self._decode_row(preds, 0)

    def _decode_row(
        self,
        preds: Mapping[str, Mapping[str, torch.Tensor]],
        row: int,
    ) -> tuple[dict[str, str], dict[str, float]]:
        metrics: dict[str, str] = {}
        confidence: dict[str, float] = {}
        for metric in self.METRIC_ORDER:
            pred = preds[metric]
            label_idx = int(pred["label_idx"][row].item())
            conf = float(pred["confidence"][row].item())
            metrics[metric] = self.LABEL_MAPS[metric][label_idx]
            confidence[metric] = conf
        return metrics, confidence

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _load_metric_classes(config_path: str) -> dict[str, int]:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        stage2 = cfg.get("stage2", {})
        classes = stage2.get("metric_classes")
        if not classes:
            raise ValueError(f"В {config_path} не найдена секция stage2.metric_classes")
        # Сохраняем канонический порядок V4_METRIC_ORDER.
        return {m: int(classes[m]) for m in V4_METRIC_ORDER}

    @classmethod
    def _split_descriptions(
        cls,
        description: str | None,
        description_lang: str | None,
        description_ru: str | None,
    ) -> tuple[str | None, str | None]:
        """Возвращает кортеж ``(d_ru, d_en)`` для TextProcessor.

        Если переданы оба ``description`` и ``description_ru`` — ``description``
        трактуется как английский (поскольку русский указан отдельно).
        Иначе для одиночного ``description`` язык определяется автоматически
        либо берётся из ``description_lang``.
        """
        d_ru: str | None = description_ru if _is_nonempty_str(description_ru) else None
        d_en: str | None = None

        if _is_nonempty_str(description):
            lang = description_lang
            if d_ru is not None:
                # Если уже передан отдельный русский — основное считаем английским.
                d_en = description
            else:
                if lang is None:
                    lang = "ru" if cls._is_russian(description) else "en"
                if lang == "ru":
                    d_ru = description
                else:
                    d_en = description
        return d_ru, d_en

    @staticmethod
    def _is_russian(text: str, threshold: float = 0.3) -> bool:
        """Простая эвристика: считаем русским, если доля кириллицы > threshold."""
        if not text:
            return False
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        cyr = sum(1 for c in letters if "Ѐ" <= c <= "ӿ")
        return (cyr / len(letters)) > threshold


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


__all__ = ["VulnerabilityPredictor"]
