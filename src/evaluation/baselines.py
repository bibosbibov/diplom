"""Простые baseline-предсказатели для сравнения с обученной моделью.

Нужны разделу 2.3.6 ВКР: метрики модели имеют смысл только в сравнении с
тривиальными стратегиями. Реализованы:

    * ``majority`` — всегда предсказывать самый частый класс метрики (оценивает
      «насколько разбалансирован датасет»);
    * ``random`` — равномерно случайный класс (нижняя граница, F1 ≈ 1/K);
    * ``TfidfBaseline`` — обёртка над TF-IDF + (RandomForest / LogReg). Это
      «реальный» классический бейзлайн, показывающий, какой прирост даёт
      трансформер. Учится на 11 базовых метриках (без ``E``).

Все функции работают с уже распарсенными словарями метрик CVSS v4.0
(``{"AV": "N", ...}``), поэтому их выход можно подавать прямо в
:mod:`src.evaluation.metrics`.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data_preparation.cvss_vector_parser import (
    V4_LABEL_MAPS,
    V4_METRIC_ORDER,
    parse_v4_vector,
)

from .metrics import (
    BASE_VECTOR_METRICS,
    compute_per_metric_scores,
    compute_vector_accuracy,
)

logger = logging.getLogger(__name__)


def _vectors_to_columns(vectors: Sequence[Mapping[str, str]]) -> dict[str, list]:
    """Транспонирует список словарей метрик в ``{метрика: [значения по записям]}``."""
    columns: dict[str, list] = {metric: [] for metric in V4_METRIC_ORDER}
    for vec in vectors:
        for metric in V4_METRIC_ORDER:
            columns[metric].append(vec.get(metric))
    return columns


def predict_majority_class(train_values: Sequence, n: int) -> list:
    """Возвращает список длины ``n`` из самого частого значения ``train_values``.

    ``None`` в обучающих значениях игнорируются. Если непустых значений нет —
    возвращается список ``None``.
    """
    counts = Counter(v for v in train_values if v is not None)
    if not counts:
        return [None] * n
    most_common = counts.most_common(1)[0][0]
    return [most_common] * n


def predict_random_class(classes: Sequence, n: int, seed: int = 42) -> list:
    """Возвращает ``n`` равномерно случайных элементов из ``classes`` (seed фиксирован)."""
    rng = random.Random(seed)
    classes = list(classes)
    if not classes:
        return [None] * n
    return [rng.choice(classes) for _ in range(n)]


def majority_class_baseline(
    train_vectors: Sequence[Mapping[str, str]],
    n_predictions: int,
) -> list[dict[str, str]]:
    """Строит ``n_predictions`` одинаковых векторов из самых частых классов.

    Args:
        train_vectors: распарсенные v4-векторы обучающей выборки — из них
            берётся самый частый класс по каждой метрике.
        n_predictions: сколько предсказаний (= размер тестовой выборки).

    Returns:
        Список словарей ``{"AV": ..., ...}`` длины ``n_predictions``
        (все элементы идентичны).
    """
    columns = _vectors_to_columns(train_vectors)
    per_metric_value = {
        metric: predict_majority_class(values, 1)[0] for metric, values in columns.items()
    }
    vector = {m: v for m, v in per_metric_value.items() if v is not None}
    return [dict(vector) for _ in range(n_predictions)]


def random_baseline(n_predictions: int, seed: int = 42) -> list[dict[str, str]]:
    """Строит ``n_predictions`` векторов со случайными классами по каждой метрике.

    Каждая метрика семплируется из своего набора классов (``V4_LABEL_MAPS``);
    у разных метрик — независимые генераторы со смещённым seed для
    воспроизводимости.
    """
    per_metric = {
        metric: predict_random_class(classes, n_predictions, seed=seed + i)
        for i, (metric, classes) in enumerate(V4_LABEL_MAPS.items())
    }
    return [
        {metric: per_metric[metric][row] for metric in V4_METRIC_ORDER}
        for row in range(n_predictions)
    ]


#: Метрики, на которые TF-IDF-бейзлайны умеют предсказывать.
#:
#: ``E`` (Exploit Maturity) исключена: она почти всегда отсутствует в
#: размеченных датасетах и не входит в base vector — её предсказание
#: требует отдельных признаков (EPSS/KEV/exploit).
_BASELINE_METRICS: tuple[str, ...] = BASE_VECTOR_METRICS


def _pick_text(row: pd.Series) -> str:
    """Берёт английское описание; если оно пустое — русское.

    БДУ-записи часто без английского, NVD-записи — без русского. Бейзлайн на
    TF-IDF ngram (1,2) одинаково работает с любым языком (это просто
    мешок биграмм), поэтому конкатенация языков не требуется.
    """
    en = row.get("d_en")
    ru = row.get("d_ru")
    if isinstance(en, str) and en.strip():
        return en
    if isinstance(ru, str) and ru.strip():
        return ru
    return ""


def _row_to_text(row: pd.Series) -> str:
    """Описание + ``[SEP]`` + cwe_name — тот же формат, что и для трансформера."""
    text = _pick_text(row)
    cwe_name = row.get("cwe_name") or ""
    if isinstance(cwe_name, str) and cwe_name.strip():
        return f"{text} [SEP] {cwe_name}".strip()
    return text.strip()


def _df_to_texts(df: pd.DataFrame) -> list[str]:
    return [_row_to_text(row) for _, row in df.iterrows()]


def _df_to_label_matrix(df: pd.DataFrame) -> dict[str, list[str | None]]:
    """Парсит ``cvss_v4_vector`` каждой строки в 11 метрик ``{AV: "N", ...}``."""
    columns: dict[str, list[str | None]] = {m: [] for m in _BASELINE_METRICS}
    for vec in df.get("cvss_v4_vector", []):
        try:
            parsed = parse_v4_vector(vec) if isinstance(vec, str) else {}
        except ValueError:
            parsed = {}
        for metric in _BASELINE_METRICS:
            columns[metric].append(parsed.get(metric))
    return columns


# --------------------------------------------------------------- TF-IDF baseline


@dataclass
class TfidfBaseline:
    """TF-IDF (1-2 gram) + один классификатор на каждую из 11 базовых метрик.

    Атрибуты заполняются при обучении (:func:`train_tfidf_random_forest` или
    :func:`train_tfidf_logreg`). Сам объект сериализуем через ``pickle``, но
    обычно бейзлайны переобучаются с нуля — это быстро.
    """

    name: str = "tfidf"
    vectorizer: TfidfVectorizer | None = None
    classifiers: dict[str, Any] = field(default_factory=dict)
    fallback_class: dict[str, str] = field(default_factory=dict)

    def predict_metric_dict(self, text: str, cwe_name: str | None = None) -> dict[str, str]:
        """Предсказание 11 метрик для одной записи (E не предсказывается)."""
        if self.vectorizer is None or not self.classifiers:
            raise RuntimeError("Бейзлайн не обучен — нет vectorizer/classifiers")
        joined = f"{text} [SEP] {cwe_name}".strip() if cwe_name else (text or "")
        features = self.vectorizer.transform([joined])
        out: dict[str, str] = {}
        for metric in _BASELINE_METRICS:
            clf = self.classifiers.get(metric)
            if clf is None:
                out[metric] = self.fallback_class.get(metric, V4_LABEL_MAPS[metric][0])
                continue
            out[metric] = str(clf.predict(features)[0])
        return out

    def predict_dataframe(self, df: pd.DataFrame) -> list[dict[str, str]]:
        """Векторное предсказание по DataFrame — быстрее, чем по одной строке."""
        if self.vectorizer is None or not self.classifiers:
            raise RuntimeError("Бейзлайн не обучен")
        texts = _df_to_texts(df)
        features = self.vectorizer.transform(texts)
        per_metric: dict[str, np.ndarray] = {}
        for metric in _BASELINE_METRICS:
            clf = self.classifiers.get(metric)
            if clf is None:
                fallback = self.fallback_class.get(metric, V4_LABEL_MAPS[metric][0])
                per_metric[metric] = np.array([fallback] * features.shape[0])
            else:
                per_metric[metric] = clf.predict(features)
        return [
            {metric: str(per_metric[metric][i]) for metric in _BASELINE_METRICS}
            for i in range(features.shape[0])
        ]


def _train_tfidf_baseline(
    train_df: pd.DataFrame,
    classifier_factory,
    name: str,
) -> TfidfBaseline:
    """Общая «шапка» обучения TF-IDF-бейзлайна с произвольным classifier_factory.

    Берутся только строки с распарсенным cvss_v4_vector. Для каждой из 11
    базовых метрик обучается отдельный экземпляр classifier_factory().
    """
    df = train_df[train_df["cvss_v4_vector"].notna()].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("Нет записей с cvss_v4_vector — нечего учить")

    texts = _df_to_texts(df)
    labels = _df_to_label_matrix(df)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    features = vectorizer.fit_transform(texts)

    classifiers: dict[str, Any] = {}
    fallback_class: dict[str, str] = {}
    for metric in _BASELINE_METRICS:
        y = labels[metric]
        mask = [v is not None for v in y]
        if not any(mask):
            logger.warning("Нет меток для %s — голова пропущена", metric)
            continue
        X_m = features[mask]
        y_m = [v for v, keep in zip(y, mask) if keep]
        counts = Counter(y_m)
        most_common, _ = counts.most_common(1)[0]
        fallback_class[metric] = most_common
        if len(counts) < 2:
            # один класс на всю обучающую — sklearn падает; запоминаем константу.
            logger.info(
                "Метрика %s имеет один класс %r — fallback без классификатора",
                metric,
                most_common,
            )
            continue
        clf = classifier_factory()
        clf.fit(X_m, y_m)
        classifiers[metric] = clf

    return TfidfBaseline(
        name=name,
        vectorizer=vectorizer,
        classifiers=classifiers,
        fallback_class=fallback_class,
    )


def train_tfidf_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
) -> TfidfBaseline:
    """Обучает TF-IDF + 11 RandomForestClassifier'ов (по одному на каждую метрику).

    Args:
        train_df: train.parquet с колонками ``d_ru``, ``d_en``, ``cwe_name``,
            ``cvss_v4_vector``.
        val_df: оставлен для совместимости API — сам по себе не используется,
            оценивайте через :func:`evaluate_baseline`.
        n_estimators: число деревьев в каждом лесе.
        random_state: для воспроизводимости.
        n_jobs: параллелизм sklearn.

    Returns:
        Обученный :class:`TfidfBaseline`.
    """
    del val_df  # API-параметр для симметрии; не используется
    factory = lambda: RandomForestClassifier(  # noqa: E731
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return _train_tfidf_baseline(train_df, factory, name="tfidf_rf")


def train_tfidf_logreg(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    max_iter: int = 1000,
    random_state: int = 42,
) -> TfidfBaseline:
    """Обучает TF-IDF + 11 LogisticRegression (по одному на каждую метрику)."""
    del val_df
    factory = lambda: LogisticRegression(  # noqa: E731
        max_iter=max_iter,
        random_state=random_state,
    )
    return _train_tfidf_baseline(train_df, factory, name="tfidf_logreg")


def train_tfidf_random_forest_v4_only(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
) -> TfidfBaseline:
    """То же, что :func:`train_tfidf_random_forest`, но явно фильтрует на v4-only.

    Семантически эквивалентно — :func:`train_tfidf_random_forest` уже отбирает
    только записи с ``cvss_v4_vector``. Существует отдельной функцией ради
    нарратива ВКР («честный бейзлайн без двухэтапной стратегии»).
    """
    v4_only = train_df[train_df["cvss_v4_vector"].notna()].reset_index(drop=True)
    logger.info("v4-only обучающая выборка: %d записей", len(v4_only))
    return train_tfidf_random_forest(
        v4_only,
        val_df=val_df,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def evaluate_baseline(
    baseline_model: TfidfBaseline,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Оценивает TF-IDF-бейзлайн на тестовом наборе.

    Возвращает структуру, частично совместимую с :meth:`Evaluator.evaluate`
    (только per-metric F1 и vector_accuracy — балл/severity отсутствуют, так
    как E не предсказывается).

    Returns:
        ``{"per_metric": {...}, "aggregated": {"macro_f1", "vector_accuracy",
        "samples_evaluated"}}``.
    """
    df = test_df[test_df["cvss_v4_vector"].notna()].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("В test_df нет записей с cvss_v4_vector")

    true_labels = _df_to_label_matrix(df)
    true_vectors: list[dict[str, str]] = []
    for i in range(len(df)):
        true_vectors.append(
            {
                metric: true_labels[metric][i]
                for metric in _BASELINE_METRICS
                if true_labels[metric][i] is not None
            }
        )

    pred_vectors = baseline_model.predict_dataframe(df)

    per_metric: dict[str, dict[str, Any]] = {}
    f1_macros: list[float] = []
    for metric in _BASELINE_METRICS:
        labels = V4_LABEL_MAPS[metric]
        pairs = [
            (true_vectors[i].get(metric), pred_vectors[i][metric])
            for i in range(len(df))
            if true_vectors[i].get(metric) is not None
        ]
        y_true_m = [t for t, _ in pairs]
        y_pred_m = [p for _, p in pairs]
        scores = compute_per_metric_scores(y_true_m, y_pred_m, labels=labels)
        per_metric[metric] = {**scores, "support": len(pairs)}
        if pairs:
            f1_macros.append(scores["f1_macro"])

    aggregated = {
        "macro_f1": float(np.mean(f1_macros)) if f1_macros else 0.0,
        "vector_accuracy": compute_vector_accuracy(true_vectors, pred_vectors),
        "samples_evaluated": len(df),
    }
    return {"per_metric": per_metric, "aggregated": aggregated}


__all__ = [
    "predict_majority_class",
    "predict_random_class",
    "majority_class_baseline",
    "random_baseline",
    "TfidfBaseline",
    "train_tfidf_random_forest",
    "train_tfidf_logreg",
    "train_tfidf_random_forest_v4_only",
    "evaluate_baseline",
]
