# Техническое описание системы автоматической оценки критичности уязвимостей CVSS v4.0

Документ описывает архитектуру, математический аппарат, ключевые функции, контракты данных и метрики качества системы, реализованной в репозитории `diplom/`. Объект работы — магистерская ВКР «Разработка системы автоматической оценки критичности уязвимостей программного обеспечения на основе CVSS v4.0 с применением трансформерной модели mBERT».

---

## 1. Архитектурные решения

### 1.1. Общая структура проекта

Система реализована в виде монорепозитория с послойной декомпозицией. Каждому слою соответствует свой пакет в `src/` и набор unit-тестов в `tests/`. Слои сцеплены через явные интерфейсы (DTO, parquet-файлы, JSON-словари), что обеспечивает независимое тестирование и взаимозаменяемость реализаций.

| Слой | Пакет | Ответственность |
|---|---|---|
| Сбор данных | [src/data_collection/](src/data_collection/) | Скачивание и парсинг БДУ ФСТЭК, NVD, EPSS, CISA KEV, ExploitDB, CWE MITRE |
| Подготовка | [src/data_preparation/](src/data_preparation/) | Токенизация mBERT, кодирование CWE, нормализация числовых признаков, разбор CVSS-векторов |
| Модель | [src/model/](src/model/) | mBERT + FeaturesMLP + FusionLayer + 12 classification heads |
| Обучение | [src/training/](src/training/) | Двухэтапный pipeline (v3.1 → v4.0), MultiTaskLoss, early stopping |
| CVSS-калькулятор | [src/cvss_calculator/](src/cvss_calculator/) | Собственная реализация алгоритма FIRST CVSS v4.0 |
| Оценка | [src/evaluation/](src/evaluation/) | per-metric / vector / score / severity метрики, baselines, confusion matrices |
| Инференс | [src/inference/](src/inference/) | End-to-end predictor + CLI |
| API | [src/api/](src/api/) | FastAPI-сервис с UI |

### 1.2. Архитектура нейросетевой модели

Файл: [src/model/cvss_model.py](src/model/cvss_model.py).

```
input_ids, attention_mask ──▶ mBERT (bert-base-multilingual-cased)
                                  │
                                  ▼
                          h_text = H[:, 0, :]  (768)

cwe_idx ──▶ Embedding(num_cwe, 64) ──┐
features  ──────────────────────────┴── concat (67)
                                  │
                          Linear(67→128) → ReLU
                          Linear(128→64) → ReLU
                                  │
                                  ▼
                              h_feat (64)

  Fusion:  concat(h_text, h_feat) → Linear(832→512) → ReLU → Dropout(0.1)
                                  │
                                  ▼
                              h_fused (512)

  12 голов: Linear(512 → num_classes_i) для каждой метрики CVSS v4.0
```

**Ключевые элементы:**

- **Backbone** — `bert-base-multilingual-cased` (Hugging Face). Размер `hidden_size = 768`. Берётся представление `[CLS]`-токена ([src/model/cvss_model.py:127-142](src/model/cvss_model.py#L127-L142)).
- **FeaturesMLP** ([src/model/features_mlp.py](src/model/features_mlp.py)) — двухслойный MLP `67 → 128 → 64` с ReLU. CWE-id кодируется через `nn.Embedding(num_cwe, 64)` с `padding_idx=0`; числовые признаки `[epss, kev, exploit]` подаются как есть, маркер отсутствия `-1` модель учится интерпретировать самостоятельно.
- **FusionLayer** ([src/model/fusion_layer.py](src/model/fusion_layer.py)) — конкатенация и сжатие `832 → 512` с Dropout 0.1.
- **ClassificationHeads** ([src/model/classification_heads.py](src/model/classification_heads.py)) — `nn.ModuleDict` из 12 линейных голов. Канонический порядок и число классов фиксированы в `DEFAULT_METRIC_CLASSES` (`AV=4, AC=2, AT=2, PR=3, UI=3, VC/VI/VA/SC/SI/SA=3, E=3`).

Общее количество выходов: `4+2+2+3+3+3+3+3+3+3+3+3 = 35` логитов на запись.

### 1.3. Двухэтапное обучение

Файл: [src/training/trainer.py](src/training/trainer.py).

| Параметр | Stage 1 (CVSS v3.1) | Stage 2 (CVSS v4.0) |
|---|---|---|
| Активных голов | 8 (общих с v3.1) | 12 (полный вектор v4.0) |
| Размер train | ≈122 750 | ≈4 715 |
| Learning rate | 2e-5 | 1e-5 |
| Batch size | 32 | 16 |
| Max epochs | 10 | 20 |
| Dropout / weight decay | 0.1 / 0.01 | 0.1 / 0.01 |
| Early-stopping patience | 3 | 3 |

Между этапами выполняется trasfer learning:

1. `_load_stage1_weights_if_present` ([src/training/trainer.py:383-407](src/training/trainer.py#L383-L407)) — загрузка `best_stage1.pt` со `strict=False` и фильтрацией по формам тензоров.
2. `_reinit_heads_for_stage2` ([src/training/trainer.py:333-379](src/training/trainer.py#L333-L379)) — головы `AT, SC, SI, SA` создаются с нуля (в v3.1 их не было), голова `E` пересоздаётся (в v3.1 было 5 классов, в v4.0 — 3). Остальные головы наследуют веса с этапа 1. Инициализация Xavier для `weight`, нули для `bias`.

### 1.4. Сервисный контур

- **Predictor** ([src/inference/predictor.py](src/inference/predictor.py)) — лениво грузит модель, токенизатор, словарь CWE, нормализатор признаков и калькулятор. Возвращает `{vector, score, severity, metrics, low_confidence_metrics, inference_time_ms}`. Порог уверенности конфигурируем (по умолчанию 0.7).
- **CLI** ([src/inference/cli.py](src/inference/cli.py)) — Click-обёртка.
- **REST API** ([src/api/main.py](src/api/main.py)) — FastAPI с эндпоинтами `/predict`, `/predict/batch`, `/health`, `/model/info`; статика UI в `src/api/static/`. Pydantic-валидация в [src/api/schemas.py](src/api/schemas.py).

---

## 2. Математические алгоритмы

### 2.1. Функция потерь

Файл: [src/training/loss.py](src/training/loss.py).

Для активных метрик `A ⊆ {AV, AC, ..., E}` (`|A|=8` на этапе 1, `|A|=12` на этапе 2):

```
L(θ) = Σ_{m ∈ A}  CE(logits_m(x; θ), y_m;  w_m, ignore_index=-100)
```

- `CE` — стандартная категориальная кросс-энтропия PyTorch с поддержкой `ignore_index` и весов классов.
- `w_m ∈ ℝ^{C_m}` — веса классов по стратегии **balanced** (`sklearn.utils.class_weight.compute_class_weight`), отдельно для каждой метрики (см. `compute_class_weights`, [src/training/loss.py:105-139](src/training/loss.py#L105-L139)). Для классов, отсутствующих в трейне, вес = 1 (нейтральный).
- Метки `−100` (стандартный ignore-индекс PyTorch) исключаются из расчёта; если в батче *все* метки метрики помечены ignore, эта метрика пропускается без NaN ([src/training/loss.py:83-93](src/training/loss.py#L83-L93)).

### 2.2. Оптимизатор и scheduler

- **AdamW** с раздельными группами параметров: bias и LayerNorm — без weight decay; остальные — `weight_decay=0.01` ([src/training/trainer.py:104-128](src/training/trainer.py#L104-L128)).
- **Linear warmup + linear decay** ([src/training/trainer.py:130-142](src/training/trainer.py#L130-L142)):
  - `total_steps = epochs · len(train_loader)`
  - `num_warmup = ⌊warmup_ratio · total_steps⌋` (по умолчанию 10%).
- Gradient clipping по L2-норме `‖∇‖₂ ≤ 1.0` ([src/training/trainer.py:199, 211](src/training/trainer.py#L199)).
- Опциональный AMP (`torch.amp.GradScaler` + `autocast("cuda")`) включается только при наличии CUDA.

### 2.3. Алгоритм расчёта итогового балла CVSS v4.0

Файлы: [src/cvss_calculator/core.py](src/cvss_calculator/core.py), [src/cvss_calculator/calculator.py](src/cvss_calculator/calculator.py).

Реализация **собственная** по спецификации FIRST (запрет на внешние калькуляторы — раздел 1.3.6 ВКР). Четыре этапа:

1. **MacroVector** ([src/cvss_calculator/core.py:298-363](src/cvss_calculator/core.py#L298-L363)) — 12 метрик группируются в 6 групп эквивалентности EQ1–EQ6 (например, `EQ1=0`, если `AV=N ∧ PR=N ∧ UI=N`; `EQ3=0`, если `VC=H ∧ VI=H`; и т.д.). Результат — строка из 6 цифр.
2. **Базовый балл** — `value = CVSS_LOOKUP_GLOBAL[macroVector]` (lookup-таблица из 264 значений, [src/cvss_calculator/constants.py](src/cvss_calculator/constants.py)).
3. **Интерполяция (Severity Distances)** ([src/cvss_calculator/core.py:461-624](src/cvss_calculator/core.py#L461-L624)) — для каждой группы EQ_i считается:

   ```
   current_distance_i = Σ_{m ∈ EQ_i} (level(m) − level(max_vector_m))
   available_distance_i = value − score(EQ_i_next_lower_macro)
   percent_i = current_distance_i / max_severity_i
   normalized_i = available_distance_i · percent_i
   ```

   Поправка:

   ```
   mean_distance = (Σ normalized_i) / n_existing_lower
   value ← max(0, min(10, value − mean_distance))
   ```

4. **Округление**: `final_rounding(x) = Decimal(x + ε).quantize(0.1, ROUND_HALF_UP)` ([src/cvss_calculator/core.py:93-104](src/cvss_calculator/core.py#L93-L104)) — half-away-from-zero, что отличается от дефолтного банковского округления Python.

**Severity rating** ([src/cvss_calculator/calculator.py:103-123](src/cvss_calculator/calculator.py#L103-L123)):

| Score | Severity |
|---|---|
| 0.0 | None |
| 0.1–3.9 | Low |
| 4.0–6.9 | Medium |
| 7.0–8.9 | High |
| 9.0–10.0 | Critical |

### 2.4. Метрики оценки качества

Файл: [src/evaluation/metrics.py](src/evaluation/metrics.py).

- **Per-metric F1 (macro)** — `f1_score(y_true, y_pred, average="macro", zero_division=0)`, считается отдельно для каждой из 12 метрик.
- **Vector Accuracy** — доля примеров, у которых *все 11 обязательных* метрик предсказаны верно (E не входит, т.к. часто не определена).
- **Partial Accuracy** — среднее число правильно предсказанных метрик из 11.
- **Score MAE / RMSE** — ошибка по итоговому числовому баллу 0–10:

  ```
  MAE  = (1/N) Σ |ŷ_score − y_score|
  RMSE = sqrt( (1/N) Σ (ŷ_score − y_score)² )
  ```

- **Severity Accuracy** — точное совпадение уровня (None/Low/Medium/High/Critical).
- **Severity Within ±1** — допускается отклонение на одну ступень.

---

## 3. Ключевые функции и контракты

### 3.1. Сбор данных

| Класс / функция | Назначение | Файл |
|---|---|---|
| `BDUCollector` | парсинг `vullist.xml`/`.xlsx` ФСТЭК | [src/data_collection/bdu_collector.py](src/data_collection/bdu_collector.py) |
| `NVDCollector` | пагинированный NVD API 2.0 с tenacity-ретраями, rate-limit 5/30 сек (или 50/30 при ключе) | [src/data_collection/nvd_collector.py](src/data_collection/nvd_collector.py) |
| `EPSSCollector` | пакетный запрос `api.first.org/data/v1/epss` | [src/data_collection/epss_collector.py](src/data_collection/epss_collector.py) |
| `KEVCollector`, `ExploitDBCollector`, `CWENames` | загрузка эксплуатационных каталогов | `src/data_collection/` |
| `DataIntegrator.collect_dataset` | оркестратор: на `ThreadPoolExecutor(max_workers=5)` собирает запись `{id, d_ru, d_en, cwe_id, cwe_name, epss, kev, exploit, cvss_vector}` | [src/data_collection/data_integrator.py:60-80](src/data_collection/data_integrator.py#L60-L80) |
| `split_data` | stratified split по уникальному CVE-id (защита от data leakage между stage1 и stage2) | [src/data_collection/split_data.py](src/data_collection/split_data.py) |

### 3.2. Подготовка данных

| Компонент | Контракт I/O |
|---|---|
| `TextProcessor` | `(d_ru, d_en) → text` — берёт русское описание, если есть, иначе английское |
| `CVSSTokenizer` | `(text, cwe_name) → {input_ids[L], attention_mask[L]}`, `max_length=512`, `[SEP]` между описанием и именем CWE |
| `CWEEncoder` | `cwe_id → int`, словарь сохраняется в `data/processed/cwe_vocab.json`; индексы 0=`<PAD>`, 1=`<UNK>` |
| `FeaturesEncoder` | `(epss, kev, exploit) → FloatTensor[3]` с маркером `-1` для пропусков |
| `parse_v4_vector` / `parse_v3_vector` | CVSS-строка → `{metric: class_idx}` (label maps в [src/data_preparation/cvss_vector_parser.py:18-58](src/data_preparation/cvss_vector_parser.py#L18-L58)) |
| `CVSSDataset` | `torch.utils.data.Dataset`, возвращает батчи в формате, ожидаемом Trainer |

### 3.3. Модель

| Метод | Сигнатура | Что возвращает |
|---|---|---|
| `CVSSModel.forward` | `(input_ids[B,L], attention_mask[B,L], cwe_idx[B], features[B,3])` | `dict[str, FloatTensor[B, C_m]]` — логиты 12 голов |
| `CVSSModel.encode_text` | `(input_ids, attention_mask)` | `FloatTensor[B, 768]` — `[CLS]`-вектор |
| `CVSSModel.predict` | те же входы | `dict[str, {label_idx, confidence, probs}]` |

### 3.4. Обучение

- `Trainer.train_stage1(train_loader, val_loader, train_df)` → `history` (списки `train_loss`, `val_loss`, `macro_f1`, `per_metric` по эпохам, лучший чекпоинт `best_stage1.pt`).
- `Trainer.train_stage2(...)` → аналогично, с предварительным transfer learning и переинициализацией голов.
- `MultiTaskLoss.forward(logits, labels) → (total_loss, per_metric_loss)`.
- `EarlyStopping.step(val_score, model) → bool` ([src/training/early_stopping.py](src/training/early_stopping.py)) — мониторит macro-F1 по валидации, сохраняет лучшие веса.
- `save_checkpoint` / `load_checkpoint` — полные state: model + optimizer + scheduler + global_step + config.

### 3.5. Калькулятор и инференс

- `CVSSCalculator.calculate(metrics: dict[str, str]) → (score, severity, vector)`.
- `VulnerabilityPredictor.predict_one(text, cwe_id, epss, kev, exploit) → PredictionResult`.
- `VulnerabilityPredictor.predict_batch(items) → list[PredictionResult]`.

### 3.6. Оценка качества

- `Evaluator.evaluate(test_loader) → {macro_f1, vector_accuracy, partial_accuracy, score_mae, score_rmse, severity_accuracy, severity_within_one, per_metric}` ([src/evaluation/evaluator.py](src/evaluation/evaluator.py)).
- `train_tfidf_random_forest`, `train_tfidf_logreg`, `majority_class_baseline`, `random_baseline` — baseline-предсказатели для сравнения ([src/evaluation/baselines.py](src/evaluation/baselines.py)).
- `plot_all_per_metric_matrices` — confusion matrices для 12 голов и для severity ([src/evaluation/confusion_matrices.py](src/evaluation/confusion_matrices.py)).
- `plot_training_curves` — кривые train/val loss и F1 из TensorBoard-логов ([src/evaluation/training_curves.py](src/evaluation/training_curves.py)).

### 3.7. API

| Эндпоинт | Метод | Тело | Ответ |
|---|---|---|---|
| `/predict` | POST | `PredictionRequest{description, cwe_id, epss?, kev?, exploit?}` | `PredictionResponse{vector, score, severity, metrics[12], low_confidence_metrics, inference_time_ms}` |
| `/predict/batch` | POST | `BatchPredictionRequest{items[]}` | `list[PredictionResponse]` |
| `/health` | GET | — | `HealthResponse{status, model_loaded}` |
| `/model/info` | GET | — | `ModelInfoResponse{model_name, num_cwe, metric_classes, device}` |
| `/` | GET | — | статика — HTML-форма |

---

## 4. Входные и выходные данные

### 4.1. Объём датасета

| Показатель | train | val | test |
|---|---|---|---|
| Размер | 122 913 | 26 385 | 26 348 |
| CVSS v3.1 | 122 750 (99.9%) | 26 341 (99.8%) | 26 317 (99.9%) |
| CVSS v4.0 | 4 715 (3.8%) | 1 041 (3.9%) | **972 (3.7%)** |
| Описания d_ru | 46 918 (38.2%) | 10 038 (38.0%) | 10 150 (38.5%) |
| Описания d_en | 109 560 (89.1%) | 23 545 (89.2%) | 23 479 (89.1%) |
| EPSS | 99.7% | 99.8% | 99.8% |
| CISA KEV | 0.8% | 0.8% | 0.8% |
| ExploitDB | 2.0% | 2.1% | 1.9% |

### 4.2. Контракт батча

```python
batch = {
    "input_ids":      LongTensor[B, 512],
    "attention_mask": LongTensor[B, 512],
    "cwe_idx":        LongTensor[B],
    "features":       FloatTensor[B, 3],   # [epss, kev, exploit]; -1 для пропусков
    "labels":         dict[str, LongTensor[B]],  # ключи — имена метрик; -100 = ignore
}
```

### 4.3. Выход инференса

```jsonc
{
  "vector": "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N/E:P",
  "score": 9.3,
  "severity": "Critical",
  "metrics": {
    "AV": {"value": "N", "label": "Network", "confidence": 0.97},
    // … 11 остальных метрик
  },
  "low_confidence_metrics": ["SC", "SI"],
  "inference_time_ms": 84.2
}
```

---

## 5. Метрики производительности

### 5.1. Итоговые показатели качества (test set, 972 записи CVSS v4.0)

| Показатель | Значение |
|---|---|
| **Macro-F1 (12 метрик)** | **0.7641** |
| **Vector Accuracy (11 метрик)** | **0.4763** |
| Среднее число правильных метрик | 9.63 / 11 (87.6%) |
| **MAE по баллу CVSS** | **1.01** (10.1% шкалы) |
| RMSE по баллу CVSS | 1.86 |
| **Severity Accuracy** | **0.7130** |
| Severity Within ±1 | 0.9486 |

### 5.2. Per-metric качество (test)

| Метрика | F1 (macro) | Accuracy | Support |
|---|---:|---:|---:|
| AV | 0.7626 | 0.9290 | 972 |
| AC | 0.7842 | 0.9239 | 972 |
| AT | 0.7651 | 0.9012 | 972 |
| PR | 0.7314 | 0.7860 | 972 |
| UI | 0.7131 | 0.8889 | 972 |
| VC | 0.8244 | 0.8292 | 972 |
| VI | 0.8471 | 0.8508 | 972 |
| VA | 0.8351 | 0.8405 | 972 |
| SC | 0.6458 | 0.8827 | 972 |
| SI | 0.6928 | 0.9023 | 972 |
| SA | 0.6650 | 0.8971 | 972 |
| **E** | **0.9022** | **0.9566** | 553 |

Лидеры по F1 — Exploit Maturity (0.90) и импакт-метрики Vulnerable System (VC/VI/VA, 0.82–0.85). Слабее всех — группа Subsequent System (SC/SI/SA, 0.65–0.69) из-за разреженного сигнала о вторичном воздействии.

### 5.3. Стабильность val vs test

Максимальное отклонение F1 между лучшей эпохой валидации (epoch 19) и тестом — `|Δ| ≤ 0.048` по всем 12 метрикам, среднее `|Δ| = 0.023` ([reports/final_results.md](reports/final_results.md#L37-L52)), что свидетельствует об отсутствии переобучения на val.

### 5.4. Сравнение с baseline (Macro-F1 / Vector Accuracy на test v4.0)

| Модель | Macro-F1 | Vector Accuracy | Время обучения, с |
|---|---:|---:|---:|
| TF-IDF + RandomForest | 0.6905 | 0.4393 | 6.4 |
| TF-IDF + LogReg | 0.6528 | 0.3930 | 2.1 |
| **mBERT (Stage 1 + Stage 2 + DAPT)** | **0.7641** | **0.4763** | — |

Финальная модель опережает сильнейший baseline (RandomForest) и по Macro-F1 (`+0.0736`, ≈ +7.4 п.п.), и по Vector Accuracy (`+0.037`, ≈ +3.7 п.п.). Прирост относительно майской модели (0.7090) разложен в `reports/dapt_experiment/chapter3_summary.md`: +0.034 от очистки корпуса и +0.021 от DAPT.

### 5.5. Прочие характеристики

- **Backbone**: ≈177 M параметров (mBERT 12L/768H/12A).
- **Дополнительная голова**: FeaturesMLP + Fusion + 12 heads ≈ 470 K параметров.
- **Max sequence length**: 512 токенов.
- **Inference latency**: ≈80–120 мс на GPU, ≈700–900 мс на CPU (одна запись, batch=1, mBERT-base).
- **Воспроизводимость**: `seed=42` через `torch.manual_seed`, `numpy.random.seed`, `random.seed`, `transformers.set_seed` (см. [src/training/utils.py](src/training/utils.py)).
- **TensorBoard**: пишутся batch-loss, learning rate, GPU memory, val-F1/accuracy по метрикам ([src/training/trainer.py:221-240](src/training/trainer.py#L221-L240)).

---

## 6. Сводка соответствия модулей разделам ВКР

| Модуль кода | Раздел отчёта |
|---|---|
| [src/data_collection/](src/data_collection/) | 2.3.1 Алгоритм сбора и интеграции данных |
| [src/data_preparation/](src/data_preparation/) | 2.3.2 Алгоритм подготовки данных |
| [src/model/](src/model/) | 2.2.5 Модель классификации |
| [src/training/](src/training/) | 2.2.4, 2.3.3 Модель и алгоритм обучения |
| [src/cvss_calculator/](src/cvss_calculator/) | 2.2.6, 2.3.5 Расчёт балла |
| [src/evaluation/](src/evaluation/) | 2.2.7, 2.3.6 Оценка качества |
| [src/inference/](src/inference/) | Глава 3 — реализация |
| [src/api/](src/api/) | Глава 3 — программное средство |

---

## 7. Расширение системы: режим CVSS v3.1

Базовая система оценивает уязвимости по CVSS v4.0. Однако подавляющее большинство исторических записей в БДУ ФСТЭК и NVD размечено в **CVSS v3.1**, и на практике пользователю часто нужна оценка именно в этой версии. В рамках расширения реализован полноценный режим CVSS v3.1, переиспользующий уже обученный backbone, и проведена сквозная оценка его качества, сопоставимая с v4.0.

### 7.1. Идея: transfer learning поверх stage 1

Обучение модели двухэтапное: stage 1 — предобучение на корпусе CVSS v3.1 (8 голов `AV, AC, PR, UI, VC, VI, VA, E`), stage 2 — дообучение на CVSS v4.0 (12 голов). Базовый вектор CVSS v3.1 состоит из 8 метрик: `AV, AC, PR, UI, S, C, I, A`. Семь из них напрямую соответствуют головам stage 1 (`VC/VI/VA` — это импакты `C/I/A`), и недостаёт только метрики **Scope (S)**, которой нет ни в stage 1, ни в v4.0 (в v4.0 Scope упразднён).

Поэтому вместо обучения отдельной v3.1-модели применён классический transfer learning: **backbone stage 1 замораживается**, а для Scope обучается единственная линейная голова `Linear(512, 2)` поверх выхода FusionLayer (`h_fused`). mBERT, FeaturesMLP, Fusion и 8 v3-голов при этом не меняются.

```
                 (замороженный stage 1 backbone)
описание+CWE+features ─▶ mBERT+MLP+Fusion ─▶ h_fused (512)
                                               ├─▶ 8 v3-голов ─▶ AV,AC,PR,UI,C(=VC),I(=VI),A(=VA)
                                               └─▶ Linear(512,2) ─▶ S (Scope)   ← обучается отдельно
```

Файлы: [src/training/train_scope_head.py](src/training/train_scope_head.py) (обучение), [src/inference/predictor_v31.py](src/inference/predictor_v31.py) (инференс).

### 7.2. Обучение Scope-головы

Алгоритм (раздел «классический transfer learning»):

1. Один проход замороженного backbone по корпусу, кэширование представлений `h_fused` (`FloatTensor[N, 512]`) и меток Scope.
2. Обучение `Linear(512, 2)` поверх кэша: AdamW, `lr=1e-3`, 10 эпох, batch 256, ранний останов по val-accuracy (patience 3). Сам линейный слой обучается за доли секунды на эпоху — всё время занимает однократное кэширование.

**Оптимизация кэширования.** Наивная реализация (`padding="max_length"=512`, fp32, `num_workers=0`) давала ≈2.3 с/батч на T4 — ≈88 мин на полный корпус. Внесены три правки в `cache_fused_features`:

- **динамический padding** (`_trim_pad_collate`) — обрезка pad-хвоста до батч-максимума реальной длины; результат идентичен (pad-позиции занулены attention-маской), но forward не тратится на пустые токены;
- **fp16 autocast** на CUDA — задействует tensor cores T4;
- **`num_workers`** + `pin_memory` — токенизация идёт в фоне, не блокирует GPU.

Итог: кэширование ускорено до ≈10–20 мин на T4. Метрики качества Scope-головы (валидация, ≈26 тыс. записей): **accuracy 0.913, F1-macro 0.850**.

### 7.3. Калькулятор базового балла CVSS v3.1

Файл: [src/cvss_calculator/cvss31.py](src/cvss_calculator/cvss31.py). В отличие от v4.0 (форк библиотеки FIRST/Red Hat) для v3.1 достаточно базового балла, поэтому реализована компактная самостоятельная формула из раздела 7.1 спецификации [FIRST CVSS v3.1](https://www.first.org/cvss/v3.1/specification-document):

```
ISS          = 1 − (1−C)·(1−I)·(1−A)
Impact       = 6.42 · ISS                                       , если Scope = Unchanged
             = 7.52·(ISS−0.029) − 3.25·(ISS−0.02)^15            , если Scope = Changed
Exploitability = 8.22 · AV · AC · PR · UI    (вес PR зависит от Scope)
BaseScore    = 0                                                , если Impact ≤ 0
             = Roundup(min(Impact+Exploitability, 10))          , если Scope = Unchanged
             = Roundup(min(1.08·(Impact+Exploitability), 10))   , если Scope = Changed
```

`Roundup` — округление вверх до десятых на целочисленном масштабе 100000 (как в спецификации, чтобы исключить ошибки представления float). Severity — по шкале Qualitative Severity Rating Scale v3.1. Интерфейс совместим с v4-калькулятором: `calculate(metrics) → (score, severity, vector)`. Корректность подтверждена эталонными векторами FIRST ([tests/test_cvss31.py](tests/test_cvss31.py)): 9.8, 10.0, 7.5, 4.3, 0.0.

### 7.4. Контроль происхождения backbone (provenance)

Scope-голова валидна только с тем backbone, поверх которого она обучалась: подмена stage 1-чекпойнта изменит `h_fused`, и предсказание Scope станет случайным. Совпадение **форм** весов это не ловит (другой чекпойнт той же архитектуры грузится без ошибки).

Решение ([src/model/fingerprint.py](src/model/fingerprint.py)): при обучении в артефакт `scope_head_v3.pt` записывается **отпечаток** весов backbone — SHA-256 по байтам модулей, формирующих `h_fused` (`transformer + features_mlp + fusion`). Предиктор при загрузке пересчитывает отпечаток и сверяет; несовпадение → явный `ValueError` вместо тихого мусора. Отпечаток детерминирован и не зависит от устройства (хешируются биты весов, а не их статистики).

### 7.5. Инференс, API и веб-интерфейс

- **Предиктор** [VulnerabilityPredictorV31](src/inference/predictor_v31.py): грузит замороженный stage 1 (8 голов) + Scope-голову, делает forward → `h_fused` → 7 базовых метрик + Scope, переименовывает `VC/VI/VA → C/I/A`, отбрасывает `E` (временна́я, в базовый балл не входит) и считает балл калькулятором 3.1. Логиты `VC/VI/VA` слайсятся до 3 классов (`H/L/N`), нетренированный «X» обрезается. Реализованы `predict` и `predict_batch`.
- **API** ([src/api/schemas.py](src/api/schemas.py), [ml_service.py](src/api/ml_service.py), [main.py](src/api/main.py)): в `/predict` добавлено поле `cvss_version` (`4.0` по умолчанию / `3.1`); ответ содержит эхо версии. v3.1-предиктор загружается **лениво** при первом v3.1-запросе. Пакетный режим группирует запросы по версии.
- **Фронтенд** ([src/api/static/index.html](src/api/static/index.html)): переключатель «CVSS v4.0 / v3.1», version-aware порядок и подписи метрик, передача версии в запрос.

### 7.6. Методология сквозной оценки v3.1

Файл: [src/evaluation/evaluate_v31.py](src/evaluation/evaluate_v31.py) (`V31Evaluator`). До этого для v3.1 считался только per-head F1; итоговый балл/severity не оценивались, т.к. калькулятор был только для v4.0. `V31Evaluator` повторяет методологию v4-`Evaluator`:

- оценивается **развёрнутый пайплайн** (через предиктор, текст без `cwe_name`) — измеряется то, что реально отдаёт API;
- **истинный** балл считается из эталонного вектора тем же калькулятором 3.1 (как и в v4-эвалюаторе) — измеряется ошибка предсказания метрик, пропущенная через официальную формулу;
- метрики идентичны v4: macro-F1, vector accuracy, среднее число верных метрик, score MAE/RMSE, severity accuracy, severity ±1.

Запуск: `python -m src.evaluation.evaluate_v31`. Прогон на полном тесте — ≈16 мин на T4.

### 7.7. Результаты и сравнение v3.1 vs v4.0

Обе модели оценены по одинаковым метрикам и единой методологии (балл — из вектора через соответствующий калькулятор). v4.0 — `final_model.pt` ([reports/test_evaluation.json](reports/test_evaluation.json)); v3.1 — `dapt_mbert/best_stage1.pt` + Scope-голова ([reports/v31_results.json](reports/v31_results.json)).

Обе модели оценены в развёрнутом режиме с подстановкой `cwe_name`.

| Метрика | CVSS v3.1 | CVSS v4.0 |
|---|---:|---:|
| Размер test set, записей | 26 317 | 972 |
| Базовых метрик в векторе | 8 | 11 (+E) |
| Macro-F1 | **0.814** | 0.764 |
| Vector accuracy | **0.523** | 0.476 |
| Метрик верно в среднем | **7.07 / 8 (88.4 %)** | 9.63 / 12 (80.3 %) |
| Severity accuracy | **0.738** | 0.713 |
| Severity ±1 уровень | **0.970** | 0.949 |
| Score MAE | **0.743** | 1.014 |
| Score RMSE | **1.318** | 1.857 |

Per-metric качество v3.1 (через предиктор с `cwe_name`; тест 26 317 записей, [reports/v31_results.md](reports/v31_results.md)):

| Метрика | Полное название | F1 (macro) | Accuracy |
|---|---|---:|---:|
| AV | Attack Vector | 0.7817 | 0.9352 |
| AC | Attack Complexity | 0.7253 | 0.9197 |
| PR | Privileges Required | 0.7609 | 0.8132 |
| UI | User Interaction | 0.9063 | 0.9173 |
| S | Scope | 0.8469 | 0.9120 |
| C | Confidentiality Impact | 0.8388 | 0.8510 |
| I | Integrity Impact | 0.8502 | 0.8550 |
| A | Availability Impact | 0.7994 | 0.8638 |

### 7.8. Интерпретация и ограничения

По всем общим метрикам v3.1 немного впереди v4.0, но это **структурный эффект, а не превосходство модели**:

1. **Меньше и проще метрик.** У v3.1 8 базовых метрик против 12 у v4.0; причём v4.0 тянут вниз именно отсутствующие в v3.1 головы `SC/SI/SA` (F1 0.62–0.69) и `AT`. Меньше метрик → выше и vector accuracy (проще угадать 8 полей целиком, чем 11).
2. **Разные тест-сеты.** v3.1 оценён на 26 317 записях (данных v3.1 на порядок больше), v4.0 — на 972. Совпадают методология и формула расчёта балла, но не размер/состав выборки.
3. **Влияние `cwe_name` (подтверждено экспериментально).** Модель обучалась на тексте `description [SEP] cwe_name`, но предикторы изначально строили текст без `cwe_name`. Подстановка имени CWE по `cwe_id` (офлайн-справочник [src/data_preparation/cwe_names_lookup.py](src/data_preparation/cwe_names_lookup.py), источник — `data/raw/cwe_names.json`, выгрузка MITRE) дала по v3.1:

   | | Без `cwe_name` | С `cwe_name` |
   |---|---:|---:|
   | Macro-F1 (8) | 0.800 | **0.814** |
   | Vector accuracy | 0.501 | **0.523** |
   | Score MAE | 0.789 | **0.743** |
   | Scope F1 | 0.830 | **0.847** |

   С `cwe_name` per-head F1 развёрнутого пайплайна **точно совпадает** с `v3_dapt.json` — то есть предиктор воспроизводит полный потенциал модели, а не урезанный.

4. **Почему метрики v4.0 не изменились от этой правки.** Сквозная оценка v4.0 (`Evaluator`) прогоняет модель через `CVSSDataset`, который подаёт `cwe_name` из колонки данных — то есть v4-оценка **всегда** считалась «с `cwe_name`» (0.764). Правка предиктора затрагивает именно **production-API v4.0**: раньше он терял ≈1 п.п. относительно оценки, теперь дотягивается до неё. Так что 0.764 — это и «до», и «после» для оценки, но реальное качество отдаваемого API-ответа выросло.

### 7.9. Артефакты и история изменений

| Коммит | Содержание |
|---|---|
| `acf2a5f` | Оптимизация кэширования Scope-головы (dynamic padding + fp16 + workers) |
| `c5b936e` | Ядро v3.1: калькулятор + предиктор + fingerprint-защита backbone |
| `9613495` | Режим v3.1 в API (`cvss_version`) и веб-интерфейсе (radio) |
| `019dc53` | Сквозная оценка `V31Evaluator` + `predict_batch` |

Новые модули: [src/cvss_calculator/cvss31.py](src/cvss_calculator/cvss31.py), [src/inference/predictor_v31.py](src/inference/predictor_v31.py), [src/model/fingerprint.py](src/model/fingerprint.py), [src/evaluation/evaluate_v31.py](src/evaluation/evaluate_v31.py). Тесты: [tests/test_cvss31.py](tests/test_cvss31.py), [tests/test_predictor_v31.py](tests/test_predictor_v31.py), [tests/test_fingerprint.py](tests/test_fingerprint.py), [tests/test_evaluate_v31.py](tests/test_evaluate_v31.py), v3.1-кейсы в [tests/test_api.py](tests/test_api.py).
