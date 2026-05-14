# Обучение модели

Этот документ описывает, как обучить модель самостоятельно с нуля — на случай,
если вы хотите воспроизвести результаты ВКР, дообучить на свежих данных или
поэкспериментировать с гиперпараметрами.

## Содержание

1. [Требования к окружению](#требования-к-окружению)
2. [Шаг 1. Сбор данных](#шаг-1-сбор-данных)
3. [Шаг 2. Подготовка датасета](#шаг-2-подготовка-датасета)
4. [Шаг 3. Обучение в Google Colab](#шаг-3-обучение-в-google-colab)
5. [Шаг 3 (альтернативный). Локальное обучение](#шаг-3-альтернативный-локальное-обучение)
6. [Ожидаемые показатели](#ожидаемые-показатели)
7. [Troubleshooting](#troubleshooting)

---

## Требования к окружению

### Минимальные

| Ресурс | Значение |
|:--|:--|
| GPU | **NVIDIA Tesla T4 (16 GB VRAM)** или эквивалент |
| RAM | 16 GB |
| Диск | 30 GB (датасеты + чекпоинты + кэш mBERT) |
| Python | 3.10+ |
| PyTorch | 2.x с CUDA 11.8+ |

### Рекомендуемые

| Ресурс | Значение |
|:--|:--|
| GPU | NVIDIA A100 (40/80 GB) — обучение в 3–4 раза быстрее T4 |
| Платформа | Google Colab Pro или локальный сервер с CUDA |

> **На CPU обучить невозможно** — оценочное время ~10 суток. На MPS (Apple
> Silicon) PyTorch работает, но в 5–6 раз медленнее T4 и часть операторов
> mBERT не поддерживается.

### Платные ключи (опционально)

Без ключей сбор данных работает, но **в 3–4 раза медленнее** из-за rate limit
у NVD:

| Переменная | Получение | Назначение |
|:--|:--|:--|
| `NVD_API_KEY` | <https://nvd.nist.gov/developers/request-an-api-key> | 50 req/30s вместо 5 req/30s |
| `HF_TOKEN` | <https://huggingface.co/settings/tokens> | Более высокие лимиты на скачивание весов mBERT |

Заполняются в `.env` (скопировать из `.env.example`).

---

## Шаг 1. Сбор данных

Объединение пяти источников: NVD, БДУ ФСТЭК, EPSS, CISA KEV, ExploitDB.

```bash
python -m src.data_collection.collect --config configs/config.yaml
```

Внутри `src/data_collection/`:

| Модуль | Что делает |
|:--|:--|
| `nvd_client.py` | Запрашивает NVD API 2.0, выгружает 260 тыс. CVE с CVSS-векторами |
| `fstec_client.py` | Скачивает БДУ ФСТЭК `vullist.xml`, парсит русскоязычные описания |
| `epss_client.py` | Получает EPSS-оценки (вероятность эксплуатации) |
| `kev_client.py` | Скачивает CISA KEV JSON feed |
| `exploitdb_client.py` | Скачивает ExploitDB CSV |
| `cwe_client.py` | Скачивает CWE MITRE XML для названий типов |
| `integrator.py` | Сшивает данные по CVE-идентификатору |

Готовый скрипт-обёртка для полного цикла сбора:

```bash
python scripts/collect_full.py
```

**Ожидаемое время:** 4–6 часов без `NVD_API_KEY`, 1–2 часа с ключом.
**Размер raw-выгрузок:** ~3 GB в `data/raw/`.

Результат — `data/raw/integrated.parquet` (~155 тыс. строк со всеми
полями).

---

## Шаг 2. Подготовка датасета

Очистка, валидация, разбиение на train/val/test:

```bash
python -m src.data_preparation.prepare --config configs/config.yaml
```

Что делает:

1. Удаляет дубликаты по `cve_id`.
2. Валидирует CVSS-векторы (отсеивает повреждённые `parse_v3_vector` /
   `parse_v4_vector`).
3. Очищает текст описаний (нормализация Unicode, разбор HTML-сущностей).
4. Строит словарь CWE → индекс: `data/processed/cwe_vocab.json`.
5. Разбивает **по уникальному CVE** на 75/15/15 c фиксированным seed=42.

Результат:

```
data/processed/
├── train.parquet   # 122 913 строк (4 715 с CVSS v4.0)
├── val.parquet     #  26 385 строк
├── test.parquet    #  26 348 строк
└── cwe_vocab.json  # ~350 уникальных CWE
```

> **Critical:** разбиение по CVE-id выполняется **ДО** разделения на наборы
> Stage 1/Stage 2 — иначе одна CVE может оказаться одновременно в Stage 1 train
> и Stage 2 test (утечка данных). Подробнее — раздел 6.5 ВКР.

---

## Шаг 3. Обучение в Google Colab

Самый простой путь — открыть [notebooks/train_colab.ipynb](../notebooks/train_colab.ipynb)
в Google Colab Pro:

```text
File → Upload Notebook → выбрать train_colab.ipynb
Runtime → Change runtime type → GPU → T4 (или A100)
```

Notebook автоматически:

1. Монтирует Google Drive (укажите путь к репозиторию).
2. Устанавливает зависимости (`pip install -r requirements.txt`).
3. Загружает `train.parquet`/`val.parquet` из Drive.
4. Запускает **Stage 1** (CVSS v3.1, 8 голов).
5. Сохраняет `models/checkpoints/stage1_best.pt`.
6. Запускает **Stage 2** (CVSS v4.0, 12 голов, переинициализация AT/SC/SI/SA/E).
7. Сохраняет `models/final_model.pt` обратно в Drive.

Прогресс отслеживается в TensorBoard (запуск из Colab cell — последняя
ячейка). Файлы `logs/tensorboard/events.out.*` сохраняются в `logs/`.

### Время обучения

| Этап | Tesla T4 | A100 |
|:--|:--|:--|
| **Stage 1** (v3.1, 122 913 строк, до 10 эпох) | **4–5 ч** | 1,5 ч |
| **Stage 2** (v4.0, 4 715 строк, до 20 эпох) | **~1 ч** | 25 мин |
| Итого | ~6 ч | ~2 ч |

Early stopping обычно отрабатывает на 6–8 эпохе Stage 1 и на 14–18 эпохе
Stage 2.

---

## Шаг 3 (альтернативный). Локальное обучение

Если есть локальный GPU:

```bash
# Stage 1 — предобучение на CVSS v3.1 (8 метрик)
python -m src.training.train --config configs/train.yaml --stage 1

# Stage 2 — дообучение на CVSS v4.0 (12 метрик)
python -m src.training.train --config configs/train.yaml --stage 2
```

Все гиперпараметры — в `configs/train.yaml`:

```yaml
stage1:
  learning_rate: 2.0e-5
  batch_size: 32
  max_epochs: 10
  dropout: 0.1
  weight_decay: 0.01
  early_stopping_patience: 3

stage2:
  learning_rate: 1.0e-5
  batch_size: 16
  max_epochs: 20
  dropout: 0.1
  weight_decay: 0.01
  early_stopping_patience: 3
  reinit_heads: ["AT", "SC", "SI", "SA", "E"]

seed: 42
```

Чекпоинты сохраняются в `models/checkpoints/` (один на каждую улучшенную
эпоху + last). Финальная модель копируется в `models/final_model.pt`.

---

## Ожидаемые показатели

После двух этапов на test set (972 v4.0-записи) вы должны получить:

| Показатель | Целевой диапазон |
|:--|:--|
| Macro-F1 (12 метрик) | **0,70 – 0,72** |
| Среднее число правильных метрик | 9,3 – 9,5 / 11 |
| MAE по CVSS-баллу | 1,1 – 1,3 |
| Severity Within ±1 | 0,90 – 0,93 |

Если ваши значения сильно ниже:

* проверьте, что Stage 1 выполнен полностью (early stopping не сработал на
  1–2 эпохе из-за неправильно настроенного val);
* проверьте, что головы AT/SC/SI/SA/E были **переинициализированы** в
  Stage 2 (флаг `reinit_heads` в конфиге);
* проверьте, что split по CVE-id не позволил утечке (см. предыдущий раздел).

Полный разбор экспериментальных результатов — [reports/CHAPTER3_DRAFT.md](../reports/CHAPTER3_DRAFT.md).

---

## Troubleshooting

### `CUDA out of memory`

Самая частая проблема. Решения:

1. **Уменьшить batch size** в `configs/train.yaml`:
   `stage1.batch_size: 16` (было 32), `stage2.batch_size: 8` (было 16).
2. **Включить gradient accumulation** (если ещё не включён):
   `stage1.gradient_accumulation_steps: 2` — эффективный батч прежний,
   потребление VRAM в 2 раза меньше.
3. **Уменьшить `max_length`** в `configs/config.yaml`: с 512 до 256 —
   подавляющее большинство описаний короче. Это даст ~30% экономии.
4. **Закрыть другие процессы**, использующие GPU
   (`nvidia-smi` → kill PID, занимающие VRAM).

### Медленное обучение (< 1 it/s на T4)

* Убедитесь, что mixed precision включён: в логах должно быть
  `Using torch.amp.autocast(dtype=torch.float16)`.
* Проверьте `num_workers` в DataLoader — на Colab/Linux должно быть 2–4.
* На Windows `num_workers > 0` иногда тормозит — попробуйте 0.
* Если GPU не загружен (`nvidia-smi` показывает Util < 50%) — bottleneck
  в DataLoader: увеличьте `prefetch_factor`.

### Colab: «Превышено время ожидания» / разрыв соединения с Drive

Colab Free отключает runtime через 90 мин неактивности, Colab Pro — через
24 ч. Решения:

1. **Колаб-кипалив:** в отдельной ячейке запустить
   ```python
   import time
   while True:
       print(time.time(), flush=True)
       time.sleep(60)
   ```
2. **Сохранять чекпоинт в Drive каждую эпоху** (уже есть в нашем `trainer.py`).
3. **Использовать Colab Pro+ или A100** — отключения реже.
4. Если runtime упал в середине Stage 1, переоткройте notebook и в нём
   `Trainer.resume_from_checkpoint("models/checkpoints/stage1_last.pt")` —
   тренировка продолжится с последней сохранённой эпохи.

### `RuntimeError: weight_only_load_v2 ... not supported`

Старая версия PyTorch (<2.4). Обновитесь: `pip install --upgrade torch`.

### `OSError: ...bert-base-multilingual-cased not found`

mBERT не скачался автоматически. Принудительно:

```python
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
AutoModel.from_pretrained("bert-base-multilingual-cased")
```

При наличии `HF_TOKEN` в `.env` — скачивание стабильнее.

### `KeyError: 'cwe_id_xxx'` в инференсе

Ваш чекпоинт обучался на `cwe_vocab.json` из обучения, а в окружении лежит
другой словарь. Скопируйте `data/processed/cwe_vocab.json` рядом с чекпоинтом
или укажите путь явно:

```bash
python -m src.inference.cli predict ... --cwe-vocab /path/to/cwe_vocab.json
```

### Метрики после Stage 2 хуже, чем после Stage 1

Признак того, что Stage 2 переобучается на малом v4.0-корпусе. Проверьте:

* `stage2.early_stopping_patience` (по умолчанию 3) — должен срабатывать;
* `stage2.learning_rate` — 1e-5 для Stage 2 (НЕ 2e-5 как в Stage 1, иначе
  fine-tune ломает предобученные веса);
* что split НЕ изменился между запусками (одинаковый seed=42).

---

См. также:

- [docs/architecture.md](architecture.md) — устройство модели;
- [CLAUDE.md](../CLAUDE.md) — паспорт проекта со всеми требованиями ВКР;
- [reports/CHAPTER3_DRAFT.md](../reports/CHAPTER3_DRAFT.md) — анализ результатов экспериментов.
