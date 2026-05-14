# Автоматическая оценка критичности уязвимостей ПО на основе CVSS v4.0

Программная система, которая по текстовому описанию уязвимости и идентификатору
типа уязвимости (CWE) автоматически предсказывает 12 метрик базового вектора
**CVSS v4.0**, рассчитывает итоговый числовой балл (0,0–10,0) и определяет
уровень критичности (None / Low / Medium / High / Critical).

---

## О проекте

Магистерская ВКР по направлению **10.04.01 «Информационная безопасность»**,
направленность «Комплексная защита объектов информатизации». Система использует
трансформерную модель **mBERT** (`bert-base-multilingual-cased`) с
fusion-слоем для числовых признаков и 12 классификационными головами, обученную
двухэтапной стратегией — предобучение на 122 тыс. записей CVSS v3.1 и
дообучение на 4,7 тыс. записей CVSS v4.0. Расчёт итогового балла выполняется
**собственной реализацией** калькулятора по спецификации FIRST CVSS v4.0
(без сторонних библиотек).

**Главные достижения на test set (972 v4.0-записи):**

- **Macro-F1 = 0,7090** (среднее по 12 метрикам),
- **MAE по CVSS-баллу = 1,17** (11,7% шкалы),
- **Severity Within ±1 = 0,9208** (модель попадает в истинный уровень
  критичности или соседний в 92% случаев).

---

## Архитектура

Pipeline предсказания «от описания до Severity»:

```
   ┌──────────────────────────────────────────────┐
   │  Описание уязвимости (рус./англ.) +          │
   │  CWE-ID + (опц.) EPSS, KEV, ExploitDB        │
   └──────────────────────┬───────────────────────┘
                          ▼
            ┌───────────────────────────┐
            │      TextProcessor        │  очистка, выбор языка
            └─────────────┬─────────────┘
                          ▼
            ┌───────────────────────────┐
            │   CVSSTokenizer (mBERT)   │  input_ids, attention_mask
            └─────────────┬─────────────┘
                          │
   ┌──────────────────────┼──────────────────────┐
   │                      ▼                      │
   │           ┌──────────────────────┐          │
   │           │     mBERT (12L)      │   [CLS] →│
   │           └──────────┬───────────┘  h_text  │
   │                      │  (768)               │
   │                      │                      │
   │   ┌──────────────────┴──────────────┐       │
   │   │       Fusion Layer 832→512      │       │
   │   └──────────────────┬──────────────┘       │
   │                      ▲                      │
   │           ┌──────────┴───────────┐          │
   │           │   Features MLP 67→64 │          │
   │           └──────────┬───────────┘          │
   │                      │                      │
   │   ┌──────────────────┴──────────────┐       │
   │   │  Concat: CWE-emb (64) + EPSS,   │       │
   │   │  KEV, exploit (3)   = f_ext(67) │       │
   │   └─────────────────────────────────┘       │
   └──────────────────────┬──────────────────────┘
                          ▼
            ┌───────────────────────────┐
            │  12 классификационных     │  AV, AC, AT, PR, UI,
            │  голов: Linear → softmax  │  VC, VI, VA, SC, SI, SA, E
            └─────────────┬─────────────┘
                          ▼
            ┌───────────────────────────┐
            │     CVSSCalculator        │  MacroVector + interpolation
            │  (своя реализация v4.0)   │  + Exploit Maturity модификатор
            └─────────────┬─────────────┘
                          ▼
            CVSS-вектор + Score (0–10) + Severity
```

Подробнее — [docs/architecture.md](docs/architecture.md).

---

## Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/bibosbibov/diplom.git
cd diplom

# 2. Создать и активировать виртуальное окружение (Python 3.10+)
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt

# 4. Скопировать шаблон переменных окружения и заполнить ключи
cp .env.example .env
```

Минимальные требования: **Python 3.10+**, ~10 ГБ свободного диска (датасеты +
веса mBERT). GPU не обязателен для инференса (модель работает на CPU за
~200 мс на запрос).

---

## Использование

### 1. Python API

```python
from src.inference import VulnerabilityPredictor

predictor = VulnerabilityPredictor(
    model_path="models/final_model.pt",
    device="auto",
)

result = predictor.predict(
    description="SQL injection in the login form allows authentication bypass",
    cwe_id="CWE-89",
    epss=0.42,            # опционально
    kev=False,            # опционально
    exploit=True,         # опционально
)

print(result["vector"])    # CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/...
print(result["score"])     # 6.9
print(result["severity"])  # Medium
print(result["low_confidence_metrics"])  # ['PR', 'UI']
```

### 2. CLI

```bash
# Одиночное предсказание
python -m src.inference.cli predict \
    --description "Buffer overflow in image parser..." \
    --cwe CWE-787

# Пакетная обработка CSV
python -m src.inference.cli batch-predict input.csv output.csv

# Оценка модели на test parquet
python -m src.inference.cli evaluate data/processed/test.parquet --limit 100
```

### 3. Веб-сервис (FastAPI)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

После запуска:

- <http://localhost:8000> — веб-интерфейс с формой;
- <http://localhost:8000/docs> — Swagger UI;
- <http://localhost:8000/health> — статус сервиса.

Пример запроса:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "SQL injection in login form allows auth bypass",
       "cwe_id": "CWE-89"}'
```

Полная документация REST API — [docs/api.md](docs/api.md). Руководство для
инженера по ИБ — [docs/user_guide.md](docs/user_guide.md).

---

## Где взять модель

Готовый чекпоинт `models/final_model.pt` (440 МБ, FP32) распространяется
отдельно из-за размера. Варианты получения:

1. **Релизы GitHub:** скачать
   `final_model.pt` со страницы
   <https://github.com/bibosbibov/diplom/releases/latest>
   и положить в `models/final_model.pt`.
2. **Обучить самостоятельно** по инструкции [docs/training.md](docs/training.md).
   Время обучения на Tesla T4: ~4–5 ч на Stage 1 + ~1 ч на Stage 2.

Дополнительно потребуется `data/processed/cwe_vocab.json` — словарь
CWE → индекс, полученный на этапе подготовки данных.

---

## Результаты

Оценка проведена на отложенной тестовой выборке (972 записи с валидным
CVSS v4.0). Подробный разбор — [reports/CHAPTER3_DRAFT.md](reports/CHAPTER3_DRAFT.md).

| Показатель | Значение |
|:--|:--|
| Macro-F1 (12 метрик) | **0,7090** |
| Vector Accuracy (11 метрик) | 0,3992 |
| Среднее число правильных метрик | 9,39 / 11 (85,4%) |
| MAE по CVSS-баллу | **1,17** (11,7% шкалы 0–10) |
| RMSE по CVSS-баллу | 1,98 |
| Severity Accuracy | 0,6739 |
| **Severity Within ±1** | **0,9208** |
| Размер test set | 972 записи CVSS v4.0 |

Стабильность val → test: среднее абсолютное отклонение |Δ| F1 = 0,016 — нет
переобучения.

---

## Структура проекта

```
diplom/
├── src/
│   ├── data_collection/       # Сбор из БДУ ФСТЭК, NVD, EPSS, KEV, ExploitDB
│   ├── data_preparation/      # Токенизация, кодирование CWE, числовые признаки
│   ├── model/                 # Архитектура нейросети (mBERT + Fusion + 12 голов)
│   ├── training/              # Двухэтапное обучение, MultiTaskLoss, Trainer
│   ├── cvss_calculator/       # Собственная реализация алгоритма CVSS v4.0
│   ├── evaluation/            # Метрики качества, k-fold CV
│   ├── inference/             # End-to-end pipeline предсказания + CLI
│   └── api/                   # FastAPI веб-сервис (REST + статический UI)
├── tests/                     # pytest-тесты (≥70% покрытие)
├── data/
│   ├── raw/                   # Сырые выгрузки из API
│   └── processed/             # train/val/test.parquet, cwe_vocab.json
├── models/                    # Сохранённые чекпоинты (final_model.pt)
├── reports/                   # Метрики, графики, таблицы для главы 3 ВКР
│   ├── figures/               # PNG 300 dpi для ВКР
│   ├── error_analysis/        # Анализ ошибок, per-CWE метрики
│   └── *.md                   # Сводные таблицы для копирования в LaTeX/Word
├── notebooks/                 # 09 — датасет, 10 — анализ ошибок, 11 — итоги
├── configs/                   # train.yaml, config.yaml
├── docs/                      # Техническая документация (см. ниже)
├── logs/                      # Логи сбора и обучения, TensorBoard events
├── scripts/                   # Вспомогательные скрипты
├── CLAUDE.md                  # Паспорт проекта (фиксирует все технические решения)
├── README.md                  # Этот файл
├── LICENSE                    # MIT
├── CITATION.cff               # Цитирование
└── requirements.txt
```

---

## Документация

Развёрнутая документация — в каталоге [docs/](docs/):

| Документ | Назначение |
|:--|:--|
| [docs/architecture.md](docs/architecture.md) | Архитектура системы, обоснование решений |
| [docs/user_guide.md](docs/user_guide.md) | Руководство для инженера по ИБ |
| [docs/api.md](docs/api.md) | Полное описание REST API |
| [docs/training.md](docs/training.md) | Как обучить модель самостоятельно |

Паспорт проекта со всеми обязательными техническими решениями —
[CLAUDE.md](CLAUDE.md).

---

## Скриншоты

Скриншоты веб-интерфейса и Swagger UI:

- ![Веб-интерфейс](docs/screenshots/web_ui.png) — главная страница демо
- ![Результат предсказания](docs/screenshots/prediction_result.png) — отображение CVSS-вектора и severity
- ![Swagger UI](docs/screenshots/swagger_ui.png) — интерактивная документация API

*(Файлы заполняются перед защитой ВКР.)*

---

## Лицензия

Распространяется под лицензией [MIT](LICENSE). Использование, модификация и
коммерческое применение разрешены при сохранении уведомления об авторстве.

---

## Цитирование

При использовании результатов работы в академических публикациях, пожалуйста,
цитируйте репозиторий — формат CFF приведён в [CITATION.cff](CITATION.cff).

---

## Автор

**Артём (@bibosbibov)** — магистрант ФГБОУ ВО «Южно-Российский государственный
политехнический университет (НПИ) имени М. И. Платова», кафедра
«Информационная безопасность».

Связь: <https://github.com/bibosbibov>
