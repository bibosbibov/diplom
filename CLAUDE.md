# CLAUDE.md — Паспорт проекта

## Тема работы

Магистерская ВКР по направлению 10.04.01 «Информационная безопасность», направленность «Комплексная защита объектов информатизации».

**Тема:** «Разработка системы автоматической оценки критичности уязвимостей программного обеспечения на основе CVSS v4.0 с применением трансформерной модели mBERT».

## Цель системы

По текстовому описанию уязвимости (на русском или английском языке), идентификатору типа уязвимости CWE и дополнительным признакам эксплуатируемости (EPSS, CISA KEV, ExploitDB) автоматически предсказать значения 12 метрик базового вектора CVSS v4.0 и рассчитать итоговый числовой балл (0.0–10.0) с определением уровня критичности (None / Low / Medium / High / Critical).

## Обязательные технические решения

### Источники данных

- **БДУ ФСТЭК России** (основной источник русскоязычных описаний): `https://bdu.fstec.ru/` — файл `vullist.xml` или `vullist.xlsx`.
- **NVD API 2.0**: `https://services.nvd.nist.gov/rest/json/cves/2.0` — англоязычные описания, CVSS-векторы, CPE. Rate limit: 5 запросов/30 сек без ключа, 50/30 сек с ключом. Ключ в переменной окружения `NVD_API_KEY`.
- **EPSS API**: `https://api.first.org/data/v1/epss` — вероятность эксплуатации от 0 до 1 в ближайшие 30 дней.
- **CISA KEV Catalog**: `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json` — каталог подтверждённо эксплуатируемых уязвимостей.
- **ExploitDB**: CSV-выгрузка с `https://github.com/offensive-security/exploitdb` — наличие публичных эксплойтов.
- **CWE MITRE**: `https://cwe.mitre.org/data/xml/cwec_latest.xml.zip` — человекочитаемые названия типов уязвимостей.

### Предсказываемые метрики (12 штук)

Базовый вектор CVSS v4.0:

| Метрика | Полное название | Допустимые значения |
|---------|-----------------|---------------------|
| AV      | Attack Vector | Network (N), Adjacent (A), Local (L), Physical (P) |
| AC      | Attack Complexity | Low (L), High (H) |
| AT      | Attack Requirements | None (N), Present (P) |
| PR      | Privileges Required | None (N), Low (L), High (H) |
| UI      | User Interaction | None (N), Passive (P), Active (A) |
| VC      | Vulnerable System Confidentiality | High (H), Low (L), None (N) |
| VI      | Vulnerable System Integrity | High (H), Low (L), None (N) |
| VA      | Vulnerable System Availability | High (H), Low (L), None (N) |
| SC      | Subsequent System Confidentiality | High (H), Low (L), None (N) |
| SI      | Subsequent System Integrity | High (H), Low (L), None (N) |
| SA      | Subsequent System Availability | High (H), Low (L), None (N) |
| E       | Exploit Maturity | Attacked (A), POC (P), Unreported (U) |

### Архитектура модели

```
Входы: (описание + [SEP] + cwe_name) → токенизация → input_ids, attention_mask
       cwe_id → индекс → Embedding(num_cwe, 64)
       epss, kev, exploit → вектор размерности 3 (с маркером -1 для ∅)

Трансформер (mBERT):     input_ids, attention_mask → H[:, 0, :] = h_text (768)

MLP кодировщик признаков:
    Concat(epss, kev, exploit, cwe_emb) → f_ext (67)
    Linear(67 → 128) → ReLU → Linear(128 → 64) → ReLU → h_feat (64)

Fusion Layer:
    Concat(h_text, h_feat) → h_combined (832)
    Linear(832 → 512) → ReLU → Dropout(0.1) → h_fused (512)

12 классификационных голов:
    Для каждой метрики M_i: Linear(512 → num_classes_i) → softmax → ŷ_i, conf_i
```

**Предобученная модель:** `bert-base-multilingual-cased` (Hugging Face).

### Двухэтапная стратегия обучения

**Этап 1 — предобучение на CVSS v3.1** (8 общих метрик: AV, AC, PR, UI, VC, VI, VA, E):
- Learning rate: `2e-5`
- Batch size: `32`
- Max epochs: `10`
- Dropout: `0.1`
- Weight decay: `0.01`
- Early stopping patience: `3`

**Этап 2 — дообучение на CVSS v4.0** (все 12 метрик; головы AT, SC, SI, SA инициализируются случайно):
- Learning rate: `1e-5`
- Batch size: `16`
- Max epochs: `20`
- Dropout: `0.1`
- Weight decay: `0.01`
- Early stopping patience: `3`

**Оптимизатор:** AdamW. **Scheduler:** linear warmup (10% шагов) + linear decay. **Seed:** 42 (для воспроизводимости).

**Функция потерь:** сумма кросс-энтропий по всем обучаемым метрикам (многозадачное обучение):

```
L = Σ_{i=1}^{N} CrossEntropy(logits_i, y_i)
```

где N = 8 на этапе 1, N = 12 на этапе 2.

### Расчёт итогового балла CVSS v4.0

**Собственная реализация** по официальной спецификации FIRST (запрещено использовать сторонние библиотеки-калькуляторы — требование раздела 1.3.6 отчёта).

Четырёхэтапный алгоритм:

1. **MacroVector**: группировка 12 метрик в 6 групп эквивалентности (EQ1–EQ6) → строка из 6 цифр.
2. **Базовый балл**: lookup по таблице из 264 значений (из спецификации FIRST).
3. **Интерполяция** (Severity Distances): поправка по расстояниям от текущих метрик до наихудшего случая в группе, с ограничением `min(total_distance, 0.5)`.
4. **Модификатор Exploit Maturity**: умножение на коэффициент `k_E` (A=1.0, P=0.94, U=0.91), округление до 1 знака.

**Источник таблиц и алгоритма:** `https://www.first.org/cvss/v4.0/specification-document` и `https://github.com/FIRSTdotorg/cvss-v4-calculator`.

### Стек технологий

- **Язык:** Python 3.10+
- **Deep Learning:** PyTorch 2.x, Hugging Face Transformers
- **Данные:** Pandas, NumPy, pyarrow (parquet)
- **ML-утилиты:** scikit-learn (метрики, stratified split, k-fold)
- **Визуализация:** Matplotlib, Seaborn
- **HTTP:** requests, tenacity (ретраи)
- **API:** FastAPI, uvicorn, pydantic
- **CLI:** click, rich
- **Тесты:** pytest, pytest-cov, pytest-httpx
- **Качество кода:** black, isort, pylint, mypy
- **Логирование/мониторинг:** стандартный logging, TensorBoard
- **Контейнеризация:** Docker

### Структура проекта

```
project_root/
├── src/
│   ├── data_collection/       # Сбор из БДУ, NVD, EPSS, KEV, ExploitDB
│   ├── data_preparation/      # Токенизация, кодирование CWE, числовые признаки
│   ├── model/                 # Архитектура нейросети
│   ├── training/              # Двухэтапное обучение
│   ├── cvss_calculator/       # Собственная реализация алгоритма CVSS v4.0
│   ├── evaluation/            # Метрики качества, k-fold CV
│   ├── inference/             # End-to-end pipeline предсказания
│   └── api/                   # FastAPI веб-сервис
├── tests/                     # pytest-тесты для каждого модуля
├── data/
│   ├── raw/                   # Сырые выгрузки из API
│   └── processed/             # Train/val/test split, cwe_vocab.json
├── models/                    # Сохранённые чекпоинты
│   └── checkpoints/
├── reports/
│   ├── figures/               # Графики PNG 300 dpi для ВКР
│   └── *.json, *.md           # Результаты оценки, итоговые отчёты
├── notebooks/                 # Jupyter-ноутбуки с экспериментами
├── configs/
│   └── train.yaml             # Гиперпараметры
├── docs/                      # Техническая документация
├── logs/                      # Логи сбора данных и обучения
├── presentation/              # Материалы для защиты ВКР
├── CLAUDE.md                  # Этот файл
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── Dockerfile
└── docker-compose.yml
```

## Соглашения

- **Язык комментариев и docstrings:** русский для пользовательских описаний, английский для технических терминов без устоявшегося русского перевода.
- **Стиль docstrings:** Google.
- **Форматирование:** black (line length 100), isort.
- **Type hints:** обязательны для всех публичных функций и методов.
- **Воспроизводимость:** фиксация seed=42 в torch, numpy, random, transformers.
- **Конфигурация:** все параметры через `configs/*.yaml` и `.env`, никакого хардкодинга путей и ключей.
- **Логирование:** через стандартный `logging`, формат `%(asctime)s [%(levelname)s] %(name)s: %(message)s`, уровень INFO в файл + WARNING в консоль.
- **Тесты:** каждый новый модуль сопровождается тестами, цель покрытия ≥ 70%.
- **Коммиты:** Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`).

## Ограничения и особые требования

1. **Маркер отсутствующих признаков.** Для EPSS, KEV, exploit при отсутствии значения используется специальное значение `-1` (а не 0, 0.5 или NaN), которое модель учится интерпретировать.
2. **Выбор языка описания.** Русскоязычное описание (d_ru) приоритетнее англоязычного (d_en); если русского нет — берётся английское.
3. **Максимальная длина последовательности:** 512 токенов (`max_length=512`), truncation=True, padding до max_length.
4. **Специальные токены:** `[CLS]` в начале, `[SEP]` между описанием и cwe_name, и в конце (автоматически добавляется токенизатором).
5. **Предотвращение утечки данных.** Train/val/test разбиение должно выполняться ПО УНИКАЛЬНОМУ CVE-идентификатору до разделения на наборы для этапа 1 и этапа 2 обучения. Одна CVE не должна оказаться одновременно в обучении stage1 и тесте stage2.
6. **Собственный калькулятор CVSS v4.0.** Запрещено использовать сторонние библиотеки для расчёта итогового балла (требование раздела 1.3.6 отчёта — самостоятельная реализация по спецификации FIRST).
7. **Готовность к работе без части данных.** Система должна корректно работать, если для новой уязвимости отсутствуют EPSS/KEV/exploit (новая, только что опубликованная CVE).

## Связь с текстом ВКР

| Модуль кода | Раздел отчёта |
|-------------|---------------|
| `src/data_collection/` | 2.3.1 Алгоритм сбора и интеграции данных |
| `src/data_preparation/` | 2.3.2 Алгоритм подготовки данных |
| `src/model/` | 2.2.5 Модель классификации |
| `src/training/` | 2.2.4, 2.3.3 Модель и алгоритм обучения |
| `src/cvss_calculator/` | 2.2.6, 2.3.5 Модель и алгоритм расчёта балла |
| `src/evaluation/` | 2.2.7, 2.3.6 Модель и алгоритм оценки качества |
| `src/inference/` | Глава 3 (реализация) |
| `src/api/` | Глава 3 (программное средство) |
