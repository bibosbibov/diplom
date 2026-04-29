# Система автоматической оценки критичности уязвимостей ПО на основе CVSS v4.0

Магистерская ВКР, направление 10.04.01 «Информационная безопасность».
Система предсказывает 12 метрик базового вектора **CVSS v4.0** по текстовому
описанию уязвимости (рус./англ.), идентификатору **CWE** и признакам
эксплуатируемости (**EPSS, CISA KEV, ExploitDB**) с использованием
трансформерной модели **mBERT** и собственного калькулятора CVSS v4.0.

> Полное описание архитектуры, источников данных, гиперпараметров и связи
> с разделами отчёта — в [CLAUDE.md](CLAUDE.md).

---

## Требования

- **Python:** 3.10 или новее
- **ОС:** Windows / Linux / macOS
- **GPU (опционально):** CUDA 11.8+ для ускорения обучения
- **Свободное место:** ~10 ГБ (датасеты + чекпоинты mBERT)

---

## Установка

```bash
# 1. Клонировать репозиторий
git clone <repo-url> diplom
cd diplom

# 2. Создать и активировать виртуальное окружение
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt

# 4. Скопировать шаблон переменных окружения
cp .env.example .env
# и заполнить NVD_API_KEY, при необходимости остальные переменные
```

---

## Конфигурация

Все гиперпараметры, пути и URL источников вынесены в [configs/config.yaml](configs/config.yaml).
Секреты — только через переменные окружения (`.env`).

| Файл | Назначение |
|------|------------|
| `configs/config.yaml` | Гиперпараметры, пути, URL API |
| `.env` | API-ключи, уровни логирования (не коммитится) |
| `.env.example` | Шаблон переменных окружения |

---

## Запуск пайплайна

### 1. Сбор данных

```bash
# Загрузка из БДУ ФСТЭК, NVD, EPSS, CISA KEV, ExploitDB
python -m src.data_collection.collect --config configs/config.yaml
```

### 2. Подготовка данных

```bash
# Токенизация, кодирование CWE, формирование train/val/test
python -m src.data_preparation.prepare --config configs/config.yaml
```

### 3. Обучение модели (двухэтапное)

```bash
# Этап 1 — предобучение на CVSS v3.1 (8 общих метрик)
python -m src.training.train --config configs/config.yaml --stage 1

# Этап 2 — дообучение на CVSS v4.0 (все 12 метрик)
python -m src.training.train --config configs/config.yaml --stage 2
```

### 4. Оценка качества

```bash
python -m src.evaluation.evaluate --config configs/config.yaml \
    --checkpoint models/checkpoints/stage2_best.pt
```

### 5. Inference (одиночное предсказание)

```bash
python -m src.inference.predict --config configs/config.yaml \
    --description "Описание уязвимости..." --cwe CWE-79
```

### 6. Запуск веб-сервиса

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

После запуска:
- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

---

## Тесты

```bash
# Запуск всех тестов с покрытием
pytest --cov=src --cov-report=html tests/

# Только конкретный модуль
pytest tests/test_cvss_calculator.py -v
```

Отчёт о покрытии — `htmlcov/index.html`.

---

## Качество кода

```bash
black --line-length 100 src tests
isort src tests
pylint src
mypy src
```

---

## Структура проекта

```
diplom/
├── src/
│   ├── data_collection/    # Сбор из БДУ, NVD, EPSS, KEV, ExploitDB
│   ├── data_preparation/   # Токенизация, кодирование CWE, числовые признаки
│   ├── model/              # Архитектура mBERT + MLP + Fusion + 12 голов
│   ├── training/           # Двухэтапное обучение
│   ├── cvss_calculator/    # Собственная реализация CVSS v4.0
│   ├── evaluation/         # Метрики, k-fold CV
│   ├── inference/          # End-to-end pipeline предсказания
│   └── api/                # FastAPI веб-сервис
├── tests/                  # pytest-тесты
├── data/{raw,processed}/   # Сырые и подготовленные данные
├── models/checkpoints/     # Чекпоинты обучения
├── reports/figures/        # Графики 300 dpi для ВКР
├── notebooks/              # Эксперименты в Jupyter
├── configs/config.yaml     # Гиперпараметры и пути
├── docs/                   # Техническая документация
├── logs/                   # Логи сбора и обучения
└── presentation/           # Материалы для защиты
```

---

## Источники данных

| Источник | URL |
|----------|-----|
| БДУ ФСТЭК России | <https://bdu.fstec.ru/> |
| NVD API 2.0 | <https://services.nvd.nist.gov/rest/json/cves/2.0> |
| EPSS API | <https://api.first.org/data/v1/epss> |
| CISA KEV | <https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json> |
| ExploitDB | <https://github.com/offensive-security/exploitdb> |
| CWE MITRE | <https://cwe.mitre.org/data/xml/cwec_latest.xml.zip> |
| Спецификация CVSS v4.0 | <https://www.first.org/cvss/v4.0/specification-document> |

---

## Лицензия

Учебный проект (магистерская ВКР). Использование в коммерческих целях
не предусмотрено.
