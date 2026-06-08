# REST API

FastAPI веб-сервис предоставляет эндпоинты предсказания CVSS (v4.0 и v3.1),
оценки по Методике ФСТЭК и служебные, а также встроенный интерактивный
Swagger UI.

## Базовый адрес

После запуска `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`:

- **Web UI:** <http://localhost:8000/>
- **Swagger UI:** <http://localhost:8000/docs> — интерактивная документация, можно прямо там отправлять запросы.
- **OpenAPI JSON:** <http://localhost:8000/openapi.json>

Аутентификация не требуется (см. ограничения внизу документа).

---

## Содержание

1. [POST /predict](#post-predict)
2. [POST /predict/batch](#post-predictbatch)
3. [POST /fstec](#post-fstec)
4. [GET /fstec/options](#get-fstecoptions)
5. [GET /fstec/suggest](#get-fstecsuggest)
6. [GET /cwe](#get-cwe)
7. [GET /health](#get-health)
8. [GET /model/info](#get-modelinfo)
9. [Ошибки и их обработка](#ошибки-и-их-обработка)
10. [Ограничения](#ограничения)

---

## POST /predict

Предсказывает CVSS-вектор и итоговый балл для **одной** уязвимости.

### Request

`Content-Type: application/json`

| Поле | Тип | Обязательно | Ограничения |
|:--|:--|:--|:--|
| `description` | string | Да | min_length=10, max_length=10000 |
| `cwe_id` | string | Да | regex `^CWE-\d+$` (например, `CWE-89`) |
| `cvss_version` | string | Нет | `4.0` (по умолчанию) или `3.1`. Определяет версию вектора и набор метрик (12 для v4.0, 8 для v3.1) |
| `description_ru` | string \| null | Нет | Отдельное русскоязычное описание, если `description` английский |
| `epss` | float \| null | Нет | 0,0 ≤ epss ≤ 1,0 |
| `kev` | bool \| null | Нет | присутствие в CISA KEV |
| `exploit` | bool \| null | Нет | публичный эксплойт (ExploitDB) |

### Пример request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "description": "SQL injection in login form allows authentication bypass via crafted POST request",
    "cwe_id": "CWE-89",
    "epss": 0.42,
    "kev": false,
    "exploit": true
  }'
```

### Response

`200 OK`, `Content-Type: application/json`:

```json
{
  "cvss_version": "4.0",
  "vector": "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:L/VI:N/VA:L/SC:N/SI:N/SA:N/E:A",
  "score": 6.9,
  "severity": "Medium",
  "metrics": {
    "AV": {"value": "N", "confidence": 0.531},
    "AC": {"value": "L", "confidence": 0.933},
    "AT": {"value": "N", "confidence": 0.993},
    "PR": {"value": "N", "confidence": 0.563},
    "UI": {"value": "N", "confidence": 0.752},
    "VC": {"value": "L", "confidence": 0.829},
    "VI": {"value": "N", "confidence": 0.581},
    "VA": {"value": "L", "confidence": 0.508},
    "SC": {"value": "N", "confidence": 0.860},
    "SI": {"value": "N", "confidence": 0.913},
    "SA": {"value": "N", "confidence": 0.981},
    "E":  {"value": "A", "confidence": 0.996}
  },
  "low_confidence_metrics": ["AV", "PR", "VI", "VA"],
  "inference_time_ms": 210.47
}
```

### HTTP status codes

| Код | Когда |
|:--|:--|
| `200 OK` | Успешное предсказание |
| `422 Unprocessable Entity` | Невалидный body (пустое описание, кривой CWE, epss вне [0,1]) |
| `500 Internal Server Error` | Внутренняя ошибка (модель не загружена, ошибка инференса) |
| `503 Service Unavailable` | Сервис не готов (модель ещё загружается) |

### Поля response

| Поле | Описание |
|:--|:--|
| `cvss_version` | Версия результата: `4.0` или `3.1` (эхо запроса) |
| `vector` | Каноническая CVSS-строка `CVSS:4.0/AV:N/.../E:A` (или `CVSS:3.1/...`) |
| `score` | Итоговый балл 0,0–10,0 (рассчитан собственным калькулятором) |
| `severity` | None / Low / Medium / High / Critical |
| `metrics` | Словарь метрик с `value` и `confidence` (12 для v4.0, 8 для v3.1) |
| `low_confidence_metrics` | Метрики с confidence < 0,7 — требуют ручной проверки |
| `inference_time_ms` | Время предсказания в миллисекундах |

---

## POST /predict/batch

Пакетная обработка от 1 до 100 уязвимостей за один запрос. Эффективнее
последовательных вызовов `/predict` за счёт batch-инференса.

### Request

```json
{
  "items": [
    {"description": "Cross-site scripting...", "cwe_id": "CWE-79"},
    {"description": "SQL injection...",        "cwe_id": "CWE-89"},
    {"description": "Buffer overflow...",      "cwe_id": "CWE-787"}
  ]
}
```

Каждый элемент в `items` — полная структура `PredictionRequest` (см. выше).
Ограничения: 1 ≤ `len(items)` ≤ 100.

### Пример

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @batch.json
```

### Response

`200 OK`. Массив из N `PredictionResponse` в том же порядке, что и `items`:

```json
[
  {"vector": "CVSS:4.0/AV:N/...", "score": 6.9, "severity": "Medium", "metrics": {...}, ...},
  {"vector": "CVSS:4.0/AV:N/...", "score": 6.9, "severity": "Medium", "metrics": {...}, ...},
  {"vector": "CVSS:4.0/AV:N/...", "score": 2.1, "severity": "Low",    "metrics": {...}, ...}
]
```

### Status codes

Те же, что у `/predict`. Дополнительно:

| Код | Когда |
|:--|:--|
| `422` | `len(items) == 0` или `> 100` |

---

## POST /fstec

Рассчитывает уровень критичности уязвимости в **конкретной** информационной
системе по Методике ФСТЭК России (Методический документ от 30.06.2025):

```
V = I_cvss × I_infr × (I_at + I_imp)
```

Базовый балл `I_cvss` берётся из предсказания модели CVSS v3.1 по `description`
+ `cwe_id` (у пользователя не запрашивается). Контекстные показатели задаёт
пользователь кодами из Таблицы 1 Методики (получить каталог — `GET /fstec/options`).

### Request

| Поле | Тип | Обязательно | Описание |
|:--|:--|:--|:--|
| `description` | string | Да | Описание уязвимости (min 10, max 10000) |
| `cwe_id` | string | Да | regex `^CWE-\d+$` |
| `k` | string[] | Да | Тип компонента ИС — мультивыбор (≥1 код), берётся max |
| `l` | string[] | Да | Доля уязвимых компонентов — мультивыбор (≥1), max |
| `p` | string | Да | Влияние на периметр — один код |
| `h` | string[] | Да | Последствия воздействий — мультивыбор (≥1), max |
| `e` | string[] | Нет | Сведения об эксплуатации; если пусто — выводится из `kev`/`exploit` |
| `description_ru`, `epss`, `kev`, `exploit` | — | Нет | Как в `/predict` |

### Пример request

```bash
curl -X POST http://localhost:8000/fstec \
  -H "Content-Type: application/json" \
  -d '{
    "description": "SQL injection in login form allows authentication bypass",
    "cwe_id": "CWE-89",
    "k": ["server"],
    "l": ["from_10_to_50"],
    "p": "internet_accessible",
    "e": ["exploit_available"],
    "h": ["code_injection"]
  }'
```

### Response

`200 OK`:

```json
{
  "v": 3.84,
  "v_exact": 3.84,
  "level": "Средний",
  "i_cvss": 7.5,
  "i_infr": 0.8,
  "i_at": 0.3,
  "i_imp": 0.34,
  "breakdown": {
    "k_value": 0.7, "l_value": 0.6, "p_value": 1.1,
    "e_value": 0.3, "h_value": 0.34,
    "k_term": 0.35, "l_term": 0.12, "p_term": 0.33
  },
  "cvss31_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
  "cvss31_severity": "High",
  "inference_time_ms": 215.4
}
```

Здесь `i_infr = 0.5·0.7 + 0.2·0.6 + 0.3·1.1 = 0.80`,
`V = 7.5 · 0.80 · (0.3 + 0.34) = 3.84` → уровень «Средний».

Уровни (Таблица 2 Методики): `V > 8.0` → Критический; `5.0 ≤ V ≤ 8.0` →
Высокий; `2.0 ≤ V < 5.0` → Средний; `V < 2.0` → Низкий.

### Status codes

| Код | Когда |
|:--|:--|
| `200 OK` | Успешный расчёт |
| `400 Bad Request` | Неизвестный код показателя или пустой обязательный мультивыбор |
| `422 Unprocessable Entity` | Невалидное тело (короткое описание, кривой CWE) |
| `500 Internal Server Error` | Внутренняя ошибка инференса |

---

## GET /fstec/options

Каталог показателей Таблицы 1 Методики ФСТЭК — для построения формы UI.
Модель не загружается (каталог статичен).

```bash
curl http://localhost:8000/fstec/options
```

```json
{
  "K": {"weight": 0.5, "multiselect": true,  "options": [{"code": "server", "label": "Серверы (центральные вычислительные узлы)", "value": 0.7}, ...]},
  "L": {"weight": 0.2, "multiselect": true,  "options": [...]},
  "P": {"weight": 0.3, "multiselect": false, "options": [...]},
  "E": {"weight": 1.0, "multiselect": true,  "options": [...]},
  "H": {"weight": 1.0, "multiselect": true,  "options": [...]}
}
```

---

## GET /fstec/suggest

Предзаполнение показателей `E` и `H` (пользователь правит вручную — итоговое
решение за специалистом, п.9 Методики). `E` — по флагам `kev`/`exploit`
(CISA KEV / ExploitDB), `H` — по типу CWE. Модель не загружается.

```bash
curl "http://localhost:8000/fstec/suggest?cwe_id=CWE-89&kev=false&exploit=true"
```

```json
{
  "e": {"codes": ["exploit_available"], "source": "exploit"},
  "h": {"codes": ["code_injection"],    "source": "cwe"}
}
```

---

## GET /cwe

Список CWE (id + имя) для выпадающего списка UI: те CWE, что известны модели
(из `cwe_vocab.json`), с человекочитаемыми именами MITRE, отсортированные по
номеру. Модель не загружается.

```bash
curl http://localhost:8000/cwe
```

```json
[
  {"id": "CWE-20", "name": "Некорректная проверка вводимых данных"},
  {"id": "CWE-79", "name": "Межсайтовое выполнение сценариев"},
  {"id": "CWE-89", "name": "Внедрение SQL-кода"}
]
```

Имя отдаётся на русском (выгрузка БДУ ФСТЭК), с фолбэком на английское название
MITRE, затем на сам идентификатор.

---

## GET /health

Проверка готовности сервиса. Используется для liveness/readiness-проб.

### Request

```bash
curl http://localhost:8000/health
```

### Response (готов)

```json
{"status": "ready", "model_loaded": true, "device": "cpu"}
```

### Response (загрузка / ошибка)

```json
{"status": "loading", "model_loaded": false, "device": "cpu"}
```

```json
{"status": "error", "model_loaded": false, "device": "<traceback>"}
```

### Status codes

`200 OK` — всегда (даже при ошибке загрузки модели, чтобы probe мог получить
информативное сообщение).

---

## GET /model/info

Сводка по обученной модели — для отображения в админ-панели и для прокидывания
в страницу «О системе».

### Request

```bash
curl http://localhost:8000/model/info
```

### Response

```json
{
  "model_name": "mBERT (bert-base-multilingual-cased) + DAPT + 12 heads",
  "training_completed": "2026-06-01",
  "num_parameters": 178358563,
  "test_metrics": {
    "aggregated": {
      "macro_f1": 0.7641,
      "vector_accuracy": 0.4763,
      "metrics_correct_avg": 9.63,
      "score_mae": 1.01,
      "score_rmse": 1.86,
      "severity_accuracy": 0.7130,
      "severity_within_one": 0.9486,
      "samples_evaluated": 972
    },
    "per_metric_f1": {
      "AV": 0.7626, "AC": 0.7842, "AT": 0.7651, "PR": 0.7314,
      "UI": 0.7131, "VC": 0.8244, "VI": 0.8471, "VA": 0.8351,
      "SC": 0.6458, "SI": 0.6928, "SA": 0.6650, "E":  0.9022
    }
  }
}
```

### Status codes

| Код | Когда |
|:--|:--|
| `200 OK` | Модель загружена |
| `503 Service Unavailable` | Модель ещё не готова |

---

## Ошибки и их обработка

### 422 Unprocessable Entity

Pydantic возвращает структурированное описание невалидных полей:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "hi", "cwe_id": "bad"}'
```

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "description"],
      "msg": "String should have at least 10 characters",
      "input": "hi",
      "ctx": {"min_length": 10}
    },
    {
      "type": "string_pattern_mismatch",
      "loc": ["body", "cwe_id"],
      "msg": "String should match pattern '^CWE-\\d+$'",
      "input": "bad",
      "ctx": {"pattern": "^CWE-\\d+$"}
    }
  ]
}
```

**Как обработать:** показать пользователю список конкретных полей с ошибками.
Frontend SDK может маппить `loc[-1]` на имя инпута.

### 500 Internal Server Error

Возникает при ошибках инференса (повреждённый чекпоинт, OOM, и т.п.).
В теле — текст исключения. На production стоит:

1. Залогировать `detail`;
2. Не показывать сырой traceback пользователю — заменить на «временно недоступно»;
3. Проверить `/health` для диагностики.

### Тайм-ауты

Инференс на CPU занимает ~200 мс для одного запроса. Для batch на 100 элементов —
~10–15 с. Если используете HTTP-клиента с тайм-аутом, установите ≥30 с для
`/predict/batch`.

---

## Ограничения

> Сервис предназначен для **демонстрации** и **внутреннего использования**.
> Для open API нужны дополнительные доработки.

| Ограничение | Деталь |
|:--|:--|
| **Нет аутентификации** | Все эндпоинты открыты. Для production добавьте API-key middleware или OAuth2. |
| **Нет rate limiting** | Никаких ограничений на частоту запросов. Можно поставить `slowapi` или nginx-level лимит. |
| **Batch ≤ 100 элементов** | Hardcoded в `PredictionRequest.items.max_length`. Меняется в `src/api/schemas.py`. |
| **CORS не настроен** | По умолчанию запросы только от того же origin. Для cross-domain добавьте `CORSMiddleware`. |
| **Нет CVE-enrichment** | По задаче ВКР: эндпоинт `/predict/cve` (обогащение через NVD API) намеренно не реализован — на защите интернет может отвалиться. |

---

## Swagger UI

Полная интерактивная документация со всеми схемами Pydantic и возможностью
отправлять запросы прямо из браузера:

**<http://localhost:8000/docs>**

Альтернативный формат (ReDoc):

**<http://localhost:8000/redoc>**

---

См. также:

- [docs/user_guide.md](user_guide.md) — руководство пользователя веб-интерфейса;
- [docs/architecture.md](architecture.md) — устройство модели за этим API;
- [src/api/schemas.py](../src/api/schemas.py) — исходные Pydantic-схемы.
