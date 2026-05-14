# REST API

FastAPI веб-сервис предоставляет четыре endpoint-а и встроенный
интерактивный Swagger UI.

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
3. [GET /health](#get-health)
4. [GET /model/info](#get-modelinfo)
5. [Ошибки и их обработка](#ошибки-и-их-обработка)
6. [Ограничения](#ограничения)

---

## POST /predict

Предсказывает CVSS-вектор и итоговый балл для **одной** уязвимости.

### Request

`Content-Type: application/json`

| Поле | Тип | Обязательно | Ограничения |
|:--|:--|:--|:--|
| `description` | string | Да | min_length=10, max_length=10000 |
| `cwe_id` | string | Да | regex `^CWE-\d+$` (например, `CWE-89`) |
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
| `vector` | Каноническая CVSS-строка `CVSS:4.0/AV:N/.../E:A` |
| `score` | Итоговый балл 0,0–10,0 (рассчитан собственным калькулятором) |
| `severity` | None / Low / Medium / High / Critical |
| `metrics` | Словарь из 12 метрик с `value` и `confidence` |
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
  "model_name": "mBERT (bert-base-multilingual-cased) + 12 heads",
  "training_completed": "2026-05-11",
  "num_parameters": 178358563,
  "test_metrics": {
    "aggregated": {
      "macro_f1": 0.7090,
      "vector_accuracy": 0.3992,
      "metrics_correct_avg": 9.39,
      "score_mae": 1.17,
      "score_rmse": 1.98,
      "severity_accuracy": 0.6739,
      "severity_within_one": 0.9208,
      "samples_evaluated": 972
    },
    "per_metric_f1": {
      "AV": 0.5175, "AC": 0.7482, "AT": 0.7433, "PR": 0.6308,
      "UI": 0.6414, "VC": 0.7886, "VI": 0.8198, "VA": 0.7946,
      "SC": 0.6228, "SI": 0.6705, "SA": 0.656,  "E":  0.8751
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
