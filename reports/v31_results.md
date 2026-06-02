# Сквозная оценка CVSS v3.1 (stage 1 + Scope-голова)

> Оценён развёрнутый пайплайн (через `VulnerabilityPredictorV31`, текст без
> `cwe_name`). Истинный балл рассчитан из эталонного вектора калькулятором 3.1.

## Интегральные метрики

| Показатель | Значение |
|:-----------|:---------|
| Macro-F1 (8 метрик)        | 0.8000 |
| Vector accuracy            | 0.5008 |
| Метрик верно в среднем     | 7.00 / 8 |
| Score MAE                  | 0.7885 |
| Score RMSE                 | 1.3676 |
| Severity accuracy          | 0.7234 |
| Severity ±1 уровень        | 0.9668 |
| Размер test set            | 26317 записей |

## Per-metric качество

| Метрика | Полное название | F1 (macro) | Accuracy | Support |
|:--------|:----------------|-----------:|---------:|--------:|
| AV      | Attack Vector           |     0.7707 |   0.9333 |   26317 |
| AC      | Attack Complexity       |     0.7097 |   0.9165 |   26317 |
| PR      | Privileges Required     |     0.7502 |   0.8060 |   26317 |
| UI      | User Interaction        |     0.8953 |   0.9084 |   26317 |
| S       | Scope                   |     0.8295 |   0.9049 |   26317 |
| C       | Confidentiality Impact  |     0.8234 |   0.8369 |   26317 |
| I       | Integrity Impact        |     0.8356 |   0.8409 |   26317 |
| A       | Availability Impact     |     0.7854 |   0.8516 |   26317 |
