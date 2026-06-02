# Сквозная оценка CVSS v3.1 (stage 1 + Scope-голова)

> Оценён развёрнутый пайплайн через `VulnerabilityPredictorV31` (с подстановкой
> `cwe_name` по `cwe_id`). Истинный балл рассчитан из эталонного вектора калькулятором 3.1.

## Интегральные метрики

| Показатель | Значение |
|:-----------|:---------|
| Macro-F1 (8 метрик)        | 0.8137 |
| Vector accuracy            | 0.5229 |
| Метрик верно в среднем     | 7.07 / 8 |
| Score MAE                  | 0.7430 |
| Score RMSE                 | 1.3175 |
| Severity accuracy          | 0.7376 |
| Severity ±1 уровень        | 0.9698 |
| Размер test set            | 26317 записей |

## Per-metric качество

| Метрика | Полное название | F1 (macro) | Accuracy | Support |
|:--------|:----------------|-----------:|---------:|--------:|
| AV      | Attack Vector           |     0.7817 |   0.9352 |   26317 |
| AC      | Attack Complexity       |     0.7253 |   0.9197 |   26317 |
| PR      | Privileges Required     |     0.7609 |   0.8132 |   26317 |
| UI      | User Interaction        |     0.9063 |   0.9173 |   26317 |
| S       | Scope                   |     0.8469 |   0.9120 |   26317 |
| C       | Confidentiality Impact  |     0.8388 |   0.8510 |   26317 |
| I       | Integrity Impact        |     0.8502 |   0.8550 |   26317 |
| A       | Availability Impact     |     0.7994 |   0.8638 |   26317 |
