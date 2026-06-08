# Итоговые метрики качества (test set)

> Финальная модель — mBERT + DAPT (`models/final_model.pt`,
> `reports/test_evaluation.json`).

| Показатель                      | Значение              |
|:--------------------------------|:----------------------|
| Macro-F1 (12 метрик)            | 0.7641                |
| Vector Accuracy (11 метрик)     | 0.4763                |
| Среднее число правильных метрик | 9.63 / 11 (87.6%)     |
| MAE по CVSS-баллу               | 1.01 (10.1% шкалы)    |
| RMSE по CVSS-баллу              | 1.86                  |
| Severity Accuracy               | 0.7130                |
| Severity Within ±1              | 0.9486                |
| Размер test set                 | 972 CVSS v4.0 записей |
