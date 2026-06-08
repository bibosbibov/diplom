# Итоговые результаты экспериментов

> Финальная модель — **mBERT + DAPT** (`models/final_model.pt`). Метрики из
> `reports/test_evaluation.json`. Сравнение с baseline/майской моделью —
> `reports/dapt_experiment/chapter3_summary.md`.

## Интегральные метрики качества


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

## Per-metric качество (test set)


| Метрика   | Полное название                   |   F1 (macro) |   Accuracy |   Support |
|:----------|:----------------------------------|-------------:|-----------:|----------:|
| AV        | Attack Vector                     |       0.7626 |     0.9290 |       972 |
| AC        | Attack Complexity                 |       0.7842 |     0.9239 |       972 |
| AT        | Attack Requirements               |       0.7651 |     0.9012 |       972 |
| PR        | Privileges Required               |       0.7314 |     0.7860 |       972 |
| UI        | User Interaction                  |       0.7131 |     0.8889 |       972 |
| VC        | Vulnerable System Confidentiality |       0.8244 |     0.8292 |       972 |
| VI        | Vulnerable System Integrity       |       0.8471 |     0.8508 |       972 |
| VA        | Vulnerable System Availability    |       0.8351 |     0.8405 |       972 |
| SC        | Subsequent System Confidentiality |       0.6458 |     0.8827 |       972 |
| SI        | Subsequent System Integrity       |       0.6928 |     0.9023 |       972 |
| SA        | Subsequent System Availability    |       0.6650 |     0.8971 |       972 |
| E         | Exploit Maturity                  |       0.9022 |     0.9566 |       553 |

## Стабильность: val vs test


| Метрика   |   F1 val (epoch 19) |   F1 test |      Δ |
|:----------|--------------------:|----------:|-------:|
| AV        |               0.787 |     0.763 | -0.025 |
| AC        |               0.765 |     0.784 |  0.019 |
| AT        |               0.749 |     0.765 |  0.016 |
| PR        |               0.745 |     0.731 | -0.013 |
| UI        |               0.756 |     0.713 | -0.043 |
| VC        |               0.836 |     0.824 | -0.012 |
| VI        |               0.823 |     0.847 |  0.024 |
| VA        |               0.826 |     0.835 |  0.009 |
| SC        |               0.619 |     0.646 |  0.027 |
| SI        |               0.645 |     0.693 |  0.048 |
| SA        |               0.668 |     0.665 | -0.003 |
| E         |               0.869 |     0.902 |  0.033 |
