# Итоговые результаты экспериментов


## Интегральные метрики качества


| Показатель                      | Значение              |
|:--------------------------------|:----------------------|
| Macro-F1 (12 метрик)            | 0.7090                |
| Vector Accuracy (11 метрик)     | 0.3992                |
| Среднее число правильных метрик | 9.39 / 11 (85.4%)     |
| MAE по CVSS-баллу               | 1.17 (11.7% шкалы)    |
| RMSE по CVSS-баллу              | 1.98                  |
| Severity Accuracy               | 0.6739                |
| Severity Within ±1              | 0.9208                |
| Размер test set                 | 972 CVSS v4.0 записей |

## Per-metric качество (test set)


| Метрика   | Полное название                   |   F1 (macro) |   Accuracy |   Support |
|:----------|:----------------------------------|-------------:|-----------:|----------:|
| AV        | Attack Vector                     |       0.5175 |     0.9023 |       972 |
| AC        | Attack Complexity                 |       0.7482 |     0.93   |       972 |
| AT        | Attack Requirements               |       0.7433 |     0.8837 |       972 |
| PR        | Privileges Required               |       0.6308 |     0.7212 |       972 |
| UI        | User Interaction                  |       0.6414 |     0.8827 |       972 |
| VC        | Vulnerable System Confidentiality |       0.7886 |     0.7942 |       972 |
| VI        | Vulnerable System Integrity       |       0.8198 |     0.8241 |       972 |
| VA        | Vulnerable System Availability    |       0.7946 |     0.7994 |       972 |
| SC        | Subsequent System Confidentiality |       0.6228 |     0.8704 |       972 |
| SI        | Subsequent System Integrity       |       0.6705 |     0.8909 |       972 |
| SA        | Subsequent System Availability    |       0.656  |     0.892  |       972 |
| E         | Exploit Maturity                  |       0.8751 |     0.9512 |       553 |

## Стабильность: val vs test


| Метрика   |   F1 val (epoch 18) |   F1 test |      Δ |
|:----------|--------------------:|----------:|-------:|
| AV        |               0.534 |     0.517 | -0.017 |
| AC        |               0.724 |     0.748 |  0.024 |
| AT        |               0.744 |     0.743 | -0.001 |
| PR        |               0.629 |     0.631 |  0.002 |
| UI        |               0.629 |     0.641 |  0.012 |
| VC        |               0.805 |     0.789 | -0.016 |
| VI        |               0.813 |     0.82  |  0.007 |
| VA        |               0.823 |     0.795 | -0.028 |
| SC        |               0.648 |     0.623 | -0.025 |
| SI        |               0.635 |     0.67  |  0.035 |
| SA        |               0.66  |     0.656 | -0.004 |
| E         |               0.881 |     0.875 | -0.006 |