# Результаты оценки этапа 1 (CVSS v3.1)


## Интегральные метрики


| Показатель                | Значение                |
|:--------------------------|:------------------------|
| Macro-F1 (8 метрик)       | 0.8077                  |
| Размер test set           | 26317 CVSS v3.x записей |

## Per-metric качество (test set)


| Метрика   | Полное название         |   F1 (macro) |   Accuracy |   Support |
|:----------|:------------------------|-------------:|-----------:|----------:|
| AV        | Attack Vector           |       0.7839 |     0.9358 |     26317 |
| AC        | Attack Complexity       |       0.7256 |     0.9168 |     26317 |
| PR        | Privileges Required     |       0.7526 |     0.8056 |     26317 |
| UI        | User Interaction        |       0.9037 |     0.9147 |     26317 |
| VC        | Confidentiality Impact  |       0.8393 |     0.8511 |     26317 |
| VI        | Integrity Impact        |       0.8529 |     0.8574 |     26317 |
| VA        | Availability Impact     |       0.7957 |     0.8623 |     26317 |
| E         | Exploit Code Maturity   |       0.0000 |     0.0000 |         0 |
