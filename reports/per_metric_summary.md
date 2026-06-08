# Per-metric F1 / Accuracy на test set

> Финальная модель — mBERT + DAPT (`reports/test_evaluation.json`).

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
