# Показательные примеры предсказаний (приложение к ВКР)

## Пример 1. Идеальное предсказание (11/11 метрик)

- **CVE:** CVE-2024-6115
- **CWE:** CWE-434
- **Severity:** true = Medium, pred = Medium
- **Score:** true = 6.9, pred = 6.9
- **Метрик правильно:** 11 / 11
- **Средняя уверенность:** 0.955

**Описание:**

> A vulnerability classified as critical was found in itsourcecode Simple Online Hotel Reservation System 1.0. Affected by this vulnerability is an unknown functionality of the file add_room.php. The manipulation of the argument photo leads to unrestricted upload. The attack can be launched remotely. The exploit has been disclosed to the public and may be used. The associated identifier of this vulnerability is VDB-268867.

**True vector:** `CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:L/VI:L/VA:L/SC:N/SI:N/SA:N/E:X`  
**Pred vector:** `CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:L/VI:L/VA:L/SC:N/SI:N/SA:N/E:A`

**Ошибок нет** — модель попала во все 11 базовых метрик.

**Низкая уверенность** (<0.7): PR

---

## Пример 2. Идеальное предсказание (11/11 метрик)

- **CVE:** CVE-2024-12928
- **CWE:** CWE-74
- **Severity:** true = Medium, pred = Medium
- **Score:** true = 5.3, pred = 5.3
- **Метрик правильно:** 11 / 11
- **Средняя уверенность:** 0.961

**Описание:**

> A vulnerability, which was classified as critical, was found in code-projects Simple Admin Panel 1.0. This affects an unknown part. The manipulation of the argument c_name leads to sql injection. It is possible to initiate the attack remotely. The exploit has been disclosed to the public and may be used.

**True vector:** `CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:L/VI:L/VA:L/SC:N/SI:N/SA:N/E:X`  
**Pred vector:** `CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:L/VI:L/VA:L/SC:N/SI:N/SA:N/E:A`

**Ошибок нет** — модель попала во все 11 базовых метрик.

**Низкая уверенность** (<0.7): PR

---

## Пример 3. Частичная ошибка (8–9 / 11 метрик)

- **CVE:** CVE-2025-9235
- **CWE:** CWE-79
- **Severity:** true = Low, pred = Medium
- **Score:** true = 2.0, pred = 5.5
- **Метрик правильно:** 9 / 11
- **Средняя уверенность:** 0.841

**Описание:**

> Уязвимость модуля Compound Events многоплатформенного веб-решения для создания Scada-систем Scada-LTS связана с непринятием мер по защите структуры веб-страницы при обработке поля Name. Эксплуатация уязвимости может позволить нарушителю, действующему удалённо, проводить межсайтовые сценарные атаки

**True vector:** `AV:N/AC:L/AT:N/PR:L/UI:P/VC:N/VI:L/VA:N/SC:N/SI:N/SA:N/E:P`  
**Pred vector:** `CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:L/VA:N/SC:N/SI:N/SA:N/E:P`

**Ошибки модели:**
  - **PR:** true=`L`, pred=`N` (confidence=0.526)
  - **UI:** true=`P`, pred=`N` (confidence=0.562)

**Низкая уверенность** (<0.7): PR, UI, VI

---

## Пример 4. Частичная ошибка (8–9 / 11 метрик)

- **CVE:** CVE-2026-21989
- **CWE:** CWE-20
- **Severity:** true = High, pred = None
- **Score:** true = 8.3, pred = 0.0
- **Метрик правильно:** 8 / 11
- **Средняя уверенность:** 0.744

**Описание:**

> Уязвимость компонента Core виртуальной машины Oracle VM VirtualBox связана с недостаточной проверкой входных данных. Эксплуатация уязвимости может позволить нарушителю получить доступ на чтение, изменение или удаление защищаемой информации

**True vector:** `AV:L/AC:L/AT:N/PR:H/UI:N/VC:H/VI:H/VA:L/SC:N/SI:N/SA:N`  
**Pred vector:** `CVSS:4.0/AV:L/AC:L/AT:N/PR:H/UI:N/VC:N/VI:N/VA:N/SC:N/SI:N/SA:N/E:U`

**Ошибки модели:**
  - **VC:** true=`H`, pred=`N` (confidence=0.486)
  - **VI:** true=`H`, pred=`N` (confidence=0.756)
  - **VA:** true=`L`, pred=`N` (confidence=0.710)

**Низкая уверенность** (<0.7): PR, VC, SC, SI

---

## Пример 5. Грубая ошибка (< 6 / 11 метрик)

- **CVE:** CVE-2024-52276
- **CWE:** CWE-451
- **Severity:** true = High, pred = High
- **Score:** true = 8.2, pred = 7.1
- **Метрик правильно:** 5 / 11
- **Средняя уверенность:** 0.604

**Описание:**

> User Interface (UI) Misrepresentation of Critical Information vulnerability in DocuSign allows Content Spoofing.
1. Displayed version does not show the layer flattened version, which is provided when the "Print" option is used.
2. Displayed version does not show the layer flattened version, which is provided when the combined download option is used.
3. Displayed version does not show the layer flattened version, which is also the provided version when downloading the result in the uncombined option.
Once download, If printed (e.g. via Google Chrome -> Examine the print preview): Will render t…

**True vector:** `CVSS:4.0/AV:L/AC:L/AT:N/PR:N/UI:P/VC:N/VI:H/VA:N/SC:N/SI:H/SA:N/E:X`  
**Pred vector:** `CVSS:4.0/AV:N/AC:L/AT:P/PR:N/UI:P/VC:N/VI:L/VA:H/SC:N/SI:L/SA:H/E:A`

**Ошибки модели:**
  - **AV:** true=`L`, pred=`N` (confidence=0.657)
  - **AT:** true=`N`, pred=`P` (confidence=0.901)
  - **VI:** true=`H`, pred=`L` (confidence=0.459)
  - **VA:** true=`N`, pred=`H` (confidence=0.428)
  - **SI:** true=`H`, pred=`L` (confidence=0.439)
  - **SA:** true=`N`, pred=`H` (confidence=0.441)

**Низкая уверенность** (<0.7): AV, AC, PR, UI, VI, VA, SC, SI, SA

---
