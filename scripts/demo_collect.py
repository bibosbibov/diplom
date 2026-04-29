"""Demo-сбор: 100 случайных CVE из NVD за июнь 2024.

Делает один пагинированный запрос к NVD за период, рандомно отбирает 100
записей, обогащает их EPSS / KEV / ExploitDB / CWE, сохраняет dataset.parquet
и выполняет стратифицированное разбиение 70/15/15.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_collection import (  # noqa: E402
    CWENames,
    EPSSCollector,
    ExploitDBCollector,
    KEVCollector,
    NVDCollector,
)
from src.data_collection._config import load_config  # noqa: E402
from src.data_collection._logging import setup_logger  # noqa: E402
from src.data_collection.split_data import split_dataset  # noqa: E402


def main() -> None:
    setup_logger()
    config = load_config()
    random.seed(42)

    sources = config.get("data_sources", {})
    paths = config.get("paths", {})
    cache_dir = paths.get("data_raw", "data/raw")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # 1. Bulk-выгрузка из NVD за неделю июня 2024
    nvd = NVDCollector(sources.get("nvd"))
    print("=" * 70)
    print("Шаг 1/6. Запрос NVD bulk: pubDate 2024-06-01 .. 2024-06-08")
    print("=" * 70)
    pool: list[dict] = []
    for rec in nvd.fetch_bulk(
        start_date="2024-06-01T00:00:00.000",
        end_date="2024-06-08T00:00:00.000",
    ):
        if rec.get("id"):
            pool.append(rec)
        if len(pool) >= 600:
            break
    print(f"  Получено {len(pool)} записей из NVD")

    if not pool:
        sys.exit("Не удалось получить данные NVD")

    # 2. Случайные 100
    sampled = random.sample(pool, min(100, len(pool)))
    cve_ids = [r["id"] for r in sampled]
    print(f"\nШаг 2/6. Случайная выборка: {len(sampled)} CVE")

    # 3. Обогащение
    epss = EPSSCollector(sources.get("epss"), cache_dir=cache_dir)
    kev = KEVCollector(sources.get("cisa_kev"), cache_dir=cache_dir)
    exploit = ExploitDBCollector(sources.get("exploit_db"), cache_dir=cache_dir)
    cwe = CWENames(sources.get("cwe_mitre"), cache_dir=cache_dir)

    print("\nШаг 3/6. EPSS batch ...")
    epss_map = epss.fetch_batch(cve_ids)
    print(f"  EPSS: {sum(v is not None for v in epss_map.values())} / {len(epss_map)}")

    print("\nШаг 4/6. CISA KEV / ExploitDB / CWE ...")
    kev.refresh()
    print("  KEV catalog загружен")
    exploit.refresh()
    print("  ExploitDB CSV загружен")
    cwe_full = cwe.all()
    print(f"  CWE: {len(cwe_full)} имён")

    # 4. Сборка записей
    print("\nШаг 5/6. Формирование датасета ...")
    rows = []
    for rec in sampled:
        cid = rec["id"]
        cwe_id = rec.get("cwe_id")
        rows.append(
            {
                "id": cid,
                "cve_id": cid,
                "d_ru": None,
                "d_en": rec.get("description_en"),
                "cwe_id": cwe_id,
                "cwe_name": cwe.get(cwe_id) if cwe_id else None,
                "epss": epss_map.get(cid),
                "kev": int(kev.is_in_kev(cid)),
                "exploit": int(exploit.has_exploit(cid)),
                "cvss_v3_vector": rec.get("cvss_v3_vector"),
                "cvss_v4_vector": rec.get("cvss_v4_vector"),
                "cvss_vector": rec.get("cvss_v4_vector") or rec.get("cvss_v3_vector"),
            }
        )

    df = pd.DataFrame(rows)
    out_path = Path(cache_dir) / "dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Сохранено: {out_path} ({len(df)} строк, {len(df.columns)} колонок)")

    # 5. Анализ заполненности
    print("\n=== Заполненность полей ===")
    for col in df.columns:
        if df[col].dtype == object:
            filled = df[col].notna().sum()
        else:
            filled = (df[col].notna() & (df[col] != 0)).sum() if col in ("kev", "exploit") else df[col].notna().sum()
        pct = filled / len(df) * 100
        print(f"  {col:<22} {filled:>3}/{len(df)} ({pct:5.1f}%)")

    # 6. Три примера
    print("\n=== 3 примера записей ===")
    examples = df.sample(3, random_state=7).reset_index(drop=True)
    for i, row in examples.iterrows():
        print(f"\n--- Пример #{i + 1}: {row['cve_id']} ---")
        for k, v in row.items():
            if v is None or (isinstance(v, float) and pd.isna(v)):
                v_str = "None"
            elif isinstance(v, str) and len(v) > 110:
                v_str = v[:110] + "..."
            else:
                v_str = str(v)
            print(f"  {k:<22}: {v_str}")

    # 7. Стратифицированное разбиение
    print("\n" + "=" * 70)
    print("Шаг 6/6. Стратифицированное разбиение 70/15/15")
    print("=" * 70)
    parts = split_dataset(df, seed=42)
    total = sum(len(p) for p in parts.values())
    print(f"  Всего после дедупликации по cve_id: {total}")
    for name, frame in parts.items():
        print(f"  {name:<5}: {len(frame):>3} ({len(frame) / total * 100:5.1f}%)")

    # 8. Проверка отсутствия утечки + сохранение
    train_ids = set(parts["train"]["cve_id"])
    val_ids = set(parts["val"]["cve_id"])
    test_ids = set(parts["test"]["cve_id"])
    leak = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    print(f"\n  Пересечений по cve_id между сплитами: {len(leak)}")

    processed = Path(paths.get("data_processed", "data/processed"))
    processed.mkdir(parents=True, exist_ok=True)
    for name, frame in parts.items():
        target = processed / f"{name}.parquet"
        frame.to_parquet(target, index=False)
        print(f"  Сохранён: {target}")

    # 9. Распределение AV в каждом сплите
    print("\n=== Распределение AV в сплитах (стратификация) ===")
    import re

    av_re = re.compile(r"AV:([NALP])", re.IGNORECASE)
    for name, frame in parts.items():
        av_counts = (
            frame["cvss_vector"]
            .fillna("")
            .map(lambda s: (av_re.search(s).group(1).upper() if av_re.search(s) else "UNK"))
            .value_counts()
            .to_dict()
        )
        print(f"  {name:<5}: {av_counts}")


if __name__ == "__main__":
    main()
