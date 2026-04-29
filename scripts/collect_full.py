"""Полный сбор обучающего корпуса.

Реализует алгоритм раздела 2.3.1:
    1. Скачать БДУ ФСТЭК (vullist.xml) и распарсить.
    2. Bulk-выгрузить NVD за период [--start, --end] окнами по 120 дней.
    3. Обновить справочные каталоги: CISA KEV, ExploitDB, CWE MITRE.
    4. Получить EPSS-оценки батчами по 100 CVE.
    5. Объединить BDU↔NVD по cve_id, обогатить epss/kev/exploit/cwe_name.
    6. Сохранить data/raw/dataset.parquet.
    7. Стратифицированно разбить 70/15/15 → data/processed/{train,val,test}.parquet.

Запуск (с NVD_API_KEY в .env):
    python scripts/collect_full.py --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import click
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_collection import (  # noqa: E402
    BDUCollector,
    CWENames,
    EPSSCollector,
    ExploitDBCollector,
    KEVCollector,
    NVDCollector,
)
from src.data_collection._config import load_config  # noqa: E402
from src.data_collection._logging import get_logger, setup_logger  # noqa: E402
from src.data_collection.split_data import split_and_save  # noqa: E402

logger = get_logger("collect_full")

NVD_DT_FORMAT = "%Y-%m-%dT%H:%M:%S.000"
EPSS_BATCH_SIZE = 100


def date_windows(start_str: str, end_str: str, max_days: int = 120) -> Iterator[tuple[datetime, datetime]]:
    """NVD требует pubStartDate/pubEndDate с диапазоном ≤ 120 дней."""
    start = datetime.fromisoformat(start_str)
    end = datetime.fromisoformat(end_str)
    if end <= start:
        raise click.BadParameter("--end должен быть позже --start")
    cur = start
    while cur < end:
        win_end = min(cur + timedelta(days=max_days), end)
        yield cur, win_end
        cur = win_end + timedelta(seconds=1)


def chunked(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--start",
    default="2002-01-01",
    show_default=True,
    help="Дата начала NVD-периода (YYYY-MM-DD).",
)
@click.option(
    "--end",
    default=None,
    help="Дата конца NVD-периода (YYYY-MM-DD); по умолчанию — сегодня.",
)
@click.option(
    "--bdu-format",
    type=click.Choice(["xml", "xlsx"], case_sensitive=False),
    default="xml",
    show_default=True,
    help="Формат выгрузки БДУ ФСТЭК.",
)
@click.option(
    "--bdu-xml-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Путь к локально скачанному vullist.xml (использовать вместо HTTP-загрузки).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Ограничение числа CVE из NVD (для отладочных прогонов).",
)
@click.option(
    "--config",
    "config_path",
    default="configs/config.yaml",
    show_default=True,
    help="Путь к конфигу проекта.",
)
@click.option(
    "--skip-bdu",
    is_flag=True,
    default=False,
    help="Пропустить загрузку БДУ ФСТЭК (русские описания будут пустыми).",
)
@click.option(
    "--no-split",
    is_flag=True,
    default=False,
    help="Только собрать dataset.parquet, не делать train/val/test split.",
)
def main(
    start: str,
    end: str | None,
    bdu_format: str,
    bdu_xml_file: Path | None,
    limit: int | None,
    config_path: str,
    skip_bdu: bool,
    no_split: bool,
) -> None:
    setup_logger()
    cfg = load_config(config_path)
    sources = cfg["data_sources"]
    paths = cfg["paths"]
    cache_dir = paths.get("data_raw", "data/raw")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    start_ts = datetime.now()

    print("=" * 78)
    print(f"  Полный сбор данных: NVD период {start} .. {end}")
    print(f"  Формат БДУ: {bdu_format}; limit: {limit or '∞'}")
    print("=" * 78)

    # ------------------------------------------------------------- Шаг 1. БДУ
    bdu_by_cve: dict[str, dict] = {}
    bdu_total = 0
    if not skip_bdu:
        if bdu_xml_file is not None:
            size_mb = bdu_xml_file.stat().st_size / (1024 * 1024)
            print(f"\n[1/7] БДУ ФСТЭК — локальный XML: {bdu_xml_file} ({size_mb:.1f} МБ)")
        else:
            print("\n[1/7] БДУ ФСТЭК — полная выгрузка")
        bdu = BDUCollector(
            sources.get("bdu_fstec"),
            cache_dir=cache_dir,
            local_xml_path=bdu_xml_file,
        )
        if bdu_xml_file is not None or bdu_format.lower() == "xml":
            bdu_records = bdu.fetch_bulk_xml()
        else:
            bdu_records = bdu.fetch_bulk()
        bdu_total = len(bdu_records)
        for r in bdu_records:
            cid = r.get("cve_id")
            if cid:
                bdu_by_cve[cid] = r
        print(f"  записей БДУ: {bdu_total}, привязано к CVE: {len(bdu_by_cve)}")
    else:
        print("\n[1/7] БДУ — пропущено (--skip-bdu)")

    # ------------------------------------------------------------- Шаг 2. NVD
    print("\n[2/7] NVD — bulk-выгрузка по 120-дневным окнам")
    nvd = NVDCollector(sources.get("nvd"))
    nvd_pool: list[dict] = []
    windows = list(date_windows(start, end))
    for win_start, win_end in tqdm(windows, desc="NVD windows", unit="win"):
        try:
            for rec in nvd.fetch_bulk(
                start_date=win_start.strftime(NVD_DT_FORMAT),
                end_date=win_end.strftime(NVD_DT_FORMAT),
            ):
                if rec.get("id"):
                    nvd_pool.append(rec)
                if limit and len(nvd_pool) >= limit:
                    break
        except Exception as exc:  # noqa: BLE001 — одно битое окно не должно ронять весь сбор
            logger.exception("NVD-окно %s..%s упало: %s", win_start, win_end, exc)
        if limit and len(nvd_pool) >= limit:
            break
    if limit:
        nvd_pool = nvd_pool[:limit]
    nvd_by_cve = {r["id"]: r for r in nvd_pool}
    print(f"  CVE из NVD: {len(nvd_by_cve)}")

    # ------------------------------------------------- Шаг 3..5. Каталоги-set
    print("\n[3/7] CISA KEV")
    kev = KEVCollector(sources.get("cisa_kev"), cache_dir=cache_dir)
    kev.refresh()

    print("[4/7] ExploitDB")
    exploit = ExploitDBCollector(sources.get("exploit_db"), cache_dir=cache_dir)
    exploit.refresh()

    print("[5/7] CWE MITRE")
    cwe = CWENames(sources.get("cwe_mitre"), cache_dir=cache_dir)
    cwe_map = cwe.all()
    print(f"  CWE имён: {len(cwe_map)}")

    # ----------------------------------------------------- Шаг 6. EPSS батчи
    print("\n[6/7] EPSS — батчи по", EPSS_BATCH_SIZE, "CVE")
    all_cves = sorted(set(nvd_by_cve.keys()) | set(bdu_by_cve.keys()))
    epss = EPSSCollector(sources.get("epss"), cache_dir=cache_dir)
    epss_map: dict[str, float | None] = {}
    chunks = list(chunked(all_cves, EPSS_BATCH_SIZE))
    for chunk in tqdm(chunks, desc="EPSS", unit="batch"):
        epss_map.update(epss.fetch_batch(chunk))
    filled = sum(1 for v in epss_map.values() if v is not None)
    print(f"  EPSS: {filled}/{len(epss_map)} ({filled / max(len(epss_map), 1) * 100:.1f}%)")

    # ----------------------------------------------- Шаг 7. Сборка датасета
    print("\n[7/7] Объединение BDU↔NVD и формирование dataset.parquet")
    rows = []
    for cve_id in tqdm(all_cves, desc="merge", unit="cve"):
        nvd_rec = nvd_by_cve.get(cve_id, {})
        bdu_rec = bdu_by_cve.get(cve_id, {})
        cwe_id = bdu_rec.get("cwe_id") or nvd_rec.get("cwe_id")
        v3 = bdu_rec.get("cvss_v3_vector") or nvd_rec.get("cvss_v3_vector")
        v4 = bdu_rec.get("cvss_v4_vector") or nvd_rec.get("cvss_v4_vector")
        rows.append(
            {
                "id": bdu_rec.get("id") or cve_id,
                "cve_id": cve_id,
                "d_ru": bdu_rec.get("description_ru"),
                "d_en": nvd_rec.get("description_en"),
                "cwe_id": cwe_id,
                "cwe_name": cwe_map.get(cwe_id) if cwe_id else None,
                "epss": epss_map.get(cve_id),
                "kev": int(kev.is_in_kev(cve_id)),
                "exploit": int(exploit.has_exploit(cve_id)),
                "cvss_v3_vector": v3,
                "cvss_v4_vector": v4,
                "cvss_vector": v4 or v3,
            }
        )

    df = pd.DataFrame(rows).drop_duplicates("cve_id").reset_index(drop=True)
    out_path = Path(cache_dir) / "dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  dataset.parquet: {out_path} ({len(df)} строк)")

    # ------------------------------------------------------------------ Сводка
    elapsed = (datetime.now() - start_ts).total_seconds()
    print("\n" + "=" * 78)
    print(f"  Готово за {elapsed / 60:.1f} мин")
    print("=" * 78)
    print(f"  всего записей          : {len(df)}")
    print(f"  d_ru (БДУ описания)    : {df['d_ru'].notna().sum():>6}")
    print(f"  d_en (NVD описания)    : {df['d_en'].notna().sum():>6}")
    print(f"  cwe_id                 : {df['cwe_id'].notna().sum():>6}")
    print(f"  cwe_name               : {df['cwe_name'].notna().sum():>6}")
    print(f"  epss                   : {df['epss'].notna().sum():>6}")
    print(f"  kev = 1                : {(df['kev'] == 1).sum():>6}")
    print(f"  exploit = 1            : {(df['exploit'] == 1).sum():>6}")
    print(f"  cvss_v3_vector         : {df['cvss_v3_vector'].notna().sum():>6}")
    print(f"  cvss_v4_vector         : {df['cvss_v4_vector'].notna().sum():>6}")
    print(f"  cvss_vector (любой)    : {df['cvss_vector'].notna().sum():>6}")

    # ----------------------------------------------------- Стратифицированный split
    if no_split:
        print("\n  split пропущен (--no-split)")
        return

    print("\n  Стратифицированное разбиение 70/15/15 ...")
    targets = split_and_save(cfg, input_path=out_path)
    for name, path in targets.items():
        size = pd.read_parquet(path).shape[0]
        print(f"    {name:<5}: {path} ({size} строк)")


if __name__ == "__main__":
    main()
