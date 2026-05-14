"""CLI поверх :class:`VulnerabilityPredictor`.

Запуск::

    python -m src.inference.cli predict --description "..." --cwe CWE-79
    python -m src.inference.cli batch-predict input.csv output.csv
    python -m src.inference.cli evaluate data/processed/test.parquet --limit 100
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from .predictor import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CWE_VOCAB_PATH,
    DEFAULT_MODEL_PATH,
    VulnerabilityPredictor,
)

console = Console()

SEVERITY_STYLE = {
    "Critical": "bold red",
    "High": "bold orange3",
    "Medium": "yellow",
    "Low": "green",
    "None": "dim",
}


# ----------------------------------------------------------------- shared opts
def _shared_model_options(f):
    f = click.option("--model-path", default=DEFAULT_MODEL_PATH, show_default=True)(f)
    f = click.option("--config-path", default=DEFAULT_CONFIG_PATH, show_default=True)(f)
    f = click.option("--cwe-vocab-path", default=DEFAULT_CWE_VOCAB_PATH, show_default=True)(f)
    f = click.option(
        "--device", default="auto", show_default=True, type=click.Choice(["auto", "cuda", "cpu"])
    )(f)
    f = click.option(
        "--threshold", default=0.7, show_default=True, type=float, help="Порог низкой уверенности"
    )(f)
    return f


def _build_predictor(
    model_path: str, config_path: str, cwe_vocab_path: str, device: str, threshold: float
) -> VulnerabilityPredictor:
    with console.status("[bold cyan]Загрузка модели...", spinner="dots"):
        predictor = VulnerabilityPredictor(
            model_path=model_path,
            config_path=config_path,
            device=device,
            confidence_threshold=threshold,
            cwe_vocab_path=cwe_vocab_path,
        )
    console.print(f"[green]Модель загружена на устройство [bold]{predictor.device}[/]")
    return predictor


def _severity_text(severity: str) -> Text:
    return Text(severity, style=SEVERITY_STYLE.get(severity, "white"))


# ------------------------------------------------------------------ predict
@click.group()
def cli() -> None:
    """CLI для системы автоматической оценки CVSS v4.0."""


@cli.command("predict")
@click.option("--description", required=True, help="Описание уязвимости (auto-detect языка).")
@click.option(
    "--description-ru",
    default=None,
    help="Отдельное русское описание, если в --description английское.",
)
@click.option("--cwe", "cwe_id", required=True, help="CWE-идентификатор (например CWE-79).")
@click.option("--epss", type=float, default=None, help="EPSS-вероятность (0..1).")
@click.option("--kev", is_flag=True, default=False, help="Признак: в CISA KEV.")
@click.option("--exploit", is_flag=True, default=False, help="Признак: есть публичный эксплойт.")
@click.option(
    "--no-kev", "no_kev", is_flag=True, default=False, help="Признак: не отмечен в kev (явно)."
)
@click.option(
    "--no-exploit",
    "no_exploit",
    is_flag=True,
    default=False,
    help="Признак: эксплойт отсутствует (явно).",
)
@_shared_model_options
def predict_cmd(
    description: str,
    description_ru: str | None,
    cwe_id: str,
    epss: float | None,
    kev: bool,
    exploit: bool,
    no_kev: bool,
    no_exploit: bool,
    model_path: str,
    config_path: str,
    cwe_vocab_path: str,
    device: str,
    threshold: float,
) -> None:
    """Однократное предсказание с красивым выводом."""
    kev_val: int | None
    if kev:
        kev_val = 1
    elif no_kev:
        kev_val = 0
    else:
        kev_val = None

    exploit_val: int | None
    if exploit:
        exploit_val = 1
    elif no_exploit:
        exploit_val = 0
    else:
        exploit_val = None

    predictor = _build_predictor(model_path, config_path, cwe_vocab_path, device, threshold)
    result = predictor.predict(
        description=description,
        description_ru=description_ru,
        cwe_id=cwe_id,
        epss=epss,
        kev=kev_val,
        exploit=exploit_val,
    )
    _render_predict_result(result, threshold)


def _render_predict_result(result: dict[str, Any], threshold: float) -> None:
    console.rule("[bold]Результат предсказания CVSS v4.0")
    console.print(f"[bold]Вектор:[/] {result['vector']}")
    console.print(f"[bold]Балл:[/] {result['score']:.1f}    [bold]Severity:[/] ", end="")
    console.print(_severity_text(result["severity"]))

    table = Table(title="Метрики", show_header=True, header_style="bold")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значение", style="white")
    table.add_column("Уверенность", justify="right")
    for metric, value in result["metrics"].items():
        conf = result["confidence"][metric]
        conf_style = "red" if conf < threshold else "green"
        table.add_row(metric, value, Text(f"{conf:.3f}", style=conf_style))
    console.print(table)

    if result["low_confidence_metrics"]:
        console.print(
            f"[yellow]Низкая уверенность ({threshold:.2f}) у метрик: "
            f"{', '.join(result['low_confidence_metrics'])}[/]"
        )


# ------------------------------------------------------------- batch predict
@cli.command("batch-predict")
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_csv", type=click.Path(dir_okay=False))
@click.option("--description-col", default="description", show_default=True)
@click.option("--description-ru-col", default="description_ru", show_default=True)
@click.option("--cwe-col", default="cwe_id", show_default=True)
@click.option("--epss-col", default="epss", show_default=True)
@click.option("--kev-col", default="kev", show_default=True)
@click.option("--exploit-col", default="exploit", show_default=True)
@click.option("--batch-size", default=16, show_default=True, type=int)
@_shared_model_options
def batch_predict_cmd(
    input_csv: str,
    output_csv: str,
    description_col: str,
    description_ru_col: str,
    cwe_col: str,
    epss_col: str,
    kev_col: str,
    exploit_col: str,
    batch_size: int,
    model_path: str,
    config_path: str,
    cwe_vocab_path: str,
    device: str,
    threshold: float,
) -> None:
    """Пакетная обработка CSV-файла."""
    df = pd.read_csv(input_csv)
    console.print(f"[cyan]Загружено [bold]{len(df)}[/] записей из {input_csv}[/]")

    items = [
        _row_to_item(
            row, description_col, description_ru_col, cwe_col, epss_col, kev_col, exploit_col
        )
        for _, row in df.iterrows()
    ]

    predictor = _build_predictor(model_path, config_path, cwe_vocab_path, device, threshold)

    results: list[dict[str, Any]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Предсказание...", total=len(items))
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            results.extend(predictor.predict_batch(chunk, batch_size=batch_size))
            progress.update(task, advance=len(chunk))

    df = df.copy()
    df["vector"] = [r["vector"] for r in results]
    df["score"] = [r["score"] for r in results]
    df["severity"] = [r["severity"] for r in results]
    df["confidence_avg"] = [sum(r["confidence"].values()) / len(r["confidence"]) for r in results]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    console.print(f"[green]Сохранено в {output_csv}[/]")


def _row_to_item(
    row: pd.Series,
    description_col: str,
    description_ru_col: str,
    cwe_col: str,
    epss_col: str,
    kev_col: str,
    exploit_col: str,
) -> dict[str, Any]:
    def _get(col: str) -> Any:
        if col not in row:
            return None
        value = row[col]
        if pd.isna(value):
            return None
        return value

    return {
        "description": _get(description_col) or _get("description_en") or _get("d_en"),
        "description_ru": _get(description_ru_col) or _get("d_ru"),
        "cwe_id": _get(cwe_col),
        "epss": _get(epss_col),
        "kev": _get(kev_col),
        "exploit": _get(exploit_col),
    }


# -------------------------------------------------------------------- evaluate
@cli.command("evaluate")
@click.argument("test_parquet", type=click.Path(exists=True, dir_okay=False))
@click.option("--limit", default=20, show_default=True, type=int, help="Сколько примеров показать.")
@click.option("--batch-size", default=16, show_default=True, type=int)
@_shared_model_options
def evaluate_cmd(
    test_parquet: str,
    limit: int,
    batch_size: int,
    model_path: str,
    config_path: str,
    cwe_vocab_path: str,
    device: str,
    threshold: float,
) -> None:
    """Сравнительная таблица true vs predicted на тестовом parquet."""
    df = pd.read_parquet(test_parquet)
    df = df[df["cvss_v4_vector"].notna()].head(limit).reset_index(drop=True)
    console.print(f"[cyan]Загружено [bold]{len(df)}[/] примеров с CVSS v4 из {test_parquet}[/]")

    items: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        items.append(
            {
                "description": row.get("d_en") if pd.notna(row.get("d_en")) else None,
                "description_ru": row.get("d_ru") if pd.notna(row.get("d_ru")) else None,
                "cwe_id": row.get("cwe_id"),
                "epss": row.get("epss"),
                "kev": row.get("kev"),
                "exploit": row.get("exploit"),
            }
        )

    predictor = _build_predictor(model_path, config_path, cwe_vocab_path, device, threshold)
    results = predictor.predict_batch(items, batch_size=batch_size)

    table = Table(title="True vs Predicted (CVSS v4.0)", show_lines=True)
    table.add_column("CVE", style="cyan", no_wrap=True)
    table.add_column("True vector")
    table.add_column("Predicted vector")
    table.add_column("Match", justify="right")

    total_correct = 0
    total_metrics = 0
    for (_, row), result in zip(df.iterrows(), results):
        true_vec = str(row["cvss_v4_vector"])
        pred_vec = result["vector"]
        true_metrics = predictor._calculator.parse_vector_string(true_vec)
        pred_metrics = result["metrics"]
        compared = [m for m in pred_metrics if m in true_metrics]
        matches = sum(1 for m in compared if true_metrics[m] == pred_metrics[m])
        total_correct += matches
        total_metrics += len(compared)
        cve_id = row.get("cve_id") or row.get("id") or "?"
        table.add_row(
            str(cve_id),
            _trim_vector(true_vec),
            _trim_vector(pred_vec),
            f"{matches}/{len(compared)}",
        )
    console.print(table)
    if total_metrics:
        console.print(
            f"[bold]Суммарная точность по метрикам:[/] "
            f"{total_correct}/{total_metrics} = "
            f"{total_correct / total_metrics:.1%}"
        )


def _trim_vector(vec: str) -> str:
    # CVSS:4.0 хвост с E:X/CR:X/... отбрасываем для читаемости.
    parts = vec.split("/")
    keep: list[str] = []
    for p in parts:
        if ":X" in p and p.split(":")[0] not in {"CVSS"}:
            continue
        keep.append(p)
    return "/".join(keep)


if __name__ == "__main__":
    cli()
