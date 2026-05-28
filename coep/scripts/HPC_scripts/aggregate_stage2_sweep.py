#!/usr/bin/env python3
"""Aggregate Stage 2 sweep batches into publication-friendly summaries."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import polars as pl
import yaml


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


ALL_JURISDICTIONS_DIR_NAME = "all_jurisdictions"
BATCHES_DIR_NAME = "batches"
SWEEPS_DIR_NAME = "sweeps"
TIMESTAMP_DIR_PATTERN = re.compile(r"^\d{8}_\d{6}$")
STAGE2_BATCH_PATTERN = re.compile(
    r"_run(?P<run>\d+)_n(?P<n_results>\d+)_hyde(?P<hyde>[01])_thr(?P<threshold>\d{2})$"
)
STAGE2_OVERRIDE_PATTERN = re.compile(
    r"run(?P<run>\d+)_n(?P<n_results>\d+)_hyde(?P<hyde>[01])_thr(?P<threshold>\d{2})\.ya?ml$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Stage 2 sweep batches into publication-friendly per-batch summaries"
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument("--sweep-root", type=Path, default=None)
    parser.add_argument("--batch-id", action="append", default=[])
    return parser.parse_args()


def _resolve_from_project_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def resolve_output_dir(override: Path | None) -> Path:
    if override:
        return _resolve_from_project_root(override)
    try:
        from legiscope import config

        return _resolve_from_project_root(config.output_dir())
    except Exception:
        return project_root / "data" / "output"


def sanitize_batch_id(batch_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", batch_id.strip())
    normalized = normalized.strip("-._")
    if not normalized:
        raise ValueError("batch_id must contain at least one filename-safe character")
    return normalized


def batches_root(output_dir: Path) -> Path:
    return output_dir / ALL_JURISDICTIONS_DIR_NAME / BATCHES_DIR_NAME


def default_sweep_root(output_dir: Path) -> Path:
    return output_dir / ALL_JURISDICTIONS_DIR_NAME / SWEEPS_DIR_NAME


def batch_dir(output_dir: Path, batch_id: str) -> Path:
    return batches_root(output_dir) / sanitize_batch_id(batch_id)


def latest_batch_run_dir(batch_root: Path) -> Path | None:
    candidates = [
        child
        for child in batch_root.iterdir()
        if child.is_dir() and TIMESTAMP_DIR_PATTERN.match(child.name)
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def bool_to_label(value: Any) -> str | None:
    if value is None:
        return None
    return "on" if bool(value) else "off"


def parse_stage2_condition_from_name(name: str) -> dict[str, Any]:
    match = STAGE2_BATCH_PATTERN.search(name)
    if not match:
        match = STAGE2_OVERRIDE_PATTERN.search(name)
    if not match:
        return {}

    return {
        "stage2_run_number": int(match.group("run")),
        "n_results": int(match.group("n_results")),
        "hyde_enabled": match.group("hyde") == "1",
        "relevance_threshold": int(match.group("threshold")) / 100.0,
    }


def get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def parse_summary_text(summary_text: str) -> dict[str, Any]:
    patterns = {
        "evaluation_row_score": r"Correct evaluation row rate: ([0-9.]+)%",
        "query_weighted_score": r"Query-weighted score: ([0-9.]+)%",
        "whole_query_score_alt": (
            r"Scored-query accuracy \(alternative denominator\): "
            r"(\d+)/(\d+) fully correct \(([0-9.]+)%\)"
        ),
        "best_jurisdiction": (
            r"Best jurisdiction query-weighted score: (.+?) \((.+?), ([0-9.]+)%\)"
        ),
        "lowest_jurisdiction": (
            r"Lowest jurisdiction query-weighted score: (.+?) \((.+?), ([0-9.]+)%\)"
        ),
    }

    parsed: dict[str, Any] = {}

    match = re.search(patterns["evaluation_row_score"], summary_text)
    if match:
        parsed["evaluation_row_score"] = float(match.group(1))

    match = re.search(patterns["query_weighted_score"], summary_text)
    if match:
        parsed["query_weighted_score"] = float(match.group(1))

    match = re.search(patterns["whole_query_score_alt"], summary_text)
    if match:
        parsed["whole_query_correct_alt"] = int(match.group(1))
        parsed["whole_query_scored_alt"] = int(match.group(2))
        parsed["whole_query_score_alt"] = float(match.group(3))

    match = re.search(patterns["best_jurisdiction"], summary_text)
    if match:
        parsed["best_jurisdiction_id"] = match.group(1)
        parsed["best_jurisdiction_dataset"] = match.group(2)
        parsed["best_jurisdiction_score"] = float(match.group(3))

    match = re.search(patterns["lowest_jurisdiction"], summary_text)
    if match:
        parsed["lowest_jurisdiction_id"] = match.group(1)
        parsed["lowest_jurisdiction_dataset"] = match.group(2)
        parsed["lowest_jurisdiction_score"] = float(match.group(3))

    return parsed


def compute_metrics_fallbacks(metrics_path: Path) -> dict[str, Any]:
    df = pl.read_csv(metrics_path)
    result: dict[str, Any] = {
        "jurisdictions_completed": int(df["jurisdiction_id"].n_unique())
        if "jurisdiction_id" in df.columns
        else df.height,
    }

    if df.height == 0:
        return result

    if {"correct", "total"}.issubset(df.columns):
        total_correct = int(df["correct"].sum())
        total_rows = int(df["total"].sum())
        if total_rows > 0:
            result["evaluation_row_score"] = round((total_correct / total_rows) * 100, 2)

    if {"collapsed_query_correct", "collapsed_query_scored"}.issubset(df.columns):
        total_query_correct = int(df["collapsed_query_correct"].sum())
        total_query_scored = int(df["collapsed_query_scored"].sum())
        if total_query_scored > 0:
            result["whole_query_correct_alt"] = total_query_correct
            result["whole_query_scored_alt"] = total_query_scored
            result["whole_query_score_alt"] = round(
                (total_query_correct / total_query_scored) * 100,
                2,
            )

    if {"primary_score", "core_benchmark_queries"}.issubset(df.columns):
        total_queries = int(df["core_benchmark_queries"].sum())
        if total_queries > 0:
            weighted_score = (
                (df["primary_score"] * df["core_benchmark_queries"]).sum() / total_queries
            )
            result["query_weighted_score"] = round(float(weighted_score), 2)

        ordered = df.sort("primary_score", descending=True)
        best_row = ordered.row(0, named=True)
        worst_row = ordered.row(df.height - 1, named=True)
        result["best_jurisdiction_id"] = best_row.get("jurisdiction_id")
        result["best_jurisdiction_score"] = best_row.get("primary_score")
        result["lowest_jurisdiction_id"] = worst_row.get("jurisdiction_id")
        result["lowest_jurisdiction_score"] = worst_row.get("primary_score")

    return result


def materialize_batch_aggregate(output_dir: Path, batch_id: str) -> None:
    aggregate_script = project_root / "coep" / "scripts" / "HPC_scripts" / "aggregate_results.py"
    command = [
        sys.executable,
        str(aggregate_script),
        "--batch-id",
        batch_id,
        "--output-dir",
        str(output_dir),
    ]
    print(f"\n--- Materializing batch artifacts for {batch_id} ---")
    subprocess.run(command, check=True)


def summarize_batch(output_dir: Path, batch_id: str) -> dict[str, Any]:
    batch_root = batch_dir(output_dir, batch_id)
    if not batch_root.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_root}")

    run_dir = latest_batch_run_dir(batch_root)
    if run_dir is None:
        raise FileNotFoundError(f"No aggregate run directories found under {batch_root}")

    manifest_path = batch_root / "dispatch_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}

    override_path_raw = manifest.get("params_override_file")
    override_path = Path(override_path_raw) if override_path_raw else None
    overrides = load_yaml_mapping(override_path) if override_path and override_path.exists() else {}
    condition_from_batch_id = parse_stage2_condition_from_name(batch_id)
    condition_from_override_name = (
        parse_stage2_condition_from_name(override_path.name)
        if override_path is not None
        else {}
    )

    metrics_candidates = sorted(run_dir.glob("all_jurisdictions_metrics_*.csv"))
    if not metrics_candidates:
        raise FileNotFoundError(f"No aggregate metrics CSV found under {run_dir}")
    metrics_path = metrics_candidates[-1]

    summary_path = run_dir / "summary.txt"
    summary_metrics = (
        parse_summary_text(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    fallback_metrics = compute_metrics_fallbacks(metrics_path)

    relevance_threshold = get_nested(overrides, "retrieval", "relevance_filter", "threshold")
    if relevance_threshold is None:
        relevance_threshold = condition_from_batch_id.get(
            "relevance_threshold", condition_from_override_name.get("relevance_threshold")
        )

    row: dict[str, Any] = {
        "batch_id": batch_id,
        "stage2_run_number": condition_from_batch_id.get(
            "stage2_run_number", condition_from_override_name.get("stage2_run_number")
        ),
        "batch_run_timestamp": run_dir.name,
        "compute_mode": manifest.get("compute_mode"),
        "quantization": manifest.get("quantization"),
        "jurisdictions_expected": manifest.get("jurisdiction_count"),
        "jurisdictions_completed": fallback_metrics.get("jurisdictions_completed"),
        "params_override_file": str(override_path) if override_path else None,
        "n_results": get_nested(overrides, "retrieval", "n_results")
        if get_nested(overrides, "retrieval", "n_results") is not None
        else condition_from_batch_id.get("n_results", condition_from_override_name.get("n_results")),
        "hyde_enabled": get_nested(overrides, "retrieval", "hyde", "enabled")
        if get_nested(overrides, "retrieval", "hyde", "enabled") is not None
        else condition_from_batch_id.get("hyde_enabled", condition_from_override_name.get("hyde_enabled")),
        "relevance_filter_enabled": get_nested(
            overrides,
            "retrieval",
            "relevance_filter",
            "enabled",
        ),
        "relevance_threshold": relevance_threshold,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "metrics_path": str(metrics_path),
    }

    if row.get("relevance_filter_enabled") is None:
        row["relevance_filter_enabled"] = True

    for key in [
        "evaluation_row_score",
        "query_weighted_score",
        "whole_query_correct_alt",
        "whole_query_scored_alt",
        "whole_query_score_alt",
        "best_jurisdiction_id",
        "best_jurisdiction_dataset",
        "best_jurisdiction_score",
        "lowest_jurisdiction_id",
        "lowest_jurisdiction_dataset",
        "lowest_jurisdiction_score",
    ]:
        row[key] = summary_metrics.get(key, fallback_metrics.get(key))

    if row.get("jurisdictions_expected") is not None and row.get("jurisdictions_completed") is not None:
        row["jurisdictions_missing"] = (
            int(row["jurisdictions_expected"]) - int(row["jurisdictions_completed"])
        )
    else:
        row["jurisdictions_missing"] = None

    row["hyde_label"] = bool_to_label(row.get("hyde_enabled"))
    row["relevance_filter_label"] = bool_to_label(row.get("relevance_filter_enabled"))

    return row


def resolve_batch_ids(output_dir: Path, sweep_id: str | None, explicit_batch_ids: list[str]) -> list[str]:
    batch_ids = [sanitize_batch_id(batch_id) for batch_id in explicit_batch_ids]
    if sweep_id:
        sweep_prefix = sanitize_batch_id(sweep_id) + "_"
        discovered = [
            child.name
            for child in batches_root(output_dir).iterdir()
            if child.is_dir() and child.name.startswith(sweep_prefix)
        ]
        batch_ids.extend(discovered)

    unique_ids = sorted(set(batch_ids))
    if not unique_ids:
        raise SystemExit("No batch IDs resolved. Pass --sweep-id or one or more --batch-id values.")
    return unique_ids


def format_pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}%"


def format_int(value: Any) -> str:
    if value is None:
        return ""
    return str(int(value))


def write_markdown_table(df: pl.DataFrame, output_path: Path) -> None:
    columns = [
        ("batch_id", "Batch ID"),
        ("stage2_run_number", "Run"),
        ("n_results", "n_results"),
        ("hyde_label", "HYDE"),
        ("relevance_threshold", "Threshold"),
        ("relevance_filter_label", "Relevance filter"),
        ("query_weighted_score", "Query weighted score"),
        ("evaluation_row_score", "Evaluation row score"),
        ("whole_query_score_alt", "Whole-query score"),
        ("best_jurisdiction_id", "Best jurisdiction"),
        ("best_jurisdiction_score", "Best jurisdiction score"),
        ("lowest_jurisdiction_id", "Lowest jurisdiction"),
        ("lowest_jurisdiction_score", "Lowest jurisdiction score"),
    ]

    lines = []
    lines.append("| " + " | ".join(label for _, label in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")

    for row in df.iter_rows(named=True):
        rendered = []
        for key, _ in columns:
            value = row.get(key)
            if key in {"query_weighted_score", "evaluation_row_score", "whole_query_score_alt", "best_jurisdiction_score", "lowest_jurisdiction_score"}:
                rendered.append(format_pct(value))
            elif key in {"n_results", "stage2_run_number"}:
                rendered.append(format_int(value))
            elif key == "relevance_threshold":
                rendered.append("" if value is None else f"{float(value):.2f}")
            else:
                rendered.append("" if value is None else str(value))
        lines.append("| " + " | ".join(rendered) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_narrative_summary(df: pl.DataFrame, output_path: Path) -> None:
    top = df.row(0, named=True)
    bottom = df.row(df.height - 1, named=True)
    lines = [
        f"Stage 2 batches summarized: {df.height}",
        (
            f"Top batch by query-weighted score: {top['batch_id']} "
            f"({format_pct(top.get('query_weighted_score'))}, threshold={top.get('relevance_threshold')}, "
            f"n_results={format_int(top.get('n_results'))}, HYDE={top.get('hyde_label') or ''})"
        ),
        (
            f"Lowest batch by query-weighted score: {bottom['batch_id']} "
            f"({format_pct(bottom.get('query_weighted_score'))}, threshold={bottom.get('relevance_threshold')}, "
            f"n_results={format_int(bottom.get('n_results'))}, HYDE={bottom.get('hyde_label') or ''})"
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    batch_ids = resolve_batch_ids(output_dir, args.sweep_id, args.batch_id)

    for batch_id in batch_ids:
        materialize_batch_aggregate(output_dir, batch_id)

    rows = [summarize_batch(output_dir, batch_id) for batch_id in batch_ids]
    df = pl.DataFrame(rows).sort(["stage2_run_number"], descending=False, nulls_last=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.sweep_id:
        sweep_root = (
            _resolve_from_project_root(args.sweep_root)
            if args.sweep_root is not None
            else default_sweep_root(output_dir)
        )
        aggregate_dir = sweep_root / sanitize_batch_id(args.sweep_id) / "aggregate" / run_timestamp
    else:
        aggregate_dir = (
            output_dir
            / ALL_JURISDICTIONS_DIR_NAME
            / "stage2_batch_aggregate"
            / run_timestamp
        )

    aggregate_dir.mkdir(parents=True, exist_ok=True)

    csv_path = aggregate_dir / "stage2_batch_summary.csv"
    markdown_path = aggregate_dir / "stage2_batch_summary.md"
    text_path = aggregate_dir / "stage2_batch_summary.txt"

    df.write_csv(csv_path)
    write_markdown_table(df, markdown_path)
    write_narrative_summary(df, text_path)

    print("=" * 70)
    print("LEGISCOPE — Stage 2 Batch Summary")
    print("=" * 70)
    if args.sweep_id:
        print(f"Sweep ID: {args.sweep_id}")
    print(f"Batches summarized: {df.height}")
    print(f"Output directory: {aggregate_dir}")
    print(f"CSV summary: {csv_path}")
    print(f"Markdown table: {markdown_path}")
    print(f"Narrative summary: {text_path}")
    print("")

    preview_columns = [
        "batch_id",
        "stage2_run_number",
        "n_results",
        "hyde_label",
        "relevance_threshold",
        "query_weighted_score",
        "evaluation_row_score",
        "whole_query_score_alt",
        "best_jurisdiction_id",
        "best_jurisdiction_score",
        "lowest_jurisdiction_id",
        "lowest_jurisdiction_score",
    ]
    print(df.select(preview_columns))


if __name__ == "__main__":
    main()