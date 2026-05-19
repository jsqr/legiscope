#!/usr/bin/env python3
# ruff: noqa: E402
"""Aggregate benchmark results across all jurisdictions.

After all 50 SLURM jobs complete, this script:

1. Checks for failed/missing jurisdictions (compares DOCX files to results)
2. Collects per-jurisdiction benchmark_metrics.json into a summary table
3. Concatenates per-jurisdiction benchmark_results.csv into a single CSV
4. Prints a summary report with pass/fail counts and accuracy rankings

Usage (on HPC, with conda env activated):
    # From project root:
    python coep/scripts/HPC_scripts/aggregate_results.py

    # With explicit paths:
    python coep/scripts/HPC_scripts/aggregate_results.py \
        --output-dir data/output \
        --docx-dir /gpfs/data/cerdalab/LegalAI/docx_sources

    # Check SLURM job status alongside results:
    python coep/scripts/HPC_scripts/aggregate_results.py --check-slurm
"""

import argparse
from datetime import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import polars as pl


CANONICAL_RESULTS_NAME = "benchmark_results.csv"
TIMESTAMPED_RESULTS_GLOB = "benchmark_results_*.csv"
TIMESTAMPED_RESULTS_PATTERN = re.compile(r"^benchmark_results_(\d{8}_\d{6})\.csv$")
BATCH_RESULTS_GLOB = "benchmark_results_batch_*.csv"
CANONICAL_METRICS_NAME = "benchmark_metrics.json"
TIMESTAMPED_METRICS_GLOB = "benchmark_metrics_*.json"
TIMESTAMPED_METRICS_PATTERN = re.compile(r"^benchmark_metrics_(\d{8}_\d{6})\.json$")
BATCH_METRICS_GLOB = "benchmark_metrics_batch_*.json"
AGGREGATE_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
ALL_JURISDICTIONS_DIR_NAME = "all_jurisdictions"
BATCHES_DIR_NAME = "batches"
BATCH_MANIFEST_NAME = "dispatch_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results across all jurisdictions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory (default: data/output from config.yaml)",
    )
    parser.add_argument(
        "--docx-dir",
        type=Path,
        default=None,
        help="DOCX staging directory to cross-check expected jurisdictions",
    )
    parser.add_argument(
        "--check-slurm",
        action="store_true",
        help="Query SLURM sacct for job status (only works on HPC)",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help=(
            "Aggregate only artifacts tagged for the given dispatch batch ID. "
            "This expects benchmark_results_batch_<ID>.csv and "
            "benchmark_metrics_batch_<ID>.json per jurisdiction."
        ),
    )
    return parser.parse_args()


def _resolve_from_project_root(path: Path) -> Path:
    """Resolve a path relative to the repository root if needed."""
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def resolve_output_dir(override: Path | None) -> Path:
    """Resolve output directory from argument or config.yaml."""
    if override:
        return _resolve_from_project_root(override)
    try:
        from legiscope import config

        return _resolve_from_project_root(config.output_dir())
    except Exception:
        return project_root / "data" / "output"


def get_expected_jurisdictions(docx_dir: Path) -> list[str]:
    """Parse expected jurisdiction IDs from DOCX filenames."""
    ids = []
    for f in sorted(docx_dir.glob("*.docx")):
        parts = f.stem.split("_", 2)
        if len(parts) >= 2:
            ids.append(f"{parts[0]}-{parts[1]}")
    return ids


def _iter_jurisdiction_output_dirs(output_dir: Path) -> list[Path]:
    """Return immediate child directories that contain benchmark artifacts."""
    jurisdiction_dirs: list[Path] = []
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        has_metrics = (child / CANONICAL_METRICS_NAME).exists() or any(
            child.glob(TIMESTAMPED_METRICS_GLOB)
        ) or any(child.glob(BATCH_METRICS_GLOB))
        has_canonical_results = (child / CANONICAL_RESULTS_NAME).exists()
        has_timestamped_results = any(child.glob(TIMESTAMPED_RESULTS_GLOB)) or any(
            child.glob(BATCH_RESULTS_GLOB)
        )
        if has_metrics or has_canonical_results or has_timestamped_results:
            jurisdiction_dirs.append(child)
    return jurisdiction_dirs


def _extract_results_timestamp(results_file: Path) -> str | None:
    """Extract a benchmark timestamp from a timestamped results filename."""
    match = TIMESTAMPED_RESULTS_PATTERN.match(results_file.name)
    if not match:
        return None
    return match.group(1)


def _extract_metrics_timestamp(metrics_file: Path) -> str | None:
    """Extract a benchmark timestamp from a timestamped metrics filename."""
    match = TIMESTAMPED_METRICS_PATTERN.match(metrics_file.name)
    if not match:
        return None
    return match.group(1)


def _select_results_file(
    jurisdiction_dir: Path, batch_id: str | None = None
) -> Path | None:
    """Pick the newest available results file for a jurisdiction.

    Prefer timestamped copies because local workspaces may only have the
    historical artifacts synced back from HPC while the canonical DVC output is
    absent or stale.
    """
    if batch_id:
        batch_results = jurisdiction_dir / _batch_results_name(batch_id)
        if batch_results.exists():
            return batch_results

    timestamped_files = sorted(jurisdiction_dir.glob(TIMESTAMPED_RESULTS_GLOB))
    if timestamped_files:
        return timestamped_files[-1]

    canonical_results = jurisdiction_dir / CANONICAL_RESULTS_NAME
    if canonical_results.exists():
        return canonical_results

    return None


def _select_metrics_file(
    jurisdiction_dir: Path, batch_id: str | None = None
) -> Path | None:
    """Pick the newest available metrics file for a jurisdiction."""
    if batch_id:
        batch_metrics = jurisdiction_dir / _batch_metrics_name(batch_id)
        if batch_metrics.exists():
            return batch_metrics

    timestamped_files = sorted(jurisdiction_dir.glob(TIMESTAMPED_METRICS_GLOB))
    if timestamped_files:
        return timestamped_files[-1]

    canonical_metrics = jurisdiction_dir / CANONICAL_METRICS_NAME
    if canonical_metrics.exists():
        return canonical_metrics

    return None


def _serialize_nested_columns_for_csv(df: pl.DataFrame) -> pl.DataFrame:
    """Convert nested/object columns to JSON strings before CSV export."""
    nested_columns = [
        column_name
        for column_name, dtype in df.schema.items()
        if isinstance(dtype, (pl.List, pl.Array, pl.Struct, pl.Object))
    ]
    if not nested_columns:
        return df

    return df.with_columns(
        [
            pl.col(column_name)
            .map_elements(
                lambda value: json.dumps(value) if value is not None else None,
                return_dtype=pl.String,
            )
            .alias(column_name)
            for column_name in nested_columns
        ]
    )


def _timestamped_aggregate_output_path(
    output_dir: Path, stem: str, timestamp: str
) -> Path:
    """Build a timestamped aggregate CSV output path."""
    return output_dir / f"{stem}_{timestamp}.csv"


def _all_jurisdictions_output_dir(output_dir: Path) -> Path:
    """Return the root directory for aggregate cross-jurisdiction outputs."""
    return output_dir / ALL_JURISDICTIONS_DIR_NAME


def _sanitize_batch_id(batch_id: str) -> str:
    """Normalize a batch ID for use in filenames and directories."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", batch_id.strip())
    normalized = normalized.strip("-._")
    if not normalized:
        raise ValueError("batch_id must contain at least one filename-safe character")
    return normalized


def _batch_output_dir(output_dir: Path, batch_id: str) -> Path:
    """Return the root directory for a batch manifest and aggregate outputs."""
    return _all_jurisdictions_output_dir(output_dir) / BATCHES_DIR_NAME / _sanitize_batch_id(
        batch_id
    )


def _batch_manifest_path(output_dir: Path, batch_id: str) -> Path:
    """Return the dispatch manifest path for a batch."""
    return _batch_output_dir(output_dir, batch_id) / BATCH_MANIFEST_NAME


def _batch_results_name(batch_id: str) -> str:
    """Build the per-jurisdiction batch-tagged results filename."""
    return f"benchmark_results_batch_{_sanitize_batch_id(batch_id)}.csv"


def _batch_metrics_name(batch_id: str) -> str:
    """Build the per-jurisdiction batch-tagged metrics filename."""
    return f"benchmark_metrics_batch_{_sanitize_batch_id(batch_id)}.json"


def _aggregate_run_output_dir(
    output_dir: Path, run_timestamp: str, batch_id: str | None = None
) -> Path:
    """Return the timestamped aggregate output directory for this run."""
    if batch_id:
        return _batch_output_dir(output_dir, batch_id) / run_timestamp
    return _all_jurisdictions_output_dir(output_dir) / run_timestamp


def _load_batch_manifest_jurisdictions(output_dir: Path, batch_id: str) -> list[str]:
    """Return jurisdiction IDs recorded in the dispatch manifest, if present."""
    manifest_path = _batch_manifest_path(output_dir, batch_id)
    if not manifest_path.exists():
        return []

    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    jurisdictions = manifest.get("jurisdictions", [])
    results: list[str] = []
    for entry in jurisdictions:
        if not isinstance(entry, dict):
            continue
        jurisdiction_id = entry.get("jurisdiction_id")
        if isinstance(jurisdiction_id, str) and jurisdiction_id.strip():
            results.append(jurisdiction_id.strip())
    return results


def _prepend_jurisdiction_column(df: pl.DataFrame) -> pl.DataFrame:
    """Add a left-most jurisdiction column derived from jurisdiction_id."""
    if df.is_empty() or "jurisdiction_id" not in df.columns:
        return df

    jurisdiction_expr = (
        pl.when(pl.col("jurisdiction").cast(pl.String).fill_null("") != "")
        .then(pl.col("jurisdiction").cast(pl.String))
        .otherwise(pl.col("jurisdiction_id").cast(pl.String))
        .alias("jurisdiction")
        if "jurisdiction" in df.columns
        else pl.col("jurisdiction_id").cast(pl.String).alias("jurisdiction")
    )
    ordered_columns = [
        "jurisdiction",
        *[column for column in df.columns if column != "jurisdiction"],
    ]
    return df.with_columns(jurisdiction_expr).select(ordered_columns)


def collect_metrics(output_dir: Path, batch_id: str | None = None) -> pl.DataFrame:
    """Collect all benchmark_metrics.json files into a single DataFrame."""
    rows = []
    for jur_dir in _iter_jurisdiction_output_dirs(output_dir):
        metrics_file = _select_metrics_file(jur_dir, batch_id=batch_id)
        if metrics_file is None:
            continue
        try:
            data = json.loads(metrics_file.read_text())
            data.setdefault("jurisdiction_id", jur_dir.name)
            data["aggregate_metrics_path"] = str(metrics_file)
            data["aggregate_metrics_source_file"] = metrics_file.name
            data["aggregate_metrics_source_type"] = (
                "timestamped"
                if metrics_file.name != CANONICAL_METRICS_NAME
                else "canonical"
            )
            data["aggregate_metrics_source_timestamp"] = _extract_metrics_timestamp(
                metrics_file
            )
            rows.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Could not read {metrics_file}: {e}")
    if not rows:
        return pl.DataFrame()
    return _prepend_jurisdiction_column(pl.DataFrame(rows))


def collect_results(output_dir: Path, batch_id: str | None = None) -> pl.DataFrame:
    """Concatenate all benchmark_results.csv files."""
    frames = []
    for jur_dir in _iter_jurisdiction_output_dirs(output_dir):
        results_file = _select_results_file(jur_dir, batch_id=batch_id)
        if results_file is None:
            continue

        try:
            df = pl.read_csv(str(results_file))
            jurisdiction_id_expr = (
                pl.when(
                    pl.col("jurisdiction_id")
                    .cast(pl.String)
                    .fill_null("")
                    .str.strip_chars()
                    == ""
                )
                .then(pl.lit(jur_dir.name))
                .otherwise(pl.col("jurisdiction_id").cast(pl.String))
                .alias("jurisdiction_id")
                if "jurisdiction_id" in df.columns
                else pl.lit(jur_dir.name).alias("jurisdiction_id")
            )
            source_type = (
                "timestamped"
                if results_file.name != CANONICAL_RESULTS_NAME
                else "canonical"
            )
            df = df.with_columns(
                [
                    jurisdiction_id_expr,
                    pl.lit(str(results_file), dtype=pl.String).alias(
                        "_aggregate_source_path"
                    ),
                    pl.lit(results_file.name, dtype=pl.String).alias(
                        "_aggregate_source_file"
                    ),
                    pl.lit(source_type, dtype=pl.String).alias(
                        "_aggregate_source_type"
                    ),
                    pl.lit(
                        _extract_results_timestamp(results_file),
                        dtype=pl.String,
                    ).alias("_aggregate_source_timestamp"),
                ]
            )
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: Could not read {results_file}: {e}")
    if not frames:
        return pl.DataFrame()
    return _prepend_jurisdiction_column(pl.concat(frames, how="diagonal_relaxed"))


def check_slurm_jobs() -> str | None:
    """Query sacct for legiscope job status. Returns formatted output or None."""
    try:
        result = subprocess.run(
            [
                "sacct",
                "-u",
                subprocess.check_output(["whoami"]).decode().strip(),
                "--format=JobID,JobName%30,State,ExitCode,Elapsed",
                "--name=legiscope-jurisdiction",
                "--starttime=now-7days",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    run_timestamp = datetime.now().strftime(AGGREGATE_TIMESTAMP_FORMAT)
    aggregate_output_dir = _aggregate_run_output_dir(
        output_dir, run_timestamp, batch_id=args.batch_id
    )

    print("=" * 70)
    print("LEGISCOPE — Aggregate Benchmark Results")
    print("=" * 70)
    print(f"Jurisdiction output root: {output_dir}")
    if args.batch_id:
        print(f"Batch ID: {args.batch_id}")
    print(f"Aggregate output directory: {aggregate_output_dir}")

    if not output_dir.exists():
        print(f"\nERROR: Output directory does not exist: {output_dir}")
        sys.exit(1)

    aggregate_output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Check expected vs completed jurisdictions ──────────────
    completed_dirs = [
        d.name
        for d in _iter_jurisdiction_output_dirs(output_dir)
        if _select_results_file(d, batch_id=args.batch_id) is not None
        or _select_metrics_file(d, batch_id=args.batch_id) is not None
    ]
    manifest_expected = (
        _load_batch_manifest_jurisdictions(output_dir, args.batch_id)
        if args.batch_id
        else []
    )

    if args.docx_dir and args.docx_dir.exists():
        expected = get_expected_jurisdictions(args.docx_dir)
    elif manifest_expected:
        expected = manifest_expected
    else:
        expected = []

    if expected:
        missing = [j for j in expected if j not in completed_dirs]
        unexpected = [j for j in completed_dirs if j not in expected]

        print(f"\nExpected jurisdictions : {len(expected)}")
        print(f"Completed jurisdictions: {len(completed_dirs)}")
        if missing:
            print(f"\n  MISSING ({len(missing)}):")
            for j in missing:
                print(f"    - {j}")
        if unexpected:
            print(f"\n  UNEXPECTED ({len(unexpected)}):")
            for j in unexpected:
                print(f"    - {j}")
        if not missing and not unexpected:
            print("  All expected jurisdictions have results.")
    else:
        print(f"\nFound {len(completed_dirs)} jurisdiction result directories")
        if args.docx_dir:
            print(f"  (DOCX dir not found: {args.docx_dir})")
        elif args.batch_id:
            print(
                f"  (No dispatch manifest found at {_batch_manifest_path(output_dir, args.batch_id)})"
            )

    # ── 2. Check SLURM job status ────────────────────────────────
    if args.check_slurm:
        print("\n--- SLURM Job Status (last 7 days) ---")
        slurm_output = check_slurm_jobs()
        if slurm_output:
            # Count failures
            lines = slurm_output.strip().split("\n")
            failed = [line for line in lines if "FAILED" in line]
            print(slurm_output)
            if failed:
                print(f"\n  {len(failed)} FAILED jobs detected above.")
                print(
                    "  Check logs: tail /gpfs/data/cerdalab/LegalAI/legiscope/logs/jurisdiction_<JOBID>.err"
                )
            else:
                print("  No failed jobs detected.")
        else:
            print("  sacct not available (not on HPC?)")

    # ── 3. Collect and display metrics ────────────────────────────
    print("\n--- Per-Jurisdiction Metrics ---")
    metrics_df = collect_metrics(output_dir, batch_id=args.batch_id)

    if metrics_df.is_empty():
        print("  No benchmark_metrics.json files found.")
    else:
        has_primary_score = "primary_score" in metrics_df.columns
        sort_column = "primary_score" if has_primary_score else "accuracy_rate"
        metrics_df = metrics_df.sort(sort_column, descending=True)
        has_collapsed_accuracy = "collapsed_query_accuracy_rate" in metrics_df.columns

        # Print table
        if has_primary_score and has_collapsed_accuracy:
            print(
                f"\n  {'Jurisdiction':<25} {'Primary':>10} {'Avg Score':>10} {'Row Acc':>10} "
                f"{'Query Acc':>10} {'Correct':>8} {'Partial':>8} {'Wrong':>8} {'Total':>6}"
            )
            print("  " + "-" * 99)
        elif has_collapsed_accuracy:
            print(
                f"\n  {'Jurisdiction':<25} {'Avg Score':>10} {'Row Acc':>10} "
                f"{'Query Acc':>10} {'Correct':>8} {'Partial':>8} {'Wrong':>8} {'Total':>6}"
            )
            print("  " + "-" * 88)
        else:
            print(
                f"\n  {'Jurisdiction':<25} {'Avg Score':>10} {'Accuracy':>10} "
                f"{'Correct':>8} {'Partial':>8} {'Wrong':>8} {'Total':>6}"
            )
            print("  " + "-" * 77)

        for row in metrics_df.iter_rows(named=True):
            avg = row.get("avg_score")
            avg_str = f"{avg:.2f}" if avg is not None else "N/A"
            primary = row.get("primary_score")
            primary_str = f"{primary:>9.1f}" if primary is not None else f"{'N/A':>10}"
            if has_primary_score and has_collapsed_accuracy:
                query_accuracy = row.get("collapsed_query_accuracy_rate")
                query_accuracy_str = (
                    f"{query_accuracy:>9.1f}%"
                    if query_accuracy is not None
                    else f"{'N/A':>10}"
                )
                print(
                    f"  {row['jurisdiction_id']:<25} {primary_str} {avg_str:>10} "
                    f"{row['accuracy_rate']:>9.1f}% {query_accuracy_str} "
                    f"{row['correct']:>8} {row['partially_correct']:>8} "
                    f"{row['incorrect']:>8} {row['total']:>6}"
                )
            elif has_collapsed_accuracy:
                query_accuracy = row.get("collapsed_query_accuracy_rate")
                query_accuracy_str = (
                    f"{query_accuracy:>9.1f}%"
                    if query_accuracy is not None
                    else f"{'N/A':>10}"
                )
                print(
                    f"  {row['jurisdiction_id']:<25} {avg_str:>10} "
                    f"{row['accuracy_rate']:>9.1f}% {query_accuracy_str} "
                    f"{row['correct']:>8} {row['partially_correct']:>8} "
                    f"{row['incorrect']:>8} {row['total']:>6}"
                )
            else:
                print(
                    f"  {row['jurisdiction_id']:<25} {avg_str:>10} "
                    f"{row['accuracy_rate']:>9.1f}% "
                    f"{row['correct']:>8} {row['partially_correct']:>8} "
                    f"{row['incorrect']:>8} {row['total']:>6}"
                )

        # Aggregate summary
        total_correct = metrics_df["correct"].sum()
        total_partial = metrics_df["partially_correct"].sum()
        total_incorrect = metrics_df["incorrect"].sum()
        total_questions = metrics_df["total"].sum()
        overall_accuracy = (
            (total_correct / total_questions) * 100 if total_questions > 0 else 0
        )
        overall_avg = metrics_df["avg_score"].mean()
        overall_primary = None
        if has_primary_score and "core_benchmark_queries" in metrics_df.columns:
            total_core_queries = metrics_df["core_benchmark_queries"].sum()
            if total_core_queries and total_core_queries > 0:
                weighted_primary_points = (
                    metrics_df["primary_score"] * metrics_df["core_benchmark_queries"]
                ).sum() / total_core_queries
                overall_primary = weighted_primary_points
        overall_query_accuracy = None
        if has_collapsed_accuracy:
            total_query_correct = metrics_df["collapsed_query_correct"].sum()
            total_query_count = metrics_df["core_benchmark_queries"].sum()
            overall_query_accuracy = (
                (total_query_correct / total_query_count) * 100
                if total_query_count > 0
                else 0
            )

        print(
            "  "
            + (
                "-" * 99
                if has_primary_score and has_collapsed_accuracy
                else "-" * 88
                if has_collapsed_accuracy
                else "-" * 77
            )
        )
        avg_str = f"{overall_avg:.2f}" if overall_avg is not None else "N/A"
        primary_str = (
            f"{overall_primary:>9.1f}"
            if overall_primary is not None
            else f"{'N/A':>10}"
        )
        if has_primary_score and has_collapsed_accuracy:
            print(
                f"  {'OVERALL':<25} {primary_str} {avg_str:>10} "
                f"{overall_accuracy:>9.1f}% {overall_query_accuracy:>9.1f}% "
                f"{total_correct:>8} {total_partial:>8} "
                f"{total_incorrect:>8} {total_questions:>6}"
            )
        elif has_collapsed_accuracy:
            print(
                f"  {'OVERALL':<25} {avg_str:>10} "
                f"{overall_accuracy:>9.1f}% {overall_query_accuracy:>9.1f}% "
                f"{total_correct:>8} {total_partial:>8} "
                f"{total_incorrect:>8} {total_questions:>6}"
            )
        else:
            print(
                f"  {'OVERALL':<25} {avg_str:>10} "
                f"{overall_accuracy:>9.1f}% "
                f"{total_correct:>8} {total_partial:>8} "
                f"{total_incorrect:>8} {total_questions:>6}"
            )

        # Save metrics summary
        metrics_out = _timestamped_aggregate_output_path(
            aggregate_output_dir, "all_jurisdictions_metrics", run_timestamp
        )
        _serialize_nested_columns_for_csv(metrics_df).write_csv(str(metrics_out))
        print(f"\n  Metrics saved to: {metrics_out}")

    # ── 4. Concatenate detailed results ───────────────────────────
    print("\n--- Detailed Results ---")
    results_df = collect_results(output_dir, batch_id=args.batch_id)

    if results_df.is_empty():
        print("  No benchmark_results.csv files found.")
    else:
        combined_out = _timestamped_aggregate_output_path(
            aggregate_output_dir, "all_jurisdictions_benchmark", run_timestamp
        )
        results_df.write_csv(str(combined_out))
        timestamped_sources = results_df.filter(
            pl.col("_aggregate_source_type") == "timestamped"
        )
        print(
            f"  Combined {results_df.height} rows across {len(completed_dirs)} jurisdictions"
        )
        print(f"  Saved to: {combined_out}")
        if timestamped_sources.height > 0:
            unique_timestamped = timestamped_sources["jurisdiction_id"].n_unique()
            print(
                "  Used latest timestamped benchmark CSVs for "
                f"{unique_timestamped} jurisdiction(s)."
            )

        # Per-jurisdiction breakdown from detailed results
        if (
            "jurisdiction_id" in results_df.columns
            and "eval_label" in results_df.columns
        ):
            breakdown = (
                results_df.group_by("jurisdiction_id")
                .agg(
                    [
                        pl.col("eval_score").mean().alias("avg_score"),
                        (pl.col("eval_label") == "Correct").sum().alias("correct"),
                        pl.len().alias("total"),
                    ]
                )
                .with_columns(
                    (pl.col("correct") / pl.col("total") * 100)
                    .round(1)
                    .alias("accuracy_%")
                )
                .sort("accuracy_%", descending=True)
            )

            print("\n  Per-jurisdiction accuracy (from detailed results):")
            print(f"  {breakdown}")

    # ── 5. DVC experiments hint ───────────────────────────────────
    print("\n--- DVC Experiments ---")
    print("  To view all experiments with params + metrics side-by-side:")
    print("    dvc exp show")
    print("  To export as CSV:")
    print("    dvc exp show --csv > all_experiments.csv")

    print("\n" + "=" * 70)
    print("Aggregation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
