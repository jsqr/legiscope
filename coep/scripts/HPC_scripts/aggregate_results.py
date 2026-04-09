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
import json
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
    return parser.parse_args()


def resolve_output_dir(override: Path | None) -> Path:
    """Resolve output directory from argument or config.yaml."""
    if override:
        return override
    try:
        from legiscope import config

        return config.output_dir()
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


def collect_metrics(output_dir: Path) -> pl.DataFrame:
    """Collect all benchmark_metrics.json files into a single DataFrame."""
    rows = []
    for metrics_file in sorted(output_dir.rglob("benchmark_metrics.json")):
        try:
            data = json.loads(metrics_file.read_text())
            rows.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Could not read {metrics_file}: {e}")
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def collect_results(output_dir: Path) -> pl.DataFrame:
    """Concatenate all benchmark_results.csv files."""
    frames = []
    for jur_dir in sorted(output_dir.iterdir()):
        if not jur_dir.is_dir():
            continue
        results_file = jur_dir / "benchmark_results.csv"
        if results_file.exists():
            try:
                df = pl.read_csv(str(results_file))
                frames.append(df)
            except Exception as e:
                print(f"  WARNING: Could not read {results_file}: {e}")
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


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

    print("=" * 70)
    print("LEGISCOPE — Aggregate Benchmark Results")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    if not output_dir.exists():
        print(f"\nERROR: Output directory does not exist: {output_dir}")
        sys.exit(1)

    # ── 1. Check expected vs completed jurisdictions ──────────────
    completed_dirs = sorted(
        [d.name for d in output_dir.iterdir() if d.is_dir() and "-" in d.name]
    )

    if args.docx_dir and args.docx_dir.exists():
        expected = get_expected_jurisdictions(args.docx_dir)
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
    metrics_df = collect_metrics(output_dir)

    if metrics_df.is_empty():
        print("  No benchmark_metrics.json files found.")
    else:
        # Sort by accuracy
        metrics_df = metrics_df.sort("accuracy_rate", descending=True)

        # Print table
        print(
            f"\n  {'Jurisdiction':<25} {'Avg Score':>10} {'Accuracy':>10} "
            f"{'Correct':>8} {'Partial':>8} {'Wrong':>8} {'Total':>6}"
        )
        print("  " + "-" * 77)
        for row in metrics_df.iter_rows(named=True):
            avg = row.get("avg_score")
            avg_str = f"{avg:.2f}" if avg is not None else "N/A"
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

        print("  " + "-" * 77)
        avg_str = f"{overall_avg:.2f}" if overall_avg is not None else "N/A"
        print(
            f"  {'OVERALL':<25} {avg_str:>10} "
            f"{overall_accuracy:>9.1f}% "
            f"{total_correct:>8} {total_partial:>8} "
            f"{total_incorrect:>8} {total_questions:>6}"
        )

        # Save metrics summary
        metrics_out = output_dir / "all_jurisdictions_metrics.csv"
        metrics_df.write_csv(str(metrics_out))
        print(f"\n  Metrics saved to: {metrics_out}")

    # ── 4. Concatenate detailed results ───────────────────────────
    print("\n--- Detailed Results ---")
    results_df = collect_results(output_dir)

    if results_df.is_empty():
        print("  No benchmark_results.csv files found.")
    else:
        combined_out = output_dir / "all_jurisdictions_benchmark.csv"
        results_df.write_csv(str(combined_out))
        print(
            f"  Combined {results_df.height} rows across {len(completed_dirs)} jurisdictions"
        )
        print(f"  Saved to: {combined_out}")

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
