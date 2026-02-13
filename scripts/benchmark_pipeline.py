#!/usr/bin/env python3
"""
Benchmark pipeline script for evaluating RAG performance against a human-labeled dataset.

Implements workflow as described in BENCHMARKING.md:
1. Load queries from CSV (with question and variable_name columns)
2. Load MonQcle Standard Report and filter to target jurisdiction
3. Melt the wide-format MonQcle data to long format (one row per variable)
4. Run RAG queries and join results with ground truth by variable_name
5. Evaluate generated answers against ground truth using LLM-as-judge
6. Output detailed results with scores and metrics

Jurisdiction and retrieval/query settings are read from params.yaml.
Paths are resolved from config.yaml.

Usage:
    python scripts/benchmark_pipeline.py
    python scripts/benchmark_pipeline.py --test-limit 5 --debug
"""

import argparse
import csv
import sys
from pathlib import Path

import chromadb
import polars as pl
from loguru import logger

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from legiscope import config
from legiscope.embeddings import EMBEDDING_PROVIDER, CollectionConfig
from legiscope.eval import (
    Evaluator,
    jurisdiction_id_to_monqcle_name,
    load_and_filter_monqcle,
    melt_monqcle_to_long,
)
from legiscope.models import CodeRef
from legiscope.params import load_params
from legiscope.query import BatchQuerySettings, load_queries, run_queries


def main():
    code_ref = CodeRef.from_params()
    jurisdiction_id = code_ref.jurisdiction_id
    params = load_params()

    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation pipeline against MonQcle ground truth"
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        help="Limit number of queries for testing pipeline",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode to save intermediate results",
    )

    args = parser.parse_args()

    # Resolve paths and settings from config/params
    queries_path = config.default_queries_path()
    monqcle_path = config.monqcle_report_path()
    output_path = config.output_dir() / jurisdiction_id / "benchmark_results.csv"
    series_title = params.get("benchmark", {}).get(
        "series_title", "DPL_2025_Consolidated"
    )

    # =========================================================================
    # Step 1: Load Queries
    # =========================================================================
    # Uses shared query loading logic (returns list[QueryInput])
    query_inputs = load_queries(str(queries_path))

    if args.debug:
        debug_dir = Path(f"data/output/{jurisdiction_id}/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_queries_path = debug_dir / "loaded_queries.csv"
        with open(debug_queries_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(query_inputs)
        logger.info(f"Debug: Saved loaded queries to {debug_queries_path}")

    if args.test_limit:
        query_inputs = query_inputs[: args.test_limit]
        logger.info(f"Limiting benchmark to first {args.test_limit} queries")

    # =========================================================================
    # Step 2: Load and Filter MonQcle Ground Truth
    # =========================================================================
    monqcle_name = jurisdiction_id_to_monqcle_name(jurisdiction_id)
    monqcle_row = load_and_filter_monqcle(
        str(monqcle_path), monqcle_name, series_title
    )

    # Get variable names from query inputs for filtering ground truth
    variable_names = [q.variable_name for q in query_inputs if q.variable_name]

    # Identify combined variables
    combined_vars = [v for v in variable_names if "combined" in v]

    # Pre-process MonQcle row to include combined variables
    if combined_vars:
        logger.info(
            f"Pre-processing {len(combined_vars)} combined variables: {combined_vars}"
        )

        # Specific handling for Drug Paraphernalia combined variables
        if any(v.startswith("dp") for v in variable_names):
            val_collected = ""
            val_valid = ""

            if "dp_collected" in monqcle_row.columns:
                val = monqcle_row["dp_collected"][0]
                val_collected = str(val) if val not in [None, "-"] else ""

            if "dp_valid_imp" in monqcle_row.columns:
                val = monqcle_row["dp_valid_imp"][0]
                val_valid = str(val) if val not in [None, "-"] else ""

            combined_truth = (
                f"Collected: {val_collected}\nValid/Imp: {val_valid}".strip()
            )

            # Add columns for each combined variable name to the MonQcle row
            new_cols = [pl.lit(combined_truth).alias(v) for v in combined_vars]
            monqcle_row = monqcle_row.with_columns(new_cols)
            logger.info(f"Added combined columns to MonQcle data: {combined_vars}")

    # Melt to long format (now handles everything including combined vars)
    ground_truth_df = melt_monqcle_to_long(monqcle_row, variable_names)

    if args.debug:
        debug_dir = Path(f"data/output/{jurisdiction_id}/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        melted_path = debug_dir / "melted_monqcle.csv"
        ground_truth_df.write_csv(melted_path)
        logger.info(f"Debug: Saved melted MonQcle data to {melted_path}")

    # =========================================================================
    # Step 3: Initialize Resources
    # =========================================================================
    chroma_client = chromadb.PersistentClient(path=str(config.chroma_db_path()))
    collection_cfg = CollectionConfig(provider=EMBEDDING_PROVIDER)
    collection = chroma_client.get_collection(collection_cfg.collection_name)

    # Construct path to sections parquet using CodeRef
    parquet_path = code_ref.full_data_dir / "sections.parquet"
    if not parquet_path.exists():
        logger.error(f"Sections file not found: {parquet_path}")
        sys.exit(1)

    # =========================================================================
    # Step 4: Configure LLM Agents
    # =========================================================================
    query_settings = BatchQuerySettings()

    # Evaluator Agent (Powerful model for judging)
    evaluator = Evaluator()

    # =========================================================================
    # Step 5: Run RAG Pipeline (Generation)
    # =========================================================================
    logger.info(f"Running RAG pipeline for {len(query_inputs)} queries...")

    # Pass structured inputs directly to run_queries
    # The returned DataFrame will include 'variable_name' automatically
    gen_results_df = run_queries(
        collection=collection,
        sections_parquet_path=str(parquet_path),
        queries=query_inputs,
        jurisdiction_id=jurisdiction_id,
        settings=query_settings,
    )

    # =========================================================================
    # Step 6: Join with Ground Truth
    # =========================================================================
    logger.info("Joining generated answers with ground truth...")

    joined_df = gen_results_df.join(ground_truth_df, on="variable_name", how="left")

    if args.debug:
        debug_dir = Path(f"data/output/{jurisdiction_id}/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / "queries_and_ground_truth.csv"
        joined_df.write_csv(debug_path)
        logger.info(f"Debug: Saved queries and ground truth to {debug_path}")

    # Check for missing ground truth (Null, empty, or "-")
    missing_truth = joined_df.filter(
        pl.col("ground_truth").is_null()
        | (pl.col("ground_truth").str.strip_chars() == "")
        | (pl.col("ground_truth") == "-")
    )
    if len(missing_truth) > 0:
        logger.warning(
            f"{len(missing_truth)} queries have no ground truth: "
            f"{missing_truth['variable_name'].to_list()}"
        )

    # =========================================================================
    # Step 7: Run Evaluation Pipeline
    # =========================================================================
    logger.info("Evaluating generated answers against ground truth...")

    # Filter to only rows with ground truth for evaluation
    eval_df = joined_df.filter(
        pl.col("ground_truth").is_not_null()
        & (pl.col("ground_truth") != "")
        & (pl.col("ground_truth") != "-")
    )

    if len(eval_df) == 0:
        logger.error("No rows with ground truth to evaluate!")
        sys.exit(1)

    # Construct comprehensive answer context for evaluation
    eval_df = eval_df.with_columns(
        pl.concat_str(
            [
                pl.col("short_answer"),
                pl.lit("\n\nReasoning: "),
                pl.col("reasoning"),
                pl.lit("\n\nSupporting Passages: "),
                pl.col("supporting_passages"),
            ]
        ).alias("comprehensive_answer")
    )

    eval_df = evaluator.evaluate_batch(
        eval_df,
        question_col="query",
        answer_col="comprehensive_answer",
        truth_col="ground_truth",
    )

    # Add jurisdiction metadata
    eval_df = eval_df.with_columns(
        pl.lit(jurisdiction_id).alias("jurisdiction_id")
    )

    # =========================================================================
    # Step 8: Compute Summary Metrics
    # =========================================================================
    avg_score = eval_df["eval_score"].mean()
    correct_count = eval_df.filter(pl.col("eval_label") == "Correct").height
    partial_count = eval_df.filter(
        pl.col("eval_label") == "Partially Correct"
    ).height
    incorrect_count = eval_df.filter(pl.col("eval_label") == "Incorrect").height
    total_count = eval_df.height
    accuracy_rate = (correct_count / total_count) * 100 if total_count > 0 else 0

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)
    print(f"Total Questions Evaluated: {total_count}")
    print(f"Average Quality Score: {avg_score:.2f} / 10")
    print(f"Correct: {correct_count} ({accuracy_rate:.1f}%)")
    print(f"Partially Correct: {partial_count}")
    print(f"Incorrect: {incorrect_count}")
    print("=" * 60 + "\n")

    # =========================================================================
    # Step 9: Save Results
    # =========================================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if str(output_path).endswith(".parquet"):
        eval_df.write_parquet(str(output_path))
    else:
        eval_df.write_csv(str(output_path))

    logger.info(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
