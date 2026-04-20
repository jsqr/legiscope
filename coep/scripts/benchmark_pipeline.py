#!/usr/bin/env python3
# ruff: noqa: E402
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
    python scripts/benchmark_pipeline.py --state CA --locality LosAngeles --code-slug municipal-code
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import chromadb
import polars as pl
from loguru import logger

# Add project root and src to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from legiscope import config
from legiscope.embeddings import EMBEDDING_PROVIDER, CollectionConfig
from legiscope.models import CodeRef
from legiscope.params import load_params
from legiscope.query import BatchQuerySettings, load_queries, run_queries
from coep.src.eval import (
    Evaluator,
    expand_combined_variables,
    jurisdiction_id_to_monqcle_name,
    load_and_filter_monqcle,
    melt_monqcle_to_long,
    prioritize_ground_truth_matches,
)
from coep.src.query import adjust_drug_paraphernalia_queries
from coep.src.retrieval_guidance import get_drug_paraphernalia_retrieval_guidance


def main():
    config.setup_logging()

    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation pipeline against MonQcle ground truth"
    )
    parser.add_argument("--state", help="Two-letter state abbreviation")
    parser.add_argument("--locality", help="Locality name")
    parser.add_argument("--code-slug", help="Code slug identifier")
    parser.add_argument(
        "--test-limit",
        type=int,
        help="Limit number of queries for testing pipeline",
    )

    args = parser.parse_args()

    # Build CodeRef from CLI args (DVC stage) or params.yaml (standalone)
    if args.state and args.code_slug:
        code_ref = CodeRef.from_dvc_vars(
            state=args.state,
            locality=args.locality,
            code_slug=args.code_slug,
        )
    else:
        code_ref = CodeRef.from_params()

    jurisdiction_id = code_ref.jurisdiction_id
    output_dir_name = code_ref.jurisdiction.output_dir_name
    params = load_params()

    # Read debug flag from params.yaml (shared with run_queries.py)
    debug_enabled = params.get("retrieval", {}).get("debug", False)
    debug_dir = None
    if debug_enabled:
        debug_dir = config.output_dir() / output_dir_name / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug mode enabled, writing debug files to {debug_dir}")

    # Resolve paths and settings from config/params
    queries_path = config.default_queries_path()
    monqcle_path = config.monqcle_report_path()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir() / output_dir_name
    # DVC-tracked output (deterministic name for dvc.yaml outs:)
    output_path = output_dir / "benchmark_results.csv"
    # Timestamped copy for historical tracking (not DVC-tracked)
    timestamped_path = output_dir / f"benchmark_results_{timestamp}.csv"
    # DVC metrics file
    metrics_path = output_dir / "benchmark_metrics.json"
    series_title = params.get("benchmark", {}).get(
        "series_title", "DPL_2025_Consolidated"
    )

    # =========================================================================
    # Step 1: Load Queries
    # =========================================================================
    # Uses shared query loading logic (returns list[QueryInput])
    query_inputs = load_queries(
        str(queries_path),
        adjust_for_dataset=True,
        query_adjuster=adjust_drug_paraphernalia_queries,
    )

    if args.test_limit:
        query_inputs = query_inputs[: args.test_limit]
        logger.info(f"Limiting benchmark to first {args.test_limit} queries")

    # =========================================================================
    # Step 2: Load and Filter MonQcle Ground Truth
    # =========================================================================
    monqcle_name = jurisdiction_id_to_monqcle_name(jurisdiction_id)
    monqcle_row = load_and_filter_monqcle(str(monqcle_path), monqcle_name, series_title)

    # Get variable names from query inputs for filtering ground truth
    variable_names = [q.variable_name for q in query_inputs if q.variable_name]

    # Expand any combined variables (e.g. dp_collected_combined, dp_state_fed_combined)
    monqcle_row = expand_combined_variables(monqcle_row, variable_names)

    # Melt to long format (now handles everything including combined vars)
    ground_truth_df = melt_monqcle_to_long(monqcle_row, variable_names)

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
    query_settings = BatchQuerySettings(
        debug_dir=debug_dir,
        debug_timestamp=timestamp,
        retrieval_guidance_provider=get_drug_paraphernalia_retrieval_guidance,
    )

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

    joined_df = (
        gen_results_df.join(ground_truth_df, on="variable_name", how="left")
        .with_row_count("benchmark_row_id")
        .with_columns(
            (
                pl.col("ground_truth").is_not_null()
                & (pl.col("ground_truth").str.strip_chars() != "")
                & (pl.col("ground_truth") != "-")
            ).alias("ground_truth_available")
        )
        .with_columns(
            pl.when(pl.col("ground_truth_available"))
            .then(pl.lit("scored"))
            .otherwise(pl.lit("missing_ground_truth"))
            .alias("evaluation_status")
        )
    )

    if debug_enabled and debug_dir:
        query_stage_path = debug_dir / f"query_stage_{timestamp}.csv"
        if query_stage_path.exists():
            query_stage_df = pl.read_csv(query_stage_path)
            ground_truth_cols = [
                col
                for col in [
                    "variable_name",
                    "ground_truth",
                    "ground_truth_citation",
                    "comprehensive_answer",
                ]
                if col in joined_df.columns
            ]
            if ground_truth_cols:
                ground_truth_debug = joined_df.select(ground_truth_cols).unique(
                    subset=["variable_name"]
                )
                query_stage_df = query_stage_df.join(
                    ground_truth_debug,
                    on="variable_name",
                    how="left",
                )
                query_stage_df.write_csv(query_stage_path)
                logger.info(
                    f"Debug: Enriched query stage CSV with ground truth at {query_stage_path}"
                )

    # Check for missing ground truth (Null, empty, or "-")
    missing_truth = joined_df.filter(~pl.col("ground_truth_available"))
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
    eval_input_df = joined_df.filter(pl.col("ground_truth_available"))

    if len(eval_input_df) == 0:
        logger.error("No rows with ground truth to evaluate!")
        sys.exit(1)

    # Construct comprehensive answer context for evaluation
    eval_scored_df = eval_input_df.with_columns(
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

    eval_scored_df = evaluator.evaluate_batch(
        eval_scored_df,
        question_col="query",
        answer_col="comprehensive_answer",
        truth_col="ground_truth",
    )

    final_df = joined_df.join(
        eval_scored_df.select(
            [
                "benchmark_row_id",
                "comprehensive_answer",
                "eval_score",
                "eval_reason",
                "eval_label",
            ]
        ),
        on="benchmark_row_id",
        how="left",
    ).with_columns(pl.lit(jurisdiction_id).alias("jurisdiction_id"))
    final_df = prioritize_ground_truth_matches(final_df)

    # =========================================================================
    # Step 8: Compute Summary Metrics
    # =========================================================================
    avg_score = eval_scored_df["eval_score"].mean()
    correct_count = eval_scored_df.filter(pl.col("eval_label") == "Correct").height
    partial_count = eval_scored_df.filter(
        pl.col("eval_label") == "Partially Correct"
    ).height
    incorrect_count = eval_scored_df.filter(pl.col("eval_label") == "Incorrect").height
    processed_count = final_df.height
    scored_count = eval_scored_df.height
    unscored_count = processed_count - scored_count
    accuracy_rate = (correct_count / scored_count) * 100 if scored_count > 0 else 0

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)
    print(f"Total Queries Processed: {processed_count}")
    print(f"Queries Scored Against Ground Truth: {scored_count}")
    print(f"Queries Unscored (missing/excluded ground truth): {unscored_count}")
    print(f"Average Quality Score: {avg_score:.2f} / 10")
    print(f"Correct: {correct_count} ({accuracy_rate:.1f}%)")
    print(f"Partially Correct: {partial_count}")
    print(f"Incorrect: {incorrect_count}")
    print("=" * 60 + "\n")

    # =========================================================================
    # Step 9: Save Results
    # =========================================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write DVC-tracked output (deterministic path)
    final_df.write_csv(str(output_path))
    logger.info(f"Results saved to {output_path}")

    # Write timestamped copy for historical reference
    final_df.write_csv(str(timestamped_path))
    logger.info(f"Timestamped copy saved to {timestamped_path}")

    # Write DVC metrics JSON
    metrics = {
        "jurisdiction_id": jurisdiction_id,
        "avg_score": round(avg_score, 4) if avg_score is not None else None,
        "accuracy_rate": round(accuracy_rate, 2),
        "correct": correct_count,
        "partially_correct": partial_count,
        "incorrect": incorrect_count,
        "processed_queries": processed_count,
        "scored_queries": scored_count,
        "unscored_queries": unscored_count,
        "total": scored_count,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
