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

"""

import argparse
import sys
import os
from pathlib import Path
import polars as pl
import chromadb
from loguru import logger
import csv

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from legiscope.llm_config import Config
from legiscope.utils import LLMConfig
from legiscope.query import BatchQuerySettings, run_queries, load_queries
from legiscope.eval import (
    Evaluator, 
    load_and_filter_monqcle, 
    melt_monqcle_to_long, 
    jurisdiction_id_to_monqcle_name
)

# Default paths - can be overridden via CLI
DEFAULT_MONQCLE_PATH = "data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv"
DEFAULT_CHROMA_PATH = "./data/chroma_db"


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation pipeline against MonQcle ground truth"
    )
    parser.add_argument(
        "--queries-path", 
        required=True, 
        help="Path to queries CSV with 'question' and 'variable_name' columns"
    )
    parser.add_argument(
        "--jurisdiction-id", 
        required=True, 
        help="Jurisdiction ID (e.g., CA-LosAngeles)"
    )
    parser.add_argument(
        "--monqcle-path",
        default=DEFAULT_MONQCLE_PATH,
        help="Path to MonQcle Standard Report CSV"
    )
    parser.add_argument(
        "--output", 
        default="data/output/benchmark_results.csv", 
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--n-results", 
        type=int, 
        default=10, 
        help="Number of embedding segments to retrieve per query"
    )
    parser.add_argument(
        "--test-limit", 
        type=int, 
        help="Limit number of queries for testing pipeline"
    )
    parser.add_argument(
        "--series-title",
        default="DPL_2025_Consolidated",
        help="MonQcle series title to use for ground truth"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode to save intermediate results"
    )
    
    args = parser.parse_args()

    # =========================================================================
    # Step 1: Load Queries
    # =========================================================================
    # Uses shared query loading logic (returns list[QueryInput])
    query_inputs = load_queries(args.queries_path)

    if args.debug:
        debug_dir = Path(f"data/output/{args.jurisdiction_id}/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        queries_path = debug_dir / "loaded_queries.csv"
        with open(queries_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(query_inputs)
        logger.info(f"Debug: Saved loaded queries to {queries_path}")
    
    if args.test_limit:
        query_inputs = query_inputs[:args.test_limit]
        logger.info(f"Limiting benchmark to first {args.test_limit} queries")

    # =========================================================================
    # Step 2: Load and Filter MonQcle Ground Truth
    # =========================================================================
    monqcle_name = jurisdiction_id_to_monqcle_name(args.jurisdiction_id)
    monqcle_row = load_and_filter_monqcle(
        args.monqcle_path, 
        monqcle_name,
        args.series_title
    )
    
    # Get variable names from query inputs for filtering ground truth
    variable_names = [q.variable_name for q in query_inputs if q.variable_name]
    
    # Identify combined variables
    combined_vars = [v for v in variable_names if "combined" in v]

    # Pre-process MonQcle row to include combined variables
    if combined_vars:
        logger.info(f"Pre-processing {len(combined_vars)} combined variables: {combined_vars}")
        
        # Specific handling for Drug Paraphernalia combined variables
        if any(v.startswith("dp") for v in variable_names):
            # Get values for the constituent columns
            # Note: This logic assumes 'dp_collected' and 'dp_valid_imp' are the targets for combination
            # regardless of the specific combined variable name
            
            val_collected = ""
            val_valid = ""
            
            if "dp_collected" in monqcle_row.columns:
                val = monqcle_row["dp_collected"][0]
                val_collected = str(val) if val not in [None, "-"] else ""
                
            if "dp_valid_imp" in monqcle_row.columns:
                val = monqcle_row["dp_valid_imp"][0]
                val_valid = str(val) if val not in [None, "-"] else ""
                
            combined_truth = f"Collected: {val_collected}\nValid/Imp: {val_valid}".strip()
            
            # Add columns for each combined variable name to the MonQcle row
            new_cols = [pl.lit(combined_truth).alias(v) for v in combined_vars]
            monqcle_row = monqcle_row.with_columns(new_cols)
            logger.info(f"Added combined columns to MonQcle data: {combined_vars}")

    # Melt to long format (now handles everything including combined vars)
    ground_truth_df = melt_monqcle_to_long(monqcle_row, variable_names)

    if args.debug:
        debug_dir = Path(f"data/output/{args.jurisdiction_id}/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        melted_path = debug_dir / "melted_monqcle.csv"
        ground_truth_df.write_csv(melted_path)
        logger.info(f"Debug: Saved melted MonQcle data to {melted_path}")

    # =========================================================================
    # Step 3: Initialize Resources
    # =========================================================================
    chroma_client = chromadb.PersistentClient(path=DEFAULT_CHROMA_PATH)
    collection_name = os.getenv("LEGISCOPE_COLLECTION_NAME", "legal_code_all")
    collection = chroma_client.get_collection(collection_name)
    
    # Construct path to sections parquet
    parquet_path = Path(f"data/laws/{args.jurisdiction_id}/tables/sections.parquet")
    if not parquet_path.exists():
        logger.error(f"Sections file not found: {parquet_path}")
        sys.exit(1)

    # =========================================================================
    # Step 4: Configure LLM Agents
    # =========================================================================
    # Query Agent (Powerful model for legal analysis)
    query_llm = LLMConfig(
        client=Config.get_powerful_client(), 
        model=Config.get_powerful_model(),
        temperature=0.0
    )
    query_settings = BatchQuerySettings(
        llm=query_llm, 
        n_results=args.n_results, 
        filter_relevance=True,
        relevance_threshold=0.5
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
        jurisdiction_id=args.jurisdiction_id,
        settings=query_settings
    )
    
    # =========================================================================
    # Step 6: Join with Ground Truth
    # =========================================================================
    logger.info("Joining generated answers with ground truth...")
    
    joined_df = gen_results_df.join(
        ground_truth_df,
        on="variable_name",
        how="left"
    )

    if args.debug:
        debug_dir = Path(f"data/output/{args.jurisdiction_id}/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / "queries_and_ground_truth.csv"
        joined_df.write_csv(debug_path)
        logger.info(f"Debug: Saved queries and ground truth to {debug_path}")
    
    # Check for missing ground truth
    missing_truth = joined_df.filter(pl.col("ground_truth").is_null())
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
        pl.col("ground_truth").is_not_null() & 
        (pl.col("ground_truth") != "")
    )
    
    if len(eval_df) == 0:
        logger.error("No rows with ground truth to evaluate!")
        sys.exit(1)
    
    eval_df = evaluator.evaluate_batch(
        eval_df,
        question_col="query",
        answer_col="short_answer",
        truth_col="ground_truth"
    )
    
    # Add jurisdiction metadata
    eval_df = eval_df.with_columns(pl.lit(args.jurisdiction_id).alias("jurisdiction_id"))

    # =========================================================================
    # Step 8: Compute Summary Metrics
    # =========================================================================
    avg_score = eval_df["eval_score"].mean()
    correct_count = eval_df.filter(pl.col("eval_label") == "Correct").height
    partial_count = eval_df.filter(pl.col("eval_label") == "Partially Correct").height
    incorrect_count = eval_df.filter(pl.col("eval_label") == "Incorrect").height
    total_count = eval_df.height
    accuracy_rate = (correct_count / total_count) * 100 if total_count > 0 else 0

    print("\n" + "="*60)
    print("BENCHMARK COMPLETED")
    print("="*60)
    print(f"Total Questions Evaluated: {total_count}")
    print(f"Average Quality Score: {avg_score:.2f} / 10")
    print(f"Correct: {correct_count} ({accuracy_rate:.1f}%)")
    print(f"Partially Correct: {partial_count}")
    print(f"Incorrect: {incorrect_count}")
    print("="*60 + "\n")

    # =========================================================================
    # Step 9: Save Results
    # =========================================================================
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.output.endswith('.parquet'):
        eval_df.write_parquet(args.output)
    else:
        eval_df.write_csv(args.output)
    
    logger.info(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
    main()
