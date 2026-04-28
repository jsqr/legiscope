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
import ast
import json
import shutil
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
from legiscope.query import (
    BatchQuerySettings,
    _clean_response_options,
    _is_scalar_date_response_options,
    _is_status_date_response_options,
    _normalize_option_text,
    _normalize_structured_short_answer,
    _split_response_options,
    load_queries,
    run_queries,
)
from coep.src.eval import (
    Evaluator,
    jurisdiction_id_to_monqcle_name,
    load_and_filter_monqcle,
    prepare_ground_truth_for_variables,
    prioritize_ground_truth_matches,
)
from coep.src.query import adjust_drug_paraphernalia_queries
from coep.src.retrieval_guidance import get_drug_paraphernalia_retrieval_guidance


_BENCHMARK_RESULT_QUERY_COLUMNS_TO_DROP = [
    "coding_instructions",
    "query_text",
    "question_number",
    "response_options",
]


def _should_drop_benchmark_output_column(column_name: str) -> bool:
    """Return whether a benchmark output column should be removed as export noise."""
    normalized = column_name.strip().lower()

    if column_name in _BENCHMARK_RESULT_QUERY_COLUMNS_TO_DROP:
        return True
    if not normalized:
        return True
    if normalized.startswith("_duplicated_"):
        return True
    if "deprecated" in normalized:
        return True

    return False


def _drop_redundant_query_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop query subfields already captured in the composed query and metadata."""
    columns_to_drop = [
        column for column in df.columns if _should_drop_benchmark_output_column(column)
    ]
    if not columns_to_drop:
        return df
    return df.drop(columns_to_drop)


def _ensure_generation_outcome_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure benchmark exports always carry comparable query-outcome flags."""
    derived_columns: list[pl.Expr] = []

    if "generated_abstention" not in df.columns:
        derived_columns.append(
            pl.col("short_answer")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.starts_with("i cannot answer your question")
            .alias("generated_abstention")
        )

    if "generated_error_response" not in df.columns:
        derived_columns.append(
            pl.col("short_answer")
            .cast(pl.Utf8)
            .str.starts_with("Error:")
            .alias("generated_error_response")
        )

    if "no_retrieval_units_found" not in df.columns:
        derived_columns.append(
            pl.col("query_stage_status")
            .cast(pl.Utf8)
            .eq("no_sections")
            .alias("no_retrieval_units_found")
        )

    if "all_retrieval_units_filtered_out" not in df.columns:
        derived_columns.append(
            pl.col("query_stage_status")
            .cast(pl.Utf8)
            .eq("no_sections_after_filtering")
            .alias("all_retrieval_units_filtered_out")
        )

    if not derived_columns:
        return df

    return df.with_columns(derived_columns)


def _has_supporting_passage_validation_drift(value: object) -> bool:
    """Return True when any supporting-passage validation score falls below 0.9."""
    if value is None:
        return False

    scores = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        try:
            scores = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return False

    if not isinstance(scores, (list, tuple)):
        return False

    for score in scores:
        try:
            if float(score) < 0.9:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _ensure_supporting_passage_validation_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure benchmark exports surface supporting-passage validation drift."""
    if "supporting_passage_validation_drift" in df.columns:
        return df
    if "supporting_passage_validation_scores" not in df.columns:
        return df

    return df.with_columns(
        pl.col("supporting_passage_validation_scores")
        .map_elements(
            _has_supporting_passage_validation_drift,
            return_dtype=pl.Boolean,
        )
        .alias("supporting_passage_validation_drift")
    )


def _build_query_metadata_df(query_inputs) -> pl.DataFrame:
    """Return per-variable benchmark query metadata needed for evaluation."""
    records: list[dict[str, object]] = []
    seen: set[str] = set()

    for query_input in query_inputs:
        variable_name = str(query_input.variable_name or "").strip()
        if not variable_name or variable_name in seen:
            continue

        metadata = query_input.metadata or {}
        records.append(
            {
                "variable_name": variable_name,
                "question_number": metadata.get("question_number"),
                "query_text": metadata.get("query_text"),
                "response_options": metadata.get("response_options"),
                "coding_instructions": metadata.get("coding_instructions"),
            }
        )
        seen.add(variable_name)

    if not records:
        return pl.DataFrame(
            schema={
                "variable_name": pl.String,
                "question_number": pl.String,
                "query_text": pl.String,
                "response_options": pl.String,
                "coding_instructions": pl.String,
            }
        )

    return pl.DataFrame(records)


def _attach_query_metadata_columns(df: pl.DataFrame, query_inputs) -> pl.DataFrame:
    """Reattach structured query metadata that run_queries intentionally omits."""
    metadata_df = _build_query_metadata_df(query_inputs)
    if metadata_df.is_empty():
        return df

    overlapping_columns = [
        column
        for column in metadata_df.columns
        if column != "variable_name" and column in df.columns
    ]
    enriched = df.join(metadata_df, on="variable_name", how="left", suffix="_query")

    if not overlapping_columns:
        return enriched

    coalesced_columns: list[pl.Expr] = []
    duplicate_columns: list[str] = []
    for column in overlapping_columns:
        duplicate_column = f"{column}_query"
        if duplicate_column not in enriched.columns:
            continue
        coalesced_columns.append(
            pl.coalesce(pl.col(column), pl.col(duplicate_column)).alias(column)
        )
        duplicate_columns.append(duplicate_column)

    if coalesced_columns:
        enriched = enriched.with_columns(coalesced_columns)
    if duplicate_columns:
        enriched = enriched.drop(duplicate_columns)
    return enriched


def _expandable_response_options(response_options: object) -> list[str] | None:
    """Return discrete options suitable for AND/OR option-level evaluation."""
    cleaned = _clean_response_options(response_options)
    if not cleaned:
        return None
    if _is_scalar_date_response_options(cleaned):
        return None
    if _is_status_date_response_options(cleaned):
        return None

    options, separator = _split_response_options(cleaned)
    if separator != " AND/OR " or len(options) < 2:
        return None

    if any("<" in option and ">" in option for option in options):
        return None

    return options


def _selected_option_keys(
    answer: object,
    *,
    variable_name: object,
    response_options: object,
    coding_instructions: object,
) -> set[str]:
    """Return canonical option keys selected by an answer."""
    cleaned_response_options = _clean_response_options(response_options)
    if not cleaned_response_options:
        return set()

    metadata = {
        "response_options": cleaned_response_options,
        "coding_instructions": str(coding_instructions or ""),
    }
    canonical_answer = _normalize_structured_short_answer(
        str(answer or ""),
        str(variable_name or "").strip() or None,
        metadata,
    )
    if not canonical_answer:
        return set()

    if cleaned_response_options == "Yes, <citation> OR No":
        normalized_answer = _normalize_option_text(canonical_answer)
        if normalized_answer.startswith("yes"):
            return {_normalize_option_text("Yes")}
        if normalized_answer == _normalize_option_text("No"):
            return {_normalize_option_text("No")}
        return set()

    selected_options, _separator = _split_response_options(canonical_answer)
    return {
        _normalize_option_text(str(option).strip())
        for option in selected_options
        if str(option).strip()
    }


def _build_option_level_question(base_question: str, option: str) -> str:
    """Return the judge-facing subquestion for a discrete response option."""
    return "\n".join(
        [
            f"Original question: {base_question}",
            f'Response option under evaluation: "{option}"',
            (
                "Evaluate only whether this specific option should be listed. "
                "The subquestion is correct only if the generated answer includes "
                "the option when the ground truth includes it, and omits it when "
                "the ground truth omits it."
            ),
        ]
    )


def _build_option_level_ground_truth(
    option: str,
    *,
    expected_present: bool,
    original_ground_truth: object,
) -> str:
    """Return the ground-truth payload for an option-level evaluation row."""
    lines = [
        f"Response option: {option}",
        f"Expected presence: {'Present' if expected_present else 'Absent'}",
    ]
    ground_truth_text = str(original_ground_truth or "").strip()
    if ground_truth_text:
        lines.append(f"Original ground truth answer: {ground_truth_text}")
    return "\n".join(lines)


def _build_option_level_generated_answer(
    row: dict[str, object],
    *,
    option: str,
    generated_present: bool,
) -> str:
    """Return the generated-answer payload for an option-level evaluation row."""
    lines = [
        f"Response option: {option}",
        f"Generated presence: {'Present' if generated_present else 'Absent'}",
    ]

    short_answer = str(row.get("short_answer") or "").strip()
    if short_answer:
        lines.append(f"Original generated short answer: {short_answer}")

    raw_short_answer = str(row.get("raw_short_answer") or "").strip()
    if raw_short_answer and raw_short_answer != short_answer:
        lines.append(f"Original raw short answer: {raw_short_answer}")

    reasoning = str(row.get("reasoning") or "").strip()
    if reasoning:
        lines.append(f"Reasoning: {reasoning}")

    supporting_passages = str(row.get("supporting_passages") or "").strip()
    if supporting_passages:
        lines.append(f"Supporting Passages: {supporting_passages}")

    return "\n\n".join(lines)


def _expand_option_level_evaluation_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Explode discrete benchmark rows into per-option evaluation subquestions."""
    if df.is_empty():
        return df

    expanded_rows: list[dict[str, object]] = []
    for row in df.to_dicts():
        row_copy = dict(row)
        row_copy.setdefault("evaluation_mode", "whole_answer")
        row_copy.setdefault("evaluation_subquestion_index", 0)
        row_copy.setdefault("evaluation_option", None)
        row_copy.setdefault("evaluation_expected_present", None)
        row_copy.setdefault("evaluation_generated_present", None)
        row_copy.setdefault(
            "evaluation_question",
            str(row.get("query_text") or row.get("query") or "").strip(),
        )
        row_copy.setdefault("evaluation_ground_truth", row.get("ground_truth"))
        row_copy.setdefault("evaluation_generated_answer", None)

        if row.get("evaluation_status") not in {"scored_llm", "scored_skipped"}:
            expanded_rows.append(row_copy)
            continue
        if not row.get("ground_truth_available"):
            expanded_rows.append(row_copy)
            continue

        options = _expandable_response_options(row.get("response_options"))
        if not options:
            expanded_rows.append(row_copy)
            continue

        response_options = row.get("response_options")
        coding_instructions = row.get("coding_instructions")
        variable_name = row.get("variable_name")
        expected_option_keys = _selected_option_keys(
            row.get("ground_truth"),
            variable_name=variable_name,
            response_options=response_options,
            coding_instructions=coding_instructions,
        )
        generated_option_keys = _selected_option_keys(
            row.get("short_answer"),
            variable_name=variable_name,
            response_options=response_options,
            coding_instructions=coding_instructions,
        )
        base_question = str(row.get("query_text") or row.get("query") or "").strip()

        for option_index, option in enumerate(options):
            option_key = _normalize_option_text(str(option).strip())
            expected_present = option_key in expected_option_keys
            generated_present = option_key in generated_option_keys
            expanded_row = dict(row)
            expanded_row.update(
                {
                    "evaluation_mode": "response_option",
                    "evaluation_subquestion_index": option_index,
                    "evaluation_option": option,
                    "evaluation_expected_present": expected_present,
                    "evaluation_generated_present": generated_present,
                    "evaluation_question": _build_option_level_question(
                        base_question,
                        option,
                    ),
                    "evaluation_ground_truth": _build_option_level_ground_truth(
                        option,
                        expected_present=expected_present,
                        original_ground_truth=row.get("ground_truth"),
                    ),
                    "evaluation_generated_answer": _build_option_level_generated_answer(
                        row,
                        option=option,
                        generated_present=generated_present,
                    ),
                }
            )
            expanded_rows.append(expanded_row)

    return pl.DataFrame(expanded_rows)


def _ensure_evaluation_prompt_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Populate evaluation prompt columns for both whole-answer and subquestion rows."""
    if df.is_empty():
        return df

    comprehensive_answer_expr = pl.concat_str(
        [
            pl.col("short_answer"),
            pl.lit("\n\nReasoning: "),
            pl.col("reasoning"),
            pl.lit("\n\nSupporting Passages: "),
            pl.col("supporting_passages"),
        ]
    ).alias("_fallback_comprehensive_answer")

    fallback_question = pl.when(pl.col("query_text").is_not_null())
    fallback_question = fallback_question.then(pl.col("query_text")).otherwise(
        pl.col("query")
    )

    return (
        df.with_columns(comprehensive_answer_expr)
        .with_columns(
            [
                pl.coalesce(
                    pl.col("evaluation_question"),
                    fallback_question,
                ).alias("evaluation_question"),
                pl.coalesce(
                    pl.col("evaluation_ground_truth"),
                    pl.col("ground_truth"),
                ).alias("evaluation_ground_truth"),
                pl.coalesce(
                    pl.col("evaluation_generated_answer"),
                    pl.col("_fallback_comprehensive_answer"),
                ).alias("evaluation_generated_answer"),
            ]
        )
        .drop("_fallback_comprehensive_answer")
    )


def _score_skipped_queries(df: pl.DataFrame) -> pl.DataFrame:
    """Score skipped rows deterministically instead of sending them to the judge LLM."""
    if df.is_empty():
        return df

    has_option_presence = "evaluation_expected_present" in df.columns
    evaluation_expected_present = (
        pl.col("evaluation_expected_present")
        if has_option_presence
        else pl.lit(None, dtype=pl.Boolean)
    )

    return df.with_columns(
        [
            pl.when(pl.col("ground_truth_available"))
            .then(
                pl.when(evaluation_expected_present.is_not_null())
                .then(
                    pl.when(evaluation_expected_present)
                    .then(pl.lit(0))
                    .otherwise(pl.lit(10))
                )
                .otherwise(pl.lit(0))
            )
            .otherwise(pl.lit(10))
            .alias("eval_score"),
            pl.when(pl.col("ground_truth_available"))
            .then(
                pl.when(evaluation_expected_present.is_not_null())
                .then(
                    pl.when(evaluation_expected_present)
                    .then(
                        pl.lit(
                            "The query was skipped by an explicit dependency rule, and this option should have been listed according to the ground truth."
                        )
                    )
                    .otherwise(
                        pl.lit(
                            "The query was skipped by an explicit dependency rule, and this option should not have been listed according to the ground truth."
                        )
                    )
                )
                .otherwise(
                    pl.lit(
                        "The query was skipped by an explicit dependency rule even though ground truth was present."
                    )
                )
            )
            .otherwise(
                pl.lit(
                    "The query was skipped by an explicit dependency rule and the corresponding ground truth was blank or unavailable."
                )
            )
            .alias("eval_reason"),
            pl.when(pl.col("ground_truth_available"))
            .then(
                pl.when(evaluation_expected_present.is_not_null())
                .then(
                    pl.when(evaluation_expected_present)
                    .then(pl.lit("Incorrect"))
                    .otherwise(pl.lit("Correct"))
                )
                .otherwise(pl.lit("Incorrect"))
            )
            .otherwise(pl.lit("Correct"))
            .alias("eval_label"),
            pl.lit("other").alias("eval_error_type"),
        ]
    )


def _requested_variable_names(query_inputs) -> list[str]:
    """Return distinct benchmark variable names in query order."""
    seen: set[str] = set()
    variable_names: list[str] = []
    for query_input in query_inputs:
        variable_name = query_input.variable_name
        if not variable_name or variable_name in seen:
            continue
        seen.add(variable_name)
        variable_names.append(variable_name)
    return variable_names


def _build_ground_truth_df(
    monqcle_row: pl.DataFrame,
    variable_names: list[str],
) -> pl.DataFrame:
    """Prepare long-form ground truth with split variables primary."""
    return prepare_ground_truth_for_variables(monqcle_row, variable_names)


def _summarize_eval_error_types(df: pl.DataFrame) -> dict[str, int]:
    """Return compact counts for evaluation error types."""
    if "eval_error_type" not in df.columns or df.is_empty():
        return {}

    counts: dict[str, int] = {}
    for row in df.group_by("eval_error_type").len().iter_rows(named=True):
        error_type = str(row["eval_error_type"] or "")
        counts[error_type] = int(row["len"])
    return counts


def _summarize_scoring_methods(df: pl.DataFrame) -> dict[str, int]:
    """Return counts describing whole-answer versus AND/OR option-level scoring."""
    if df.is_empty() or "evaluation_mode" not in df.columns:
        return {
            "whole_answer_rows": 0,
            "response_option_rows": 0,
            "and_or_questions_scored_option_level": 0,
        }

    whole_answer_rows = df.filter(pl.col("evaluation_mode") == "whole_answer").height
    response_option_rows = df.filter(
        pl.col("evaluation_mode") == "response_option"
    ).height

    and_or_questions_scored_option_level = 0
    if "benchmark_row_id" in df.columns:
        and_or_questions_scored_option_level = (
            df.filter(pl.col("evaluation_mode") == "response_option")
            .select("benchmark_row_id")
            .n_unique()
        )

    return {
        "whole_answer_rows": int(whole_answer_rows),
        "response_option_rows": int(response_option_rows),
        "and_or_questions_scored_option_level": int(
            and_or_questions_scored_option_level
        ),
    }


def _materialize_benchmark_outputs(
    *,
    final_df: pl.DataFrame,
    output_path: Path,
    timestamped_path: Path,
    metrics: dict[str, object],
    metrics_path: Path,
) -> None:
    """Write benchmark outputs and ensure the canonical DVC out is materialized."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_df.write_csv(str(output_path))
    logger.info(f"Results saved to {output_path}")

    if timestamped_path != output_path:
        shutil.copy2(output_path, timestamped_path)
        logger.info(f"Timestamped copy saved to {timestamped_path}")

    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics saved to {metrics_path}")


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
    variable_names = _requested_variable_names(query_inputs)
    ground_truth_df = _build_ground_truth_df(monqcle_row, variable_names)

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
    gen_results_df = _ensure_generation_outcome_columns(gen_results_df)
    gen_results_df = _ensure_supporting_passage_validation_columns(gen_results_df)
    gen_results_df = _attach_query_metadata_columns(gen_results_df, query_inputs)

    # =========================================================================
    # Step 6: Join with Ground Truth
    # =========================================================================
    logger.info("Joining generated answers with ground truth...")

    joined_df = (
        gen_results_df.join(ground_truth_df, on="variable_name", how="left")
        .with_row_index("benchmark_row_id")
        .with_columns(
            (
                pl.col("ground_truth").is_not_null()
                & (pl.col("ground_truth").str.strip_chars() != "")
                & (pl.col("ground_truth") != "-")
            ).alias("ground_truth_available")
        )
        .with_columns(
            pl.when(pl.col("query_status").cast(pl.Utf8).eq("skipped"))
            .then(pl.lit("scored_skipped"))
            .when(pl.col("ground_truth_available"))
            .then(pl.lit("scored_llm"))
            .otherwise(pl.lit("missing_ground_truth"))
            .alias("evaluation_status")
        )
    )
    joined_df = _expand_option_level_evaluation_rows(joined_df)
    joined_df = _ensure_evaluation_prompt_columns(joined_df).with_row_index(
        "evaluation_row_id"
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

    llm_eval_input_df = joined_df.filter(pl.col("evaluation_status") == "scored_llm")
    skipped_eval_df = _score_skipped_queries(
        joined_df.filter(pl.col("evaluation_status") == "scored_skipped")
    )

    if len(llm_eval_input_df) == 0 and len(skipped_eval_df) == 0:
        logger.error("No rows with ground truth to evaluate!")
        sys.exit(1)

    llm_eval_scored_df = llm_eval_input_df
    if len(llm_eval_input_df) > 0:
        llm_eval_scored_df = evaluator.evaluate_batch(
            llm_eval_scored_df,
            question_col="evaluation_question",
            answer_col="evaluation_generated_answer",
            truth_col="evaluation_ground_truth",
        )
    else:
        llm_eval_scored_df = pl.DataFrame(
            schema={
                "evaluation_row_id": pl.Int64,
                "evaluation_generated_answer": pl.String,
                "eval_score": pl.Int64,
                "eval_reason": pl.String,
                "eval_label": pl.String,
                "eval_error_type": pl.String,
            }
        )

    skipped_eval_df = skipped_eval_df.with_columns(
        pl.col("evaluation_generated_answer")
        .cast(pl.String)
        .alias("evaluation_generated_answer")
    )
    eval_scored_df = pl.concat(
        [
            llm_eval_scored_df.select(
                [
                    "evaluation_row_id",
                    "evaluation_generated_answer",
                    "eval_score",
                    "eval_reason",
                    "eval_label",
                    "eval_error_type",
                ]
            ),
            skipped_eval_df.select(
                [
                    "evaluation_row_id",
                    "evaluation_generated_answer",
                    "eval_score",
                    "eval_reason",
                    "eval_label",
                    "eval_error_type",
                ]
            ),
        ],
        how="vertical_relaxed",
    )

    final_df = joined_df.join(
        eval_scored_df.select(
            [
                "evaluation_row_id",
                "evaluation_generated_answer",
                "eval_score",
                "eval_reason",
                "eval_label",
                "eval_error_type",
            ]
        ),
        on="evaluation_row_id",
        how="left",
    ).with_columns(
        [
            pl.lit(jurisdiction_id).alias("jurisdiction_id"),
            pl.col("evaluation_generated_answer").alias("comprehensive_answer"),
        ]
    )
    final_df = prioritize_ground_truth_matches(final_df)
    final_df = _drop_redundant_query_columns(final_df)

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
    no_retrieval_units_count = final_df.filter(
        pl.col("no_retrieval_units_found")
    ).height
    filtered_out_all_units_count = final_df.filter(
        pl.col("all_retrieval_units_filtered_out")
    ).height
    abstention_count = final_df.filter(pl.col("generated_abstention")).height
    error_response_count = final_df.filter(pl.col("generated_error_response")).height
    supporting_passage_validation_drift_count = final_df.filter(
        pl.col("supporting_passage_validation_drift")
    ).height
    accuracy_rate = (correct_count / scored_count) * 100 if scored_count > 0 else 0
    eval_error_type_counts = _summarize_eval_error_types(eval_scored_df)
    scoring_method_counts = _summarize_scoring_methods(eval_scored_df)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)
    print(f"Total Queries Processed: {processed_count}")
    print(f"Queries Scored Against Ground Truth: {scored_count}")
    print(f"Queries Unscored (missing/excluded ground truth): {unscored_count}")
    print(f"Whole-answer scored rows: {scoring_method_counts['whole_answer_rows']}")
    print(
        "AND/OR option-level scored rows: "
        f"{scoring_method_counts['response_option_rows']}"
    )
    print(
        "Original AND/OR questions scored option-level: "
        f"{scoring_method_counts['and_or_questions_scored_option_level']}"
    )
    print(f"Average Quality Score: {avg_score:.2f} / 10")
    print(f"Correct: {correct_count} ({accuracy_rate:.1f}%)")
    print(f"Partially Correct: {partial_count}")
    print(f"Incorrect: {incorrect_count}")
    print("=" * 60 + "\n")

    # =========================================================================
    # Step 9: Save Results
    # =========================================================================
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
        "queries_with_no_retrieval_units": no_retrieval_units_count,
        "queries_filtered_to_zero_units": filtered_out_all_units_count,
        "abstained_queries": abstention_count,
        "error_response_queries": error_response_count,
        "supporting_passage_validation_drift_queries": supporting_passage_validation_drift_count,
        "whole_answer_scored_rows": scoring_method_counts["whole_answer_rows"],
        "and_or_option_level_scored_rows": scoring_method_counts[
            "response_option_rows"
        ],
        "and_or_questions_scored_option_level": scoring_method_counts[
            "and_or_questions_scored_option_level"
        ],
        "eval_error_type_counts": eval_error_type_counts,
        "total": scored_count,
    }
    _materialize_benchmark_outputs(
        final_df=final_df,
        output_path=output_path,
        timestamped_path=timestamped_path,
        metrics=metrics,
        metrics_path=metrics_path,
    )


if __name__ == "__main__":
    main()
