from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import polars as pl


EXCLUDED_DATE_ANALYSIS_VARIABLES = {
    "ssp_enacted",
    "ssp_effective_dt",
    "ssp_collected",
    "ssp_current_imp",
    "dp_enacted",
    "dp_effective_dt",
    "dp_collected",
    "dp_valid_imp",
}

EXCLUDED_STATE_FED_ANALYSIS_VARIABLES = {
    "ssp_state_fed_reference",
    "ssp_state_fed_citation",
    "dp_state_fed_reference",
    "dp_state_fed_citation",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = normalize_text(value).lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def first_non_empty_text(*values: Any) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def parse_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    text = normalize_text(value)
    if not text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _row_variable_names(row: dict[str, Any]) -> set[str]:
    names = {
        normalize_text(row.get("variable_name")),
        normalize_text(row.get("query_id")),
        normalize_text(row.get("analysis_variable")),
        normalize_text(row.get("variable")),
    }
    metadata = parse_mapping(row.get("query_metadata"))
    for key in ("variable_name", "query_id", "question", "question_name", "name", "variable"):
        names.add(normalize_text(metadata.get(key)))
    return {name for name in names if name}


def _is_excluded_row(row: dict[str, Any]) -> bool:
    names = _row_variable_names(row)
    return bool(names & EXCLUDED_DATE_ANALYSIS_VARIABLES) or bool(
        names & EXCLUDED_STATE_FED_ANALYSIS_VARIABLES
    )


def filter_results_for_analysis(results_df: pl.DataFrame) -> pl.DataFrame:
    if results_df.is_empty():
        return results_df
    keep_mask = [not _is_excluded_row(row) for row in results_df.to_dicts()]
    if not any(keep_mask):
        return results_df.head(0)
    return results_df.filter(pl.Series(keep_mask))


def _counts_toward_query_metrics(row: dict[str, Any]) -> bool:
    explicit = row.get("counts_toward_query_metrics")
    normalized = normalize_bool(explicit)
    if normalized is not None:
        return normalized
    status = normalize_text(row.get("query_status")).lower()
    return status not in {"skipped", "not_scored", "not scored", "na", "n/a", "exclude"}


def _query_instance_id(row: dict[str, Any]) -> str:
    jurisdiction_id = normalize_text(row.get("jurisdiction_id"))
    identifier = first_non_empty_text(
        row.get("query_instance_id"),
        row.get("query_id"),
        row.get("variable_name"),
        row.get("benchmark_row_id"),
        row.get("evaluation_row_id"),
        row.get("question_name"),
    )
    if not identifier:
        identifier = f"row-{row.get('row_number', '')}"
    return f"{jurisdiction_id}::{identifier}" if jurisdiction_id else identifier


def _score_label(row: dict[str, Any]) -> str:
    return normalize_text(row.get("eval_label")).lower()


def summarize_jurisdiction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    filtered_rows = [row for row in rows if not _is_excluded_row(row)]
    if not filtered_rows:
        return {}

    jurisdiction_id = first_non_empty_text(
        filtered_rows[0].get("jurisdiction_id"),
        filtered_rows[0].get("jurisdiction"),
    )
    jurisdiction = first_non_empty_text(
        filtered_rows[0].get("jurisdiction"),
        filtered_rows[0].get("jurisdiction_id"),
    )

    scored_rows = [row for row in filtered_rows if _score_label(row)]
    scored_row_count = len(scored_rows)
    correct_rows = sum(1 for row in scored_rows if _score_label(row) == "correct")
    partial_rows = sum(1 for row in scored_rows if _score_label(row) == "partially correct")
    incorrect_rows = sum(1 for row in scored_rows if _score_label(row) == "incorrect")

    eval_scores: list[float] = []
    for row in scored_rows:
        raw_score = row.get("eval_score")
        if raw_score in (None, ""):
            continue
        try:
            eval_scores.append(float(raw_score))
        except (TypeError, ValueError):
            continue

    processed_query_ids: set[str] = set()
    query_groups: dict[str, dict[str, int]] = {}
    for row in filtered_rows:
        query_id = _query_instance_id(row)
        if _counts_toward_query_metrics(row):
            processed_query_ids.add(query_id)
        label = _score_label(row)
        if not label:
            continue
        group = query_groups.setdefault(query_id, {"scored": 0, "correct": 0})
        group["scored"] += 1
        if label == "correct":
            group["correct"] += 1

    scored_query_count = len(query_groups)
    fully_correct_queries = sum(
        1 for group in query_groups.values() if group["scored"] > 0 and group["correct"] == group["scored"]
    )
    query_credit_sum = sum(
        (group["correct"] / group["scored"])
        for group in query_groups.values()
        if group["scored"] > 0
    )

    row_accuracy = (correct_rows / scored_row_count) * 100 if scored_row_count else 0.0
    query_weighted_score = (query_credit_sum / scored_query_count) * 100 if scored_query_count else 0.0
    query_accuracy = (
        (fully_correct_queries / len(processed_query_ids)) * 100 if processed_query_ids else 0.0
    )
    collapsed_query_accuracy = (
        (fully_correct_queries / scored_query_count) * 100 if scored_query_count else 0.0
    )

    eval_error_type_counts = Counter(
        normalize_text(row.get("eval_error_type"))
        for row in scored_rows
        if normalize_text(row.get("eval_error_type"))
    )
    eval_error_type_refined_counts = Counter(
        normalize_text(row.get("eval_error_type_refined"))
        for row in scored_rows
        if normalize_text(row.get("eval_error_type_refined"))
    )

    summary: dict[str, Any] = {
        "jurisdiction": jurisdiction,
        "jurisdiction_id": jurisdiction_id,
        "primary_score": round(query_weighted_score, 2),
        "primary_score_label": "weighted_query_score",
        "query_weighted_score": round(query_weighted_score, 2),
        "weighted_query_score": round(query_weighted_score, 2),
        "weighted_query_score_percent": round(query_weighted_score, 2),
        "weighted_query_points_per_query": round(100 / scored_query_count, 2) if scored_query_count else 0.0,
        "weighted_query_scored_point_ceiling": 100.0 if scored_query_count else 0.0,
        "weighted_query_scored": scored_query_count,
        "weighted_query_unscored": max(len(processed_query_ids) - scored_query_count, 0),
        "avg_score": round(mean(eval_scores), 2) if eval_scores else None,
        "accuracy_rate": round(row_accuracy, 2),
        "expanded_accuracy_rate": round(row_accuracy, 2),
        "correct": correct_rows,
        "partially_correct": partial_rows,
        "incorrect": incorrect_rows,
        "processed_queries": len(processed_query_ids),
        "scored_queries": scored_query_count,
        "unscored_queries": max(len(processed_query_ids) - scored_query_count, 0),
        "core_benchmark_queries": scored_query_count,
        "collapsed_query_accuracy_rate": round(collapsed_query_accuracy, 2),
        "collapsed_query_correct": fully_correct_queries,
        "collapsed_query_incorrect": max(scored_query_count - fully_correct_queries, 0),
        "collapsed_query_scored": scored_query_count,
        "collapsed_query_unscored": max(len(processed_query_ids) - scored_query_count, 0),
        "collapsed_query_coverage_rate": (
            round((scored_query_count / len(processed_query_ids)) * 100, 2)
            if processed_query_ids
            else 0.0
        ),
        "query_accuracy_rate": round(query_accuracy, 2),
        "total": scored_row_count,
        "whole_answer_scored_rows": scored_row_count,
        "and_or_option_level_scored_rows": 0,
        "and_or_questions_scored_option_level": 0,
        "eval_error_type_counts": dict(sorted(eval_error_type_counts.items())),
        "eval_error_type_refined_counts": dict(sorted(eval_error_type_refined_counts.items())),
    }

    for key in ("slurm_job_id", "code_slug", "docx_path", "batch_id", "batch_submitted_at"):
        summary[key] = first_non_empty_text(*(row.get(key) for row in filtered_rows)) or None

    return summary


def summarize_benchmark_results(raw_df: pl.DataFrame) -> pl.DataFrame:
    if raw_df.is_empty():
        return raw_df

    filtered_df = filter_results_for_analysis(raw_df)
    if filtered_df.is_empty():
        return pl.DataFrame()

    rows: list[dict[str, Any]] = []
    jurisdiction_ids = sorted(
        {
            normalize_text(jurisdiction_id)
            for jurisdiction_id in filtered_df.get_column("jurisdiction_id").to_list()
            if normalize_text(jurisdiction_id)
        }
    )
    for jurisdiction_id in jurisdiction_ids:
        jurisdiction_rows = [
            row
            for row in filtered_df.to_dicts()
            if normalize_text(row.get("jurisdiction_id")) == jurisdiction_id
        ]
        summary = summarize_jurisdiction_rows(jurisdiction_rows)
        if summary:
            rows.append(summary)

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def summarize_batch_metrics(metrics_df: pl.DataFrame) -> dict[str, Any]:
    if metrics_df.is_empty():
        return {
            "evaluation_row_score": 0.0,
            "query_weighted_score": 0.0,
            "whole_query_correct_alt": 0,
            "whole_query_scored_alt": 0,
            "whole_query_score_alt": 0.0,
            "best_jurisdiction_id": None,
            "best_jurisdiction_dataset": None,
            "best_jurisdiction_score": None,
            "lowest_jurisdiction_id": None,
            "lowest_jurisdiction_dataset": None,
            "lowest_jurisdiction_score": None,
        }

    result: dict[str, Any] = {}
    if {"correct", "total"}.issubset(metrics_df.columns):
        total_rows = float(metrics_df["total"].sum())
        correct_rows = float(metrics_df["correct"].sum())
        if total_rows > 0:
            result["evaluation_row_score"] = round((correct_rows / total_rows) * 100, 2)

    if {"primary_score", "core_benchmark_queries"}.issubset(metrics_df.columns):
        total_queries = float(metrics_df["core_benchmark_queries"].sum())
        if total_queries > 0:
            weighted_score = (
                (metrics_df["primary_score"] * metrics_df["core_benchmark_queries"]).sum()
                / total_queries
            )
            result["query_weighted_score"] = round(float(weighted_score), 2)

    if {"collapsed_query_correct", "collapsed_query_scored"}.issubset(metrics_df.columns):
        total_query_correct = float(metrics_df["collapsed_query_correct"].sum())
        total_query_scored = float(metrics_df["collapsed_query_scored"].sum())
        result["whole_query_correct_alt"] = int(total_query_correct)
        result["whole_query_scored_alt"] = int(total_query_scored)
        if total_query_scored > 0:
            result["whole_query_score_alt"] = round((total_query_correct / total_query_scored) * 100, 2)

    if "query_weighted_score" in metrics_df.columns and metrics_df.height > 0:
        ordered = metrics_df.sort("query_weighted_score", descending=True)
        best_row = ordered.row(0, named=True)
        worst_row = ordered.row(ordered.height - 1, named=True)
        result["best_jurisdiction_id"] = best_row.get("jurisdiction_id")
        result["best_jurisdiction_dataset"] = best_row.get("jurisdiction")
        result["best_jurisdiction_score"] = best_row.get("query_weighted_score")
        result["lowest_jurisdiction_id"] = worst_row.get("jurisdiction_id")
        result["lowest_jurisdiction_dataset"] = worst_row.get("jurisdiction")
        result["lowest_jurisdiction_score"] = worst_row.get("query_weighted_score")

    return result


def load_benchmark_csv(benchmark_path: Path) -> pl.DataFrame:
    return pl.read_csv(benchmark_path)