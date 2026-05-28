"""Regression tests for filtered benchmark metric aggregation."""

import importlib.util
import sys
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "coep" / "scripts" / "HPC_scripts"
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src", SCRIPT_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

_MODULE_PATH = SCRIPT_DIR / "accuracy_metrics.py"
_SPEC = importlib.util.spec_from_file_location("test_accuracy_metrics_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
accuracy_metrics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(accuracy_metrics)


def test_summarize_benchmark_results_excludes_date_and_state_fed_rows():
    raw_df = pl.DataFrame(
        {
            "jurisdiction_id": ["CA-LosAngeles"] * 5,
            "jurisdiction": ["CA-LosAngeles"] * 5,
            "variable_name": [
                "valid_var_1",
                "valid_var_1",
                "valid_var_2",
                "dp_enacted",
                "dp_state_fed_reference",
            ],
            "query_id": ["q1", "q1", "q2", "q3", "q4"],
            "benchmark_row_id": ["1", "2", "3", "4", "5"],
            "eval_label": ["Correct", "Incorrect", "Correct", "Correct", "Correct"],
            "eval_score": [10, 0, 10, 10, 10],
            "query_status": ["answered", "answered", "answered", "answered", "answered"],
            "counts_toward_query_metrics": [True, True, True, True, True],
        }
    )

    metrics_df = accuracy_metrics.summarize_benchmark_results(raw_df)

    assert metrics_df.height == 1
    row = metrics_df.to_dicts()[0]
    assert row["total"] == 3
    assert row["correct"] == 2
    assert row["accuracy_rate"] == 66.67
    assert row["core_benchmark_queries"] == 2
    assert row["query_weighted_score"] == 75.0
    assert row["collapsed_query_accuracy_rate"] == 50.0


def test_summarize_batch_metrics_uses_scored_query_denominator():
    metrics_df = pl.DataFrame(
        [
            {
                "jurisdiction_id": "CA-LosAngeles",
                "jurisdiction": "CA-LosAngeles",
                "primary_score": 75.0,
                "query_weighted_score": 75.0,
                "core_benchmark_queries": 2,
                "correct": 2,
                "total": 3,
                "collapsed_query_correct": 1,
                "collapsed_query_scored": 2,
            }
        ]
    )

    summary = accuracy_metrics.summarize_batch_metrics(metrics_df)

    assert summary["evaluation_row_score"] == 66.67
    assert summary["query_weighted_score"] == 75.0
    assert summary["whole_query_score_alt"] == 50.0