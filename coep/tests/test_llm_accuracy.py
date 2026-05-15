"""Regression tests for benchmark accuracy analysis question typing."""

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

_MODULE_PATH = PROJECT_ROOT / "coep" / "analysis" / "LLM_accuracy.py"
_SPEC = importlib.util.spec_from_file_location("test_llm_accuracy_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
llm_accuracy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = llm_accuracy
_SPEC.loader.exec_module(llm_accuracy)


def _row(
    *,
    response_options: str,
    evaluation_mode: str = "whole_answer",
    query_text: str = "",
) -> dict[str, object]:
    return {
        "evaluation_mode": evaluation_mode,
        "query": query_text,
        "query_metadata": json.dumps(
            {
                "query_text": query_text,
                "response_options": response_options,
            }
        ),
    }


class TestDeriveQuestionType:
    def test_classifies_binary_yes_no_questions(self):
        question_type = llm_accuracy.derive_question_type(
            _row(
                response_options="Responses: Yes OR No",
                query_text="Does the jurisdiction have a law?",
            )
        )

        assert question_type == "Binary"

    def test_classifies_scalar_date_questions(self):
        question_type = llm_accuracy.derive_question_type(
            _row(
                response_options="Responses: <enactment date> OR Unknown",
                query_text="On which date was the law enacted?",
            )
        )

        assert question_type == "Date"

    def test_classifies_single_option_current_through_date_questions(self):
        question_type = llm_accuracy.derive_question_type(
            _row(
                response_options="Responses: <current-through date>",
                query_text="What is the current-through date of the ordinance?",
            )
        )

        assert question_type == "Date"

    def test_classifies_ssp_current_imp_as_categorical(self):
        question_type = llm_accuracy.derive_question_type(
            _row(
                response_options=(
                    "Responses: Known, current through date published in ordinance OR "
                    "Known, partial current through date published in ordinance (month or day imputed) OR "
                    "Unknown, reflects date of data collection"
                ),
                query_text="Is the current-through date known or imputed?",
            )
        )

        assert question_type == "Categorical"

    def test_classifies_ssp_restrict_as_multi_select(self):
        question_type = llm_accuracy.derive_question_type(
            _row(
                response_options=(
                    "Responses: Cap on total number of programs or sites AND/OR "
                    "Restrictions on mobile sites AND/OR No restrictions listed"
                ),
                evaluation_mode="response_option",
                query_text="Does the ordinance require any restrictions on SSPs?",
            )
        )

        assert question_type == "Multi-select"

    def test_classifies_citation_or_unknown_as_categorical(self):
        question_type = llm_accuracy.derive_question_type(
            _row(
                response_options="Responses: <citation> OR Unknown",
                query_text="If yes, what is the citation of the relevant law?",
            )
        )

        assert question_type == "Categorical"


class TestAggregateQueryScoping:
    def test_summarize_metrics_scopes_queries_by_jurisdiction(self):
        scored_df = pl.DataFrame(
            {
                "query_instance_id": [
                    "A::benchmark:0",
                    "B::benchmark:0",
                    "B::benchmark:0",
                ],
                "eval_label": ["Correct", "Correct", "Incorrect"],
            }
        )
        dimension_df = pl.DataFrame(
            {
                "query_instance_id": ["A::benchmark:0", "B::benchmark:0"],
                "dataset": ["DPL", "DPL"],
                "jurisdiction": ["A", "B"],
            }
        )

        metrics = llm_accuracy.summarize_metrics(scored_df, dimension_df)

        assert metrics.processed_queries == 2
        assert metrics.fully_correct_queries == 1
        assert round(metrics.query_accuracy_pct, 2) == 50.0
        assert round(metrics.query_weighted_score_pct, 2) == 75.0

    def test_error_summary_excludes_none_like_error_types(self):
        scored_df = pl.DataFrame(
            {
                "eval_label": [
                    "Incorrect",
                    "Incorrect",
                    "Partially Correct",
                    "Incorrect",
                ],
                "eval_error_type": [
                    "none",
                    "retrieval_failure",
                    "Unspecified",
                    "hallucination",
                ],
            }
        )

        summary = llm_accuracy.make_error_type_summary(scored_df, top_n=10)

        rows = {row["eval_error_type"]: row["count"] for row in summary.to_dicts()}
        assert rows == {
            "retrieval_failure": 1,
            "hallucination": 1,
        }

    def test_jurisdiction_score_summary_uses_query_weighted_score(self):
        query_credit_df = pl.DataFrame(
            {
                "dataset": ["DPL", "DPL", "DPL"],
                "jurisdiction": ["A", "A", "B"],
                "query_credit": [1.0, 0.5, 0.25],
            }
        )

        summary = llm_accuracy.make_jurisdiction_score_summary(query_credit_df)

        rows = {
            (row["dataset"], row["jurisdiction"]): row
            for row in summary.to_dicts()
        }
        assert rows[("DPL", "A")]["query_count"] == 2
        assert rows[("DPL", "A")]["query_weighted_score_pct"] == 75.0
        assert rows[("DPL", "B")]["query_weighted_score_pct"] == 25.0

    def test_summarize_metrics_excludes_dependency_skipped_queries_from_denominator(self):
        scored_df = pl.DataFrame(
            {
                "query_instance_id": ["A::benchmark:0"],
                "eval_label": ["Correct"],
            }
        )
        dimension_df = pl.DataFrame(
            {
                "query_instance_id": ["A::benchmark:0", "A::benchmark:1"],
                "dataset": ["SSP", "SSP"],
                "jurisdiction": ["A", "A"],
                "counts_toward_query_metrics": [True, False],
            }
        )

        metrics = llm_accuracy.summarize_metrics(scored_df, dimension_df)

        assert metrics.processed_queries == 1
        assert metrics.scored_queries == 1
        assert metrics.fully_correct_queries == 1
        assert round(metrics.query_accuracy_pct, 2) == 100.0
