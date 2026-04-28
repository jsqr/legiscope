"""Regression tests for benchmark pipeline output helpers."""

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl

from legiscope.query import QueryInput


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

_MODULE_PATH = PROJECT_ROOT / "coep" / "scripts" / "benchmark_pipeline.py"
_SPEC = importlib.util.spec_from_file_location(
    "test_benchmark_pipeline_module", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_pipeline)


class TestBenchmarkPipelineHelpers:
    def test_ensure_generation_outcome_columns_derives_filtered_and_abstention_flags(
        self,
    ):
        df = pl.DataFrame(
            {
                "short_answer": [
                    "I cannot answer your question as no relevant legal provisions were found after filtering.",
                    "Error: timeout",
                ],
                "query_stage_status": ["no_sections_after_filtering", "error"],
            }
        )

        enriched = benchmark_pipeline._ensure_generation_outcome_columns(df)

        assert enriched[0, "generated_abstention"] is True
        assert enriched[0, "all_retrieval_units_filtered_out"] is True
        assert enriched[0, "no_retrieval_units_found"] is False
        assert enriched[1, "generated_error_response"] is True

    def test_ensure_supporting_passage_validation_columns_flags_drift_below_threshold(
        self,
    ):
        df = pl.DataFrame(
            {
                "supporting_passage_validation_scores": [
                    "[1.0, 0.95, 0.9]",
                    "[1.0, 0.89]",
                    "[]",
                ]
            }
        )

        enriched = benchmark_pipeline._ensure_supporting_passage_validation_columns(df)

        assert enriched[0, "supporting_passage_validation_drift"] is False
        assert enriched[1, "supporting_passage_validation_drift"] is True
        assert enriched[2, "supporting_passage_validation_drift"] is False

    def test_drop_redundant_query_columns_preserves_composed_query_and_metadata(self):
        df = pl.DataFrame(
            {
                "query": ["Question: ...\\n\\nCoding instructions: ..."],
                "query_metadata": ['{"question_number": "Q1.2"}'],
                "question_number": ["Q1.2"],
                "query_text": ["On which date was the ordinance enacted?"],
                "response_options": ["Responses: <enactment date> OR Unknown"],
                "coding_instructions": ["Use the enacted date if known."],
                "Deprecated": ["legacy"],
                "Deprecated Query Field": ["legacy2"],
                "": [""],
                "_duplicated_0": [""],
                "variable_name": ["dp_enacted"],
            }
        )

        cleaned = benchmark_pipeline._drop_redundant_query_columns(df)

        assert "query" in cleaned.columns
        assert "query_metadata" in cleaned.columns
        assert "variable_name" in cleaned.columns
        assert "question_number" not in cleaned.columns
        assert "query_text" not in cleaned.columns
        assert "response_options" not in cleaned.columns
        assert "coding_instructions" not in cleaned.columns
        assert "Deprecated" not in cleaned.columns
        assert "Deprecated Query Field" not in cleaned.columns
        assert "" not in cleaned.columns
        assert "_duplicated_0" not in cleaned.columns

    def test_materialize_benchmark_outputs_writes_canonical_timestamped_and_metrics(
        self, tmp_path
    ):
        final_df = pl.DataFrame({"variable_name": ["dp_enacted"], "eval_score": [8]})
        output_path = tmp_path / "benchmark_results.csv"
        timestamped_path = tmp_path / "benchmark_results_20260421_120000.csv"
        metrics_path = tmp_path / "benchmark_metrics.json"
        metrics = {"avg_score": 8.0, "processed_queries": 1}

        benchmark_pipeline._materialize_benchmark_outputs(
            final_df=final_df,
            output_path=output_path,
            timestamped_path=timestamped_path,
            metrics=metrics,
            metrics_path=metrics_path,
        )

        assert output_path.exists()
        assert timestamped_path.exists()
        assert metrics_path.exists()
        assert output_path.read_text() == timestamped_path.read_text()
        assert json.loads(metrics_path.read_text()) == metrics

    def test_score_skipped_queries_uses_deterministic_blank_vs_nonblank_ground_truth(
        self,
    ):
        df = pl.DataFrame(
            {
                "query_status": ["skipped", "skipped"],
                "ground_truth_available": [False, True],
            }
        )

        scored = benchmark_pipeline._score_skipped_queries(df)

        assert scored[0, "eval_label"] == "Correct"
        assert scored[0, "eval_score"] == 10
        assert scored[1, "eval_label"] == "Incorrect"
        assert scored[1, "eval_score"] == 0

    def test_attach_query_metadata_columns_restores_structured_csv_fields(self):
        df = pl.DataFrame(
            {
                "variable_name": ["dp_activity", "dp_type"],
                "short_answer": ["Sales", "Pipes"],
            }
        )
        query_inputs = [
            QueryInput(
                question="Question: Which activities are prohibited?",
                variable_name="dp_activity",
                metadata={
                    "question_number": "Q1.8",
                    "query_text": "Which activities are prohibited?",
                    "response_options": "Sales AND/OR Use",
                    "coding_instructions": "Select all that apply.",
                },
            )
        ]

        enriched = benchmark_pipeline._attach_query_metadata_columns(df, query_inputs)

        assert enriched[0, "question_number"] == "Q1.8"
        assert enriched[0, "query_text"] == "Which activities are prohibited?"
        assert enriched[0, "response_options"] == "Sales AND/OR Use"
        assert enriched[0, "coding_instructions"] == "Select all that apply."
        assert enriched[1, "response_options"] is None

    def test_expand_option_level_evaluation_rows_splits_multi_select_answers(self):
        df = pl.DataFrame(
            {
                "benchmark_row_id": [0],
                "query": ["Question: Which activities are prohibited?"],
                "query_text": ["Which activities are prohibited?"],
                "variable_name": ["dp_activity"],
                "response_options": ["Sales AND/OR Use AND/OR Other"],
                "coding_instructions": ["Select all that apply."],
                "short_answer": ["Sales AND/OR Use"],
                "raw_short_answer": ["Sales AND/OR Use"],
                "reasoning": ["The ordinance prohibits sale and use."],
                "supporting_passages": ["['sale', 'use']"],
                "ground_truth": ["Use AND/OR Other"],
                "ground_truth_available": [True],
                "evaluation_status": ["scored_llm"],
            }
        )

        expanded = benchmark_pipeline._expand_option_level_evaluation_rows(df)

        assert expanded["evaluation_mode"].to_list() == [
            "response_option",
            "response_option",
            "response_option",
        ]
        assert expanded["evaluation_option"].to_list() == ["Sales", "Use", "Other"]
        assert expanded["evaluation_expected_present"].to_list() == [
            False,
            True,
            True,
        ]
        assert expanded["evaluation_generated_present"].to_list() == [
            True,
            True,
            False,
        ]
        assert (
            'Response option under evaluation: "Sales"'
            in expanded[0, "evaluation_question"]
        )
        assert "Expected presence: Present" in expanded[1, "evaluation_ground_truth"]
        assert (
            "Generated presence: Absent" in expanded[2, "evaluation_generated_answer"]
        )

    def test_expand_option_level_evaluation_rows_does_not_split_plain_or_questions(
        self,
    ):
        df = pl.DataFrame(
            {
                "benchmark_row_id": [0],
                "query": ["Question: Does the jurisdiction have a paraphernalia law?"],
                "query_text": ["Does the jurisdiction have a paraphernalia law?"],
                "variable_name": ["dp_law"],
                "response_options": ["Yes OR No"],
                "coding_instructions": ["Code YES if found."],
                "short_answer": ["Yes"],
                "raw_short_answer": ["Yes"],
                "reasoning": ["A prohibition is present."],
                "supporting_passages": ["['section']"],
                "ground_truth": ["Yes"],
                "ground_truth_available": [True],
                "evaluation_status": ["scored_llm"],
            }
        )

        expanded = benchmark_pipeline._expand_option_level_evaluation_rows(df)

        assert expanded.height == 1
        assert expanded[0, "evaluation_mode"] == "whole_answer"
        assert expanded[0, "evaluation_option"] is None

    def test_score_skipped_queries_uses_option_expectation_for_subquestions(self):
        df = pl.DataFrame(
            {
                "query_status": ["skipped", "skipped"],
                "ground_truth_available": [True, True],
                "evaluation_expected_present": [True, False],
            }
        )

        scored = benchmark_pipeline._score_skipped_queries(df)

        assert scored[0, "eval_score"] == 0
        assert scored[0, "eval_label"] == "Incorrect"
        assert scored[1, "eval_score"] == 10
        assert scored[1, "eval_label"] == "Correct"

    def test_expand_option_level_evaluation_rows_skips_open_ended_date_questions(self):
        df = pl.DataFrame(
            {
                "benchmark_row_id": [0],
                "query": ["Question: On which date was the ordinance enacted?"],
                "query_text": ["On which date was the ordinance enacted?"],
                "variable_name": ["dp_enacted"],
                "response_options": ["Responses: <enactment date> OR Unknown"],
                "coding_instructions": ["Use the enacted date if known."],
                "short_answer": ["01/01/2024"],
                "reasoning": ["The ordinance was adopted on January 1, 2024."],
                "supporting_passages": ["[]"],
                "ground_truth": ["01/01/2024"],
                "ground_truth_available": [True],
                "evaluation_status": ["scored_llm"],
            }
        )

        expanded = benchmark_pipeline._expand_option_level_evaluation_rows(df)

        assert expanded.height == 1
        assert expanded[0, "evaluation_mode"] == "whole_answer"
        assert expanded[0, "evaluation_option"] is None

    def test_summarize_scoring_methods_reports_whole_vs_option_level_rows(self):
        df = pl.DataFrame(
            {
                "benchmark_row_id": [0, 0, 1],
                "evaluation_mode": [
                    "response_option",
                    "response_option",
                    "whole_answer",
                ],
            }
        )

        summary = benchmark_pipeline._summarize_scoring_methods(df)

        assert summary == {
            "whole_answer_rows": 1,
            "response_option_rows": 2,
            "and_or_questions_scored_option_level": 1,
        }

    def test_requested_variable_names_preserves_query_order_without_duplicates(self):
        query_inputs = [
            QueryInput(question="Q1", variable_name="dp_law"),
            QueryInput(question="Q2", variable_name="dp_collected"),
            QueryInput(question="Q3", variable_name="dp_law"),
        ]

        assert benchmark_pipeline._requested_variable_names(query_inputs) == [
            "dp_law",
            "dp_collected",
        ]

    def test_build_ground_truth_df_prefers_split_variables(self):
        monqcle_row = pl.DataFrame(
            {
                "dp_collected": ["Yes"],
                "dp_valid_imp": ["By officer"],
            }
        )

        ground_truth_df = benchmark_pipeline._build_ground_truth_df(
            monqcle_row,
            ["dp_collected", "dp_valid_imp"],
        )

        assert ground_truth_df["variable_name"].to_list() == [
            "dp_collected",
            "dp_valid_imp",
        ]

    def test_build_ground_truth_df_supports_legacy_combined_variables(self):
        monqcle_row = pl.DataFrame(
            {
                "dp_collected": ["Yes"],
                "dp_valid_imp": ["By officer"],
            }
        )

        ground_truth_df = benchmark_pipeline._build_ground_truth_df(
            monqcle_row,
            ["dp_collected_combined"],
        )

        assert ground_truth_df["variable_name"].to_list() == ["dp_collected_combined"]
        assert (
            ground_truth_df[0, "ground_truth"]
            == "Collected: Yes\nValid/Imp: By officer"
        )
