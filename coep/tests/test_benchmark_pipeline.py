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

    def test_ensure_supporting_passage_validation_columns_flags_drift_and_not_found(
        self,
    ):
        df = pl.DataFrame(
            {
                "supporting_passage_validation_match_types": [
                    '["exact", "exact"]',
                    '["exact", "near_exact"]',
                    '["exact", "not_found"]',
                    "[]",
                ]
            }
        )

        enriched = benchmark_pipeline._ensure_supporting_passage_validation_columns(df)

        assert enriched[0, "supporting_passage_validation_drift"] is False
        assert enriched[1, "supporting_passage_validation_drift"] is True
        assert enriched[2, "supporting_passage_validation_drift"] is False
        assert enriched[0, "supporting_passage_validation_not_found"] is False
        assert enriched[1, "supporting_passage_validation_not_found"] is False
        assert enriched[2, "supporting_passage_validation_not_found"] is True
        assert enriched[3, "supporting_passage_validation_not_found"] is False

    def test_ensure_supporting_passage_validation_columns_falls_back_to_scores(self):
        df = pl.DataFrame(
            {
                "supporting_passage_validation_scores": [
                    "[1.0, 0.95]",
                    "[1.0, 0.89]",
                    "[]",
                ]
            }
        )

        enriched = benchmark_pipeline._ensure_supporting_passage_validation_columns(df)

        assert enriched[0, "supporting_passage_validation_drift"] is True
        assert enriched[0, "supporting_passage_validation_not_found"] is False
        assert enriched[1, "supporting_passage_validation_drift"] is False
        assert enriched[1, "supporting_passage_validation_not_found"] is True
        assert enriched[2, "supporting_passage_validation_drift"] is False
        assert enriched[2, "supporting_passage_validation_not_found"] is False

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

    def test_materialize_benchmark_outputs_serializes_nested_columns_for_csv(
        self, tmp_path
    ):
        final_df = pl.DataFrame(
            {
                "variable_name": ["dp_activity"],
                "match_types": [["exact", "near_exact"]],
                "answer_payload": [{"label": "Sales", "score": 1.0}],
            }
        )
        output_path = tmp_path / "benchmark_results.csv"
        timestamped_path = output_path
        metrics_path = tmp_path / "benchmark_metrics.json"

        benchmark_pipeline._materialize_benchmark_outputs(
            final_df=final_df,
            output_path=output_path,
            timestamped_path=timestamped_path,
            metrics={"processed_queries": 1},
            metrics_path=metrics_path,
        )

        written_df = pl.read_csv(output_path)

        assert written_df[0, "variable_name"] == "dp_activity"
        assert json.loads(written_df[0, "match_types"]) == ["exact", "near_exact"]
        assert json.loads(written_df[0, "answer_payload"]) == {
            "label": "Sales",
            "score": 1.0,
        }

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

    def test_score_skipped_queries_empty_frame_keeps_eval_schema(self):
        df = pl.DataFrame(
            schema={
                "evaluation_row_id": pl.Int64,
                "evaluation_generated_answer": pl.String,
                "ground_truth_available": pl.Boolean,
            }
        )

        scored = benchmark_pipeline._score_skipped_queries(df)

        assert scored.is_empty()
        assert "eval_score" in scored.columns
        assert "eval_reason" in scored.columns
        assert "eval_label" in scored.columns
        assert "eval_error_type" in scored.columns

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

    def test_expand_option_level_evaluation_rows_ignores_new_suffix_in_ground_truth(
        self,
    ):
        df = pl.DataFrame(
            {
                "benchmark_row_id": [0],
                "query": [
                    "Question: Does the ordinance specify any penalties for violations?"
                ],
                "query_text": [
                    "Does the ordinance specify any penalties for violations?"
                ],
                "variable_name": ["dp_penalties"],
                "response_options": ["Civil Fine AND/OR Other"],
                "coding_instructions": ["Select all that apply."],
                "short_answer": ["Civil Fine"],
                "raw_short_answer": ["Civil Fine"],
                "reasoning": ["The ordinance imposes a civil fine."],
                "supporting_passages": ["['civil fine']"],
                "ground_truth": ["Civil Fine (NEW)"],
                "ground_truth_available": [True],
                "evaluation_status": ["scored_llm"],
            }
        )

        expanded = benchmark_pipeline._expand_option_level_evaluation_rows(df)

        assert expanded["evaluation_option"].to_list() == ["Civil Fine", "Other"]
        assert expanded["evaluation_expected_present"].to_list() == [True, False]
        assert expanded["evaluation_generated_present"].to_list() == [True, False]

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

    def test_score_option_level_queries_uses_presence_flags_deterministically(self):
        df = pl.DataFrame(
            {
                "benchmark_row_id": [0, 1, 2],
                "evaluation_mode": [
                    "response_option",
                    "response_option",
                    "response_option",
                ],
                "evaluation_row_id": [10, 11, 12],
                "evaluation_generated_answer": ["a", "b", "c"],
                "evaluation_expected_present": [True, False, False],
                "evaluation_generated_present": [True, False, True],
            }
        )

        scored = benchmark_pipeline._score_option_level_queries(df)

        assert scored["eval_score"].to_list() == [10, 10, 0]
        assert scored["eval_label"].to_list() == [
            "Correct",
            "Correct",
            "Incorrect",
        ]
        assert scored["eval_error_type"].to_list() == ["none", "none", "other"]
        assert "correctly omitted this option" in scored[1, "eval_reason"]
        assert (
            "included this option even though the ground truth omits it"
            in scored[2, "eval_reason"]
        )

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

    def test_summarize_collapsed_query_accuracy_requires_all_subrows_correct(self):
        df = pl.DataFrame(
            {
                "benchmark_row_id": [0, 0, 1, 2],
                "eval_label": [
                    "Correct",
                    "Correct",
                    "Incorrect",
                    "Partially Correct",
                ],
            }
        )

        summary = benchmark_pipeline._summarize_collapsed_query_accuracy(
            df,
            total_queries=4,
        )

        assert summary["processed_queries"] == 4
        assert summary["scored_queries"] == 3
        assert summary["unscored_queries"] == 1
        assert summary["correct_queries"] == 1
        assert summary["incorrect_queries"] == 3
        assert round(summary["accuracy_rate"], 2) == 25.00

    def test_summarize_weighted_query_score_awards_partial_credit_by_option(self):
        df = pl.DataFrame(
            {
                "benchmark_row_id": [0, 0, 1],
                "eval_label": [
                    "Correct",
                    "Incorrect",
                    "Correct",
                ],
            }
        )

        summary = benchmark_pipeline._summarize_weighted_query_score(
            df,
            total_queries=3,
        )

        assert summary["processed_queries"] == 3
        assert summary["scored_queries"] == 2
        assert summary["unscored_queries"] == 1
        assert round(summary["points_per_query"], 4) == 33.3333
        assert round(summary["scored_point_ceiling"], 2) == 66.67
        assert round(summary["earned_points"], 2) == 50.00
        assert round(summary["score_percent"], 2) == 50.00

    def test_summarize_weighted_query_score_treats_unscored_queries_as_zero_points(
        self,
    ):
        df = pl.DataFrame(
            {
                "benchmark_row_id": [1],
                "eval_label": ["Correct"],
            }
        )

        summary = benchmark_pipeline._summarize_weighted_query_score(
            df,
            total_queries=4,
        )

        assert summary["scored_queries"] == 1
        assert summary["unscored_queries"] == 3
        assert round(summary["earned_points"], 2) == 25.00
        assert round(summary["score_percent"], 2) == 25.00

    def test_eval_concat_preserves_columns_needed_for_scoring_summary(self):
        llm_eval_scored_df = pl.DataFrame(
            {
                "benchmark_row_id": [0],
                "evaluation_mode": ["response_option"],
                "evaluation_row_id": [10],
                "evaluation_generated_answer": ["option payload"],
                "eval_score": [10],
                "eval_reason": ["matched"],
                "eval_label": ["Correct"],
                "eval_error_type": ["other"],
            }
        )
        skipped_eval_df = pl.DataFrame(
            {
                "benchmark_row_id": [1],
                "evaluation_mode": ["whole_answer"],
                "evaluation_row_id": [11],
                "evaluation_generated_answer": ["whole answer payload"],
                "eval_score": [0],
                "eval_reason": ["skipped despite ground truth"],
                "eval_label": ["Incorrect"],
                "eval_error_type": ["other"],
            }
        )

        eval_scored_df = pl.concat(
            [
                llm_eval_scored_df.select(
                    [
                        "benchmark_row_id",
                        "evaluation_mode",
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
                        "benchmark_row_id",
                        "evaluation_mode",
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

        assert benchmark_pipeline._summarize_scoring_methods(eval_scored_df) == {
            "whole_answer_rows": 1,
            "response_option_rows": 1,
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
