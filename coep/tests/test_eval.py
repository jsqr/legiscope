"""
Tests for the eval module.
"""

import os
import tempfile
import threading
import time
from unittest.mock import Mock, patch

import polars as pl
import pytest

from coep.src.eval import (
    Evaluator,
    EvaluationResult,
    expected_series_title_for_monqcle_report,
    expand_combined_variables,
    jurisdiction_id_to_monqcle_name,
    load_and_filter_monqcle,
    melt_monqcle_to_long,
    prepare_ground_truth_for_variables,
    prioritize_ground_truth_matches,
)


class TestJurisdictionMapping:
    """Test jurisdiction ID to name mapping."""

    def test_known_jurisdiction(self):
        """Test mapping for known jurisdiction."""
        name = jurisdiction_id_to_monqcle_name("CA-LosAngeles")
        assert name == "Los Angeles, Los Angeles County, California, United States"

    def test_pa_philadelphia_mapping(self):
        """Test mapping for Philadelphia jurisdiction."""
        name = jurisdiction_id_to_monqcle_name("PA-Philadelphia")
        assert name == "Philadelphia, Philadelphia County, Pennsylvania, United States"

    def test_unknown_jurisdiction(self):
        """Test error for unknown jurisdiction."""
        with pytest.raises(ValueError, match="Unknown jurisdiction ID"):
            jurisdiction_id_to_monqcle_name("Unknown-Place")


class TestMonqcleLoading:
    """Test loading and filtering of MonQcle data."""

    def test_expected_series_title_for_known_report_names(self):
        assert (
            expected_series_title_for_monqcle_report(
                "coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report_20260501.csv"
            )
            == "DPL_2025_Consolidated"
        )
        assert (
            expected_series_title_for_monqcle_report(
                "coep/data/monqcle_data/SSP_Laws_Standard_Report_20260513.csv"
            )
            == "SSP_2025_Consolidated"
        )

    def test_load_and_filter_success(self):
        """Test successful loading and filtering."""
        # Create dummy CSV
        df = pl.DataFrame(
            {
                "name": ["Jurisdiction A", "Jurisdiction B"],
                "series_title": ["Series 1", "Series 1"],
                "var1": ["val1", "val2"],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name

        try:
            result = load_and_filter_monqcle(
                temp_path, "Jurisdiction A", series_title="Series 1"
            )
            assert len(result) == 1
            assert result["name"][0] == "Jurisdiction A"
            assert result["var1"][0] == "val1"
        finally:
            os.unlink(temp_path)

    def test_load_and_filter_not_found(self):
        """Test error when record not found."""
        df = pl.DataFrame({"name": ["Jurisdiction A"], "series_title": ["Series 1"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="No records found"):
                load_and_filter_monqcle(
                    temp_path, "Jurisdiction B", series_title="Series 1"
                )
        finally:
            os.unlink(temp_path)

    def test_load_and_filter_falls_back_to_only_available_series(self):
        """A single-series report should still load when the requested series differs."""
        df = pl.DataFrame(
            {
                "name": ["Jurisdiction A"],
                "series_title": ["Series SSP"],
                "var1": ["val1"],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name

        try:
            result = load_and_filter_monqcle(
                temp_path, "Jurisdiction A", series_title="Series DPL"
            )
            assert len(result) == 1
            assert result["series_title"][0] == "Series SSP"
        finally:
            os.unlink(temp_path)

    def test_load_and_filter_prefers_more_recent_through_to(self):
        df = pl.DataFrame(
            {
                "name": ["Jurisdiction A", "Jurisdiction A"],
                "series_title": ["Series 1", "Series 1"],
                "through_to": ["2024-01-31", "2024-12-31"],
                "var1": ["older", "newer"],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name

        try:
            result = load_and_filter_monqcle(
                temp_path, "Jurisdiction A", series_title="Series 1"
            )
            assert len(result) == 1
            assert result["through_to"][0] == "2024-12-31"
            assert result["var1"][0] == "newer"
        finally:
            os.unlink(temp_path)


class TestMonqcleMelting:
    """Test reshaping of MonQcle data."""

    def test_melt_success(self):
        """Test basic melting functionality."""
        row = pl.DataFrame(
            {
                "name": ["Jusrisdiction A"],
                "var1": ["answer1"],
                "var2": ["answer2"],
                "unused": ["ignore"],
            }
        )

        variable_names = ["var1", "var2"]

        result = melt_monqcle_to_long(row, variable_names)

        assert len(result) == 2
        assert result.schema["variable_name"] == pl.String
        assert result.schema["ground_truth"] == pl.String
        assert result.schema["ground_truth_citation"] == pl.String

        # Check content
        var1_row = result.filter(pl.col("variable_name") == "var1")
        assert var1_row["ground_truth"][0] == "answer1"
        assert var1_row["ground_truth_citation"][0] == ""

        var2_row = result.filter(pl.col("variable_name") == "var2")
        assert var2_row["ground_truth"][0] == "answer2"
        assert var2_row["ground_truth_citation"][0] == ""

    def test_melt_includes_ground_truth_citations_when_present(self):
        """Citation companion columns should be propagated into long-format output."""
        row = pl.DataFrame(
            {
                "var1": ["answer1"],
                "_citations_var1": ["Section 1"],
            }
        )

        result = melt_monqcle_to_long(row, ["var1"])

        assert result["ground_truth_citation"][0] == "Section 1"

    def test_melt_null_values(self):
        """Test handling of null/dash values."""
        row = pl.DataFrame({"var1": ["-"], "var2": [None]})

        result = melt_monqcle_to_long(row, ["var1", "var2"])

        assert result.filter(pl.col("variable_name") == "var1")["ground_truth"][0] == ""
        assert result.filter(pl.col("variable_name") == "var2")["ground_truth"][0] == ""
        assert (
            result.filter(pl.col("variable_name") == "var1")["ground_truth_citation"][0]
            == ""
        )
        assert (
            result.filter(pl.col("variable_name") == "var2")["ground_truth_citation"][0]
            == ""
        )

    def test_melt_missing_columns(self):
        """Test behavior when requested variables are missing from data."""
        row = pl.DataFrame({"var1": ["val1"]})

        # 'var2' is missing from data
        result = melt_monqcle_to_long(row, ["var1", "var2"])

        # Should only return var1
        assert len(result) == 1
        assert result["variable_name"][0] == "var1"


class TestCombinedVariableExpansion:
    """Test expansion of combined MonQcle query variables."""

    def test_expand_adds_requested_combined_columns(self):
        """Requested combined variables should be added as new columns."""
        row = pl.DataFrame(
            {
                "dp_collected": ["Yes"],
                "dp_valid_imp": ["By officer"],
                "dp_state_fed_reference": ["Yes"],
                "dp_state_fed_citation": ["21 U.S.C. 863"],
            }
        )

        result = expand_combined_variables(
            row,
            ["dp_collected_combined", "dp_state_fed_combined"],
        )

        assert "dp_collected_combined" in result.columns
        assert "dp_state_fed_combined" in result.columns
        assert "_citations_dp_collected_combined" in result.columns
        assert "_citations_dp_state_fed_combined" in result.columns
        assert (
            result["dp_collected_combined"][0]
            == "Collected: Yes\nValid/Imp: By officer"
        )
        assert result["dp_state_fed_combined"][0] == (
            "References state/federal law: Yes\nCitation: 21 U.S.C. 863"
        )

    def test_expand_combined_variables_propagates_source_citations(self):
        """Combined variables should carry merged citation provenance when available."""
        row = pl.DataFrame(
            {
                "dp_collected": ["Yes"],
                "dp_valid_imp": ["By officer"],
                "_citations_dp_collected": ["Header note"],
                "_citations_dp_valid_imp": ["Edition footer"],
            }
        )

        result = expand_combined_variables(row, ["dp_collected_combined"])

        assert result["_citations_dp_collected_combined"][0] == (
            "Collected: Header note\nValid/Imp: Edition footer"
        )

    def test_expand_handles_dash_and_none_values(self):
        """Dash and null source values should be normalized to empty strings."""
        row = pl.DataFrame(
            {
                "dp_collected": ["-"],
                "dp_valid_imp": [None],
            }
        )

        result = expand_combined_variables(row, ["dp_collected_combined"])

        assert result["dp_collected_combined"][0] == "Collected: \nValid/Imp:"

    def test_expand_is_noop_when_no_combined_variables_requested(self):
        """When no combined variables are requested, the input row is unchanged."""
        row = pl.DataFrame({"var1": ["value1"], "var2": ["value2"]})

        result = expand_combined_variables(row, ["var1"])

        assert result.equals(row)

    def test_prepare_ground_truth_prefers_split_variables(self):
        row = pl.DataFrame(
            {
                "dp_collected": ["Yes"],
                "dp_valid_imp": ["By officer"],
            }
        )

        result = prepare_ground_truth_for_variables(
            row,
            ["dp_collected", "dp_valid_imp"],
        )

        assert result["variable_name"].to_list() == ["dp_collected", "dp_valid_imp"]
        assert result["ground_truth"].to_list() == ["Yes", "By officer"]

    def test_prepare_ground_truth_supports_legacy_combined_compatibility(self):
        row = pl.DataFrame(
            {
                "dp_collected": ["Yes"],
                "dp_valid_imp": ["By officer"],
            }
        )

        result = prepare_ground_truth_for_variables(
            row,
            ["dp_collected_combined"],
        )

        assert result["variable_name"].to_list() == ["dp_collected_combined"]
        assert result["ground_truth"][0] == "Collected: Yes\nValid/Imp: By officer"


class TestBenchmarkResultOrdering:
    """Test ordering of benchmark result rows after the left join."""

    def test_prioritize_ground_truth_matches_places_scored_rows_first(self):
        results = pl.DataFrame(
            {
                "benchmark_row_id": [2, 0, 3, 1],
                "variable_name": [
                    "var_missing_1",
                    "var_scored_1",
                    "var_missing_2",
                    "var_scored_2",
                ],
                "ground_truth_available": [False, True, False, True],
            }
        )

        ordered = prioritize_ground_truth_matches(results)

        assert ordered["variable_name"].to_list() == [
            "var_scored_1",
            "var_scored_2",
            "var_missing_1",
            "var_missing_2",
        ]
        assert ordered["benchmark_row_id"].to_list() == [0, 1, 2, 3]

    def test_prioritize_ground_truth_matches_preserves_option_subquestion_order(self):
        results = pl.DataFrame(
            {
                "benchmark_row_id": [0, 0, 0, 1],
                "evaluation_subquestion_index": [2, 0, 1, 0],
                "variable_name": [
                    "var_scored_1",
                    "var_scored_1",
                    "var_scored_1",
                    "var_scored_2",
                ],
                "ground_truth_available": [True, True, True, True],
            }
        )

        ordered = prioritize_ground_truth_matches(results)

        assert ordered["benchmark_row_id"].to_list() == [0, 0, 0, 1]
        assert ordered["evaluation_subquestion_index"].to_list() == [0, 1, 2, 0]


class TestEvaluator:
    """Test Evaluator class."""

    def test_evaluation_result_schema_like_payload_is_unwrapped(self):
        """Schema-wrapped payloads from vLLM should parse as EvaluationResult."""
        payload = {
            "description": "Structured output for the evaluation of a single query response.",
            "properties": {
                "score": 0,
                "reasoning": "The generated answer is incorrect.",
                "accuracy_label": "Incorrect",
                "error_type": "retrieval_failure",
            },
            "title": "EvaluationResult",
            "type": "object",
        }

        result = EvaluationResult.model_validate(payload)

        assert result.score == 0
        assert result.reasoning == "The generated answer is incorrect."
        assert result.accuracy_label == "Incorrect"
        assert result.error_type == "retrieval_failure"

    def test_evaluation_result_accepts_new_error_types(self):
        result = EvaluationResult.model_validate(
            {
                "score": 0,
                "reasoning": "The model timed out.",
                "accuracy_label": "Incorrect",
                "error_type": "llm_failure",
            }
        )

        assert result.error_type == "llm_failure"

    def test_evaluate_response(self):
        """Test single response evaluation with mocked LLM."""
        mock_client = Mock()

        # Config is imported inside Evaluator.__init__, so patch at source
        with patch("legiscope.llm_config.Config") as mock_config:
            mock_config.get_powerful_client.return_value = mock_client
            mock_config.get_llm_params.return_value = {
                "temperature": 0.0,
                "max_retries": 3,
            }

            evaluator = Evaluator()

            # Setup mock response
            expected_result = EvaluationResult(
                score=10,
                reasoning="Perfect match",
                accuracy_label="Correct",
                error_type="none",
            )
            mock_client.chat.completions.create.return_value = expected_result

            result = evaluator.evaluate_response(
                question="Q", generated_answer="A", ground_truth="A"
            )

            assert result == expected_result
            mock_client.chat.completions.create.assert_called_once()

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        mock_client = Mock()

        with patch("legiscope.llm_config.Config") as mock_config:
            mock_config.get_powerful_client.return_value = mock_client
            mock_config.get_llm_params.return_value = {
                "temperature": 0.0,
                "max_retries": 3,
            }

            evaluator = Evaluator()

            # Mock successful response
            mock_client.chat.completions.create.return_value = EvaluationResult(
                score=10,
                reasoning="Good",
                accuracy_label="Correct",
                error_type="none",
            )

            df = pl.DataFrame({"q": ["q1"], "a": ["a1"], "truth": ["t1"]})

            result_df = evaluator.evaluate_batch(
                df, question_col="q", answer_col="a", truth_col="truth"
            )

            assert "eval_score" in result_df.columns
            assert "eval_reason" in result_df.columns
            assert "eval_label" in result_df.columns
            assert "eval_error_type" in result_df.columns
            assert result_df["eval_score"][0] == 10
            assert result_df["eval_label"][0] == "Correct"
            assert result_df["eval_error_type"][0] == "none"

    def test_evaluate_batch_parallel_preserves_row_order(self):
        """Parallel evaluation should keep outputs aligned to input row order."""
        mock_client = Mock()

        with patch("legiscope.llm_config.Config") as mock_config:
            mock_config.get_powerful_client.return_value = mock_client
            mock_config.get_llm_params.return_value = {
                "temperature": 0.0,
                "max_retries": 3,
            }

            evaluator = Evaluator(max_concurrency=4)

        delays = {"q1": 0.05, "q2": 0.01, "q3": 0.03}
        score_by_question = {"q1": 1, "q2": 2, "q3": 3}

        def fake_evaluate_response(question, generated_answer, ground_truth):
            time.sleep(delays[question])
            return EvaluationResult(
                score=score_by_question[question],
                reasoning=f"reason-{question}",
                accuracy_label="Correct",
                error_type="none",
            )

        evaluator.evaluate_response = fake_evaluate_response

        df = pl.DataFrame(
            {
                "q": ["q1", "q2", "q3"],
                "a": ["a1", "a2", "a3"],
                "truth": ["t1", "t2", "t3"],
            }
        )

        result_df = evaluator.evaluate_batch(
            df, question_col="q", answer_col="a", truth_col="truth"
        )

        assert result_df["eval_score"].to_list() == [1, 2, 3]
        assert result_df["eval_reason"].to_list() == [
            "reason-q1",
            "reason-q2",
            "reason-q3",
        ]

    def test_evaluate_batch_parallel_executes_more_than_one_request_at_once(self):
        """Parallel evaluation should issue overlapping judge calls when enabled."""
        mock_client = Mock()

        with patch("legiscope.llm_config.Config") as mock_config:
            mock_config.get_powerful_client.return_value = mock_client
            mock_config.get_llm_params.return_value = {
                "temperature": 0.0,
                "max_retries": 3,
            }

            evaluator = Evaluator(max_concurrency=4)

        state = {
            "active": 0,
            "max_active": 0,
        }
        lock = threading.Lock()
        release = threading.Event()

        def fake_evaluate_response(question, generated_answer, ground_truth):
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                if state["max_active"] >= 2:
                    release.set()

            release.wait(timeout=0.2)

            with lock:
                state["active"] -= 1

            return EvaluationResult(
                score=10,
                reasoning=f"ok-{question}",
                accuracy_label="Correct",
                error_type="none",
            )

        evaluator.evaluate_response = fake_evaluate_response

        df = pl.DataFrame(
            {
                "q": ["q1", "q2", "q3"],
                "a": ["a1", "a2", "a3"],
                "truth": ["t1", "t2", "t3"],
            }
        )

        result_df = evaluator.evaluate_batch(
            df, question_col="q", answer_col="a", truth_col="truth"
        )

        assert result_df["eval_label"].to_list() == [
            "Correct",
            "Correct",
            "Correct",
        ]
        assert state["max_active"] >= 2

    def test_evaluator_disables_parallelism_for_external_default_source(self):
        """Default evaluator concurrency should stay off for external APIs."""
        mock_client = Mock()

        with patch("legiscope.llm_config.Config") as mock_config:
            mock_config.uses_self_hosted_llm.return_value = False
            mock_config.get_powerful_client.return_value = mock_client
            mock_config.get_llm_params.return_value = {
                "temperature": 0.0,
                "max_retries": 3,
            }

            evaluator = Evaluator(max_concurrency=4)

        assert evaluator.max_concurrency == 1

    def test_evaluator_allows_parallelism_for_self_hosted_custom_llm(self):
        """Custom self-hosted configs can opt into concurrency with an explicit factory."""
        from legiscope.utils import LLMConfig

        factory = Mock(return_value=Mock())
        evaluator = Evaluator(
            llm_config=LLMConfig(
                client=Mock(),
                model="test-model",
                source="self_hosted",
                client_factory=factory,
            ),
            max_concurrency=4,
        )

        assert evaluator.max_concurrency == 4
