"""
Tests for the eval module.
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch
import polars as pl

from legiscope.coep.eval import (
    Evaluator,
    EvaluationResult,
    jurisdiction_id_to_monqcle_name,
    load_and_filter_monqcle,
    melt_monqcle_to_long,
)


class TestJurisdictionMapping:
    """Test jurisdiction ID to name mapping."""

    def test_known_jurisdiction(self):
        """Test mapping for known jurisdiction."""
        name = jurisdiction_id_to_monqcle_name("CA-LosAngeles")
        assert name == "Los Angeles, Los Angeles County, California, United States"

    def test_unknown_jurisdiction(self):
        """Test error for unknown jurisdiction."""
        with pytest.raises(ValueError, match="Unknown jurisdiction ID"):
            jurisdiction_id_to_monqcle_name("Unknown-Place")


class TestMonqcleLoading:
    """Test loading and filtering of MonQcle data."""

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

        # Check content
        var1_row = result.filter(pl.col("variable_name") == "var1")
        assert var1_row["ground_truth"][0] == "answer1"

        var2_row = result.filter(pl.col("variable_name") == "var2")
        assert var2_row["ground_truth"][0] == "answer2"

    def test_melt_null_values(self):
        """Test handling of null/dash values."""
        row = pl.DataFrame({"var1": ["-"], "var2": [None]})

        result = melt_monqcle_to_long(row, ["var1", "var2"])

        assert result.filter(pl.col("variable_name") == "var1")["ground_truth"][0] == ""
        assert result.filter(pl.col("variable_name") == "var2")["ground_truth"][0] == ""

    def test_melt_missing_columns(self):
        """Test behavior when requested variables are missing from data."""
        row = pl.DataFrame({"var1": ["val1"]})

        # 'var2' is missing from data
        result = melt_monqcle_to_long(row, ["var1", "var2"])

        # Should only return var1
        assert len(result) == 1
        assert result["variable_name"][0] == "var1"


class TestEvaluator:
    """Test Evaluator class."""

    def test_evaluate_response(self):
        """Test single response evaluation with mocked LLM."""
        mock_client = Mock()

        # Create evaluator with mock client
        # We need to mock Config.get_powerful_client if we don't pass llm_config
        with patch("legiscope.coep.eval.Config") as mock_config:
            mock_config.get_powerful_client.return_value = mock_client
            mock_config.get_powerful_model.return_value = "mock_model"

            evaluator = Evaluator()

            # Setup mock response
            expected_result = EvaluationResult(
                score=10, reasoning="Perfect match", accuracy_label="Correct"
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

        with patch("legiscope.coep.eval.Config") as mock_config:
            mock_config.get_powerful_client.return_value = mock_client
            mock_config.get_powerful_model.return_value = "mock_model"

            evaluator = Evaluator()

            # Mock successful response
            mock_client.chat.completions.create.return_value = EvaluationResult(
                score=10, reasoning="Good", accuracy_label="Correct"
            )

            df = pl.DataFrame({"q": ["q1"], "a": ["a1"], "truth": ["t1"]})

            result_df = evaluator.evaluate_batch(
                df, question_col="q", answer_col="a", truth_col="truth"
            )

            assert "eval_score" in result_df.columns
            assert "eval_reason" in result_df.columns
            assert "eval_label" in result_df.columns
            assert result_df["eval_score"][0] == 10
            assert result_df["eval_label"][0] == "Correct"
