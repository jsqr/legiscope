"""Tests for COEP-specific query adjustments."""

import os
import tempfile

import polars as pl

from coep.src.query import adjust_drug_paraphernalia_queries
from legiscope.query import load_queries


class TestCoepQueryAdjustments:
    """Validate drug paraphernalia query adjustment behavior."""

    def test_query_composed_from_structured_columns(self):
        df = pl.DataFrame(
            {
                "question_number": ["Q1", "Q2"],
                "variable_name": ["dp_law", "dp_type"],
                "prepend_text": ["This is about drug paraphernalia.", ""],
                "query_text": [
                    "Does the jurisdiction ban paraphernalia?",
                    "What types?",
                ],
                "response_options": ["Yes OR No", "Syringes AND/OR Pipes"],
                "coding_instructions": ["Code YES if found.", "Select all that apply."],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name

        try:
            queries = load_queries(
                temp_path,
                adjust_for_dataset=True,
                query_adjuster=adjust_drug_paraphernalia_queries,
            )
            assert len(queries) == 2
            # Check that structured parts are present with headers
            assert "Context: This is about drug paraphernalia." in queries[0].question
            assert (
                "Question: Does the jurisdiction ban paraphernalia?"
                in queries[0].question
            )
            assert "Coding instructions: Code YES if found." in queries[0].question
            assert "Response options: Yes OR No" in queries[0].question
            # variable renamed to variable_name
            assert queries[0].variable_name == "dp_law"
            # Empty prepend_text should be omitted
            assert "Context:" not in queries[1].question
            assert "Question: What types?" in queries[1].question
        finally:
            os.unlink(temp_path)

    def test_exclusion_for_monqcle_metadata_variables(self):
        df = pl.DataFrame(
            {
                "question_number": ["Q1", "Q2", "Q3", "Q4"],
                "variable_name": ["normal", "dp_database", "dp_url", "dp_note"],
                "prepend_text": ["ctx", "ctx", "ctx", "ctx"],
                "query_text": ["Q1", "Q2", "Q3", "Q4"],
                "response_options": ["", "", "", ""],
                "coding_instructions": ["", "", "", ""],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name

        try:
            queries = load_queries(
                temp_path,
                adjust_for_dataset=True,
                query_adjuster=adjust_drug_paraphernalia_queries,
            )
            assert len(queries) == 1
            assert queries[0].variable_name == "normal"
        finally:
            os.unlink(temp_path)
