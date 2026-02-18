"""Tests for COEP-specific query adjustments."""

import os
import tempfile

import polars as pl

from coep.src.query import adjust_drug_paraphernalia_queries
from legiscope.query import load_queries


class TestCoepQueryAdjustments:
    """Validate drug paraphernalia query adjustment behavior."""

    def test_context_added_for_drug_paraphernalia_dataset(self):
        df = pl.DataFrame(
            {
                "question": ["Is drug paraphernalia allowed?", "Other question"],
                "variable_name": ["q1", "q2"],
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
            assert "ordinance that prohibits drug paraphernalia" in queries[0].question
            assert "ordinance that prohibits drug paraphernalia" in queries[1].question
        finally:
            os.unlink(temp_path)

    def test_exclusion_for_monqcle_metadata_variables(self):
        df = pl.DataFrame(
            {
                "question": ["drug paraphernalia Q1", "Q2", "Q3", "Q4"],
                "variable_name": ["normal", "dp_database", "dp_url", "dp_note"],
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
