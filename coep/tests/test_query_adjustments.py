"""Tests for COEP-specific query adjustments."""

import os
import tempfile

import polars as pl

from coep.src.query import adjust_drug_paraphernalia_queries
from legiscope.query import load_queries
from legiscope.query_hierarchy import (
    REQUIRES_DATA_COLUMN,
    REQUIRES_LABELS_COLUMN,
    REQUIRES_YES_COLUMN,
)


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
            # Check that completion-oriented parts are present with headers
            assert (
                "Question: Does the jurisdiction ban paraphernalia?"
                in queries[0].question
            )
            assert "Coding instructions: Code YES if found." in queries[0].question
            assert "Response options: Yes OR No" in queries[0].question
            # variable renamed to variable_name
            assert queries[0].variable_name == "dp_law"
            # prepend_text remains in metadata, not the composed question
            assert (
                queries[0].metadata["prepend_text"]
                == "This is about drug paraphernalia."
            )
            assert "Context:" not in queries[0].question
            assert "Context:" not in queries[1].question
            assert "Question: What types?" in queries[1].question
        finally:
            os.unlink(temp_path)

    def test_query_loader_normalizes_title_cased_coep_headers(self):
        df = pl.DataFrame(
            {
                "Question": ["Fallback question that should be replaced"],
                "Variable": ["dp_law"],
                "Prepend text": ["This is about drug paraphernalia."],
                "Query text": ["Does the jurisdiction ban paraphernalia?"],
                "Response options": ["Yes OR No"],
                "Coding instructions": ["Code YES if found."],
                'Requires ""yes"" from upstream question:': [""],
                "Requires data from upstream question:": [""],
                "Requires label(s) from upstream question:": [""],
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
            assert (
                "Question: Does the jurisdiction ban paraphernalia?"
                in queries[0].question
            )
            assert queries[0].variable_name == "dp_law"
            assert (
                queries[0].metadata["prepend_text"]
                == "This is about drug paraphernalia."
            )
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

    def test_split_query_dependency_columns_survive_adjustment(self):
        df = pl.DataFrame(
            {
                "question_number": ["Q0", "Q1", "Q1.1", "Q2", "Q2.1"],
                "variable_name": [
                    "dp_type",
                    "dp_exemption",
                    "dp_exempt_can_activity",
                    "dp_collected",
                    "dp_valid_imp",
                ],
                "prepend_text": [
                    "Context.",
                    "Context.",
                    "Context.",
                    "Context.",
                    "Context.",
                ],
                "query_text": [
                    "What types of paraphernalia are covered?",
                    "Which exemptions exist?",
                    "If cannabis paraphernalia is exempted, which activities are exempted?",
                    "What is the current through date?",
                    "Is the current-through date known or imputed?",
                ],
                "response_options": [
                    "Syringes AND/OR Pipes",
                    "Paraphernalia for consumption of cannabis, generally OR None",
                    "Sales AND/OR Use",
                    "01/01/2024 OR Unknown",
                    "Known OR Imputed",
                ],
                "coding_instructions": [
                    "Select all that apply.",
                    "Select the best option.",
                    "Select all that apply.",
                    "Enter the date if known.",
                    "Select the best option.",
                ],
                REQUIRES_YES_COLUMN: ["", "", "dp_exemption", "", ""],
                REQUIRES_DATA_COLUMN: [
                    "",
                    "dp_type",
                    "dp_exemption||dp_activity",
                    "",
                    "dp_collected",
                ],
                REQUIRES_LABELS_COLUMN: [
                    "",
                    "",
                    "dp_exemption => Paraphernalia for consumption of cannabis, generally",
                    "",
                    "",
                ],
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

            assert [query.query_id for query in queries] == [
                "Q0",
                "Q1",
                "Q1.1",
                "Q2",
                "Q2.1",
            ]
            exemption_query = queries[1]
            child_query = queries[2]
            same_surface_child = queries[4]
            assert (
                exemption_query.metadata["disable_inherited_retrieval_from"]
                == "dp_type"
            )
            assert (
                child_query.metadata["disable_inherited_retrieval_from"]
                == "dp_exemption||dp_activity"
            )
            assert (
                same_surface_child.metadata["disable_inherited_retrieval_from"] is None
            )
            assert child_query.metadata["hierarchy"] == {
                "query_id": "Q1.1",
                "parent_ids": ["dp_exemption", "dp_activity"],
                "boolean_parent_ids": ["dp_exemption"],
                "context_parent_ids": ["dp_exemption", "dp_activity"],
                "pass_parent_question": True,
                "pass_parent_short_answer": True,
                "label_blockers": [
                    {
                        "parent_query_id": "dp_exemption",
                        "blocker_labels": [
                            "Paraphernalia for consumption of cannabis, generally"
                        ],
                    }
                ],
                "inherit_parent_retrieval": True,
            }
            assert (
                "Question: If cannabis paraphernalia is exempted, which activities are exempted?"
                in child_query.question
            )
        finally:
            os.unlink(temp_path)
