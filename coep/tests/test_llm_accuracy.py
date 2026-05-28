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
    def test_filter_results_for_analysis_excludes_requested_variables(self):
        raw_df = pl.DataFrame(
            {
                "variable_name": ["ssp_enacted", None, "dp_possession"],
                "query_id": [None, "dp_state_fed_reference", None],
                "query_metadata": [
                    json.dumps({"query_id": "ssp_enacted"}),
                    json.dumps({"query_id": "dp_state_fed_reference"}),
                    json.dumps({"query_id": "dp_possession"}),
                ],
                "jurisdiction_id": ["A", "A", "A"],
                "evaluation_mode": ["whole_answer", "whole_answer", "whole_answer"],
                "eval_label": ["Correct", "Incorrect", "Correct"],
                "eval_score": [1.0, 0.0, 1.0],
            }
        )

        filtered = llm_accuracy.filter_results_for_analysis(raw_df)

        assert filtered["variable_name"].to_list() == ["dp_possession"]

    def test_filter_results_for_analysis_can_include_date_variables(self):
        raw_df = pl.DataFrame(
            {
                "variable_name": ["ssp_enacted", "dp_possession"],
                "query_id": [None, None],
                "query_metadata": [
                    json.dumps({"query_id": "ssp_enacted"}),
                    json.dumps({"query_id": "dp_possession"}),
                ],
                "jurisdiction_id": ["A", "A"],
                "evaluation_mode": ["whole_answer", "whole_answer"],
                "eval_label": ["Correct", "Correct"],
                "eval_score": [1.0, 1.0],
            }
        )

        filtered = llm_accuracy.filter_results_for_analysis(
            raw_df,
            exclude_date_variables=False,
            exclude_state_fed_reference_variables=True,
        )

        assert filtered["variable_name"].to_list() == ["ssp_enacted", "dp_possession"]

    def test_filter_results_for_analysis_can_include_state_fed_reference_variables(self):
        raw_df = pl.DataFrame(
            {
                "variable_name": ["dp_state_fed_reference", "dp_possession"],
                "query_id": [None, None],
                "query_metadata": [
                    json.dumps({"query_id": "dp_state_fed_reference"}),
                    json.dumps({"query_id": "dp_possession"}),
                ],
                "jurisdiction_id": ["A", "A"],
                "evaluation_mode": ["whole_answer", "whole_answer"],
                "eval_label": ["Correct", "Correct"],
                "eval_score": [1.0, 1.0],
            }
        )

        filtered = llm_accuracy.filter_results_for_analysis(
            raw_df,
            exclude_date_variables=True,
            exclude_state_fed_reference_variables=False,
        )

        assert filtered["variable_name"].to_list() == [
            "dp_state_fed_reference",
            "dp_possession",
        ]

    def test_make_scored_rows_preserves_refined_error_type(self):
        scored = llm_accuracy.make_scored_rows(
            pl.DataFrame(
                {
                    "jurisdiction_id": ["A"],
                    "dataset": ["DPL"],
                    "variable_name": ["dp_law"],
                    "query_text": ["Does the jurisdiction have a law?"],
                    "response_options": ["Responses: Yes OR No"],
                    "evaluation_mode": ["whole_answer"],
                    "eval_label": ["Incorrect"],
                    "eval_score": [0],
                    "eval_error_type": ["retrieval_noise"],
                    "eval_error_type_refined": ["off_topic_context"],
                }
            )
        )

        assert scored[0, "eval_error_type"] == "retrieval_noise"
        assert scored[0, "eval_error_type_refined"] == "off_topic_context"

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

    def test_error_summary_prefers_refined_error_types_when_present(self):
        scored_df = pl.DataFrame(
            {
                "eval_label": ["Incorrect", "Incorrect", "Correct"],
                "eval_error_type": ["retrieval_failure", "hallucination", "none"],
                "eval_error_type_refined": [
                    "off_topic_context",
                    "scope_error",
                    "none",
                ],
            }
        )

        summary = llm_accuracy.make_error_type_summary(scored_df, top_n=10)

        rows = {row["eval_error_type"]: row["count"] for row in summary.to_dicts()}
        assert rows == {
            "off_topic_context": 1,
            "scope_error": 1,
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

    def test_make_citation_comparison_df_classifies_exact_family_and_missing(self):
        raw_df = pl.DataFrame(
            {
                "jurisdiction_id": ["A", "A", "A"],
                "dataset": ["DPL", "DPL", "DPL"],
                "variable_name": ["dp_law", "dp_possession", "dp_type"],
                "benchmark_row_id": [1, 2, 3],
                "query_id": ["dp_law", "dp_possession", "dp_type"],
                "query_metadata": [
                    json.dumps({"query_id": "dp_law"}),
                    json.dumps({"query_id": "dp_possession"}),
                    json.dumps({"query_id": "dp_type"}),
                ],
                "ground_truth_citation": [
                    "Sec. 11-40.",
                    "Sec. 12-4-10.",
                    "Sec. 9-101.",
                ],
                "citations": [
                    "['Sec. 11-40.']",
                    "['Sec. 12-4-10(3)']",
                    "[]",
                ],
                "evaluation_mode": ["whole_answer", "whole_answer", "whole_answer"],
                "eval_label": ["Correct", "Correct", "Incorrect"],
                "eval_score": [1.0, 1.0, 0.0],
            }
        )

        citation_df = llm_accuracy.make_citation_comparison_df(raw_df)

        by_variable = {
            row["variable_name"]: row["citation_match_type"]
            for row in citation_df.to_dicts()
        }
        assert by_variable == {
            "dp_law": "exact_unit_match",
            "dp_possession": "family_match",
            "dp_type": "llm_missing",
        }

    def test_make_citation_comparison_df_deduplicates_query_rows(self):
        raw_df = pl.DataFrame(
            {
                "jurisdiction_id": ["A", "A"],
                "dataset": ["SSP", "SSP"],
                "variable_name": ["ssp_law", "ssp_law"],
                "benchmark_row_id": [10, 10],
                "query_id": ["ssp_law", "ssp_law"],
                "query_metadata": [
                    json.dumps({"query_id": "ssp_law"}),
                    json.dumps({"query_id": "ssp_law"}),
                ],
                "ground_truth_citation": ["§ 607.17", "§ 607.17"],
                "citations": ["['§ 607.17']", "['§ 607.17']"],
                "evaluation_mode": ["whole_answer", "response_option"],
                "eval_label": ["Correct", "Correct"],
                "eval_score": [1.0, 1.0],
            }
        )

        citation_df = llm_accuracy.make_citation_comparison_df(raw_df)

        assert citation_df.height == 1
        assert citation_df[0, "citation_match_type"] == "exact_unit_match"

    def test_classify_citation_match_handles_bare_numeric_citations(self):
        match_type, ground_truth_units, llm_units = llm_accuracy.classify_citation_match(
            "9.10.100   Clean needle a",
            ["9.10.100", "9.10.100.A"],
        )

        assert match_type == "exact_unit_match"
        assert ground_truth_units == ["9.10.100"]
        assert llm_units == ["9.10.100"]

    def test_classify_citation_match_handles_bare_hyphenated_citations(self):
        match_type, ground_truth_units, llm_units = llm_accuracy.classify_citation_match(
            "ARTICLE III. DRUG PARAPHE",
            ["35-52(B)(7)", "35-52(D)"],
        )

        assert match_type == "mismatch"
        assert ground_truth_units == ["iii"]
        assert llm_units == ["35-52(b)(7)", "35-52(d)"]

    def test_make_citation_match_summary_adds_share_excluding_unparseable(self):
        citation_df = pl.DataFrame(
            {
                "citation_match_type": [
                    "exact_unit_match",
                    "family_match",
                    "llm_unparseable",
                    "ground_truth_unparseable",
                    "mismatch",
                ]
            }
        )

        summary = llm_accuracy.make_citation_match_summary(citation_df)

        rows = {
            row["citation_match_type"]: row
            for row in summary.to_dicts()
        }
        assert round(rows["exact_unit_match"]["share_pct"], 2) == 20.0
        assert round(
            rows["exact_unit_match"]["share_pct_excluding_unparseable"], 2
        ) == 33.33
        assert round(rows["family_match"]["share_pct_excluding_unparseable"], 2) == 33.33
        assert round(rows["mismatch"]["share_pct_excluding_unparseable"], 2) == 33.33
        assert rows["llm_unparseable"]["share_pct_excluding_unparseable"] is None
        assert rows["ground_truth_unparseable"]["share_pct_excluding_unparseable"] is None


class TestOutputRouting:
    def test_extract_results_timestamp_from_aggregate_file(self):
        timestamp = llm_accuracy.extract_results_timestamp(
            Path("data/output/all_jurisdictions/20260515_104348/all_jurisdictions_benchmark_20260515_104348.csv")
        )

        assert timestamp == "20260515_104348"

    def test_resolve_output_dir_uses_matching_all_jurisdictions_run_folder(self):
        input_path = (
            PROJECT_ROOT
            / "data"
            / "output"
            / "all_jurisdictions"
            / "20260515_104348"
            / "all_jurisdictions_benchmark_20260515_104348.csv"
        )

        output_dir = llm_accuracy.resolve_output_dir(input_path, None)

        assert output_dir == input_path.parent

    def test_resolve_output_dir_routes_timestamped_input_to_all_jurisdictions_folder(self):
        input_path = PROJECT_ROOT / "data" / "output" / "PA-Philadelphia" / "benchmark_results_20260515_104348.csv"

        output_dir = llm_accuracy.resolve_output_dir(input_path, None)

        assert output_dir == (
            PROJECT_ROOT
            / "data"
            / "output"
            / "all_jurisdictions"
            / "20260515_104348"
        )

    def test_resolve_output_dir_uses_batch_aggregate_folder_when_present(self):
        input_path = (
            PROJECT_ROOT
            / "data"
            / "output"
            / "all_jurisdictions"
            / "batches"
            / "batch-50"
            / "20260515_104348"
            / "all_jurisdictions_benchmark_20260515_104348.csv"
        )

        output_dir = llm_accuracy.resolve_output_dir(input_path, None)

        assert output_dir == input_path.parent

    def test_resolve_input_path_prefers_newest_batch_aggregate(self, tmp_path, monkeypatch):
        project_root = tmp_path / "project"
        batch_dir = (
            project_root
            / "data"
            / "output"
            / "all_jurisdictions"
            / "batches"
            / "batch-50"
        )
        older_dir = batch_dir / "20260515_104348"
        newer_dir = batch_dir / "20260516_114500"
        older_dir.mkdir(parents=True)
        newer_dir.mkdir(parents=True)
        older_file = older_dir / "all_jurisdictions_benchmark_20260515_104348.csv"
        newer_file = newer_dir / "all_jurisdictions_benchmark_20260516_114500.csv"
        older_file.write_text("value\n1\n")
        newer_file.write_text("value\n2\n")

        monkeypatch.setattr(
            llm_accuracy,
            "__file__",
            str(project_root / "coep" / "analysis" / "LLM_accuracy.py"),
        )

        resolved = llm_accuracy.resolve_input_path(None, batch_id="batch-50")

        assert resolved == newer_file
