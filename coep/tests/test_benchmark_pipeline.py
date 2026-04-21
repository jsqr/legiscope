"""Regression tests for benchmark pipeline output helpers."""

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

_MODULE_PATH = PROJECT_ROOT / "coep" / "scripts" / "benchmark_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("test_benchmark_pipeline_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_pipeline)


class TestBenchmarkPipelineHelpers:
    def test_ensure_generation_outcome_columns_derives_filtered_and_abstention_flags(self):
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

    def test_ensure_supporting_passage_validation_columns_flags_drift_below_threshold(self):
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