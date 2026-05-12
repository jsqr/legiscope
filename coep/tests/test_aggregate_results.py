"""Regression tests for the HPC benchmark aggregation helper."""

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

_MODULE_PATH = (
    PROJECT_ROOT / "coep" / "scripts" / "HPC_scripts" / "aggregate_results.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_aggregate_results_module", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
aggregate_results = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(aggregate_results)


class TestAggregateResults:
    def test_timestamped_aggregate_output_path_uses_run_timestamp(self, tmp_path):
        output_path = aggregate_results._timestamped_aggregate_output_path(
            tmp_path, "all_jurisdictions_metrics", "20260508_153000"
        )

        assert output_path == tmp_path / "all_jurisdictions_metrics_20260508_153000.csv"

    def test_select_results_file_prefers_latest_timestamped_copy(self, tmp_path):
        jurisdiction_dir = tmp_path / "PA-Philadelphia"
        jurisdiction_dir.mkdir()
        (jurisdiction_dir / "benchmark_results.csv").write_text("value\n1\n")
        (jurisdiction_dir / "benchmark_results_20260501_010101.csv").write_text(
            "value\n2\n"
        )
        latest_file = jurisdiction_dir / "benchmark_results_20260502_020202.csv"
        latest_file.write_text("value\n3\n")

        selected = aggregate_results._select_results_file(jurisdiction_dir)

        assert selected == latest_file
        assert (
            aggregate_results._extract_results_timestamp(latest_file)
            == "20260502_020202"
        )

    def test_collect_results_uses_latest_available_file_and_adds_source_metadata(
        self, tmp_path
    ):
        output_dir = tmp_path / "output"
        pa_dir = output_dir / "PA-Philadelphia"
        ca_dir = output_dir / "CA-LosAngeles"
        pa_dir.mkdir(parents=True)
        ca_dir.mkdir(parents=True)

        pl.DataFrame({"variable_name": ["dp_law"], "eval_score": [8]}).write_csv(
            pa_dir / "benchmark_results_20260501_023513.csv"
        )
        pl.DataFrame(
            {
                "jurisdiction": ["CA-LosAngeles"],
                "jurisdiction_id": [""],
                "variable_name": ["dp_type"],
                "eval_label": ["Correct"],
            }
        ).write_csv(ca_dir / "benchmark_results.csv")

        combined = aggregate_results.collect_results(output_dir)

        assert combined.height == 2
        assert combined.columns[0] == "jurisdiction"
        assert set(combined["jurisdiction_id"].to_list()) == {
            "PA-Philadelphia",
            "CA-LosAngeles",
        }
        assert set(combined["jurisdiction"].to_list()) == {
            "PA-Philadelphia",
            "CA-LosAngeles",
        }
        assert "_aggregate_source_path" in combined.columns
        assert "_aggregate_source_timestamp" in combined.columns

        rows = {row["jurisdiction_id"]: row for row in combined.to_dicts()}
        assert rows["PA-Philadelphia"]["_aggregate_source_type"] == "timestamped"
        assert (
            rows["PA-Philadelphia"]["_aggregate_source_timestamp"] == "20260501_023513"
        )
        assert rows["CA-LosAngeles"]["jurisdiction"] == "CA-LosAngeles"
        assert rows["CA-LosAngeles"]["_aggregate_source_type"] == "canonical"

    def test_collect_metrics_backfills_jurisdiction_id_and_source_path(self, tmp_path):
        output_dir = tmp_path / "output"
        jurisdiction_dir = output_dir / "PA-Philadelphia"
        jurisdiction_dir.mkdir(parents=True)
        metrics_path = jurisdiction_dir / "benchmark_metrics.json"
        metrics_path.write_text(json.dumps({"avg_score": 7.5, "total": 10}))

        metrics_df = aggregate_results.collect_metrics(output_dir)

        assert metrics_df.height == 1
        assert metrics_df.columns[0] == "jurisdiction"
        row = metrics_df.to_dicts()[0]
        assert row["jurisdiction"] == "PA-Philadelphia"
        assert row["jurisdiction_id"] == "PA-Philadelphia"
        assert row["aggregate_metrics_path"] == str(metrics_path)

    def test_collect_metrics_prefers_latest_timestamped_copy(self, tmp_path):
        output_dir = tmp_path / "output"
        jurisdiction_dir = output_dir / "TX-Dallas"
        jurisdiction_dir.mkdir(parents=True)
        (jurisdiction_dir / "benchmark_metrics.json").write_text(
            json.dumps({"avg_score": 6.0, "total": 8})
        )
        (jurisdiction_dir / "benchmark_metrics_20260501_010101.json").write_text(
            json.dumps({"avg_score": 7.0, "total": 8})
        )
        latest_metrics = jurisdiction_dir / "benchmark_metrics_20260502_020202.json"
        latest_metrics.write_text(json.dumps({"avg_score": 8.0, "total": 8}))

        metrics_df = aggregate_results.collect_metrics(output_dir)

        assert metrics_df.height == 1
        row = metrics_df.to_dicts()[0]
        assert row["jurisdiction_id"] == "TX-Dallas"
        assert row["avg_score"] == 8.0
        assert row["aggregate_metrics_path"] == str(latest_metrics)
        assert row["aggregate_metrics_source_type"] == "timestamped"
        assert row["aggregate_metrics_source_timestamp"] == "20260502_020202"
