"""Structural validation of DVC config coherence."""

from __future__ import annotations

import stat
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).parent.parent


class TestDvcParamsCoherence:
    def test_dvc_params_references_exist_in_params_yaml(self):
        """Every params: reference in dvc.yaml stages has a matching top-level key in params.yaml."""
        dvc_path = PROJECT_ROOT / "dvc.yaml"
        params_path = PROJECT_ROOT / "params.yaml"

        with open(dvc_path) as f:
            dvc_config = yaml.safe_load(f)
        with open(params_path) as f:
            params_config = yaml.safe_load(f)

        params_top_keys = set(params_config.keys())

        for stage_name, stage_def in dvc_config.get("stages", {}).items():
            for param_ref in stage_def.get("params", []):
                # param_ref can be a string like "jurisdiction" or
                # a dotted path like "embeddings.default_provider"
                top_key = param_ref.split(".")[0]
                assert top_key in params_top_keys, (
                    f"Stage '{stage_name}' references param '{param_ref}' "
                    f"but top-level key '{top_key}' not found in params.yaml. "
                    f"Available keys: {params_top_keys}"
                )

    def test_jurisdiction_vars_have_required_keys(self):
        """jurisdiction in params.yaml has state, locality, and code_slug."""
        params_path = PROJECT_ROOT / "params.yaml"

        with open(params_path) as f:
            params_config = yaml.safe_load(f)

        jurisdiction = params_config.get("jurisdiction", {})
        assert "state" in jurisdiction, "jurisdiction missing 'state' key"
        assert "locality" in jurisdiction, "jurisdiction missing 'locality' key"
        assert "code_slug" in jurisdiction, "jurisdiction missing 'code_slug' key"

    def test_dvc_repro_script_is_executable(self):
        """scripts/dvc_repro.sh has the executable permission bit set."""
        script = PROJECT_ROOT / "scripts" / "dvc_repro.sh"
        assert script.exists(), "scripts/dvc_repro.sh not found"
        mode = script.stat().st_mode
        assert mode & stat.S_IXUSR, (
            "scripts/dvc_repro.sh is not executable (missing user execute bit)"
        )

    def test_no_vars_json_references(self):
        """No --vars-json references remain in dvc.yaml or scripts/dvc_repro.sh."""
        files_to_check = [
            PROJECT_ROOT / "dvc.yaml",
            PROJECT_ROOT / "scripts" / "dvc_repro.sh",
        ]

        for filepath in files_to_check:
            if filepath.exists():
                content = filepath.read_text()
                assert "--vars-json" not in content, (
                    f"Found deprecated --vars-json in {filepath.name}. "
                    f"Should use -S flag instead."
                )

    def test_dvc_stages_use_param_interpolation(self):
        """DVC stages use ${jurisdiction.*} interpolation in cmd and deps."""
        dvc_path = PROJECT_ROOT / "dvc.yaml"

        with open(dvc_path) as f:
            content = f.read()

        assert "${jurisdiction.state}" in content, (
            "dvc.yaml missing ${jurisdiction.state} interpolation"
        )
        assert "${jurisdiction.locality}" in content, (
            "dvc.yaml missing ${jurisdiction.locality} interpolation"
        )
        assert "${jurisdiction.code_slug}" in content, (
            "dvc.yaml missing ${jurisdiction.code_slug} interpolation"
        )

    def test_dvc_stages_completeness(self):
        """All expected pipeline stages exist in dvc.yaml."""
        dvc_path = PROJECT_ROOT / "dvc.yaml"

        with open(dvc_path) as f:
            dvc_config = yaml.safe_load(f)

        stages = set(dvc_config.get("stages", {}).keys())
        expected = {"parse", "segment", "embed", "index"}
        assert expected.issubset(stages), (
            f"Missing stages: {expected - stages}. Found: {stages}"
        )

    def test_index_stage_has_no_outs(self):
        """Index stage should not have outs (ChromaDB is not DVC-tracked)."""
        dvc_path = PROJECT_ROOT / "dvc.yaml"

        with open(dvc_path) as f:
            dvc_config = yaml.safe_load(f)

        index_stage = dvc_config["stages"]["index"]
        assert "outs" not in index_stage, (
            "Index stage should not have 'outs' — ChromaDB is persistent, not DVC-tracked"
        )
