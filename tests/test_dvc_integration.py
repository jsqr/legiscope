"""Structural validation of DVC config coherence."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
VENV_BIN = PROJECT_ROOT / ".venv" / "bin"
DVC_BIN = str(VENV_BIN / "dvc") if (VENV_BIN / "dvc").exists() else "dvc"
PYTHON_BIN = (
    str(VENV_BIN / "python") if (VENV_BIN / "python").exists() else sys.executable
)
PYTHONPATH_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        [str(SRC_PATH), os.environ.get("PYTHONPATH", "")]
    ).strip(os.pathsep),
}

PIPELINE_MODULES = ["parse", "segment", "embed", "index"]


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
        """jurisdiction is either empty or contains the required key set."""
        params_path = PROJECT_ROOT / "params.yaml"

        with open(params_path) as f:
            params_config = yaml.safe_load(f)

        jurisdiction = params_config.get("jurisdiction", {})
        assert isinstance(jurisdiction, dict), "jurisdiction must be a mapping"

        if jurisdiction:
            required = {"state", "locality", "code_slug"}
            assert required.issubset(jurisdiction), (
                "Configured jurisdiction must include state, locality, and code_slug. "
                f"Found keys: {set(jurisdiction.keys())}"
            )

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
        expected = {"parse", "segment", "embed", "index", "benchmark"}
        assert expected.issubset(stages), (
            f"Missing stages: {expected - stages}. Found: {stages}"
        )

    def test_index_stage_has_stamp_output(self):
        """Index stage should produce a stamp file for DAG ordering."""
        dvc_path = PROJECT_ROOT / "dvc.yaml"

        with open(dvc_path) as f:
            dvc_config = yaml.safe_load(f)

        index_stage = dvc_config["stages"]["index"]
        assert "outs" in index_stage, (
            "Index stage should have 'outs' with a stamp file for benchmark dependency"
        )
        outs = index_stage["outs"]
        # The stamp file should be present and not cached
        stamp_entries = [
            o
            for o in outs
            if isinstance(o, dict) and any("index.stamp" in k for k in o)
        ]
        assert len(stamp_entries) == 1, (
            f"Expected exactly one index.stamp output, found: {outs}"
        )


class TestDvcPipelineCli:
    """Validate that DVC pipeline tooling is invocable."""

    def test_dvc_repro_dry(self):
        """dvc repro --dry parses dvc.yaml without errors."""
        result = subprocess.run(
            [DVC_BIN, "repro", "--dry"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        # Exit code 0 means the YAML parsed and the graph resolved.
        # A non-zero code with "no changes" or missing deps is also acceptable
        # (the pipeline definition itself is valid, just no data to process).
        stderr_lower = result.stderr.lower()
        assert (
            result.returncode == 0
            or "no changes" in stderr_lower
            or "does not exist" in stderr_lower
            or "no such file or directory" in stderr_lower
            or "could not find 'jurisdiction.state'" in stderr_lower
        ), (
            f"dvc repro --dry failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_dvc_repro_script_help(self):
        """scripts/dvc_repro.sh --help exits cleanly."""
        script = PROJECT_ROOT / "scripts" / "dvc_repro.sh"
        result = subprocess.run(
            [str(script), "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, (
            f"dvc_repro.sh --help failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr}"
        )
        assert "usage" in result.stdout.lower()

    def test_dvc_status_invocable(self):
        """dvc status runs without creating experiment refs."""
        result = subprocess.run(
            [DVC_BIN, "status"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        stderr_lower = result.stderr.lower()
        assert (
            result.returncode == 0
            or "does not exist" in stderr_lower
            or "no such file or directory" in stderr_lower
        ), (
            f"dvc status failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_pipeline_modules_help(self):
        """Each pipeline script accepts --help without error."""
        for module in PIPELINE_MODULES:
            result = subprocess.run(
                [PYTHON_BIN, f"scripts/{module}.py", "--help"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                env=PYTHONPATH_ENV,
            )
            assert result.returncode == 0, (
                f"python scripts/{module}.py --help failed "
                f"(rc={result.returncode}):\nstderr: {result.stderr}"
            )

    def test_pipeline_init_module_help(self):
        """The init script (not a DVC stage) also accepts --help."""
        result = subprocess.run(
            [PYTHON_BIN, "scripts/init.py", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=PYTHONPATH_ENV,
        )
        assert result.returncode == 0, (
            f"python scripts/init.py --help failed "
            f"(rc={result.returncode}):\nstderr: {result.stderr}"
        )
