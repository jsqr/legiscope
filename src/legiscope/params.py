"""
Pipeline parameter loader for legiscope.

Loads ``params.yaml`` (DVC-tracked hyperparameters) and provides helpers
for merging per-code overrides.

Loading strategy:
1. Try ``dvc.api.params_show()`` (works inside a DVC-tracked repo).
2. Fall back to a direct ``yaml.safe_load`` of ``params.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PARAMS_FILENAME = "params.yaml"


def _find_params_path() -> Path:
    """Walk up from this file's directory to locate ``params.yaml``."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = current / _PARAMS_FILENAME
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        f"Could not find {_PARAMS_FILENAME} in any parent directory of {Path(__file__).resolve()}"
    )


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_params(code_data_dir: str | Path | None = None) -> dict[str, Any]:
    """Load global params, optionally merging per-code overrides.

    Args:
        code_data_dir: Path to a code's data directory.  If a
            ``params.yaml`` exists inside it, those values are deep-merged
            on top of the global params.

    Returns:
        Merged parameter dictionary.
    """
    # --- Try DVC first ---------------------------------------------------
    try:
        import dvc.api

        params = dvc.api.params_show()
        logger.debug("Loaded params via dvc.api.params_show()")
    except Exception:
        # DVC not available or not in a DVC repo – fall back to direct load
        path = _find_params_path()
        with open(path) as f:
            params = yaml.safe_load(f) or {}
        logger.debug(f"Loaded params from {path}")

    # --- Per-code overrides ----------------------------------------------
    if code_data_dir is not None:
        override_path = Path(code_data_dir) / _PARAMS_FILENAME
        if override_path.exists():
            with open(override_path) as f:
                overrides = yaml.safe_load(f) or {}
            params = _deep_merge(params, overrides)
            logger.debug(f"Merged per-code overrides from {override_path}")

    return params
