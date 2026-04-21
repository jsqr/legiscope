"""
Pipeline parameter loader for legiscope.

Loads ``params.yaml`` (DVC-tracked hyperparameters) and provides helpers
for merging per-code overrides.

Loading strategy:
1. Try ``dvc.api.params_show()`` (works inside a DVC-tracked repo).
2. Fall back to a direct ``yaml.safe_load`` of ``params.yaml``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PARAMS_FILENAME = "params.yaml"
_GLOBAL_PARAMS_CACHE: dict[str, Any] | None = None
_GLOBAL_PARAMS_CACHE_SOURCE: str | None = None
_GLOBAL_PARAMS_CACHE_LOCK = Lock()
_GLOBAL_PARAMS_LOGGED_SOURCES: set[str] = set()


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


def _log_params_load_once(message: str) -> None:
    """Log params load source once per process to avoid repeated noise."""
    if message in _GLOBAL_PARAMS_LOGGED_SOURCES:
        return
    logger.debug(message)
    _GLOBAL_PARAMS_LOGGED_SOURCES.add(message)


def _load_global_params_uncached() -> tuple[dict[str, Any], str]:
    """Load global params without cache and return the source description."""
    try:
        import dvc.api

        return dvc.api.params_show(), "Loaded params via dvc.api.params_show()"
    except Exception:
        path = _find_params_path()
        with open(path) as f:
            params = yaml.safe_load(f) or {}
        return params, f"Loaded params from {path}"


def _load_global_params() -> dict[str, Any]:
    """Load and cache global params for the current process."""
    global _GLOBAL_PARAMS_CACHE, _GLOBAL_PARAMS_CACHE_SOURCE

    if _GLOBAL_PARAMS_CACHE is not None:
        return deepcopy(_GLOBAL_PARAMS_CACHE)

    with _GLOBAL_PARAMS_CACHE_LOCK:
        if _GLOBAL_PARAMS_CACHE is None:
            params, source = _load_global_params_uncached()
            _GLOBAL_PARAMS_CACHE = params
            _GLOBAL_PARAMS_CACHE_SOURCE = source
            _log_params_load_once(source)

    return deepcopy(_GLOBAL_PARAMS_CACHE)


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
    params = _load_global_params()

    # --- Per-code overrides ----------------------------------------------
    if code_data_dir is not None:
        override_path = Path(code_data_dir) / _PARAMS_FILENAME
        if override_path.exists():
            with open(override_path) as f:
                overrides = yaml.safe_load(f) or {}
            params = _deep_merge(params, overrides)
            _log_params_load_once(f"Merged per-code overrides from {override_path}")

    return params
