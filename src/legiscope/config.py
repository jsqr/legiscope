"""
Infrastructure configuration loader for legiscope.

Loads ``config.yaml`` (deployment/environment-specific settings) and exposes
a singleton ``Config`` accessor with dot-path key lookup.

Most path helpers are rooted at ``data_dir()`` and therefore follow the
``LEGISCOPE_DATA_DIR`` override. ``monqcle_report_path()`` is the intentional
exception: it points to a COEP-specific dataset outside the main data root, so
relative values are resolved from the repository/config root instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_config: dict[str, Any] | None = None
_CONFIG_FILENAME = "config.yaml"


def _find_config_path() -> Path:
    """Walk up from this file's directory to locate ``config.yaml``."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = current / _CONFIG_FILENAME
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        f"Could not find {_CONFIG_FILENAME} in any parent directory of {Path(__file__).resolve()}"
    )


def _load() -> dict[str, Any]:
    global _config
    if _config is None:
        path = _find_config_path()
        with open(path) as f:
            _config = yaml.safe_load(f) or {}
    assert _config is not None
    return _config


def reset() -> None:
    """Clear cached config (useful in tests)."""
    global _config
    _config = None


def get(key_path: str, default: Any = None) -> Any:
    """Dot-separated key lookup into the config dict.

    >>> get("paths.data_dir")
    'data'
    >>> get("database.chromadb.default_collection")
    'legal_code'
    """
    data = _load()
    keys = key_path.split(".")
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key)
        if data is None:
            return default
    return data


# ---------------------------------------------------------------------------
# Convenience properties
# ---------------------------------------------------------------------------


def data_dir() -> Path:
    """Return the root data directory, respecting ``LEGISCOPE_DATA_DIR``."""
    env = os.getenv("LEGISCOPE_DATA_DIR")
    if env:
        return Path(env)
    return Path(get("paths.data_dir", "data"))


def laws_dir() -> Path:
    """Return the laws subdirectory under :func:`data_dir`."""
    return data_dir() / get("paths.laws_dir", "laws")


def chroma_db_path() -> Path:
    """Return the ChromaDB persistence directory."""
    return data_dir() / get("paths.chroma_db_dir", "chroma_db")


def queries_dir() -> Path:
    """Return the queries subdirectory under :func:`data_dir`."""
    return data_dir() / get("paths.queries_dir", "queries")


def output_dir() -> Path:
    """Return the output subdirectory under :func:`data_dir`."""
    return data_dir() / get("paths.output_dir", "output")


def default_queries_path() -> Path:
    """Return the default queries CSV path."""
    return queries_dir() / get("paths.default_queries_file", "queries.csv")


def monqcle_report_path() -> Path:
    """Return the COEP MonQcle report path.

    Unlike the other convenience path helpers, this location is intentionally
    not nested under ``data_dir()`` because the benchmark fixture lives in the
    repository's COEP data tree. Absolute config values are returned as-is;
    relative values are resolved from the directory containing ``config.yaml``.
    """
    raw_path = Path(
        get(
            "paths.monqcle_report",
            "coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv",
        )
    )
    if raw_path.is_absolute():
        return raw_path
    return _find_config_path().parent / raw_path
