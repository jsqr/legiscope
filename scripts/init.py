"""Initialize jurisdiction directory structure and registries.

Usage::

    # All defaults from params.yaml
    python scripts/init.py

    # Override code type or jurisdiction display name
    python scripts/init.py --code-type zoning
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from loguru import logger

from legiscope.config import setup_logging
from legiscope.models import (
    CODES_SCHEMA,
    JURISDICTIONS_SCHEMA,
    CodeRef,
    JurisdictionRef,
    codes_parquet,
    jurisdictions_parquet,
)
from legiscope.params import load_params
from legiscope.utils import create_code_structure

# ------------------------------------------------------------------
# Registry helpers
# ------------------------------------------------------------------


def _load_or_create_parquet(path: Path, schema: dict) -> pl.DataFrame:
    """Load a Parquet registry file, or create an empty DataFrame if missing."""
    if path.exists():
        return pl.read_parquet(path)
    return pl.DataFrame(schema=schema)


def _append_jurisdiction(ref: JurisdictionRef, name: str) -> None:
    """Append a jurisdiction to the registry if it doesn't already exist."""
    parquet_path = jurisdictions_parquet()
    df = _load_or_create_parquet(parquet_path, JURISDICTIONS_SCHEMA)

    if df.filter(pl.col("jurisdiction_id") == ref.jurisdiction_id).height > 0:
        logger.info("Jurisdiction {} already registered", ref.jurisdiction_id)
        return

    parent = ref.state if ref.locality else None
    if ref.level == "state":
        parent = None

    row = pl.DataFrame(
        [
            {
                "jurisdiction_id": ref.jurisdiction_id,
                "state": ref.state,
                "locality": ref.locality,
                "level": ref.level,
                "name": name,
                "parent_jurisdiction": parent,
            }
        ],
        schema=JURISDICTIONS_SCHEMA,
    )
    df = pl.concat([df, row])
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    logger.info("Registered jurisdiction: {}", ref.jurisdiction_id)


def _append_code(code_ref: CodeRef, name: str, code_type: str) -> None:
    """Append a code to the registry if it doesn't already exist."""
    parquet_path = codes_parquet()
    df = _load_or_create_parquet(parquet_path, CODES_SCHEMA)

    if df.filter(pl.col("code_id") == code_ref.code_id).height > 0:
        logger.info("Code {} already registered", code_ref.code_id)
        return

    row = pl.DataFrame(
        [
            {
                "code_id": code_ref.code_id,
                "jurisdiction_id": code_ref.jurisdiction_id,
                "code_slug": code_ref.code_slug,
                "name": name,
                "code_type": code_type,
                "level": code_ref.jurisdiction.level,
            }
        ],
        schema=CODES_SCHEMA,
    )
    df = pl.concat([df, row])
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    logger.info("Registered code: {}", code_ref.code_id)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    """Create jurisdiction/code directory structure and update registries."""
    setup_logging()

    params = load_params()
    jur = params.get("jurisdiction", {})

    code_name = jur.get("code_name")
    if not code_name:
        raise SystemExit("Error: jurisdiction.code_name must be set in params.yaml")

    code_ref = CodeRef.from_params(params)

    parser = argparse.ArgumentParser(
        description="Initialize directory structure for a legal code",
        epilog="Jurisdiction is read from params.yaml.",
    )
    parser.add_argument(
        "--code-type", default="municipal", help="Code type (default: municipal)"
    )
    parser.add_argument(
        "--jurisdiction-name",
        default=None,
        help="Display name for jurisdiction (auto-generated if omitted)",
    )
    args = parser.parse_args()

    jurisdiction_name = args.jurisdiction_name
    if jurisdiction_name is None:
        if code_ref.jurisdiction.locality:
            jurisdiction_name = f"City of {code_ref.jurisdiction.locality}"
        else:
            jurisdiction_name = code_ref.jurisdiction.state

    code_dir = create_code_structure(code_ref)
    _append_jurisdiction(code_ref.jurisdiction, jurisdiction_name)
    _append_code(code_ref, code_name, args.code_type)

    logger.info(f"Created structure: {code_dir}")


if __name__ == "__main__":
    main()
