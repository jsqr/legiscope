"""Initialize jurisdiction directory structure and registries.

Usage::

    python -m legiscope.pipeline.init \\
        --state CA --municipality LosAngeles \\
        --code-slug municipal-code --name "Los Angeles Municipal Code"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from loguru import logger

from legiscope.models import (
    CODES_PARQUET,
    CODES_SCHEMA,
    JURISDICTIONS_PARQUET,
    JURISDICTIONS_SCHEMA,
    CodeRef,
    JurisdictionRef,
)
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
    df = _load_or_create_parquet(JURISDICTIONS_PARQUET, JURISDICTIONS_SCHEMA)

    if df.filter(pl.col("jurisdiction_id") == ref.jurisdiction_id).height > 0:
        logger.info("Jurisdiction {} already registered", ref.jurisdiction_id)
        return

    parent = ref.state if ref.municipality else None
    if ref.level == "state":
        parent = None

    row = pl.DataFrame(
        [
            {
                "jurisdiction_id": ref.jurisdiction_id,
                "state": ref.state,
                "municipality": ref.municipality,
                "level": ref.level,
                "name": name,
                "parent_jurisdiction": parent,
            }
        ],
        schema=JURISDICTIONS_SCHEMA,
    )
    df = pl.concat([df, row])
    JURISDICTIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(JURISDICTIONS_PARQUET)
    logger.info("Registered jurisdiction: {}", ref.jurisdiction_id)


def _append_code(code_ref: CodeRef, name: str, code_type: str) -> None:
    """Append a code to the registry if it doesn't already exist."""
    df = _load_or_create_parquet(CODES_PARQUET, CODES_SCHEMA)

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
    CODES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(CODES_PARQUET)
    logger.info("Registered code: {}", code_ref.code_id)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    """Create jurisdiction/code directory structure and update registries."""
    parser = argparse.ArgumentParser(
        description="Initialize directory structure for a legal code",
    )
    parser.add_argument("--state", required=True, help="Two-letter state abbreviation")
    parser.add_argument(
        "--municipality", default=None, help="Municipality name (omit for state-level)"
    )
    parser.add_argument("--code-slug", required=True, help="Code slug identifier")
    parser.add_argument("--name", required=True, help="Display name for the code")
    parser.add_argument(
        "--code-type", default="municipal", help="Code type (default: municipal)"
    )
    parser.add_argument(
        "--jurisdiction-name",
        default=None,
        help="Display name for jurisdiction (auto-generated if omitted)",
    )

    args = parser.parse_args()

    code_ref = CodeRef.from_dvc_vars(
        state=args.state,
        municipality=args.municipality,
        code_slug=args.code_slug,
    )

    jurisdiction_name = args.jurisdiction_name
    if jurisdiction_name is None:
        if code_ref.jurisdiction.municipality:
            jurisdiction_name = f"City of {code_ref.jurisdiction.municipality}"
        else:
            jurisdiction_name = code_ref.jurisdiction.state

    code_dir = create_code_structure(code_ref)
    _append_jurisdiction(code_ref.jurisdiction, jurisdiction_name)
    _append_code(code_ref, args.name, args.code_type)

    logger.info(f"Created structure: {code_dir}")


if __name__ == "__main__":
    main()
