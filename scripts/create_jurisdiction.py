#!/usr/bin/env python3
"""
Command-line script to create jurisdiction and code directory structures,
and register them in the global Parquet registries.
"""

import argparse
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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

    parent = ref.state if ref.locality else None
    # For state-level, don't set a parent
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


def main():
    """Create jurisdiction/code directory structure and update registries."""
    parser = argparse.ArgumentParser(
        description="Create directory structure for a new jurisdiction and code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --state CA --code-slug penal-code --name "California Penal Code" --code-type statute
  %(prog)s --state CA --locality LosAngeles --code-slug municipal-code --name "Los Angeles Municipal Code" --code-type municipal
        """,
    )

    parser.add_argument(
        "--state",
        required=True,
        help="Two-letter state abbreviation (e.g., CA, IL)",
    )
    parser.add_argument(
        "--locality",
        default=None,
        help="Locality name (e.g., LosAngeles, Chicago). Omit for state-level.",
    )
    parser.add_argument(
        "--code-slug",
        required=True,
        help="URL-friendly code identifier (e.g., penal-code, municipal-code)",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Display name for the code (e.g., 'California Penal Code')",
    )
    parser.add_argument(
        "--code-type",
        default="municipal",
        help="Code type (e.g., statute, municipal). Default: municipal",
    )
    parser.add_argument(
        "--jurisdiction-name",
        default=None,
        help="Display name for the jurisdiction. Auto-generated if omitted.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        jurisdiction = JurisdictionRef(state=args.state, locality=args.locality)
        code_ref = CodeRef(jurisdiction=jurisdiction, code_slug=args.code_slug)

        # Auto-generate jurisdiction name if not provided
        jurisdiction_name = args.jurisdiction_name
        if jurisdiction_name is None:
            if jurisdiction.locality:
                jurisdiction_name = f"City of {jurisdiction.locality}"
            else:
                jurisdiction_name = jurisdiction.state

        # Create directory structure
        code_dir = create_code_structure(code_ref)

        # Update registries
        _append_jurisdiction(jurisdiction, jurisdiction_name)
        _append_code(code_ref, args.name, args.code_type)

        print(f"Successfully created structure: {code_dir}")
        print(f"  Jurisdiction: {jurisdiction.jurisdiction_id}")
        print(f"  Code: {code_ref.code_id}")

        if args.verbose:
            print(f"  Directory: {code_dir}")
            print(f"  Raw files: {code_dir / 'raw'}")

    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
