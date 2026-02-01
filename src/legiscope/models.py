"""
Data models for jurisdiction and code identification.

Provides dataclasses for constructing globally unique IDs and resolving
file system paths for the legiscope data pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Data directory root (relative to project root)
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
LAWS_DIR = DATA_DIR / "laws"

# ---------------------------------------------------------------------------
# Parquet schema constants
# ---------------------------------------------------------------------------

JURISDICTIONS_SCHEMA = {
    "jurisdiction_id": pl.String,
    "state": pl.String,
    "municipality": pl.String,
    "level": pl.String,
    "name": pl.String,
    "parent_jurisdiction": pl.String,
}

CODES_SCHEMA = {
    "code_id": pl.String,
    "jurisdiction_id": pl.String,
    "code_slug": pl.String,
    "name": pl.String,
    "code_type": pl.String,
    "level": pl.String,
}

RELATIONS_SCHEMA = {
    "relation_id": pl.String,
    "code_id": pl.String,
    "source_section_id": pl.String,
    "target_section_id": pl.String,
    "relation_type": pl.String,
    "target_text": pl.String,
    "scope": pl.String,
    "confidence": pl.Float64,
}

EXTERNAL_REFERENCES_SCHEMA = {
    "reference_id": pl.String,
    "code_id": pl.String,
    "source_section_id": pl.String,
    "target_jurisdiction": pl.String,
    "target_code": pl.String,
    "target_citation": pl.String,
    "target_section_id": pl.String,
    "citation_format": pl.String,
    "relation_type": pl.String,
    "confidence": pl.Float64,
}


# ---------------------------------------------------------------------------
# ID models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JurisdictionRef:
    """Reference to a jurisdiction (state or state + municipality).

    Attributes:
        state: Two-letter state abbreviation (e.g. ``"CA"``).
        municipality: Municipality name in PascalCase, or ``None`` for
            state-level jurisdictions (e.g. ``"LosAngeles"``).
    """

    state: str
    municipality: str | None = None

    def __post_init__(self):
        if not self.state or not self.state.strip():
            raise ValueError("state cannot be empty")
        # Normalize state to uppercase via object.__setattr__ (frozen dataclass)
        object.__setattr__(self, "state", self.state.strip().upper())
        if self.municipality is not None:
            stripped = self.municipality.strip().replace(" ", "")
            if not stripped:
                raise ValueError("municipality cannot be empty string")
            object.__setattr__(self, "municipality", stripped)

    @property
    def jurisdiction_id(self) -> str:
        """Globally unique jurisdiction identifier.

        Format: ``"{STATE}"`` or ``"{STATE}-{Municipality}"``.
        """
        if self.municipality:
            return f"{self.state}-{self.municipality}"
        return self.state

    @property
    def level(self) -> str:
        """``"state"`` or ``"local"``."""
        return "local" if self.municipality else "state"


@dataclass(frozen=True)
class CodeRef:
    """Reference to a specific legal code within a jurisdiction.

    Attributes:
        jurisdiction: The jurisdiction this code belongs to.
        code_slug: URL-friendly code identifier (e.g. ``"penal-code"``).
    """

    jurisdiction: JurisdictionRef
    code_slug: str

    def __post_init__(self):
        if not self.code_slug or not self.code_slug.strip():
            raise ValueError("code_slug cannot be empty")
        object.__setattr__(self, "code_slug", self.code_slug.strip())

    @property
    def code_id(self) -> str:
        """Globally unique code identifier.

        Format: ``"{state}:{subdivision}:{code_slug}"``.
        The subdivision is ``"state"`` for state-level codes, or the
        municipality name for local codes.
        """
        subdivision = self.jurisdiction.municipality or "state"
        return f"{self.jurisdiction.state}:{subdivision}:{self.code_slug}"

    @property
    def jurisdiction_id(self) -> str:
        """Shortcut for ``self.jurisdiction.jurisdiction_id``."""
        return self.jurisdiction.jurisdiction_id

    def section_id(self, ordinal: int) -> str:
        """Globally unique section identifier.

        Format: ``"{code_id}:s{ordinal}"``.
        """
        return f"{self.code_id}:s{ordinal}"

    def segment_id(self, ordinal: int) -> str:
        """Globally unique segment identifier.

        Format: ``"{code_id}:g{ordinal}"``.
        """
        return f"{self.code_id}:g{ordinal}"

    @property
    def data_dir(self) -> Path:
        """Relative path from ``data/laws/`` to this code's directory.

        For state-level codes: ``{STATE}/State/{code_slug}/``
        For local codes: ``{STATE}/{Municipality}/{code_slug}/``
        """
        subdivision = self.jurisdiction.municipality or "State"
        return Path(self.jurisdiction.state) / subdivision / self.code_slug

    @property
    def full_data_dir(self) -> Path:
        """Full relative path from project root to this code's directory."""
        return LAWS_DIR / self.data_dir


# ---------------------------------------------------------------------------
# Registry file paths
# ---------------------------------------------------------------------------

JURISDICTIONS_PARQUET = DATA_DIR / "jurisdictions.parquet"
CODES_PARQUET = DATA_DIR / "codes.parquet"
