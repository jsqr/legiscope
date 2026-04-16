"""
Data models for jurisdiction and code identification.

Provides dataclasses for constructing globally unique IDs and resolving
file system paths for the legiscope data pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from legiscope import config as cfg


# ---------------------------------------------------------------------------
# Data directory helpers (read from config.yaml at access time)
# ---------------------------------------------------------------------------
def data_dir() -> Path:
    return cfg.data_dir()


def laws_dir() -> Path:
    return cfg.laws_dir()


# ---------------------------------------------------------------------------
# Parquet schema constants
# ---------------------------------------------------------------------------

JURISDICTIONS_SCHEMA = {
    "jurisdiction_id": pl.String,
    "state": pl.String,
    "locality": pl.String,
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
    """Reference to a jurisdiction (state or state + locality).

    Attributes:
        state: Two-letter state abbreviation (e.g. ``"CA"``).
        locality: Locality name in PascalCase, or ``None`` for
            state-level jurisdictions (e.g. ``"LosAngeles"``).
    """

    state: str
    locality: str | None = None

    def __post_init__(self):
        if not self.state or not self.state.strip():
            raise ValueError("state cannot be empty")
        # Normalize state to uppercase via object.__setattr__ (frozen dataclass)
        object.__setattr__(self, "state", self.state.strip().upper())
        if self.locality is not None:
            stripped = self.locality.strip().replace(" ", "")
            if not stripped:
                raise ValueError("locality cannot be empty string")
            object.__setattr__(self, "locality", stripped)

    @property
    def jurisdiction_id(self) -> str:
        """Globally unique jurisdiction identifier.

        Format: ``"{STATE}"`` or ``"{STATE}-{Locality}"``.
        """
        if self.locality:
            return f"{self.state}-{self.locality}"
        return self.state

    @property
    def level(self) -> str:
        """``"state"`` or ``"local"``."""
        return "local" if self.locality else "state"

    @property
    def output_dir_name(self) -> str:
        """Directory name for output files, matching DVC conventions.

        Format: ``"{STATE}-{Locality}"`` or ``"{STATE}-State"``.
        Uses the ``State`` sentinel for state-level jurisdictions so
        that output paths stay in sync with ``dvc.yaml`` interpolation.
        """
        return f"{self.state}-{self.locality or 'State'}"


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
        locality name for local codes.
        """
        subdivision = self.jurisdiction.locality or "state"
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

    def chunk_id(self, ordinal: int) -> str:
        """Globally unique chunk identifier.

        Format: ``"{code_id}:c{ordinal}"``.
        """
        return f"{self.code_id}:c{ordinal}"

    @property
    def data_dir(self) -> Path:
        """Relative path from ``data/laws/`` to this code's directory.

        For state-level codes: ``{STATE}/State/{code_slug}/``
        For local codes: ``{STATE}/{Locality}/{code_slug}/``
        """
        subdivision = self.jurisdiction.locality or "State"
        return Path(self.jurisdiction.state) / subdivision / self.code_slug

    @property
    def full_data_dir(self) -> Path:
        """Full relative path from project root to this code's directory."""
        return laws_dir() / self.data_dir

    @classmethod
    def from_dvc_vars(
        cls,
        state: str | None = None,
        locality: str | None = None,
        code_slug: str | None = None,
    ) -> "CodeRef":
        """Create a ``CodeRef`` from DVC pipeline variables.

        DVC stages pass ``${jurisdiction.state}``,
        ``${jurisdiction.locality}``, and ``${jurisdiction.code_slug}``
        as CLI arguments.  This factory mirrors that convention and raises
        :class:`ValueError` for any missing field.

        The sentinel value ``"State"`` for *locality* is normalised to
        ``None`` so that DVC pipelines (which cannot omit an interpolated
        argument) can represent state-level codes.

        Args:
            state: Two-letter state abbreviation.
            locality: Locality name, ``"State"`` for state-level,
                or ``None``.
            code_slug: URL-friendly code identifier.

        Returns:
            A fully initialised ``CodeRef``.
        """
        if not state:
            raise ValueError("state is required")
        if not code_slug:
            raise ValueError("code_slug is required")
        # Normalise the DVC sentinel: "State" means state-level (no locality)
        if locality is not None and locality.strip() == "State":
            locality = None
        jurisdiction = JurisdictionRef(state=state, locality=locality)
        return cls(jurisdiction=jurisdiction, code_slug=code_slug)

    @classmethod
    def from_params(cls, params: dict | None = None) -> "CodeRef":
        """Create a ``CodeRef`` from ``params.yaml`` jurisdiction settings.

        Args:
            params: Pre-loaded params dict.  When *None*, calls
                :func:`legiscope.params.load_params` automatically.

        Returns:
            A fully initialised ``CodeRef``.
        """
        if params is None:
            from legiscope.params import load_params

            params = load_params()
        jur = params.get("jurisdiction", {})
        return cls.from_dvc_vars(
            state=jur.get("state"),
            locality=jur.get("locality"),
            code_slug=jur.get("code_slug"),
        )


# ---------------------------------------------------------------------------
# Registry file paths (dynamic to respect LEGISCOPE_DATA_DIR)
# ---------------------------------------------------------------------------


def jurisdictions_parquet() -> Path:
    return data_dir() / "jurisdictions.parquet"


def codes_parquet() -> Path:
    return data_dir() / "codes.parquet"
