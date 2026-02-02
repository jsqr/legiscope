"""Shared test fixtures for legiscope pipeline tests."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from legiscope.models import CodeRef, JurisdictionRef


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "windytown"


@pytest.fixture
def windytown_fixture_dir() -> Path:
    """Path to the WindyTown test fixture directory."""
    return FIXTURE_DIR


@pytest.fixture
def sample_code_ref() -> CodeRef:
    """CodeRef for the WindyTown municipal-code fixture."""
    return CodeRef(
        jurisdiction=JurisdictionRef(state="IL", locality="WindyTown"),
        code_slug="municipal-code",
    )


@pytest.fixture
def sample_sections_df() -> pl.DataFrame:
    """Sections DataFrame loaded from fixture."""
    return pl.read_parquet(FIXTURE_DIR / "sections.parquet")


@pytest.fixture
def sample_segments_df() -> pl.DataFrame:
    """Segments DataFrame loaded from fixture."""
    return pl.read_parquet(FIXTURE_DIR / "segments.parquet")


@pytest.fixture
def sample_headings_df() -> pl.DataFrame:
    """Headings DataFrame loaded from fixture."""
    return pl.read_parquet(FIXTURE_DIR / "headings.parquet")


@pytest.fixture
def sample_embeddings_df() -> pl.DataFrame:
    """Embeddings DataFrame loaded from fixture."""
    return pl.read_parquet(FIXTURE_DIR / "embeddings.parquet")


@pytest.fixture
def mock_cli_args(monkeypatch):
    """Factory to monkeypatch sys.argv for CLI-based pipeline modules.

    Usage::

        mock_cli_args(["--state", "IL", "--locality", "WindyTown",
                        "--code-slug", "municipal-code"])
    """

    def _set(args: list[str]) -> None:
        monkeypatch.setattr("sys.argv", ["prog"] + args)

    return _set
