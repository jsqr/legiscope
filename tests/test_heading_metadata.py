"""Tests for heading metadata extraction and propagation (issue #58)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from legiscope.parse.headings import HEADINGS_SCHEMA
from legiscope.models import CodeRef, JurisdictionRef
from legiscope.segment import (
    add_parent_relationships,
    create_segments_df,
    divide_into_sections,
    enrich_sections,
    parse_frontmatter,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "windytown"


# ---------------------------------------------------------------------------
# headings.parquet schema and content
# ---------------------------------------------------------------------------


class TestHeadingsParquetSchema:
    """Verify headings.parquet has correct columns and types."""

    def test_schema(self, sample_headings_df: pl.DataFrame):
        for col, dtype in HEADINGS_SCHEMA.items():
            assert col in sample_headings_df.columns, f"Missing column: {col}"
            assert sample_headings_df.schema[col] == dtype, (
                f"Column {col}: expected {dtype}, got {sample_headings_df.schema[col]}"
            )

    def test_non_empty(self, sample_headings_df: pl.DataFrame):
        assert len(sample_headings_df) > 0


class TestHeadingsParquetLineNumbers:
    """Verify line_number values actually correspond to headings in code.md."""

    def test_line_numbers_match_headings(self, sample_headings_df: pl.DataFrame):
        code_md = (FIXTURE_DIR / "code.md").read_text(encoding="utf-8")
        all_lines = code_md.split("\n")

        for row in sample_headings_df.to_dicts():
            ln = row["line_number"]
            assert 1 <= ln <= len(all_lines), f"Line {ln} out of range"
            actual_line = all_lines[ln - 1]
            # The actual line should start with '#' (it's a markdown heading)
            assert actual_line.lstrip().startswith("#"), (
                f"Line {ln} is not a heading: {actual_line!r}"
            )


class TestHeadingsParquetTypeAndNumber:
    """Verify section_type and section_number are extracted correctly."""

    def test_chapter_has_type_and_number(self, sample_headings_df: pl.DataFrame):
        chapters = sample_headings_df.filter(pl.col("heading_level") == 1)
        assert len(chapters) >= 1
        for row in chapters.to_dicts():
            assert row["section_type"] == "chapter"
            assert row["section_number"] is not None

    def test_section_has_type_and_number(self, sample_headings_df: pl.DataFrame):
        sections = sample_headings_df.filter(pl.col("heading_level") == 2)
        assert len(sections) >= 1
        for row in sections.to_dicts():
            assert row["section_type"] == "section"
            assert row["section_number"] is not None

    def test_paragraph_has_type_and_number(self, sample_headings_df: pl.DataFrame):
        paragraphs = sample_headings_df.filter(pl.col("heading_level") == 3)
        assert len(paragraphs) >= 1
        for row in paragraphs.to_dicts():
            assert row["section_type"] == "paragraph"
            assert row["section_number"] is not None


# ---------------------------------------------------------------------------
# divide_into_sections() — line_number column
# ---------------------------------------------------------------------------


class TestDivideIntoSectionsLineNumbers:
    """Verify the new line_number column in divide_into_sections."""

    def test_has_line_number_column(self):
        text = "# Title\n\nBody."
        df = divide_into_sections(text)
        assert "line_number" in df.columns
        assert df.schema["line_number"] == pl.Int64

    def test_line_numbers_correct(self):
        text = "preamble\n\n# Title\n\nBody.\n\n## Section\n\nMore."
        df = divide_into_sections(text)
        # "# Title" is on line 3 (1-based)
        assert df["line_number"][0] == 3
        # "## Section" is on line 7
        assert df["line_number"][1] == 7

    def test_empty_returns_line_number_column(self):
        df = divide_into_sections("")
        assert "line_number" in df.columns
        assert len(df) == 0

    def test_consecutive_headings(self):
        text = "# A\n## B\n### C"
        df = divide_into_sections(text)
        assert df["line_number"].to_list() == [1, 2, 3]


# ---------------------------------------------------------------------------
# parse_frontmatter()
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    """Verify frontmatter parsing and line counting."""

    def test_no_frontmatter(self):
        body, count = parse_frontmatter("# Title\n\nBody.")
        assert count == 0
        assert body == "# Title\n\nBody."

    def test_with_frontmatter(self):
        content = "---\nkey: value\n---\n# Title\n\nBody."
        body, count = parse_frontmatter(content)
        assert count == 3  # lines 1-3 are frontmatter
        assert body.strip() == "# Title\n\nBody."

    def test_line_offset_correct(self):
        """After adding frontmatter_line_count to body line numbers,
        absolute line numbers should match the original file."""
        content = "---\na: 1\nb: 2\n---\n\n# Heading\n\nBody."
        body, fm_count = parse_frontmatter(content)
        assert fm_count == 4

        df = divide_into_sections(body)
        # Heading is on line 2 of body (line 1 is empty)
        relative_line = df["line_number"][0]
        absolute_line = relative_line + fm_count
        # In original content, "# Heading" is on line 6
        assert absolute_line == 6


# ---------------------------------------------------------------------------
# segment_legal_code() — headings join
# ---------------------------------------------------------------------------


class TestSegmentJoinsHeadings:
    """Verify sections.parquet gets true level, section_type, section_number."""

    def test_sections_have_metadata_columns(self, sample_sections_df: pl.DataFrame):
        assert "section_type" in sample_sections_df.columns
        assert "section_number" in sample_sections_df.columns
        assert "line_number" in sample_sections_df.columns
        assert "heading_level" in sample_sections_df.columns

    def test_section_types_populated(self, sample_sections_df: pl.DataFrame):
        types = sample_sections_df["section_type"].to_list()
        # All sections should have a type (no None from the join)
        assert all(t is not None for t in types)

    def test_section_numbers_populated(self, sample_sections_df: pl.DataFrame):
        numbers = sample_sections_df["section_number"].to_list()
        assert all(n is not None for n in numbers)

    def test_heading_levels_are_true_levels(self, sample_sections_df: pl.DataFrame):
        """Heading levels from headings.parquet should be the true structural
        levels, not just the markdown # count."""
        levels = sample_sections_df["heading_level"].to_list()
        # In the WindyTown fixture, levels should be 1, 2, and 3
        assert set(levels) == {1, 2, 3}


class TestSegmentsPropagate:
    """Verify segments.parquet propagates new metadata columns."""

    def test_segments_have_metadata(self, sample_segments_df: pl.DataFrame):
        assert "section_type" in sample_segments_df.columns
        assert "section_number" in sample_segments_df.columns
        assert "line_number" in sample_segments_df.columns


# ---------------------------------------------------------------------------
# Backward compatibility — no headings.parquet
# ---------------------------------------------------------------------------


class TestBackwardCompatNoHeadingsParquet:
    """segment_legal_code works without headings.parquet (None metadata)."""

    def test_no_headings_parquet(self, tmp_path: Path):
        """Without headings.parquet, sections get None for section_type/number."""
        code_md = (FIXTURE_DIR / "code.md").read_text(encoding="utf-8")

        # Read content and strip frontmatter
        body, fm_count = parse_frontmatter(code_md)

        sections_df = divide_into_sections(body)
        if fm_count > 0:
            sections_df = sections_df.with_columns(
                (pl.col("line_number") + fm_count).alias("line_number")
            )

        # Simulate no headings.parquet: add None columns
        sections_df = sections_df.with_columns(
            pl.lit(None, dtype=pl.String).alias("section_type"),
            pl.lit(None, dtype=pl.String).alias("section_number"),
        )

        sections_df = add_parent_relationships(sections_df)

        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="IL", locality="WindyTown"),
            code_slug="municipal-code",
        )
        sections_df = enrich_sections(sections_df, code_ref)

        # Should work without error
        assert len(sections_df) > 0
        assert sections_df["section_type"][0] is None
        assert sections_df["section_number"][0] is None

        # Segments should also work
        segments_df = create_segments_df(sections_df)
        assert "section_type" in segments_df.columns
        assert "section_number" in segments_df.columns
