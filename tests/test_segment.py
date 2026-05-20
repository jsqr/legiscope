"""Tests for legiscope.segment module."""

from pathlib import Path

import legiscope.segment as segment_mod
import polars as pl
import pytest

from legiscope.models import CodeRef, JurisdictionRef
from legiscope.parse.convert import text2md
from legiscope.parse.headings import HeadingLevel, HeadingStructure
from legiscope.segment import (
    _derive_chunk_token_limit,
    _estimate_token_count,
    _split_by_token_budget,
    add_parent_relationships,
    build_chunks_df,
    create_segments_df,
    divide_into_sections,
    enrich_sections,
    get_section_text,
    segment_legal_code,
    segment_text,
)


class TestDivideIntoSections:
    """Test cases for divide_into_sections function."""

    def test_basic_markdown_with_headings(self):
        """Test basic markdown with H1, H2, and H3 headings."""
        markdown_text = """# Main Title

This is the introduction paragraph.

## Section 1

This is the content of section 1.
It has multiple lines.

### Subsection 1.1

This is a subsection content.

## Section 2

Content of section 2."""

        result = divide_into_sections(markdown_text)

        # Check DataFrame structure
        assert len(result) == 4
        assert result.columns == [
            "section_ordinal",
            "heading_level",
            "heading_text",
            "body_text",
            "line_number",
        ]

        # Check section indices
        assert result["section_ordinal"].to_list() == [0, 1, 2, 3]

        # Check heading levels
        assert result["heading_level"].to_list() == [1, 2, 3, 2]

        # Check heading texts
        expected_headings = [
            "# Main Title",
            "## Section 1",
            "### Subsection 1.1",
            "## Section 2",
        ]
        assert result["heading_text"].to_list() == expected_headings

        # Check body texts
        expected_bodies = [
            "This is the introduction paragraph.",
            "This is the content of section 1.\nIt has multiple lines.",
            "This is a subsection content.",
            "Content of section 2.",
        ]
        assert result["body_text"].to_list() == expected_bodies

    def test_sections_without_body_text(self):
        """Test handling of consecutive headings without body text."""
        markdown_text = """# Title 1
## Title 2
### Title 3

Some content here.

## Title 4"""

        result = divide_into_sections(markdown_text)

        assert len(result) == 4

        # Check that sections without body have None
        bodies = result["body_text"].to_list()
        assert bodies[0] is None  # # Title 1 has no body
        assert bodies[1] is None  # ## Title 2 has no body
        assert bodies[2] == "Some content here."  # ### Title 3 has body
        assert bodies[3] is None  # ## Title 4 has no body

    def test_empty_input(self):
        """Test handling of empty input."""
        # Empty string
        result = divide_into_sections("")
        assert len(result) == 0
        assert result.columns == [
            "section_ordinal",
            "heading_level",
            "heading_text",
            "body_text",
            "line_number",
        ]

        # Whitespace only
        result = divide_into_sections("   \n  \n  ")
        assert len(result) == 0

    def test_no_headings(self):
        """Test handling of text without any markdown headings."""
        markdown_text = """This is just plain text.
It has no headings at all.
Just regular paragraphs."""

        result = divide_into_sections(markdown_text)
        assert len(result) == 0

    def test_single_heading(self):
        """Test text with only one heading."""
        markdown_text = """# Only Heading

This is the content under the single heading."""

        result = divide_into_sections(markdown_text)

        assert len(result) == 1
        assert result["section_ordinal"][0] == 0
        assert result["heading_level"][0] == 1
        assert result["heading_text"][0] == "# Only Heading"
        assert result["body_text"][0] == "This is the content under the single heading."

    def test_complex_markdown_content(self):
        """Test handling of complex markdown content in body."""
        markdown_text = """# Code Examples

Here's some code:

```python
def hello():
    print("Hello, World!")
```

And here's a list:

- Item 1
- Item 2
- Item 3

## More Content

This includes **bold text** and *italic text*."""

        result = divide_into_sections(markdown_text)

        assert len(result) == 2

        # First section body should include code block and list
        body1 = result["body_text"][0]
        assert "```python" in body1
        assert "def hello():" in body1
        assert "- Item 1" in body1

        # Second section body should include formatted text
        body2 = result["body_text"][1]
        assert "**bold text**" in body2
        assert "*italic text*" in body2

    def test_heading_levels_1_to_6(self):
        """Test all markdown heading levels."""
        markdown_text = """# H1 Heading
## H2 Heading
### H3 Heading
#### H4 Heading
##### H5 Heading
###### H6 Heading

Content after H6."""

        result = divide_into_sections(markdown_text)

        assert len(result) == 6
        assert result["heading_level"].to_list() == [1, 2, 3, 4, 5, 6]

        # Check heading texts
        expected_headings = [
            "# H1 Heading",
            "## H2 Heading",
            "### H3 Heading",
            "#### H4 Heading",
            "##### H5 Heading",
            "###### H6 Heading",
        ]
        assert result["heading_text"].to_list() == expected_headings

        # Only last section should have body text
        bodies = result["body_text"].to_list()
        assert bodies[:5] == [None, None, None, None, None]
        assert bodies[5] == "Content after H6."

    def test_preamble_ignored(self):
        """Test that text before first heading is ignored."""
        markdown_text = """This is preamble text.
It should be ignored.

# First Real Heading

This content should be captured."""

        result = divide_into_sections(markdown_text)

        assert len(result) == 1
        assert result["heading_text"][0] == "# First Real Heading"
        assert result["body_text"][0] == "This content should be captured."

    def test_whitespace_handling(self):
        """Test proper handling of whitespace in body text."""
        markdown_text = """# Title

   This line has leading spaces.

   This line has trailing spaces.

   This line has both.

## Next

No extra spaces here."""

        result = divide_into_sections(markdown_text)

        # Body text should be stripped of leading/trailing whitespace
        body1 = result["body_text"][0]
        expected_body1 = "This line has leading spaces.\n\nThis line has trailing spaces.\n\nThis line has both."
        assert body1 == expected_body1

        body2 = result["body_text"][1]
        assert body2 == "No extra spaces here."

    def test_invalid_input_types(self):
        """Test error handling for invalid input types."""
        # Test non-string input
        with pytest.raises(TypeError, match="markdown_text must be a string"):
            divide_into_sections(123)

        with pytest.raises(TypeError, match="markdown_text must be a string"):
            divide_into_sections(None)

        with pytest.raises(TypeError, match="markdown_text must be a string"):
            divide_into_sections(["list", "of", "strings"])

    def test_dataframe_schema(self):
        """Test that the returned DataFrame has the correct schema."""
        markdown_text = """# Test

Some content."""

        result = divide_into_sections(markdown_text)

        # Check column names
        assert result.columns == [
            "section_ordinal",
            "heading_level",
            "heading_text",
            "body_text",
            "line_number",
        ]

        # Check column types
        schema = result.schema
        assert schema["section_ordinal"] == pl.Int64
        assert schema["heading_level"] == pl.Int64
        assert schema["heading_text"] == pl.String
        assert schema["body_text"] == pl.String
        assert schema["line_number"] == pl.Int64

    def test_large_document(self):
        """Test handling of a larger document with many sections."""
        # Create a document with many sections
        sections = []
        for i in range(100):
            sections.append(f"## Section {i}")
            sections.append(f"Content for section {i}.")

        markdown_text = "\n\n".join(sections)

        result = divide_into_sections(markdown_text)

        assert len(result) == 100
        assert result["section_ordinal"].to_list() == list(range(100))
        assert result["heading_level"].to_list() == [2] * 100

        # Check a few sample sections
        for i in [0, 25, 50, 99]:
            assert result["heading_text"][i] == f"## Section {i}"
            assert result["body_text"][i] == f"Content for section {i}."

    def test_edge_case_empty_lines_between_headings(self):
        """Test handling of multiple empty lines between headings."""
        markdown_text = """# Title 1



## Title 2


### Title 3



Content here."""

        result = divide_into_sections(markdown_text)

        assert len(result) == 3
        assert result["body_text"][0] is None
        assert result["body_text"][1] is None
        assert result["body_text"][2] == "Content here."

    def test_unicode_line_separators_do_not_bloat_heading(self):
        """U+2028 line separators should be treated as real newlines during parsing."""
        markdown_text = (
            "#### 601   14401 Friar St.   Van Nuys\u2028"
            "602   11320 Chandler Blvd.   North Hollywood\u2028"
            "603   14521 Friar St.   Van Nuys\n\n"
            "B. Installation of Regulatory Signs."
        )

        result = divide_into_sections(markdown_text)

        assert len(result) == 1
        assert result["heading_text"][0] == "#### 601   14401 Friar St.   Van Nuys"
        assert "602   11320 Chandler Blvd." in (result["body_text"][0] or "")
        assert "603   14521 Friar St." in (result["body_text"][0] or "")


class TestAddParentRelationships:
    """Test cases for add_parent_relationships function."""

    def test_basic_hierarchy(self):
        """Test basic H1 -> H2 -> H3 hierarchy."""
        # First create sections DataFrame
        markdown_text = """# Main Title

Introduction content.

## Section 1

Section 1 content.

### Subsection 1.1

Subsection content."""

        sections = divide_into_sections(markdown_text)
        result = add_parent_relationships(sections)

        # Check DataFrame structure
        assert len(result) == 3
        assert "parent" in result.columns
        assert result.columns == [
            "section_ordinal",
            "heading_level",
            "heading_text",
            "body_text",
            "line_number",
            "parent",
            "children",
            "depth",
            "ancestor_path",
        ]

        # Check parent relationships
        parents = result["parent"].to_list()
        assert parents[0] is None  # H1 has no parent
        assert parents[1] == 0  # H2 parent is H1 (idx 0)
        assert parents[2] == 1  # H3 parent is H2 (idx 1)

        # Check children
        children = result["children"].to_list()
        assert children[0] == [1]  # H1 has H2 as child
        assert children[1] == [2]  # H2 has H3 as child
        assert children[2] == []  # H3 is a leaf

        # Check depth
        assert result["depth"].to_list() == [0, 1, 2]

        # Check ancestor_path
        assert result["ancestor_path"].to_list() == ["0", "0/1", "0/1/2"]

    def test_multiple_branches(self):
        """Test multiple branches from same parent."""
        markdown_text = """# Main Title

## Section 1

Content 1.

## Section 2

Content 2.

## Section 3

Content 3."""

        sections = divide_into_sections(markdown_text)
        result = add_parent_relationships(sections)

        # All H2 sections should have H1 as parent
        parents = result["parent"].to_list()
        assert parents[0] is None  # H1 has no parent
        assert parents[1] == 0  # First H2 parent is H1
        assert parents[2] == 0  # Second H2 parent is H1
        assert parents[3] == 0  # Third H2 parent is H1


class TestSegmentLegalCodeCanonicalFiltering:
    """Regression tests for filtering canonical sections by parse-region metadata."""

    @staticmethod
    def _write_parse_outputs(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        input_text: str,
        structure: HeadingStructure,
    ) -> CodeRef:
        """Generate code.md/headings/regions artifacts in a temporary laws dir."""
        base_laws_dir = tmp_path / "laws"
        monkeypatch.setattr("legiscope.models.laws_dir", lambda: base_laws_dir)

        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="PA", locality="TestCity"),
            code_slug="municipal-code",
        )
        code_dir = code_ref.full_data_dir
        code_dir.mkdir(parents=True, exist_ok=True)

        input_path = code_dir / "code.txt"
        input_path.write_text(input_text, encoding="utf-8")
        text2md(
            structure,
            str(input_path),
            str(code_dir / "code.md"),
            "PA",
            "TestCity",
        )
        return code_ref

    def test_segment_legal_code_excludes_toc_sections_using_regions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """regions.parquet should exclude TOC headings from canonical sections."""
        input_text = """ARTICLE I GUIDE TO THE CHARTER

1-100 Purpose

1-200 Definitions

PREAMBLE
This charter establishes the basic organization of city government under state law.

ARTICLE I

1-100. Purpose.

This section sets out the basic powers of the city.

ARTICLE II

2-100. Council.

The council exercises legislative authority through local ordinances."""

        structure = HeadingStructure(
            levels=[
                HeadingLevel(
                    level=1,
                    regex_pattern=r"^ARTICLE\s+[IVXLCDM]+(?:\s+.+)?$",
                    markdown_prefix="#",
                    example_heading="ARTICLE I GUIDE TO THE CHARTER",
                    type_label="article",
                    number_regex=r"[IVXLCDM]+",
                ),
                HeadingLevel(
                    level=2,
                    regex_pattern=r"^\d+(?:-\d+)+\.\s+.+$",
                    markdown_prefix="##",
                    example_heading="1-100. Purpose.",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                ),
            ],
            total_levels=2,
            file_sample_size=10,
            code_start_element_id=0,
            code_start_line=1,
        )

        code_ref = TestSegmentLegalCodeCanonicalFiltering._write_parse_outputs(
            tmp_path,
            monkeypatch,
            input_text,
            structure,
        )
        sections_df, _ = segment_legal_code(code_ref, embedding_model_token_limit=128)

        assert sections_df["heading_text"].to_list() == [
            "# ARTICLE I",
            "## 1-100. Purpose.",
            "# ARTICLE II",
            "## 2-100. Council.",
        ]

    def test_segment_legal_code_writes_chunks_with_legal_intro_regions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """chunks.parquet should retain legal intro material but not TOC blocks."""
        input_text = """TABLE OF CONTENTS

ARTICLE I GUIDE TO THE CHARTER

1-100 Purpose

PREAMBLE
This charter establishes the basic organization of city government under state law.

ARTICLE I

1-100. Purpose.

This section sets out the basic powers of the city."""

        structure = HeadingStructure(
            levels=[
                HeadingLevel(
                    level=1,
                    regex_pattern=r"^ARTICLE\s+[IVXLCDM]+(?:\s+.+)?$",
                    markdown_prefix="#",
                    example_heading="ARTICLE I GUIDE TO THE CHARTER",
                    type_label="article",
                    number_regex=r"[IVXLCDM]+",
                ),
                HeadingLevel(
                    level=2,
                    regex_pattern=r"^\d+(?:-\d+)+\.\s+.+$",
                    markdown_prefix="##",
                    example_heading="1-100. Purpose.",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                ),
            ],
            total_levels=2,
            file_sample_size=10,
            code_start_element_id=0,
            code_start_line=1,
        )

        code_ref = TestSegmentLegalCodeCanonicalFiltering._write_parse_outputs(
            tmp_path,
            monkeypatch,
            input_text,
            structure,
        )
        _, segments_df = segment_legal_code(
            code_ref,
            embedding_model_token_limit=128,
            llm_context_limit=32768,
        )

        chunks_path = code_ref.full_data_dir / "chunks.parquet"
        assert chunks_path.exists()

        chunks_df = pl.read_parquet(chunks_path)
        assert "chunk_id" in chunks_df.columns
        assert "region_role" in chunks_df.columns
        assert "source_kind" in chunks_df.columns
        assert "chunk_id" in segments_df.columns

        intro_chunks = chunks_df.filter(pl.col("region_role") == "legal_intro")
        assert len(intro_chunks) == 1
        assert (
            "This charter establishes the basic organization"
            in intro_chunks["body_text"][0]
        )

        all_chunk_text = "\n".join(chunks_df["body_text"].fill_null("").to_list())
        assert "GUIDE TO THE CHARTER" not in all_chunk_text

    def test_segment_legal_code_writes_chunks_with_current_through_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """chunks.parquet should retain pre-code current-through metadata but not publisher boilerplate."""
        input_text = """Current through Ordinance 24-11.

Published by Civic Publishing.

ARTICLE I ADMINISTRATION

1-100. Executive Branch.

The executive authority is vested in a mayor and administrative departments."""

        structure = HeadingStructure(
            levels=[
                HeadingLevel(
                    level=1,
                    regex_pattern=r"^ARTICLE\s+[IVXLCDM]+(?:\s+.+)?$",
                    markdown_prefix="#",
                    example_heading="ARTICLE I ADMINISTRATION",
                    type_label="article",
                    number_regex=r"[IVXLCDM]+",
                ),
                HeadingLevel(
                    level=2,
                    regex_pattern=r"^\d+(?:-\d+)+\.\s+.+$",
                    markdown_prefix="##",
                    example_heading="1-100. Executive Branch.",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                ),
            ],
            total_levels=2,
            file_sample_size=5,
            code_start_element_id=0,
            code_start_line=1,
        )

        code_ref = TestSegmentLegalCodeCanonicalFiltering._write_parse_outputs(
            tmp_path,
            monkeypatch,
            input_text,
            structure,
        )
        segment_legal_code(
            code_ref,
            embedding_model_token_limit=128,
            llm_context_limit=32768,
        )

        chunks_df = pl.read_parquet(code_ref.full_data_dir / "chunks.parquet")
        region_chunks = chunks_df.filter(pl.col("source_kind") == "region")
        region_text = "\n".join(region_chunks["body_text"].fill_null("").to_list())

        assert "Current through Ordinance 24-11." in region_text
        assert "Published by Civic Publishing." not in region_text

    def test_segment_legal_code_keeps_heading_only_current_through_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Heading-only current-through metadata should remain retrievable."""
        input_text = """Contains 2025 S-51, current through

Published by Civic Publishing.

ARTICLE I ADMINISTRATION

1-100. Executive Branch.

The executive authority is vested in a mayor and administrative departments."""

        structure = HeadingStructure(
            levels=[
                HeadingLevel(
                    level=1,
                    regex_pattern=r"^ARTICLE\s+[IVXLCDM]+(?:\s+.+)?$",
                    markdown_prefix="#",
                    example_heading="ARTICLE I ADMINISTRATION",
                    type_label="article",
                    number_regex=r"[IVXLCDM]+",
                ),
                HeadingLevel(
                    level=2,
                    regex_pattern=r"^\d+(?:-\d+)+\.\s+.+$",
                    markdown_prefix="##",
                    example_heading="1-100. Executive Branch.",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                ),
            ],
            total_levels=2,
            file_sample_size=5,
            code_start_element_id=0,
            code_start_line=1,
        )

        code_ref = TestSegmentLegalCodeCanonicalFiltering._write_parse_outputs(
            tmp_path,
            monkeypatch,
            input_text,
            structure,
        )
        segment_legal_code(
            code_ref,
            embedding_model_token_limit=128,
            llm_context_limit=32768,
        )

        chunks_df = pl.read_parquet(code_ref.full_data_dir / "chunks.parquet")
        region_chunks = chunks_df.filter(pl.col("source_kind") == "region")
        region_text = "\n".join(region_chunks["body_text"].fill_null("").to_list())

        assert "Contains 2025 S-51, current through" in region_text
        assert "Published by Civic Publishing." not in region_text


class TestBuildChunks:
    """Unit tests for derived chunk construction."""

    def test_chunk_budget_uses_retrieval_result_budget(self, monkeypatch):
        """Derived chunk size should honor the configured worst-case retrieval fan-out."""
        monkeypatch.setattr(
            segment_mod,
            "DEFAULT_EMBEDDING_MODEL_TOKEN_LIMIT",
            20000,
        )
        assert _derive_chunk_token_limit(32768) == 2457
        assert _derive_chunk_token_limit(32768, target_retrieved_chunks=5) == 4915

    def test_build_chunks_splits_oversized_section_body(self, tmp_path: Path):
        """Oversized section bodies should split into multiple derived chunks."""
        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="PA", locality="TestCity"),
            code_slug="municipal-code",
        )
        code_dir = tmp_path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        long_body = " ".join(["word"] * 1800)
        markdown_text = f"""# ARTICLE I

## 1-100. Purpose.

{long_body}

## 1-200. Scope.

Short supporting text."""

        sections_df = divide_into_sections(markdown_text)
        sections_df = add_parent_relationships(sections_df)
        sections_df = enrich_sections(sections_df, code_ref)

        chunks_df = build_chunks_df(
            sections_df,
            code_ref,
            markdown_text,
            code_dir,
            llm_context_limit=3000,
            target_retrieved_chunks=5,
        )

        split_chunks = chunks_df.filter(pl.col("source_kind") == "section_body_split")
        assert len(split_chunks) >= 2
        assert split_chunks["chunk_count"].min() >= 2
        assert split_chunks["chunk_id"].n_unique() == len(split_chunks)

    def test_build_chunks_packs_short_paragraphs_up_to_budget(self, tmp_path: Path):
        """Short neighboring paragraphs should share chunk budget before splitting."""
        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="PA", locality="PackedTown"),
            code_slug="municipal-code",
        )
        code_dir = tmp_path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        paragraph = " ".join(["word"] * 18)
        body_text = "\n\n".join([paragraph] * 5)
        markdown_text = f"""# ARTICLE I

## 1-100. Purpose.

{body_text}
"""

        sections_df = divide_into_sections(markdown_text)
        sections_df = add_parent_relationships(sections_df)
        sections_df = enrich_sections(sections_df, code_ref)

        chunks_df = build_chunks_df(
            sections_df,
            code_ref,
            markdown_text,
            code_dir,
            llm_context_limit=4400,
            target_retrieved_chunks=5,
        )

        split_chunks = chunks_df.filter(pl.col("source_kind") == "section_body_split")
        assert len(split_chunks) == 2
        assert split_chunks["chunk_count"].to_list() == [2, 2]
        assert split_chunks["body_text"][0].count("\n\n") >= 2

    def test_build_chunks_carry_canonical_embedding_heading_text(self, tmp_path: Path):
        """Canonical chunks should preserve the exact markdown heading stack used for embedding."""
        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="PA", locality="HeadingTown"),
            code_slug="municipal-code",
        )
        code_dir = tmp_path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        body_text = " ".join(["word"] * 600)
        markdown_text = f"""# TITLE I

## CHAPTER 1

### 1-100. Purpose.

{body_text}
"""

        sections_df = divide_into_sections(markdown_text)
        sections_df = add_parent_relationships(sections_df)
        sections_df = enrich_sections(sections_df, code_ref)

        chunks_df = build_chunks_df(
            sections_df,
            code_ref,
            markdown_text,
            code_dir,
            llm_context_limit=2200,
            target_retrieved_chunks=5,
        )

        leaf_chunks = chunks_df.filter(pl.col("section_ordinal") == 2)
        assert len(leaf_chunks) >= 1
        assert "embedding_heading_text" in chunks_df.columns
        assert leaf_chunks["embedding_heading_text"][0] == (
            "# TITLE I\n\n## CHAPTER 1\n\n### 1-100. Purpose."
        )

    def test_build_chunks_packs_adjacent_child_sections_up_to_budget(
        self, tmp_path: Path
    ):
        """Adjacent child sections should pack under the parent when it is too large."""
        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="PA", locality="PackedTown"),
            code_slug="municipal-code",
        )
        code_dir = tmp_path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        section_body = " ".join(["word"] * 20)
        markdown_text = f"""# ARTICLE I

## 1-100. Purpose.

{section_body}

## 1-200. Scope.

{section_body}

## 1-300. Definitions.

{section_body}

## 1-400. Administration.

{section_body}
"""

        sections_df = divide_into_sections(markdown_text)
        sections_df = add_parent_relationships(sections_df)
        sections_df = enrich_sections(sections_df, code_ref)

        chunks_df = build_chunks_df(
            sections_df,
            code_ref,
            markdown_text,
            code_dir,
            llm_context_limit=4400,
            target_retrieved_chunks=5,
        )

        section_chunks = chunks_df.filter(
            pl.col("source_kind") == "section_packed_split"
        )
        assert len(section_chunks) == 2
        assert section_chunks["heading_text"].to_list() == [
            "# ARTICLE I (Part 1)",
            "# ARTICLE I (Part 2)",
        ]
        assert section_chunks["chunk_count"].to_list() == [2, 2]
        assert section_chunks["section_type"].to_list() == ["article", "article"]

        bodies = section_chunks["body_text"].to_list()
        assert "## 1-100. Purpose." in bodies[0]
        assert "## 1-200. Scope." in bodies[0]
        assert "## 1-300. Definitions." in bodies[1]
        assert "## 1-400. Administration." in bodies[1]

    def test_build_chunks_does_not_pack_across_chapter_boundaries(self, tmp_path: Path):
        """When a title is too large, chapter boundaries should remain separate."""
        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="PA", locality="BoundaryTown"),
            code_slug="municipal-code",
        )
        code_dir = tmp_path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        section_body = " ".join(["word"] * 18)
        markdown_text = f"""# TITLE I

## CHAPTER 1

### 1-100. Purpose.

{section_body}

### 1-200. Scope.

{section_body}

## CHAPTER 2

### 2-100. Purpose.

{section_body}

### 2-200. Scope.

{section_body}
"""

        sections_df = divide_into_sections(markdown_text)
        sections_df = add_parent_relationships(sections_df)
        sections_df = enrich_sections(sections_df, code_ref)

        chunks_df = build_chunks_df(
            sections_df,
            code_ref,
            markdown_text,
            code_dir,
            llm_context_limit=4300,
            target_retrieved_chunks=5,
        )

        section_chunks = chunks_df.filter(pl.col("source_kind") == "section_subtree")
        assert len(section_chunks) == 2
        assert section_chunks["heading_text"].to_list() == [
            "## CHAPTER 1",
            "## CHAPTER 2",
        ]
        assert section_chunks["section_type"].to_list() == ["chapter", "chapter"]

        first_body, second_body = section_chunks["body_text"].to_list()
        assert "### 1-100. Purpose." in first_body
        assert "### 1-200. Scope." in first_body
        assert "### 2-100. Purpose." not in first_body
        assert "### 2-100. Purpose." in second_body
        assert "### 2-200. Scope." in second_body

    def test_build_chunks_prefers_largest_fitting_ancestor(self, tmp_path: Path):
        """If the full ancestor subtree fits, it should become the chunk target."""
        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="PA", locality="AncestorTown"),
            code_slug="municipal-code",
        )
        code_dir = tmp_path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        section_body = " ".join(["word"] * 18)
        markdown_text = f"""# TITLE I

## CHAPTER 1

### 1-100. Purpose.

{section_body}

### 1-200. Scope.

{section_body}

## CHAPTER 2

### 2-100. Purpose.

{section_body}

### 2-200. Scope.

{section_body}
"""

        sections_df = divide_into_sections(markdown_text)
        sections_df = add_parent_relationships(sections_df)
        sections_df = enrich_sections(sections_df, code_ref)

        chunks_df = build_chunks_df(
            sections_df,
            code_ref,
            markdown_text,
            code_dir,
            llm_context_limit=4700,
            target_retrieved_chunks=5,
        )

        section_chunks = chunks_df.filter(pl.col("source_kind") == "section_subtree")
        assert len(section_chunks) == 1
        assert section_chunks["heading_text"].to_list() == ["# TITLE I"]
        assert section_chunks["section_type"].to_list() == ["title"]

        body_text = section_chunks["body_text"].item(0)
        assert "## CHAPTER 1" in body_text
        assert "## CHAPTER 2" in body_text

    def test_segment_legal_code_falls_back_to_code_start_without_regions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When regions.parquet is absent, code_start.output_line should gate sections."""
        input_text = """ARTICLE I GUIDE TO THE CHARTER

1-100 Purpose

1-200 Definitions

PREAMBLE
This charter establishes the basic organization of city government under state law.

ARTICLE I

1-100. Purpose.

This section sets out the basic powers of the city.

ARTICLE II

2-100. Council.

The council exercises legislative authority through local ordinances."""

        structure = HeadingStructure(
            levels=[
                HeadingLevel(
                    level=1,
                    regex_pattern=r"^ARTICLE\s+[IVXLCDM]+(?:\s+.+)?$",
                    markdown_prefix="#",
                    example_heading="ARTICLE I GUIDE TO THE CHARTER",
                    type_label="article",
                    number_regex=r"[IVXLCDM]+",
                ),
                HeadingLevel(
                    level=2,
                    regex_pattern=r"^\d+(?:-\d+)+\.\s+.+$",
                    markdown_prefix="##",
                    example_heading="1-100. Purpose.",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                ),
            ],
            total_levels=2,
            file_sample_size=10,
            code_start_element_id=4,
            code_start_line=10,
        )

        code_ref = TestSegmentLegalCodeCanonicalFiltering._write_parse_outputs(
            tmp_path,
            monkeypatch,
            input_text,
            structure,
        )
        (code_ref.full_data_dir / "regions.parquet").unlink()

        sections_df, _ = segment_legal_code(code_ref, embedding_model_token_limit=128)

        assert sections_df["heading_text"].to_list() == [
            "# ARTICLE I",
            "## 1-100. Purpose.",
            "# ARTICLE II",
            "## 2-100. Council.",
        ]

    def test_level_jumps(self):
        """Test handling of level jumps (H1 -> H3 -> H2)."""
        markdown_text = """# Main Title

## Section 1

### Deep Subsection

## Section 2

#### Very Deep

### Back to Level 3"""

        sections = divide_into_sections(markdown_text)
        result = add_parent_relationships(sections)

        parents = result["parent"].to_list()

        # Expected: H1(None) -> H2(H1) -> H3(H2) -> H2(H1) -> H4(H2) -> H3(H2)
        expected_parents = [None, 0, 1, 0, 3, 3]
        assert parents == expected_parents

    def test_complex_nested_structure(self):
        """Test complex nested structure with multiple levels."""
        markdown_text = """# Title 1

## Section 1.1

### Subsection 1.1.1

#### Deep 1.1.1.1

### Subsection 1.1.2

## Section 1.2

### Subsection 1.2.1

# Title 2

## Section 2.1"""

        sections = divide_into_sections(markdown_text)
        result = add_parent_relationships(sections)

        parents = result["parent"].to_list()

        # Verify complex hierarchy
        expected_parents = [None, 0, 1, 2, 1, 0, 5, None, 7]
        assert parents == expected_parents

    def test_single_section(self):
        """Test single section (no parent)."""
        markdown_text = """# Only Title

Content here."""

        sections = divide_into_sections(markdown_text)
        result = add_parent_relationships(sections)

        assert len(result) == 1
        assert result["parent"][0] is None
        assert result["heading_level"][0] == 1

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pl.DataFrame(
            schema={
                "section_ordinal": pl.Int64,
                "heading_level": pl.Int64,
                "heading_text": pl.String,
                "body_text": pl.String,
            }
        )

        result = add_parent_relationships(empty_df)

        assert len(result) == 0
        assert "parent" in result.columns
        assert "children" in result.columns
        assert "depth" in result.columns
        assert "ancestor_path" in result.columns
        assert result.schema["parent"] == pl.Int64

    def test_same_level_sections(self):
        """Test consecutive sections at same level."""
        markdown_text = """# Title

## Section 1

## Section 2

## Section 3"""

        sections = divide_into_sections(markdown_text)
        result = add_parent_relationships(sections)

        parents = result["parent"].to_list()
        # All H2 sections should have H1 as parent
        assert parents == [None, 0, 0, 0]

    def test_deep_hierarchy(self):
        """Test deep hierarchy (H1 through H6)."""
        markdown_text = """# Level 1

## Level 2

### Level 3

#### Level 4

##### Level 5

###### Level 6"""

        sections = divide_into_sections(markdown_text)
        result = add_parent_relationships(sections)

        parents = result["parent"].to_list()
        # Each level should have immediate parent
        expected_parents = [None, 0, 1, 2, 3, 4]
        assert parents == expected_parents

    def test_invalid_dataframe(self):
        """Test error handling for invalid DataFrame."""
        # DataFrame missing required columns
        invalid_df = pl.DataFrame(
            {
                "section_ordinal": [0, 1],
                "heading_level": [1, 2],
                # Missing heading_text and body_text
            }
        )

        with pytest.raises(ValueError, match="DataFrame missing required columns"):
            add_parent_relationships(invalid_df)

    def test_dataframe_schema(self):
        """Test that returned DataFrame has correct schema."""
        markdown_text = """# Title

## Section

Content."""

        sections = divide_into_sections(markdown_text)
        result = add_parent_relationships(sections)

        # Check column names
        expected_columns = [
            "section_ordinal",
            "heading_level",
            "heading_text",
            "body_text",
            "line_number",
            "parent",
            "children",
            "depth",
            "ancestor_path",
        ]
        assert result.columns == expected_columns

        # Check column types
        schema = result.schema
        assert schema["section_ordinal"] == pl.Int64
        assert schema["heading_level"] == pl.Int64
        assert schema["heading_text"] == pl.String
        assert schema["body_text"] == pl.String
        assert schema["line_number"] == pl.Int64
        assert schema["parent"] == pl.Int64
        assert schema["children"] == pl.List(pl.Int64)
        assert schema["depth"] == pl.Int64
        assert schema["ancestor_path"] == pl.String

    def test_chained_usage(self):
        """Test typical usage pattern with function chaining."""
        markdown_text = """# Main

## Section 1

### Subsection

## Section 2"""

        # Test chaining
        result = add_parent_relationships(divide_into_sections(markdown_text))

        assert len(result) == 4
        assert "parent" in result.columns

        # Verify hierarchy
        parents = result["parent"].to_list()
        assert parents == [None, 0, 1, 0]

    def test_performance_large_document(self):
        """Test performance with larger document."""
        # Create a document with many sections
        sections = ["# Main Title"]
        for i in range(100):
            sections.append(f"## Section {i}")
            sections.append(f"Content for section {i}.")

        markdown_text = "\n\n".join(sections)

        sections_df = divide_into_sections(markdown_text)
        result = add_parent_relationships(sections_df)

        assert len(result) == 101

        # All H2 sections should have H1 as parent
        parents = result["parent"].to_list()
        assert parents[0] is None  # H1
        assert all(p == 0 for p in parents[1:])  # All H2 sections

    def test_handles_missing_heading_level_with_heading_text_fallback(self):
        """If heading_level is null, derive it from heading_text markers."""
        df = pl.DataFrame(
            {
                "section_ordinal": [0, 1, 2],
                "heading_level": [None, None, 3],
                "heading_text": ["# Root", "## Child", "### Grandchild"],
                "body_text": ["a", "b", "c"],
                "line_number": [1, 2, 3],
            }
        )

        result = add_parent_relationships(df)

        assert result["parent"].to_list() == [None, 0, 1]
        assert result["heading_level"].to_list() == [1, 2, 3]


class TestSegmentText:
    """Test cases for segment_text function."""

    def test_short_text_under_limit(self):
        """Test text that fits within token limit."""
        text = "This is a short text that should fit in one segment."

        segments = segment_text(text, token_limit=100)

        assert len(segments) == 1
        assert segments[0] == text

    def test_empty_text(self):
        """Test handling of empty text."""
        # Empty string
        segments = segment_text("")
        assert segments == []

        # Whitespace only
        segments = segment_text("   \n  \t  ")
        assert segments == []

    def test_text_requiring_multiple_segments(self):
        """Test text that needs to be split into multiple segments."""
        # Create text with many sentences
        sentences = ["This is sentence one."] * 50
        text = " ".join(sentences)

        segments = segment_text(text, token_limit=50)

        assert len(segments) > 1
        # Each segment should be under the token limit
        for segment in segments:
            assert _estimate_token_count(segment) <= 50

    def test_sentence_boundary_preservation(self):
        """Test that sentence boundaries are preserved when possible."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        segments = segment_text(text, token_limit=20)

        # Should preserve sentence boundaries
        for segment in segments:
            # Each segment should end with sentence-ending punctuation when possible
            if len(segment.split()) > 1:
                assert segment.endswith((".", "!", "?"))

    def test_very_long_sentence(self):
        """Test handling of very long sentences that exceed limit."""
        # Create a very long single sentence
        long_sentence = "This is an extremely long sentence that goes on and on and contains many words and should definitely exceed the token limit when processed by the segmentation function."

        segments = segment_text(long_sentence, token_limit=20)

        # Should split the long sentence
        assert len(segments) >= 1
        for segment in segments:
            assert _estimate_token_count(segment) <= 20

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Non-string text
        with pytest.raises(TypeError, match="text must be a string"):
            segment_text(123)

        with pytest.raises(TypeError, match="text must be a string"):
            segment_text(None)

        # Invalid token_limit
        with pytest.raises(ValueError, match="token_limit must be a positive number"):
            segment_text("text", token_limit=0)

        with pytest.raises(ValueError, match="token_limit must be a positive number"):
            segment_text("text", token_limit=-5)

    def test_text_with_various_punctuation(self):
        """Test text with various punctuation and sentence endings."""
        text = "First sentence! Second sentence? Third sentence. Fourth sentence; Fifth sentence:"

        segments = segment_text(text, token_limit=30)

        # Should handle different punctuation correctly
        assert len(segments) >= 1
        for segment in segments:
            assert isinstance(segment, str)
            assert len(segment.strip()) > 0

    def test_text_with_newlines_and_whitespace(self):
        """Test text with various whitespace patterns."""
        text = """First sentence.

        Second sentence with extra spaces.

        Third sentence."""

        segments = segment_text(text, token_limit=50)

        # Should handle whitespace correctly
        assert len(segments) >= 1
        for segment in segments:
            # No leading/trailing whitespace in segments
            assert segment == segment.strip()
            # No double spaces within segments
            assert "  " not in segment

    def test_exact_token_limit_boundary(self):
        """Test text that exactly matches token limit."""
        # Create text that should exactly fit
        target_words = int(50 * 0.75)  # word_limit for token_limit=50
        words = ["word"] * target_words
        text = " ".join(words)

        segments = segment_text(text, token_limit=50)

        # Should create exactly one segment
        assert len(segments) == 1
        assert len(segments[0].split()) == target_words

    def test_paragraph_preservation_basic(self):
        """Test that paragraph boundaries are preserved when possible."""
        text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph here."""

        segments = segment_text(text, token_limit=50)

        # Should preserve each paragraph as separate segments
        assert len(segments) == 3
        assert "First paragraph" in segments[0]
        assert "Second paragraph" in segments[1]
        assert "Third paragraph" in segments[2]

    def test_paragraph_under_limit_preserved(self):
        """Test that paragraphs under token limit are kept intact."""
        text = """Short paragraph.

Another short paragraph.

Final short paragraph."""

        segments = segment_text(text, token_limit=100)

        # Each paragraph should be a separate segment
        assert len(segments) == 3
        for segment in segments:
            # Each segment should be a complete paragraph
            assert segment.endswith(".")
            assert len(segment.split()) < 100  # Well under limit

    def test_paragraph_over_limit_split_by_sentences(self):
        """Test that paragraphs exceeding limit are split by sentences."""
        # Create a long paragraph with many sentences
        sentences = ["This is sentence one."] * 20
        long_paragraph = " ".join(sentences)

        text = f"""Short paragraph.

{long_paragraph}

Another short paragraph."""

        segments = segment_text(text, token_limit=50)

        # Should have more than 3 segments due to long paragraph splitting
        assert len(segments) > 3

        # First and last segments should be the short paragraphs
        assert "Short paragraph" in segments[0]
        assert "Another short paragraph" in segments[-1]

    def test_mixed_paragraph_scenarios(self):
        """Test mixed scenarios with some paragraphs under and some over limit."""
        text = """Short intro paragraph.

This is a very long paragraph that contains many sentences and should definitely exceed the token limit when processed. It has multiple sentences. Each sentence adds more words. This ensures it gets split properly.

Short conclusion paragraph."""

        segments = segment_text(text, token_limit=30)

        # Should have multiple segments
        assert len(segments) >= 3

        # First segment should be the short intro
        assert "Short intro paragraph" in segments[0]

        # Last segment should contain the conclusion
        assert any("Short conclusion paragraph" in seg for seg in segments)

    def test_single_long_paragraph(self):
        """Test single paragraph that exceeds token limit."""
        # Create a single long paragraph
        sentences = ["Sentence with content."] * 15
        text = " ".join(sentences)

        segments = segment_text(text, token_limit=40)

        # Should split the single paragraph into multiple segments
        assert len(segments) > 1

        # All segments should be under the token limit
        for segment in segments:
            assert _estimate_token_count(segment) <= 40

    def test_paragraph_with_various_whitespace(self):
        """Test paragraph handling with various whitespace patterns."""
        text = """First paragraph.

        Second paragraph with extra spaces.

        Third paragraph."""

        segments = segment_text(text, token_limit=50)

        # Should handle whitespace correctly and create 3 segments
        assert len(segments) == 3

        # Segments should be stripped of extra whitespace
        for segment in segments:
            assert segment == segment.strip()
            assert "  " not in segment

    def test_empty_paragraphs_ignored(self):
        """Test that empty paragraphs are ignored."""
        text = """First paragraph.


        Second paragraph.



        Third paragraph."""

        segments = segment_text(text, token_limit=50)

        # Should ignore empty paragraphs and create 3 segments
        assert len(segments) == 3
        assert "First paragraph" in segments[0]
        assert "Second paragraph" in segments[1]
        assert "Third paragraph" in segments[2]

    def test_dense_numeric_text_respects_token_budget(self):
        """Dense numeric/punctuation text should still be split under token budget."""
        text = " ".join(f"{600 + i} 14401 Friar St. Van Nuys" for i in range(120))

        segments = segment_text(text, token_limit=64)

        assert len(segments) > 1
        for segment in segments:
            assert _estimate_token_count(segment) <= 64

    def test_single_unbroken_alphanumeric_blob_is_split(self):
        """A single very long blob (no whitespace) should not remain one huge segment."""
        blob = "60114401FriarStVanNuys" * 80

        segments = segment_text(blob, token_limit=32)

        assert len(segments) > 1
        for segment in segments:
            assert _estimate_token_count(segment) <= 32

    def test_single_long_digit_run_is_split_under_budget(self):
        """Fallback token-budget splitting should break oversized digit runs."""
        digits = "1234567890" * 15

        chunks = _split_by_token_budget(digits, token_limit=16)

        assert len(chunks) > 1
        assert "".join(chunks) == digits
        for chunk in chunks:
            assert _estimate_token_count(chunk) <= 16


class TestCreateSegmentsDf:
    """Test cases for create_segments_df function."""

    def test_basic_flat_format(self):
        """Test basic flat format with one row per segment."""
        # Create test DataFrame with multiple paragraphs
        df = pl.DataFrame(
            {
                "section_ordinal": [0, 1],
                "heading_level": [1, 2],
                "heading_text": ["# Title", "## Section"],
                "body_text": [
                    "First paragraph. Second paragraph.",
                    "Section content here. More section content.",
                ],
            }
        )

        result = create_segments_df(df, token_limit=50)

        # Check DataFrame structure
        expected_columns = [
            "segment_ordinal",
            "section_ordinal",
            "section_heading",
            "section_level",
            "segment_position",
            "segment_text",
            "word_count",
        ]
        assert result.columns == expected_columns

        # Should have segments from both sections
        assert len(result) >= 2

        # Check segment indices are sequential
        segment_indices = result["segment_ordinal"].to_list()
        assert segment_indices == list(range(len(result)))

    def test_empty_sections_handling(self):
        """Test handling of sections with empty or null text."""
        df = pl.DataFrame(
            {
                "section_ordinal": [0, 1, 2],
                "heading_level": [1, 2, 3],
                "heading_text": ["# Title", "## Section", "### Subsection"],
                "body_text": ["Content here", None, ""],
            }
        )

        result = create_segments_df(df)

        # Should only have segments for non-empty sections
        assert len(result) == 1
        assert result["section_ordinal"][0] == 0
        assert result["segment_text"][0] == "Content here"

    def test_long_heading_adjustment(self):
        """Test token limit adjustment for long headings."""
        # min_tokens is hardcoded to 20 in implementation

        # Setup:
        # token_limit = 50
        # Heading: 35 simple words -> _estimate_token_count = 35
        # Adjusted limit: max(20, 50 - 35) = 20
        #
        # Body: 25 simple words -> _estimate_token_count = 25
        # 25 > 20 -> Should split
        # Without adjustment: 25 <= 50 -> would fit in 1 segment

        long_heading = " ".join(["Head"] * 35)
        body_text = " ".join(["Body"] * 25)

        df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_level": [1],
                "heading_text": [long_heading],
                "body_text": [body_text],
            }
        )

        result = create_segments_df(df, token_limit=50)

        # Should be split due to reduced effective limit
        assert len(result) >= 2

    def test_ancestor_heading_cost_accounted_for(self):
        """When ancestor_path is present, full ancestor heading cost is subtracted."""
        # Build a hierarchy: # Root (10 tokens) -> ## Child (10 tokens)
        # token_limit = 50
        # Ancestor heading cost for child = Root heading (10) + Child heading (10) = 20
        # Adjusted body budget = 50 - 20 = 30
        # Body text = 35 tokens -> should be split
        #
        # Without ancestor accounting (only immediate heading = 10 tokens):
        # Adjusted body budget = 50 - 10 = 40
        # Body text = 35 tokens -> would fit in 1 segment

        root_heading = " ".join(["Root"] * 10)
        child_heading = " ".join(["Child"] * 10)
        body_text = " ".join(["Word"] * 35)

        sections = add_parent_relationships(
            pl.DataFrame(
                {
                    "section_ordinal": [0, 1],
                    "heading_level": [1, 2],
                    "heading_text": [f"# {root_heading}", f"## {child_heading}"],
                    "body_text": [None, body_text],
                }
            )
        )

        result = create_segments_df(sections, token_limit=50)

        # Body should be split because the full ancestor heading cost (20)
        # leaves only 30 tokens of budget, and the body is 35 tokens
        assert len(result) >= 2

        # Verify all segments fit within the adjusted budget
        ancestor_tokens = _estimate_token_count(
            f"# {root_heading}"
        ) + _estimate_token_count(f"## {child_heading}")
        adjusted_limit = max(20, 50 - ancestor_tokens)
        for row in result.to_dicts():
            assert _estimate_token_count(row["segment_text"]) <= adjusted_limit

    def test_segment_position_tracking(self):
        """Test that segment_position is correctly tracked within sections."""
        df = pl.DataFrame(
            {
                "section_ordinal": [0, 1],
                "heading_level": [1, 2],
                "heading_text": ["# Title", "## Section"],
                "body_text": [
                    "Paragraph one. Paragraph two. Paragraph three.",
                    "Section paragraph one. Section paragraph two.",
                ],
            }
        )

        result = create_segments_df(df, token_limit=30)

        # Check segment positions for first section (should have multiple segments)
        first_section_segments = result.filter(result["section_ordinal"] == 0)
        positions = first_section_segments["segment_position"].to_list()
        assert positions == list(range(len(first_section_segments)))

        # Check segment positions for second section
        second_section_segments = result.filter(result["section_ordinal"] == 1)
        positions = second_section_segments["segment_position"].to_list()
        assert positions == list(range(len(second_section_segments)))

    def test_section_context_preservation(self):
        """Test that section context is preserved for each segment."""
        df = pl.DataFrame(
            {
                "section_ordinal": [0, 1],
                "heading_level": [1, 3],
                "heading_text": ["# Main Title", "### Deep Section"],
                "body_text": ["Main content.", "Deep content here."],
            }
        )

        result = create_segments_df(df)

        # Check section context is preserved
        for row in result.to_dicts():
            if row["section_ordinal"] == 0:
                assert row["section_heading"] == "# Main Title"
                assert row["section_level"] == 1
            elif row["section_ordinal"] == 1:
                assert row["section_heading"] == "### Deep Section"
                assert row["section_level"] == 3

    def test_word_count_accuracy(self):
        """Test that word_count is accurate for each segment."""
        df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_level": [1],
                "heading_text": ["# Title"],
                "body_text": ["One two three four five."],
            }
        )

        result = create_segments_df(df)

        # Should have one segment with 5 words
        assert len(result) == 1
        assert result["word_count"][0] == 5

    def test_large_document_flat_format(self):
        """Test flat format with larger document requiring many segments."""
        # Create content that will require multiple segments
        long_content = "This is sentence one. " * 20  # 20 sentences

        df = pl.DataFrame(
            {
                "section_ordinal": [0, 1],
                "heading_level": [1, 2],
                "heading_text": ["# Title", "## Section"],
                "body_text": [long_content, "Short content."],
            }
        )

        result = create_segments_df(df, token_limit=50)

        # Should have multiple segments from long content plus one from short content
        assert len(result) >= 3

        # Check that all segments are under the token limit
        for row in result.to_dicts():
            assert _estimate_token_count(row["segment_text"]) <= 50

    def test_paragraph_preservation_in_flat_format(self):
        """Flat-format segmentation should pack short adjacent paragraphs when possible."""
        text = """First paragraph with content.

Second paragraph here.

Third paragraph content."""

        df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_level": [1],
                "heading_text": ["# Title"],
                "body_text": [text],
            }
        )

        result = create_segments_df(df, token_limit=100)

        # All three paragraphs should pack into one retrieval segment.
        assert len(result) == 1

        # Check packed segment contains all paragraphs in order.
        segment_texts = result["segment_text"].to_list()
        assert "First paragraph" in segment_texts[0]
        assert "Second paragraph" in segment_texts[0]
        assert "Third paragraph" in segment_texts[0]

        # All segments should belong to same section
        section_refs = result["section_ordinal"].to_list()
        assert all(ref == 0 for ref in section_refs)

    def test_create_segments_df_packs_adjacent_paragraphs_under_limit(self):
        """Retrieval segments should pack adjacent paragraphs up to the effective budget."""
        paragraph = " ".join(["alpha"] * 12) + "."
        text = "\n\n".join([paragraph] * 4)

        df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_level": [2],
                "heading_text": ["## Section"],
                "body_text": [text],
            }
        )

        result = create_segments_df(df, token_limit=16)

        # create_segments_df enforces a minimum effective body budget of 20 tokens,
        # so this lower token limit still permits only one paragraph per segment.
        assert len(result) == 4
        for row in result.to_dicts():
            assert _estimate_token_count(row["segment_text"]) <= 16

        packed = create_segments_df(df, token_limit=32)
        assert len(packed) == 2
        assert packed["segment_text"][0].count("alpha") == 24
        assert packed["segment_text"][1].count("alpha") == 24

    def test_create_segments_df_prefers_internal_section_boundaries(self):
        """Packed chunk bodies should prefer embedded markdown section boundaries."""
        text = """## 1-100. Purpose.

Purpose text for the first section.

## 1-200. Scope.

Scope text for the second section.

## 1-300. Definitions.

Definitions text for the third section."""

        df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_level": [1],
                "heading_text": ["# ARTICLE I"],
                "body_text": [text],
                "source_kind": ["section_packed_split"],
            }
        )

        result = create_segments_df(df, token_limit=40)

        assert len(result) == 2
        assert "## 1-100. Purpose." in result["segment_text"][0]
        assert "## 1-200. Scope." in result["segment_text"][0]
        assert "## 1-300. Definitions." in result["segment_text"][1]
        adjusted_limit = max(20, 40 - _estimate_token_count("# ARTICLE I"))
        assert all(
            _estimate_token_count(text) <= adjusted_limit
            for text in result["segment_text"].to_list()
        )

    def test_create_segments_df_respects_total_embedding_budget(self):
        """Segment bodies should leave room for the ancestor headings prepended later."""
        body = "\n\n".join(
            [
                " ".join(["alpha"] * 10) + ".",
                " ".join(["beta"] * 10) + ".",
                " ".join(["gamma"] * 10) + ".",
            ]
        )
        df = pl.DataFrame(
            {
                "section_ordinal": [0, 1],
                "heading_level": [1, 2],
                "heading_text": ["# TITLE I", "## Section 1"],
                "body_text": ["", body],
                "parent": [None, 0],
                "children": [[1], []],
                "depth": [0, 1],
                "ancestor_path": ["0", "0/1"],
            }
        )

        result = create_segments_df(df, token_limit=20)

        assert len(result) >= 2
        for row in result.filter(pl.col("section_ordinal") == 1).to_dicts():
            assembled = "\n\n".join(["# TITLE I", "## Section 1", row["segment_text"]])
            assert _estimate_token_count(assembled) <= 20

    def test_create_segments_df_uses_embedding_heading_text_without_ancestor_path(self):
        """Chunk-derived rows should budget against canonical heading text even without ancestor_path."""
        embedding_heading_text = "\n\n".join(
            [
                "# " + " ".join(["Root"] * 6),
                "## " + " ".join(["Child"] * 6),
                "### " + " ".join(["Leaf"] * 6),
            ]
        )
        body_text = " ".join(["Body"] * 35)

        df = pl.DataFrame(
            {
                "section_ordinal": [2],
                "heading_level": [3],
                "heading_text": ["### Leaf (Part 1)"],
                "body_text": [body_text],
                "embedding_heading_text": [embedding_heading_text],
                "context_path": ["Root > Child > Leaf"],
                "source_kind": ["section_body_split"],
            }
        )

        result = create_segments_df(df, token_limit=50)

        assert len(result) >= 2
        for row in result.to_dicts():
            assembled = "\n\n".join([embedding_heading_text, row["segment_text"]])
            assert _estimate_token_count(assembled) <= 50

    def test_mixed_scenarios_flat_format(self):
        """Test mixed scenarios with various section lengths."""
        df = pl.DataFrame(
            {
                "section_ordinal": [0, 1, 2],
                "heading_level": [1, 2, 3],
                "heading_text": ["# Title", "## Section", "### Subsection"],
                "body_text": [
                    "Short.",
                    "Medium length content with multiple sentences here.",
                    "Very long content that will definitely need to be split into multiple segments for processing. "
                    * 3,
                ],
            }
        )

        result = create_segments_df(df, token_limit=50)

        # Should have segments from all sections
        assert len(result) >= 3

        # Check section distribution
        section_0_segments = result.filter(result["section_ordinal"] == 0)
        section_1_segments = result.filter(result["section_ordinal"] == 1)
        section_2_segments = result.filter(result["section_ordinal"] == 2)

        assert len(section_0_segments) == 1  # Short content
        assert len(section_1_segments) >= 1  # Medium content
        assert len(section_2_segments) > 1  # Long content split

    def test_empty_dataframe_input(self):
        """Test handling of empty DataFrame input."""
        empty_df = pl.DataFrame(
            schema={
                "section_ordinal": pl.Int64,
                "heading_level": pl.Int64,
                "heading_text": pl.String,
                "body_text": pl.String,
            }
        )

        result = create_segments_df(empty_df)

        # Should return empty DataFrame with correct schema
        assert len(result) == 0
        expected_columns = [
            "segment_ordinal",
            "section_ordinal",
            "section_heading",
            "section_level",
            "segment_position",
            "segment_text",
            "word_count",
        ]
        assert result.columns == expected_columns

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Non-DataFrame input
        with pytest.raises(TypeError, match="df must be a polars DataFrame"):
            create_segments_df("not a dataframe")

        # Missing column
        df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_level": [1],
                "heading_text": ["# Title"],
                # Missing body_text
            }
        )

        with pytest.raises(ValueError, match="Column 'body_text' not found"):
            create_segments_df(df)

    def test_custom_text_column(self):
        """Test with custom text column name."""
        df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_level": [1],
                "heading_text": ["# Title"],
                "custom_text": ["Content to be segmented."],
            }
        )

        result = create_segments_df(df, text_column="custom_text")

        # Should process custom_text column
        assert len(result) == 1
        assert result["segment_text"][0] == "Content to be segmented."

    def test_schema_validation(self):
        """Test that returned DataFrame has correct schema."""
        df = pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_level": [2],
                "heading_text": ["## Section"],
                "body_text": ["Test content."],
            }
        )

        result = create_segments_df(df)

        # Check column names
        expected_columns = [
            "segment_ordinal",
            "section_ordinal",
            "section_heading",
            "section_level",
            "segment_position",
            "segment_text",
            "word_count",
        ]
        assert result.columns == expected_columns

        # Check column types
        schema = result.schema
        assert schema["segment_ordinal"] == pl.Int64
        assert schema["section_ordinal"] == pl.Int64
        assert schema["section_heading"] == pl.String
        assert schema["section_level"] == pl.Int64
        assert schema["segment_position"] == pl.Int64
        assert schema["segment_text"] == pl.String
        assert schema["word_count"] == pl.Int64

    def test_integration_with_parent_relationships(self):
        """Test integration with parent relationships in sections."""
        # Create sections with parent relationships
        base_df = pl.DataFrame(
            {
                "section_ordinal": [0, 1, 2],
                "heading_level": [1, 2, 3],
                "heading_text": ["# Title", "## Section", "### Subsection"],
                "body_text": [
                    "Main content here.",
                    "Section content with multiple sentences. More content here.",
                    "Subsection content.",
                ],
            }
        )

        # Add parent relationships first
        df_with_parents = add_parent_relationships(base_df)

        # Then create segments dataframe
        result = create_segments_df(df_with_parents)

        # Should have segments from all sections
        assert len(result) >= 3

        # Parent/context metadata should be propagated to segments
        assert "parent" in result.columns
        assert "children" in result.columns
        assert "depth" in result.columns
        assert "ancestor_path" in result.columns

        # Validate propagated parent chain for one segment from each section
        one_per_section = (
            result.group_by("section_ordinal").first().sort("section_ordinal")
        )
        assert one_per_section["parent"].to_list() == [None, 0, 1]
        assert one_per_section["depth"].to_list() == [0, 1, 2]
        assert one_per_section["ancestor_path"].to_list() == ["0", "0/1", "0/1/2"]

        # Check that section references are still preserved
        section_refs = result["section_ordinal"].to_list()
        assert set(section_refs) == {0, 1, 2}

    def test_propagates_enriched_section_ids(self):
        """Segment rows retain globally unique section/code identifiers."""
        sections = add_parent_relationships(
            divide_into_sections("# Main\n\nIntro\n\n## Child\n\nBody")
        )
        code_ref = CodeRef(
            jurisdiction=JurisdictionRef(state="CA", locality="LosAngeles"),
            code_slug="municipal-code",
        )
        enriched = enrich_sections(sections, code_ref)

        result = create_segments_df(enriched)

        assert "code_id" in result.columns
        assert "section_id" in result.columns
        assert "parent_id" in result.columns
        assert result["code_id"].n_unique() == 1
        assert result["section_id"].n_unique() >= 2


class TestEnrichSections:
    """Test cases for enrich_sections function."""

    def _make_sections(self, markdown_text: str) -> pl.DataFrame:
        return add_parent_relationships(divide_into_sections(markdown_text))

    def _make_code_ref(self) -> CodeRef:
        return CodeRef(
            jurisdiction=JurisdictionRef(state="CA", locality="LosAngeles"),
            code_slug="municipal-code",
        )

    def test_adds_code_id_section_id_parent_id(self):
        sections = self._make_sections("# Main\n\n## Child\n\nBody.")
        code_ref = self._make_code_ref()
        result = enrich_sections(sections, code_ref)

        assert "code_id" in result.columns
        assert "section_id" in result.columns
        assert "parent_id" in result.columns

        assert result["code_id"][0] == "CA:LosAngeles:municipal-code"
        assert result["section_id"][0] == "CA:LosAngeles:municipal-code:s0"
        assert result["section_id"][1] == "CA:LosAngeles:municipal-code:s1"
        assert result["parent_id"][0] is None  # root has no parent
        assert result["parent_id"][1] == "CA:LosAngeles:municipal-code:s0"

    def test_preserves_existing_columns(self):
        sections = self._make_sections("# Title\n\nBody.")
        code_ref = self._make_code_ref()
        result = enrich_sections(sections, code_ref)

        for col in [
            "section_ordinal",
            "heading_level",
            "heading_text",
            "body_text",
            "parent",
            "children",
            "depth",
            "ancestor_path",
        ]:
            assert col in result.columns


class TestGetSectionText:
    """Test cases for get_section_text function."""

    def _make_sections(self, markdown_text: str) -> pl.DataFrame:
        return add_parent_relationships(divide_into_sections(markdown_text))

    def test_leaf_section(self):
        """Leaf section returns its heading + body."""
        sections = self._make_sections("# Title\n\nBody content.")
        text = get_section_text(sections, 0)
        assert "# Title" in text
        assert "Body content." in text

    def test_leaf_no_body(self):
        """Leaf section with no body returns just heading."""
        sections = self._make_sections("# Title\n\n## Child")
        # Section 1 (## Child) is a leaf with no body
        text = get_section_text(sections, 1)
        assert text == "## Child"

    def test_section_with_children(self):
        """Parent section expands children in document order."""
        md = "# Main\n\nIntro.\n\n## A\n\nA body.\n\n## B\n\nB body."
        sections = self._make_sections(md)
        text = get_section_text(sections, 0)

        assert "# Main" in text
        assert "Intro." in text
        assert "## A" in text
        assert "A body." in text
        assert "## B" in text
        assert "B body." in text

        # Children should appear after parent
        main_pos = text.index("# Main")
        a_pos = text.index("## A")
        b_pos = text.index("## B")
        assert main_pos < a_pos < b_pos

    def test_nested_subtree(self):
        """Deep nesting is expanded recursively."""
        md = "# Root\n\n## L2\n\n### L3\n\nDeep body."
        sections = self._make_sections(md)
        text = get_section_text(sections, 0)

        assert "# Root" in text
        assert "## L2" in text
        assert "### L3" in text
        assert "Deep body." in text

    def test_partial_subtree(self):
        """Expanding a mid-level section only includes its subtree."""
        md = "# Root\n\n## A\n\n### A1\n\nA1 body.\n\n## B\n\nB body."
        sections = self._make_sections(md)

        # Expand only section A (ordinal 1)
        text = get_section_text(sections, 1)
        assert "## A" in text
        assert "### A1" in text
        assert "A1 body." in text
        # Should NOT include sibling B
        assert "## B" not in text
        assert "B body." not in text

    def test_missing_ordinal_raises(self):
        sections = self._make_sections("# Title\n\nBody.")
        with pytest.raises(KeyError, match="section_ordinal 99"):
            get_section_text(sections, 99)
