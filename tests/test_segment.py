"""Tests for legiscope.segment module."""

import pytest
import polars as pl

from legiscope.segment import (
    divide_into_sections,
    add_parent_relationships,
    segment_text,
    add_segments_to_sections,
    create_segments_df,
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
            "section_idx",
            "heading_level",
            "heading_text",
            "body_text",
        ]

        # Check section indices
        assert result["section_idx"].to_list() == [0, 1, 2, 3]

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
            "section_idx",
            "heading_level",
            "heading_text",
            "body_text",
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
        assert result["section_idx"][0] == 0
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
            "section_idx",
            "heading_level",
            "heading_text",
            "body_text",
        ]

        # Check column types
        schema = result.schema
        assert schema["section_idx"] == pl.Int64
        assert schema["heading_level"] == pl.Int64
        assert schema["heading_text"] == pl.String
        assert schema["body_text"] == pl.String

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
        assert result["section_idx"].to_list() == list(range(100))
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
            "section_idx",
            "heading_level",
            "heading_text",
            "body_text",
            "parent",
        ]

        # Check parent relationships
        parents = result["parent"].to_list()
        assert parents[0] is None  # H1 has no parent
        assert parents[1] == 0  # H2 parent is H1 (idx 0)
        assert parents[2] == 1  # H3 parent is H2 (idx 1)

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
                "section_idx": pl.Int64,
                "heading_level": pl.Int64,
                "heading_text": pl.String,
                "body_text": pl.String,
            }
        )

        result = add_parent_relationships(empty_df)

        assert len(result) == 0
        assert "parent" in result.columns
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
                "section_idx": [0, 1],
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
            "section_idx",
            "heading_level",
            "heading_text",
            "body_text",
            "parent",
        ]
        assert result.columns == expected_columns

        # Check column types
        schema = result.schema
        assert schema["section_idx"] == pl.Int64
        assert schema["heading_level"] == pl.Int64
        assert schema["heading_text"] == pl.String
        assert schema["body_text"] == pl.String
        assert schema["parent"] == pl.Int64

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
        # Each segment should be under the word limit
        word_limit = int(50 * 0.78)  # Default words_per_token
        for segment in segments:
            assert len(segment.split()) <= word_limit

    def test_sentence_boundary_preservation(self):
        """Test that sentence boundaries are preserved when possible."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        segments = segment_text(text, token_limit=20)

        # Should preserve sentence boundaries
        for segment in segments:
            # Each segment should end with sentence-ending punctuation when possible
            if len(segment.split()) > 1:
                assert segment.endswith((".", "!", "?"))

    def test_custom_words_per_token(self):
        """Test custom words per token ratio."""
        text = "This is a test text with multiple sentences. It should be split according to the custom ratio."

        # Use higher ratio (more words per token)
        segments = segment_text(text, token_limit=20, words_per_token=1.0)

        # Should create fewer segments with higher ratio
        word_limit = int(20 * 1.0)
        for segment in segments:
            assert len(segment.split()) <= word_limit

    def test_very_long_sentence(self):
        """Test handling of very long sentences that exceed limit."""
        # Create a very long single sentence
        long_sentence = "This is an extremely long sentence that goes on and on and contains many words and should definitely exceed the token limit when processed by the segmentation function."

        segments = segment_text(long_sentence, token_limit=20)

        # Should split the long sentence
        assert len(segments) >= 1
        for segment in segments:
            assert len(segment.split()) <= int(20 * 0.78)

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

        # Invalid words_per_token
        with pytest.raises(
            ValueError, match="words_per_token must be a positive number"
        ):
            segment_text("text", words_per_token=0)

        with pytest.raises(
            ValueError, match="words_per_token must be a positive number"
        ):
            segment_text("text", words_per_token=-1)

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
        target_words = int(50 * 0.78)  # word_limit for token_limit=50
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

        # All segments should be under word limit
        word_limit = int(40 * 0.78)
        for segment in segments:
            assert len(segment.split()) <= word_limit

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


class TestAddSegmentsToSections:
    """Test cases for add_segments_to_sections function."""

    def test_basic_dataframe_integration(self):
        """Test basic integration with sections DataFrame."""
        # Create test DataFrame
        # Create longer text that will definitely need multiple segments
        long_content = "This is sentence one. " * 50  # 50 sentences
        df = pl.DataFrame(
            {
                "section_idx": [0, 1],
                "heading_level": [1, 2],
                "heading_text": ["# Title", "## Section"],
                "body_text": ["Short content.", long_content],
            }
        )

        result = add_segments_to_sections(df, token_limit=50)

        # Check new columns exist
        assert "segments" in result.columns
        assert "segment_count" in result.columns
        assert "total_words" in result.columns

        # Check segment counts
        segment_counts = result["segment_count"].to_list()
        assert segment_counts[0] == 1  # Short content = 1 segment
        assert segment_counts[1] > 1  # Longer content = multiple segments

    def test_empty_body_text(self):
        """Test handling of empty body text."""
        df = pl.DataFrame(
            {
                "section_idx": [0, 1],
                "heading_level": [1, 2],
                "heading_text": ["# Title", "## Section"],
                "body_text": [None, ""],
            }
        )

        result = add_segments_to_sections(df)

        # Both should have 0 segments
        segment_counts = result["segment_count"].to_list()
        assert segment_counts == [0, 0]

        # Segments should be empty lists
        segments = result["segments"].to_list()
        assert segments == [[], []]

    def test_custom_text_column(self):
        """Test with custom text column name."""
        df = pl.DataFrame(
            {
                "section_idx": [0],
                "heading_level": [1],
                "heading_text": ["# Title"],
                "custom_text": ["Content to be segmented."],
            }
        )

        result = add_segments_to_sections(df, text_column="custom_text")

        # Should process custom_text column
        assert "segments" in result.columns
        assert result["segment_count"][0] >= 1

    def test_invalid_dataframe_input(self):
        """Test error handling for invalid inputs."""
        # Non-DataFrame input
        with pytest.raises(TypeError, match="df must be a polars DataFrame"):
            add_segments_to_sections("not a dataframe")

        # Missing column
        df = pl.DataFrame(
            {
                "section_idx": [0],
                "heading_level": [1],
                "heading_text": ["# Title"],
                # Missing body_text
            }
        )

        with pytest.raises(ValueError, match="Column 'body_text' not found"):
            add_segments_to_sections(df)

    def test_column_order_preservation(self):
        """Test that original column order is preserved."""
        df = pl.DataFrame(
            {
                "section_idx": [0],
                "heading_level": [1],
                "heading_text": ["# Title"],
                "body_text": ["Content."],
                "parent": [None],
            }
        )

        result = add_segments_to_sections(df)

        # Original columns should be preserved
        original_columns = [
            "section_idx",
            "heading_level",
            "heading_text",
            "body_text",
            "parent",
        ]
        new_columns = ["segments", "segment_count", "total_words"]
        expected_columns = ["section_idx"] + original_columns[1:] + new_columns

        assert result.columns == expected_columns

    def test_integration_with_parent_relationships(self):
        """Test integration with parent relationships."""
        # Create sections with parent relationships
        base_df = pl.DataFrame(
            {
                "section_idx": [0, 1, 2],
                "heading_level": [1, 2, 3],
                "heading_text": ["# Title", "## Section", "### Subsection"],
                "body_text": [
                    "Main content.",
                    "Section content.",
                    "Subsection content.",
                ],
            }
        )

        # Add parent relationships first
        df_with_parents = add_parent_relationships(base_df)

        # Then add segments
        result = add_segments_to_sections(df_with_parents)

        # Should have all columns
        expected_columns = [
            "section_idx",
            "heading_level",
            "heading_text",
            "body_text",
            "parent",
            "segments",
            "segment_count",
            "total_words",
        ]
        assert result.columns == expected_columns

        # Parent relationships should be preserved
        parents = result["parent"].to_list()
        assert parents == [None, 0, 1]

    def test_large_text_processing(self):
        """Test processing of large text content."""
        # Create large text with many sentences
        large_content = "This is sentence one. " * 200  # 200 sentences

        df = pl.DataFrame(
            {
                "section_idx": [0],
                "heading_level": [1],
                "heading_text": ["# Large Section"],
                "body_text": [large_content],
            }
        )

        result = add_segments_to_sections(df, token_limit=50)

        # Should create multiple segments
        segment_count = result["segment_count"][0]
        assert segment_count >= 5  # Should be several segments with 200 sentences

        # Total word count should be correct
        total_words = result["total_words"][0]
        assert total_words == len(large_content.split())

    def test_custom_parameters(self):
        """Test with custom token limit and words per token."""
        df = pl.DataFrame(
            {
                "section_idx": [0],
                "heading_level": [1],
                "heading_text": ["# Title"],
                "body_text": [
                    "Content with multiple sentences. More content here. Even more content."
                ],
            }
        )

        # Use custom parameters
        result = add_segments_to_sections(df, token_limit=50, words_per_token=1.2)

        # Should create segments based on custom parameters
        assert result["segment_count"][0] >= 1

        # Verify segments are under custom limit
        segments = result["segments"][0]
        word_limit = int(50 * 1.2)
        for segment in segments:
            assert len(segment.split()) <= word_limit


class TestCreateSegmentsDf:
    """Test cases for create_segments_df function."""

    def test_basic_flat_format(self):
        """Test basic flat format with one row per segment."""
        # Create test DataFrame with multiple paragraphs
        df = pl.DataFrame(
            {
                "section_idx": [0, 1],
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
            "segment_idx",
            "section_ref",
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
        segment_indices = result["segment_idx"].to_list()
        assert segment_indices == list(range(len(result)))

    def test_empty_sections_handling(self):
        """Test handling of sections with empty or null text."""
        df = pl.DataFrame(
            {
                "section_idx": [0, 1, 2],
                "heading_level": [1, 2, 3],
                "heading_text": ["# Title", "## Section", "### Subsection"],
                "body_text": ["Content here", None, ""],
            }
        )

        result = create_segments_df(df)

        # Should only have segments for non-empty sections
        assert len(result) == 1
        assert result["section_ref"][0] == 0
        assert result["segment_text"][0] == "Content here"

    def test_segment_position_tracking(self):
        """Test that segment_position is correctly tracked within sections."""
        df = pl.DataFrame(
            {
                "section_idx": [0, 1],
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
        first_section_segments = result.filter(result["section_ref"] == 0)
        positions = first_section_segments["segment_position"].to_list()
        assert positions == list(range(len(first_section_segments)))

        # Check segment positions for second section
        second_section_segments = result.filter(result["section_ref"] == 1)
        positions = second_section_segments["segment_position"].to_list()
        assert positions == list(range(len(second_section_segments)))

    def test_section_context_preservation(self):
        """Test that section context is preserved for each segment."""
        df = pl.DataFrame(
            {
                "section_idx": [0, 1],
                "heading_level": [1, 3],
                "heading_text": ["# Main Title", "### Deep Section"],
                "body_text": ["Main content.", "Deep content here."],
            }
        )

        result = create_segments_df(df)

        # Check section context is preserved
        for row in result.to_dicts():
            if row["section_ref"] == 0:
                assert row["section_heading"] == "# Main Title"
                assert row["section_level"] == 1
            elif row["section_ref"] == 1:
                assert row["section_heading"] == "### Deep Section"
                assert row["section_level"] == 3

    def test_word_count_accuracy(self):
        """Test that word_count is accurate for each segment."""
        df = pl.DataFrame(
            {
                "section_idx": [0],
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
                "section_idx": [0, 1],
                "heading_level": [1, 2],
                "heading_text": ["# Title", "## Section"],
                "body_text": [long_content, "Short content."],
            }
        )

        result = create_segments_df(df, token_limit=50)

        # Should have multiple segments from long content plus one from short content
        assert len(result) >= 3

        # Check that all segments are under word limit
        word_limit = int(50 * 0.78)
        for word_count in result["word_count"].to_list():
            assert word_count <= word_limit

    def test_paragraph_preservation_in_flat_format(self):
        """Test that paragraph boundaries are preserved in flat format."""
        text = """First paragraph with content.

Second paragraph here.

Third paragraph content."""

        df = pl.DataFrame(
            {
                "section_idx": [0],
                "heading_level": [1],
                "heading_text": ["# Title"],
                "body_text": [text],
            }
        )

        result = create_segments_df(df, token_limit=100)

        # Should have 3 segments (one per paragraph)
        assert len(result) == 3

        # Check each segment contains expected paragraph
        segment_texts = result["segment_text"].to_list()
        assert "First paragraph" in segment_texts[0]
        assert "Second paragraph" in segment_texts[1]
        assert "Third paragraph" in segment_texts[2]

        # All segments should belong to same section
        section_refs = result["section_ref"].to_list()
        assert all(ref == 0 for ref in section_refs)

    def test_mixed_scenarios_flat_format(self):
        """Test mixed scenarios with various section lengths."""
        df = pl.DataFrame(
            {
                "section_idx": [0, 1, 2],
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
        section_0_segments = result.filter(result["section_ref"] == 0)
        section_1_segments = result.filter(result["section_ref"] == 1)
        section_2_segments = result.filter(result["section_ref"] == 2)

        assert len(section_0_segments) == 1  # Short content
        assert len(section_1_segments) >= 1  # Medium content
        assert len(section_2_segments) > 1  # Long content split

    def test_empty_dataframe_input(self):
        """Test handling of empty DataFrame input."""
        empty_df = pl.DataFrame(
            schema={
                "section_idx": pl.Int64,
                "heading_level": pl.Int64,
                "heading_text": pl.String,
                "body_text": pl.String,
            }
        )

        result = create_segments_df(empty_df)

        # Should return empty DataFrame with correct schema
        assert len(result) == 0
        expected_columns = [
            "segment_idx",
            "section_ref",
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
                "section_idx": [0],
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
                "section_idx": [0],
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
                "section_idx": [0],
                "heading_level": [2],
                "heading_text": ["## Section"],
                "body_text": ["Test content."],
            }
        )

        result = create_segments_df(df)

        # Check column names
        expected_columns = [
            "segment_idx",
            "section_ref",
            "section_heading",
            "section_level",
            "segment_position",
            "segment_text",
            "word_count",
        ]
        assert result.columns == expected_columns

        # Check column types
        schema = result.schema
        assert schema["segment_idx"] == pl.Int64
        assert schema["section_ref"] == pl.Int64
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
                "section_idx": [0, 1, 2],
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

        # Check that section context is preserved including parent info
        # (parent info is not directly in segments but section_ref allows lookup)
        section_refs = result["section_ref"].to_list()
        assert set(section_refs) == {0, 1, 2}
