"""Tests for legiscope.convert module."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

from instructor.core.exceptions import FailedAttempt, InstructorRetryException
import yaml

if TYPE_CHECKING:
    import polars as pl
from pydantic import BaseModel

from legiscope.parse.convert import text2md
from legiscope.parse.find_code_start import ScanResult
from legiscope.parse.headings import BooleanResult, HeadingLevel, HeadingStructure
from legiscope.parse.scan import DEFAULT_TEMPERATURE, scan_legal_text
from legiscope.utils import ask


class MockResponseModel(BaseModel):
    """Simple test model for testing purposes."""

    name: str
    value: int


class TestConvertModule:
    """Test cases for convert module functionality."""

    def test_ask_function_import(self):
        """Test that ask function is properly imported from utils."""
        from legiscope.utils import ask as utils_ask

        # Module-level import should be the same function
        assert utils_ask is ask

    @patch("legiscope.llm_config.Config.get_llm_params")
    def test_ask_function_backward_compatibility(self, mock_get_llm_params):
        """Test that ask function works as expected when imported from convert."""
        mock_get_llm_params.return_value = {
            "temperature": 0.5,
            "max_retries": 3,
            "model": "gpt-4",
        }
        # Setup mock client
        mock_client = Mock()
        mock_response = MockResponseModel(name="test", value=42)
        mock_client.chat.completions.create.return_value = mock_response

        # Call function imported from convert module
        result = ask(
            client=mock_client,
            prompt="Extract name and value from this text",
            response_model=MockResponseModel,
            model="gpt-4",
            temperature=0.5,
        )

        # Verify call was made correctly
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[
                {"role": "user", "content": "Extract name and value from this text"}
            ],
            response_model=MockResponseModel,
            model="gpt-4",
            temperature=0.5,
            max_retries=3,  # Default parameter
        )

        # Verify result
        assert result == mock_response


class TestResponseModels:
    """Test cases for the predefined response models."""

    def test_boolean_result_model(self):
        """Test BooleanResult model validation."""
        result = BooleanResult(
            answer=True, explanation="This is clearly true based on the evidence."
        )
        assert result.answer is True
        assert result.explanation == "This is clearly true based on the evidence."

        # Test with None answer
        result_none = BooleanResult(
            answer=None,
            explanation="The evidence is insufficient to determine a clear answer.",
        )
        assert result_none.answer is None
        assert (
            result_none.explanation
            == "The evidence is insufficient to determine a clear answer."
        )

    def test_heading_structure_accepts_json_toc_ranges(self):
        """LLM JSON arrays for toc_line_ranges should normalize to tuple pairs."""
        structure = HeadingStructure.model_validate(
            {
                "heading_levels": [
                    {
                        "level": 1,
                        "regex_pattern": r"^CHAPTER\s+\d+.*$",
                        "regex_patterns": [r"^CHAPTER\s+\d+.*$"],
                        "markdown_prefix": "# ",
                        "example_heading": "CHAPTER 1 GENERAL PROVISIONS",
                        "type_label": "chapter",
                        "number_regex": r"\d+",
                        "multiline": False,
                        "inferred": False,
                        "outline_line_numbers": [114, 313],
                    }
                ],
                "total_levels": 1,
                "file_sample_size": 200,
                "toc_line_ranges": [[114, 313]],
                "outline_warnings": [],
                "quality_score": 0.0,
                "iterations": 0,
            }
        )

        assert structure.toc_line_ranges == [(114, 313)]


def _make_mock_client(heading_structure_response):
    """Create a mock client that returns element-based code start results
    for find_code_start (forward scan + verify) and then the given
    HeadingStructure for subsequent calls.

    When candidate element_id=0, _verify_code_start skips the LLM call
    (no preceding elements to check), so find_code_start uses only 1 call.
    """
    mock_client = Mock()
    # find_code_start: forward scan returns element_id=0 → verify skips LLM
    scan_result = ScanResult(found=True, element_id=0, reasoning="Start of document")
    # Provide heading_structure_responses for scan iterations
    mock_client.chat.completions.create.side_effect = [
        scan_result,
    ] + [heading_structure_response] * 10
    return mock_client


class TestScanLegalText:
    """Test cases for scan_legal_text function."""

    def test_scan_legal_text_success(self):
        """Test successful analysis of legal text with mock LLM response."""
        sample_text = """CHAPTER 1: GENERAL PROVISIONS

This chapter contains general provisions.

SECTION 1.1: PURPOSE

The purpose of this chapter is to establish rules.

ARTICLE 2: ADMINISTRATION

Administrative procedures are outlined here."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(sample_text)
            test_file = f.name

        try:
            mock_response = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: GENERAL PROVISIONS",
                    ),
                    HeadingLevel(
                        level=2,
                        regex_pattern=r"^(?:SECTION|ARTICLE)\s+[\d.]+:\s+.+$",
                        markdown_prefix="##",
                        example_heading="SECTION 1.1: PURPOSE",
                    ),
                ],
                total_levels=2,
                file_sample_size=10,
            )

            mock_client = _make_mock_client(mock_response)
            result = scan_legal_text(mock_client, test_file, max_lines=10)

            assert result.total_levels == 2
            assert len(result.levels) == 2
            assert result.levels[0].level == 1
            assert result.levels[0].markdown_prefix == "#"
            assert "CHAPTER" in result.levels[0].regex_pattern
            assert result.levels[1].level == 2
            assert result.levels[1].markdown_prefix == "##"

        finally:
            os.unlink(test_file)

    def test_scan_legal_text_file_not_found(self):
        """Test error handling when file doesn't exist."""
        mock_client = Mock()

        try:
            scan_legal_text(mock_client, "nonexistent_file.txt")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected

    def test_scan_legal_text_empty_file(self):
        """Test error handling when file is empty."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            test_file = f.name

        try:
            mock_client = Mock()
            scan_legal_text(mock_client, test_file)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        finally:
            os.unlink(test_file)

    def test_scan_legal_text_delegates_to_scan_headings(self):
        """Test that scan_legal_text delegates to scan_headings."""
        sample_text = """CHAPTER 1: TEST

Some body text here."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(sample_text)
            test_file = f.name

        try:
            mock_response = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: TEST",
                    ),
                ],
                total_levels=1,
                file_sample_size=3,
            )

            mock_client = _make_mock_client(mock_response)
            result = scan_legal_text(mock_client, test_file)

            # Should have called create at least 2 times
            # (find_code_start scan + heading structure LLM call)
            assert mock_client.chat.completions.create.call_count >= 2
            assert (
                mock_client.chat.completions.create.call_args_list[-1].kwargs[
                    "temperature"
                ]
                == DEFAULT_TEMPERATURE
            )
            assert result.total_levels == 1

        finally:
            os.unlink(test_file)

    def test_scan_legal_text_retries_after_generation_failure(self):
        """Test that malformed structured output is handed off to outer retries."""
        sample_text = """CHAPTER 1: TEST

Some body text here."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(sample_text)
            test_file = f.name

        try:
            mock_response = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: TEST",
                    )
                ],
                total_levels=1,
                file_sample_size=2,
            )

            retry_error = InstructorRetryException(
                "retry failed",
                n_attempts=2,
                total_usage=0,
                failed_attempts=[
                    FailedAttempt(
                        attempt_number=1,
                        exception=ValueError(
                            "3 validation errors for HeadingStructure: heading_levels missing; "
                            "total_levels missing; file_sample_size missing; input contained $defs and properties"
                        ),
                    )
                ],
            )

            mock_client = Mock()
            scan_result = ScanResult(
                found=True, element_id=0, reasoning="Start of document"
            )
            mock_client.chat.completions.create.side_effect = [
                scan_result,
                retry_error,
                mock_response,
            ]

            result = scan_legal_text(mock_client, test_file)

            assert result.total_levels == 1
            assert len(result.levels) == 1
            assert mock_client.chat.completions.create.call_count == 3

        finally:
            os.unlink(test_file)

    def test_scan_legal_text_invalid_regex_produces_warnings(self):
        """Test that invalid regex patterns produce warnings and low quality score."""
        sample_text = "CHAPTER 1: TEST"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(sample_text)
            test_file = f.name

        try:
            mock_client = Mock()
            mock_response = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"[invalid regex(",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: TEST",
                    )
                ],
                total_levels=1,
                file_sample_size=1,
            )
            # find_code_start: scan returns element_id=0 → verify skips LLM
            scan_result = ScanResult(found=True, element_id=0, reasoning="start")
            mock_client.chat.completions.create.side_effect = [
                scan_result,
            ] + [mock_response] * 10

            result = scan_legal_text(mock_client, test_file)
            # Invalid regex should produce warnings and low quality score
            assert result.quality_score < 0.7
            assert any("invalid regex" in w.lower() for w in result.outline_warnings)
        finally:
            os.unlink(test_file)


class TestText2Md:
    """Test cases for text2md function."""

    def test_text2md_basic_conversion(self):
        """Test basic heading conversion to Markdown."""
        # Create test input text
        input_text = """CHAPTER 1: GENERAL PROVISIONS

This chapter contains general provisions.

SECTION 1.1: PURPOSE

The purpose of this chapter is to establish rules.

SECTION 1.2: SCOPE

This chapter applies to all residents.

CHAPTER 2: ADMINISTRATION

Administrative procedures are outlined here."""

        # Create temporary input file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(input_text)
            input_file = f.name

        # Create temporary output file path
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_file = f.name

        try:
            # Create HeadingStructure
            structure = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: GENERAL PROVISIONS",
                    ),
                    HeadingLevel(
                        level=2,
                        regex_pattern=r"^SECTION\s+[\d.]+:\s+.+$",
                        markdown_prefix="##",
                        example_heading="SECTION 1.1: PURPOSE",
                    ),
                ],
                total_levels=2,
                file_sample_size=10,
            )

            # Convert text
            text2md(structure, input_file, output_file, "IL", "TestCity")

            # Read and verify output
            with open(output_file, "r", encoding="utf-8") as f:
                output_content = f.read()

            # Check that headings were converted
            assert "# CHAPTER 1: GENERAL PROVISIONS" in output_content
            assert "## SECTION 1.1: PURPOSE" in output_content
            assert "## SECTION 1.2: SCOPE" in output_content
            assert "# CHAPTER 2: ADMINISTRATION" in output_content

            # Check that non-heading content is preserved
            assert "This chapter contains general provisions." in output_content
            assert (
                "The purpose of this chapter is to establish rules." in output_content
            )

        finally:
            # Clean up
            os.unlink(input_file)
            os.unlink(output_file)

    def test_text2md_three_level_hierarchy(self):
        """Test conversion with three-level heading hierarchy."""
        input_text = """CHAPTER 1: RULES

ARTICLE 1: GENERAL

SECTION 1.1: BASIC RULES

These are the basic rules.

1.1.1: SPECIFIC RULE

This is a specific rule."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(input_text)
            input_file = f.name

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_file = f.name

        try:
            structure = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: RULES",
                    ),
                    HeadingLevel(
                        level=2,
                        regex_pattern=r"^ARTICLE\s+\d+:\s+.+$",
                        markdown_prefix="##",
                        example_heading="ARTICLE 1: GENERAL",
                    ),
                    HeadingLevel(
                        level=3,
                        regex_pattern=r"^SECTION\s+[\d.]+:\s+.+$|^\d+\.\d+\.\d+:\s+.+$",
                        markdown_prefix="###",
                        example_heading="SECTION 1.1: BASIC RULES",
                    ),
                ],
                total_levels=3,
                file_sample_size=8,
            )

            text2md(structure, input_file, output_file, "IL", "TestCity")

            with open(output_file, "r", encoding="utf-8") as f:
                output_content = f.read()

            # Verify all levels were converted
            assert "# CHAPTER 1: RULES" in output_content
            assert "## ARTICLE 1: GENERAL" in output_content
            assert "### SECTION 1.1: BASIC RULES" in output_content
            assert "### 1.1.1: SPECIFIC RULE" in output_content

        finally:
            os.unlink(input_file)
            os.unlink(output_file)

    def test_text2md_no_headings(self):
        """Test conversion when text contains no matching headings."""
        input_text = """This is just regular text.

It has no headings at all.

Just plain paragraphs."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(input_text)
            input_file = f.name

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_file = f.name

        try:
            structure = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: TEST",
                    )
                ],
                total_levels=1,
                file_sample_size=3,
            )

            text2md(structure, input_file, output_file, "IL", "TestCity")

            with open(output_file, "r", encoding="utf-8") as f:
                output_content = f.read()

            # Content should be unchanged (plus frontmatter)
            assert "This is just regular text." in output_content
            assert "It has no headings at all." in output_content
            assert "Just plain paragraphs." in output_content
            # Check that frontmatter is present
            assert "---" in output_content
            assert "jurisdiction:" in output_content
            assert "state: IL" in output_content
            assert "locality: TestCity" in output_content

        finally:
            os.unlink(input_file)
            os.unlink(output_file)

    def test_text2md_invalid_structure(self):
        """Test error handling with invalid HeadingStructure."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("test content")
            input_file = f.name

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_file = f.name

        try:
            # Test with empty structure
            try:
                text2md(None, input_file, output_file, "IL", "TestCity")  # type: ignore
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Invalid HeadingStructure" in str(e)

            # Test with structure with no levels
            empty_structure = HeadingStructure(
                levels=[], total_levels=0, file_sample_size=0
            )
            try:
                text2md(empty_structure, input_file, output_file, "IL", "TestCity")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "contains no levels" in str(e)

        finally:
            os.unlink(input_file)
            os.unlink(output_file)

    def test_text2md_file_errors(self):
        """Test error handling for file access issues."""
        structure = HeadingStructure(
            levels=[
                HeadingLevel(
                    level=1,
                    regex_pattern=r"^TEST:\s*.+$",
                    markdown_prefix="#",
                    example_heading="TEST: Example",
                )
            ],
            total_levels=1,
            file_sample_size=1,
        )

        # Test with non-existent input file
        try:
            text2md(structure, "nonexistent.txt", "output.md", "IL", "TestCity")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected

        # Test with invalid input path (directory)
        try:
            text2md(
                structure, "/tmp", "output.md", "IL", "TestCity"
            )  # /tmp exists but is directory
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not a file" in str(e)

    def test_text2md_invalid_regex(self):
        """Test error handling with invalid regex pattern."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("TEST: Example")
            input_file = f.name

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_file = f.name

        try:
            structure = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"[invalid regex(",  # Invalid regex
                        markdown_prefix="#",
                        example_heading="TEST: Example",
                    )
                ],
                total_levels=1,
                file_sample_size=1,
            )

            text2md(structure, input_file, output_file, "IL", "TestCity")
            assert False, "Should have raised ValueError for invalid regex"
        except ValueError as e:
            assert "Invalid regex pattern" in str(e)
        finally:
            os.unlink(input_file)

    def test_text2md_frontmatter_generation(self):
        """Test YAML frontmatter generation in output."""
        input_text = """CHAPTER 1: TEST

This is a test chapter."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(input_text)
            input_file = f.name

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_file = f.name

        try:
            structure = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: TEST",
                    )
                ],
                total_levels=1,
                file_sample_size=3,
            )

            text2md(structure, input_file, output_file, "CA", "LosAngeles")

            with open(output_file, "r", encoding="utf-8") as f:
                output_content = f.read()

            # Verify frontmatter structure
            assert "---" in output_content
            assert "jurisdiction:" in output_content
            assert "state: CA" in output_content
            assert "locality: LosAngeles" in output_content
            assert "full_name: CA - LosAngeles" in output_content
            assert "heading_patterns:" in output_content
            assert "level: 1" in output_content
            assert "regex_pattern:" in output_content
            assert "markdown_prefix: '#'" in output_content
            assert "example_heading:" in output_content
            assert "created_at:" in output_content

            # Verify YAML structure is valid
            # Extract frontmatter and parse as YAML
            frontmatter_start = output_content.find("---")
            frontmatter_end = output_content.find("---", frontmatter_start + 3)
            frontmatter_yaml = output_content[frontmatter_start + 3 : frontmatter_end]

            parsed_data = yaml.safe_load(frontmatter_yaml)
            assert parsed_data["jurisdiction"]["state"] == "CA"
            assert parsed_data["jurisdiction"]["locality"] == "LosAngeles"
            assert len(parsed_data["heading_patterns"]) == 1
            assert "created_at" in parsed_data

        finally:
            os.unlink(input_file)
            os.unlink(output_file)

    def test_text2md_paragraph_handling(self):
        """Test proper paragraph handling with single and double newlines."""
        input_text = """CHAPTER 1: GENERAL PROVISIONS

This is the first paragraph.
It has multiple lines but should be one paragraph.

This is the second paragraph.
It also has multiple lines
that should be joined together.

SECTION 1.1: PURPOSE

The purpose of this chapter is to establish rules.
These rules apply to all residents
and visitors in the municipality.

This is another paragraph in the section.
It demonstrates proper paragraph separation."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(input_text)
            input_file = f.name

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_file = f.name

        try:
            structure = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: GENERAL PROVISIONS",
                    ),
                    HeadingLevel(
                        level=2,
                        regex_pattern=r"^SECTION\s+[\d.]+:\s+.+$",
                        markdown_prefix="##",
                        example_heading="SECTION 1.1: PURPOSE",
                    ),
                ],
                total_levels=2,
                file_sample_size=15,
            )

            text2md(structure, input_file, output_file, "IL", "TestCity")

            with open(output_file, "r", encoding="utf-8") as f:
                output_content = f.read()

            # Check headings were converted
            assert "# CHAPTER 1: GENERAL PROVISIONS" in output_content
            assert "## SECTION 1.1: PURPOSE" in output_content

            # Check that lines within paragraphs are properly joined
            assert (
                "This is the first paragraph. It has multiple lines but should be one paragraph."
                in output_content
            )
            assert (
                "This is the second paragraph. It also has multiple lines that should be joined together."
                in output_content
            )
            assert (
                "The purpose of this chapter is to establish rules. These rules apply to all residents and visitors in the municipality."
                in output_content
            )
            assert (
                "This is another paragraph in the section. It demonstrates proper paragraph separation."
                in output_content
            )

        finally:
            os.unlink(input_file)
            os.unlink(output_file)

    def test_text2md_paragraph_edge_cases(self):
        """Test paragraph handling with edge cases like multiple consecutive empty lines."""
        input_text = """CHAPTER 1: TEST

Single line paragraph.

Line 1 of multi-line paragraph.
Line 2 of multi-line paragraph.
Line 3 of multi-line paragraph.


Multiple empty lines above this paragraph.

Last paragraph without trailing empty line."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(input_text)
            input_file = f.name

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_file = f.name

        try:
            structure = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: TEST",
                    ),
                ],
                total_levels=1,
                file_sample_size=10,
            )

            text2md(structure, input_file, output_file, "IL", "TestCity")

            with open(output_file, "r", encoding="utf-8") as f:
                output_content = f.read()

            # Check heading conversion
            assert "# CHAPTER 1: TEST" in output_content

            # Check single line paragraph
            assert "Single line paragraph." in output_content

            # Check multi-line paragraph is properly joined
            assert (
                "Line 1 of multi-line paragraph. Line 2 of multi-line paragraph. Line 3 of multi-line paragraph."
                in output_content
            )

            # Check paragraph after multiple empty lines
            assert "Multiple empty lines above this paragraph." in output_content

            # Check last paragraph
            assert "Last paragraph without trailing empty line." in output_content

        finally:
            os.unlink(input_file)
            os.unlink(output_file)

    def test_text2md_mixed_content_with_paragraphs(self):
        """Test conversion with headings, paragraphs, and mixed content."""
        input_text = """CHAPTER 1: INTRODUCTION

This chapter introduces the municipal code.

It provides important context
for understanding the regulations
that follow.

ARTICLE 1: DEFINITIONS

Term: Definition
Term 2: Definition 2

SECTION 1.1: GENERAL TERMS

These terms are used throughout
the entire municipal code.

Additional definitions may be found
in specific sections as needed."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(input_text)
            input_file = f.name

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_file = f.name

        try:
            structure = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: INTRODUCTION",
                    ),
                    HeadingLevel(
                        level=2,
                        regex_pattern=r"^ARTICLE\s+\d+:\s+.+$",
                        markdown_prefix="##",
                        example_heading="ARTICLE 1: DEFINITIONS",
                    ),
                    HeadingLevel(
                        level=3,
                        regex_pattern=r"^SECTION\s+[\d.]+:\s+.+$",
                        markdown_prefix="###",
                        example_heading="SECTION 1.1: GENERAL TERMS",
                    ),
                ],
                total_levels=3,
                file_sample_size=12,
            )

            text2md(structure, input_file, output_file, "IL", "TestCity")

            with open(output_file, "r", encoding="utf-8") as f:
                output_content = f.read()

            # Check all heading levels
            assert "# CHAPTER 1: INTRODUCTION" in output_content
            assert "## ARTICLE 1: DEFINITIONS" in output_content
            assert "### SECTION 1.1: GENERAL TERMS" in output_content

            # Check paragraph joining
            assert "This chapter introduces the municipal code." in output_content
            assert (
                "It provides important context for understanding the regulations that follow."
                in output_content
            )
            assert (
                "These terms are used throughout the entire municipal code."
                in output_content
            )
            assert (
                "Additional definitions may be found in specific sections as needed."
                in output_content
            )

            # Check that definition lines (non-paragraph content) are preserved
            assert "Term: Definition" in output_content
            assert "Term 2: Definition 2" in output_content

        finally:
            os.unlink(input_file)
            os.unlink(output_file)

    def test_text2md_headings_parquet_has_element_id(self):
        """Test that headings.parquet includes element_id column."""
        import polars as pl

        input_text = """CHAPTER 1: GENERAL PROVISIONS

This chapter contains general provisions.

SECTION 1.1: PURPOSE

The purpose of this chapter is to establish rules."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(input_text)
            input_file = f.name

        import tempfile as tmp_mod

        output_dir = tmp_mod.mkdtemp()
        output_file = os.path.join(output_dir, "code.md")

        try:
            structure = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: GENERAL PROVISIONS",
                    ),
                    HeadingLevel(
                        level=2,
                        regex_pattern=r"^SECTION\s+[\d.]+:\s+.+$",
                        markdown_prefix="##",
                        example_heading="SECTION 1.1: PURPOSE",
                    ),
                ],
                total_levels=2,
                file_sample_size=10,
            )

            text2md(structure, input_file, output_file, "IL", "TestCity")

            headings_path = os.path.join(output_dir, "headings.parquet")
            assert os.path.exists(headings_path)
            headings_df = pl.read_parquet(headings_path)

            # Verify element_id column exists
            assert "element_id" in headings_df.columns
            assert headings_df.schema["element_id"] == pl.Int64

            # Verify line_number still exists for backward compat
            assert "line_number" in headings_df.columns

            # Verify we got heading records
            assert len(headings_df) >= 2

        finally:
            os.unlink(input_file)
            import shutil

            shutil.rmtree(output_dir)


class TestScoreStructure:
    """Tests for score_structure quality scoring."""

    @staticmethod
    def _make_elements(lines: list[str]) -> pl.DataFrame:
        """Build a minimal elements DataFrame from a list of first-lines."""
        import polars as pl

        return pl.DataFrame(
            [
                {
                    "element_id": i,
                    "start_line": i + 1,
                    "end_line": i + 1,
                    "n_lines": 1,
                    "text": line,
                }
                for i, line in enumerate(lines)
            ],
            schema={
                "element_id": pl.Int64,
                "start_line": pl.Int64,
                "end_line": pl.Int64,
                "n_lines": pl.Int64,
                "text": pl.String,
            },
        )

    def test_low_recall_penalised(self):
        """Patterns matching few heading-like elements should get low recall."""
        from legiscope.parse.scan import score_structure

        # Simulate Philadelphia-like case: patterns only capture CHAPTER,
        # but document also has ARTICLE and § section headings.
        lines = [
            "ARTICLE I   POWERS OF THE CITY",
            "CHAPTER 1   THE COUNCIL",
            "§ 1-100. The City's Powers Defined.",
            "§ 1-101. Legislative Power.",
            "CHAPTER 2   COUNCIL PROCEDURE",
            "§ 2-100. Regular Meetings.",
            "§ 2-101. Special Meetings.",
            "ARTICLE II   LEGISLATIVE BRANCH",
            "CHAPTER 3   LEGISLATION",
            "§ 3-100. Ordinances and Resolutions.",
            "Body text that is not a heading at all.",
            "More body text describing regulations.",
        ]
        elements = self._make_elements(lines)

        # Structure that only captures CHAPTER — misses ARTICLE and §
        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=1,
                    regex_patterns=[r"^CHAPTER\s+\d+"],
                    markdown_prefix="# ",
                    example_heading="CHAPTER 1   THE COUNCIL",
                    type_label="chapter",
                ),
            ],
            total_levels=1,
            file_sample_size=len(lines),
        )

        score, errors = score_structure(elements, structure)

        # With only ~3 of ~10 heading-like elements matched, recall is ~0.3
        assert score < 0.7, f"Score {score:.3f} should be below 0.7 (low recall)"
        recall_errors = [e for e in errors if "recall" in e.lower()]
        assert len(recall_errors) > 0, "Should report low recall in errors"

    def test_high_recall_rewarded(self):
        """Patterns capturing most heading-like elements should score well."""
        from legiscope.parse.scan import score_structure

        lines = [
            "ARTICLE I   POWERS OF THE CITY",
            "CHAPTER 1   THE COUNCIL",
            "§ 1-100. The City's Powers Defined.",
            "§ 1-101. Legislative Power.",
            "CHAPTER 2   COUNCIL PROCEDURE",
            "§ 2-100. Regular Meetings.",
            "Body text that is not a heading.",
            "More body text.",
        ]
        elements = self._make_elements(lines)

        # Structure that captures all three heading types
        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=1,
                    regex_patterns=[r"^ARTICLE\s+[IVXLCDM]+"],
                    markdown_prefix="# ",
                    example_heading="ARTICLE I   POWERS OF THE CITY",
                    type_label="article",
                ),
                HeadingLevel(
                    level=2,
                    regex_patterns=[r"^CHAPTER\s+\d+"],
                    markdown_prefix="## ",
                    example_heading="CHAPTER 1   THE COUNCIL",
                    type_label="chapter",
                    number_regex=r"\d+",
                ),
                HeadingLevel(
                    level=3,
                    regex_patterns=[r"^§\s*\d+-\d+"],
                    markdown_prefix="### ",
                    example_heading="§ 1-100. The City's Powers Defined.",
                    type_label="section",
                    number_regex=r"\d+-\d+",
                ),
            ],
            total_levels=3,
            file_sample_size=len(lines),
        )

        score, errors = score_structure(elements, structure)
        assert score >= 0.7, f"Score {score:.3f} should be >= 0.7 (high recall)"

    def test_zero_matches_returns_zero(self):
        """Patterns matching nothing should score 0.0."""
        from legiscope.parse.scan import score_structure

        lines = [
            "§ 1-100. The City's Powers Defined.",
            "§ 1-101. Legislative Power.",
            "Body text.",
        ]
        elements = self._make_elements(lines)

        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=1,
                    regex_patterns=[r"^NONEXISTENT\s+\d+"],
                    markdown_prefix="# ",
                    example_heading="NONEXISTENT 1",
                    type_label="fake",
                ),
            ],
            total_levels=1,
            file_sample_size=len(lines),
        )

        score, errors = score_structure(elements, structure)
        assert score == 0.0

    def test_compound_identifiers_do_not_trigger_false_ordering_penalty(self):
        """Compound numeric ids should be compared naturally, not lexicographically."""
        from legiscope.parse.scan import score_structure

        lines = [
            "§ 1-100. First section.",
            "§ 2-100. Second section.",
            "§ 10-100. Tenth section.",
        ]
        elements = self._make_elements(lines)

        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=1,
                    regex_patterns=[r"^§\s*\d+(?:-\d+)+\.\s*.*$"],
                    markdown_prefix="# ",
                    example_heading="§ 1-100. First section.",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                ),
            ],
            total_levels=1,
            file_sample_size=len(lines),
        )

        score, errors = score_structure(elements, structure)

        assert score >= 0.8
        assert not any("out-of-order siblings" in error.lower() for error in errors)

    def test_outline_mismatch_penalises_broad_regex(self):
        """Broad regexes should be penalized when they disagree with outline ids."""
        from legiscope.parse.scan import score_structure

        lines = [
            "1-100   Proper heading",
            "1-100 body text that should not be a heading",
            "2-100   Another heading",
            "2-100 more body text that should not be a heading",
        ]
        elements = self._make_elements(lines)

        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=1,
                    regex_patterns=[r"^\d+(?:-\d+)+\s+.*$"],
                    markdown_prefix="# ",
                    example_heading="1-100   Proper heading",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                    outline_line_numbers=[0, 2],
                ),
            ],
            total_levels=1,
            file_sample_size=len(lines),
        )

        score, errors = score_structure(elements, structure)

        assert score < 0.8
        assert any("outline mismatch" in error.lower() for error in errors)


class TestScanNormalization:
    """Tests for scan-time normalization of heading structures."""

    def test_reordered_levels_also_reset_markdown_prefixes(self):
        """Markdown prefixes should follow normalized level order, not stale LLM output."""
        from legiscope.parse.scan import _normalize_scanned_structure

        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=1,
                    regex_patterns=[r"^ARTICLE\s+[IVXLCDM]+\s+.*$"],
                    markdown_prefix="#",
                    example_heading="ARTICLE I   POWERS OF THE CITY",
                    type_label="article",
                    outline_line_numbers=[116],
                ),
                HeadingLevel(
                    level=2,
                    regex_patterns=[r"^\d+(?:-\d+)\s+.*$"],
                    markdown_prefix="##",
                    example_heading="1-100   The City's Powers Defined",
                    type_label="section",
                    outline_line_numbers=list(range(117, 180)),
                ),
                HeadingLevel(
                    level=3,
                    regex_patterns=[r"^CHAPTER\s+\d+\s+.*$"],
                    markdown_prefix="###",
                    example_heading="CHAPTER 1   THE COUNCIL",
                    type_label="chapter",
                    outline_line_numbers=[120, 121, 129, 135],
                ),
            ],
            total_levels=3,
            file_sample_size=200,
        )

        normalized = _normalize_scanned_structure(structure)

        assert normalized.levels[0].type_label == "article"
        assert normalized.levels[0].markdown_prefix == "#"
        assert normalized.levels[1].type_label == "chapter"
        assert normalized.levels[1].markdown_prefix == "##"
        assert normalized.levels[2].type_label == "section"
        assert normalized.levels[2].markdown_prefix == "###"
