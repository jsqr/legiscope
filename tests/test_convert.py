"""Tests for legiscope.convert module."""

import os
import tempfile
from unittest.mock import Mock
from pydantic import BaseModel
import yaml

from legiscope.convert import (
    ask,
    BooleanResult,
    HeadingLevel,
    HeadingStructure,
    scan_legal_text,
    text2md,
)


class TestResponseModel(BaseModel):
    """Simple test model for testing purposes."""

    name: str
    value: int


class TestConvertModule:
    """Test cases for convert module functionality."""

    def test_ask_function_import(self):
        """Test that ask function is properly imported from utils."""
        # Test that we can import ask from convert module
        from legiscope.utils import ask as utils_ask
        from legiscope.convert import ask as convert_ask

        # Both should be same function
        assert utils_ask is convert_ask

    def test_ask_function_backward_compatibility(self):
        """Test that ask function works as expected when imported from convert."""
        # Setup mock client
        mock_client = Mock()
        mock_response = TestResponseModel(name="test", value=42)
        mock_client.chat.completions.create.return_value = mock_response

        # Call function imported from convert module
        result = ask(
            client=mock_client,
            prompt="Extract name and value from this text",
            response_model=TestResponseModel,
            model="gpt-4",
            temperature=0.5,
        )

        # Verify call was made correctly
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[
                {"role": "user", "content": "Extract name and value from this text"}
            ],
            response_model=TestResponseModel,
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


class TestScanLegalText:
    """Test cases for scan_legal_text function."""

    def test_scan_legal_text_success(self):
        """Test successful analysis of legal text with mock LLM response."""
        # Sample legal text
        sample_text = """CHAPTER 1: GENERAL PROVISIONS

This chapter contains general provisions.

SECTION 1.1: PURPOSE

The purpose of this chapter is to establish rules.

ARTICLE 2: ADMINISTRATION

Administrative procedures are outlined here."""

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(sample_text)
            test_file = f.name

        try:
            # Create mock client
            mock_client = Mock()

            # Create mock response
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
                        regex_pattern=r"^(SECTION|ARTICLE)\s+[\d.]+:\s+.+$",
                        markdown_prefix="##",
                        example_heading="SECTION 1.1: PURPOSE",
                    ),
                ],
                total_levels=2,
                file_sample_size=10,
            )

            mock_client.chat.completions.create.return_value = mock_response

            # Test function
            result = scan_legal_text(mock_client, test_file, max_lines=10)

            # Verify results
            assert result.total_levels == 2
            assert len(result.levels) == 2
            assert result.file_sample_size == 10

            # Check heading levels
            assert result.levels[0].level == 1
            assert result.levels[0].markdown_prefix == "#"
            assert "CHAPTER" in result.levels[0].regex_pattern

            assert result.levels[1].level == 2
            assert result.levels[1].markdown_prefix == "##"

        finally:
            # Clean up
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
        # Create empty temporary file
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

    def test_scan_legal_text_invalid_regex(self):
        """Test error handling when LLM returns invalid regex pattern."""
        sample_text = "CHAPTER 1: TEST"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(sample_text)
            test_file = f.name

        try:
            mock_client = Mock()

            # Create response with invalid regex
            mock_response = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"[invalid regex(",  # Invalid regex
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: TEST",
                    )
                ],
                total_levels=1,
                file_sample_size=1,
            )

            mock_client.chat.completions.create.return_value = mock_response

            scan_legal_text(mock_client, test_file)
            assert False, "Should have raised ValueError for invalid regex"
        except ValueError as e:
            assert "Invalid regex pattern" in str(e)
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
            assert "municipality: TestCity" in output_content

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
            assert "municipality: LosAngeles" in output_content
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
            assert parsed_data["jurisdiction"]["municipality"] == "LosAngeles"
            assert len(parsed_data["heading_patterns"]) == 1
            assert "created_at" in parsed_data

        finally:
            os.unlink(input_file)
            os.unlink(output_file)
