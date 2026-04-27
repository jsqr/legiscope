"""Tests for legiscope.convert module."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

from instructor.core.exceptions import FailedAttempt, InstructorRetryException
import pytest
import yaml

if TYPE_CHECKING:
    import polars as pl
from pydantic import BaseModel

from legiscope.parse.convert import text2md
from legiscope.parse.find_code_start import ScanResult
from legiscope.parse.headings import BooleanResult, HeadingLevel, HeadingStructure
from legiscope.parse.regions import REGIONS_SCHEMA
from legiscope.parse.scan import (
    DEFAULT_TEMPERATURE,
    SCAN_SYSTEM_PROMPT,
    _apply_example_based_pattern_refinement,
    _sample_diagnostics,
    _select_scan_sample,
    _format_exception_debug_summary,
    scan_legal_text,
    score_structure,
)
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

    def test_heading_structure_accepts_minimal_scan_payload(self):
        """Scan payloads may omit optional fields that the pipeline fills later."""
        structure = HeadingStructure.model_validate(
            {
                "heading_levels": [
                    {
                        "level": 1,
                        "regex_patterns": [r"^CHAPTER\s+\d+.*$"],
                        "example_heading": "CHAPTER 1 GENERAL PROVISIONS",
                        "type_label": "chapter",
                    }
                ]
            }
        )

        assert structure.total_levels == 0
        assert structure.file_sample_size == 0
        assert structure.levels[0].markdown_prefix == "#"
        assert not hasattr(structure.levels[0], "outline_line_numbers")


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

    def test_find_code_start_refines_late_section_candidate(self):
        """Late LLM candidates should backtrack to the nearby structure transition."""
        import polars as pl

        from legiscope.parse.find_code_start import find_code_start

        elements = pl.DataFrame(
            [
                {
                    "element_id": 0,
                    "start_line": 1,
                    "end_line": 1,
                    "n_lines": 1,
                    "text": "TABLE OF CONTENTS",
                },
                {
                    "element_id": 1,
                    "start_line": 2,
                    "end_line": 4,
                    "n_lines": 3,
                    "text": "APPENDIX\nCHAPTER A-1\nA-100 Certain Existing Departments",
                },
                {
                    "element_id": 2,
                    "start_line": 5,
                    "end_line": 9,
                    "n_lines": 5,
                    "text": (
                        "CHAPTER A-2\nA-200 Schedule\nPREAMBLE\n"
                        "Grateful to God for the freedoms we enjoy.\n"
                        "ARTICLE I\nPOWERS OF THE CITY"
                    ),
                },
                {
                    "element_id": 3,
                    "start_line": 10,
                    "end_line": 12,
                    "n_lines": 3,
                    "text": (
                        "§ 1-100. The City's Powers Defined.\n"
                        "This section grants the City broad powers and authority over municipal affairs.\n"
                        "Additional body text follows here."
                    ),
                },
                {
                    "element_id": 4,
                    "start_line": 13,
                    "end_line": 15,
                    "n_lines": 3,
                    "text": (
                        "§ 1-101. Legislative Power.\n"
                        "The legislative power of the City is vested in Council under this charter.\n"
                        "Additional body text follows here."
                    ),
                },
                {
                    "element_id": 5,
                    "start_line": 16,
                    "end_line": 17,
                    "n_lines": 2,
                    "text": "CHAPTER 1\nTHE COUNCIL",
                },
                {
                    "element_id": 6,
                    "start_line": 18,
                    "end_line": 20,
                    "n_lines": 3,
                    "text": (
                        "§ 2-100. Number, Terms and Salaries of Councilmembers.\n"
                        "The Council shall consist of the members prescribed by this charter.\n"
                        "Additional body text follows here."
                    ),
                },
            ],
            schema={
                "element_id": pl.Int64,
                "start_line": pl.Int64,
                "end_line": pl.Int64,
                "n_lines": pl.Int64,
                "text": pl.String,
            },
        )

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            ScanResult(found=True, element_id=6, reasoning="late section candidate"),
            Mock(
                correct=True, adjusted_element_id=None, reasoning="candidate accepted"
            ),
        ]

        result = find_code_start(mock_client, elements)

        assert result.element_id == 2
        assert result.start_line == 5
        assert "Refined backward" in result.reasoning

    def test_verify_code_start_highlights_earlier_boundary_candidates(self):
        """Verification should show broader prior context for late section candidates."""
        import polars as pl

        from legiscope.parse.find_code_start import _verify_code_start

        elements = pl.DataFrame(
            [
                {
                    "element_id": 0,
                    "start_line": 1,
                    "end_line": 1,
                    "n_lines": 1,
                    "text": "TABLE OF CONTENTS",
                },
                {
                    "element_id": 1,
                    "start_line": 2,
                    "end_line": 4,
                    "n_lines": 3,
                    "text": "APPENDIX\nCHAPTER A-1\nA-100 Certain Existing Departments",
                },
                {
                    "element_id": 2,
                    "start_line": 5,
                    "end_line": 9,
                    "n_lines": 5,
                    "text": (
                        "CHAPTER A-2\nA-200 Schedule\nPREAMBLE\n"
                        "Grateful to God for the freedoms we enjoy.\n"
                        "ARTICLE I\nPOWERS OF THE CITY"
                    ),
                },
                {
                    "element_id": 3,
                    "start_line": 10,
                    "end_line": 12,
                    "n_lines": 3,
                    "text": (
                        "§ 1-100. The City's Powers Defined.\n"
                        "This section grants the City broad powers and authority over municipal affairs.\n"
                        "Additional body text follows here."
                    ),
                },
                {
                    "element_id": 4,
                    "start_line": 13,
                    "end_line": 15,
                    "n_lines": 3,
                    "text": (
                        "§ 1-101. Legislative Power.\n"
                        "The legislative power of the City is vested in Council under this charter.\n"
                        "Additional body text follows here."
                    ),
                },
                {
                    "element_id": 5,
                    "start_line": 16,
                    "end_line": 17,
                    "n_lines": 2,
                    "text": "CHAPTER 1\nTHE COUNCIL",
                },
                {
                    "element_id": 6,
                    "start_line": 18,
                    "end_line": 20,
                    "n_lines": 3,
                    "text": (
                        "§ 2-100. Number, Terms and Salaries of Councilmembers.\n"
                        "The Council shall consist of the members prescribed by this charter.\n"
                        "Additional body text follows here."
                    ),
                },
            ],
            schema={
                "element_id": pl.Int64,
                "start_line": pl.Int64,
                "end_line": pl.Int64,
                "n_lines": pl.Int64,
                "text": pl.String,
            },
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            correct=True,
            adjusted_element_id=None,
            reasoning="candidate accepted",
        )

        _verify_code_start(mock_client, elements, candidate_id=6)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert "EARLIEST boundary" in messages[0]["content"]
        assert "LIKELY EARLIER BOUNDARY CANDIDATES" in messages[1]["content"]
        assert "[2]" in messages[1]["content"]
        assert "[3]" in messages[1]["content"]

    def test_find_code_start_advances_mid_toc_candidate_to_first_substantive_block(
        self,
    ):
        """TOC candidates should advance to the first substantive chapter/body block."""
        import polars as pl

        from legiscope.parse.find_code_start import find_code_start

        elements = pl.DataFrame(
            [
                {
                    "element_id": 0,
                    "start_line": 1,
                    "end_line": 1,
                    "n_lines": 1,
                    "text": "TABLE OF CONTENTS",
                },
                {
                    "element_id": 1,
                    "start_line": 2,
                    "end_line": 2,
                    "n_lines": 1,
                    "text": "ARTICLE I GENERAL PROVISIONS",
                },
                {
                    "element_id": 2,
                    "start_line": 3,
                    "end_line": 3,
                    "n_lines": 1,
                    "text": "CHAPTER 1 INTRODUCTION",
                },
                {
                    "element_id": 3,
                    "start_line": 4,
                    "end_line": 4,
                    "n_lines": 1,
                    "text": "§ 1-100. Purpose. 1",
                },
                {
                    "element_id": 4,
                    "start_line": 5,
                    "end_line": 6,
                    "n_lines": 2,
                    "text": "CHAPTER 1\nGENERAL PROVISIONS",
                },
                {
                    "element_id": 5,
                    "start_line": 7,
                    "end_line": 9,
                    "n_lines": 3,
                    "text": (
                        "§ 1-100. Purpose.\n"
                        "This chapter establishes the general provisions of the code.\n"
                        "Additional body text follows here."
                    ),
                },
            ],
            schema={
                "element_id": pl.Int64,
                "start_line": pl.Int64,
                "end_line": pl.Int64,
                "n_lines": pl.Int64,
                "text": pl.String,
            },
        )

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            ScanResult(found=True, element_id=2, reasoning="mid-TOC candidate"),
            Mock(
                correct=True, adjusted_element_id=None, reasoning="candidate accepted"
            ),
        ]

        result = find_code_start(mock_client, elements)

        assert result.element_id == 4
        assert result.start_line == 5

    def test_find_code_start_advances_past_long_toc_and_split_body_start(self):
        """Long TOCs should not trap the start boundary before a split section/body pair."""
        import polars as pl

        from legiscope.parse.find_code_start import find_code_start

        rows = [
            {
                "element_id": 0,
                "start_line": 1,
                "end_line": 1,
                "n_lines": 1,
                "text": "TABLE OF CONTENTS",
            },
            {
                "element_id": 1,
                "start_line": 2,
                "end_line": 2,
                "n_lines": 1,
                "text": "PREAMBLE",
            },
        ]

        line_number = 3
        for element_id in range(2, 610):
            rows.append(
                {
                    "element_id": element_id,
                    "start_line": line_number,
                    "end_line": line_number,
                    "n_lines": 1,
                    "text": f"§ 1-{element_id:03d} Purpose",
                }
            )
            line_number += 1

        rows.extend(
            [
                {
                    "element_id": 610,
                    "start_line": line_number,
                    "end_line": line_number,
                    "n_lines": 1,
                    "text": "ARTICLE II",
                },
                {
                    "element_id": 611,
                    "start_line": line_number + 1,
                    "end_line": line_number + 1,
                    "n_lines": 1,
                    "text": "CHAPTER 1",
                },
                {
                    "element_id": 612,
                    "start_line": line_number + 2,
                    "end_line": line_number + 2,
                    "n_lines": 1,
                    "text": "§ 2-100. Composition of Council.",
                },
                {
                    "element_id": 613,
                    "start_line": line_number + 3,
                    "end_line": line_number + 3,
                    "n_lines": 1,
                    "text": (
                        "The Council shall consist of district and at-large members and "
                        "exercise legislative authority under this charter."
                    ),
                },
                {
                    "element_id": 614,
                    "start_line": line_number + 4,
                    "end_line": line_number + 4,
                    "n_lines": 1,
                    "text": "§ 2-101. Terms of Office.",
                },
                {
                    "element_id": 615,
                    "start_line": line_number + 5,
                    "end_line": line_number + 5,
                    "n_lines": 1,
                    "text": (
                        "Councilmembers serve staggered four-year terms and remain subject "
                        "to the provisions of this charter."
                    ),
                },
            ]
        )

        elements = pl.DataFrame(
            rows,
            schema={
                "element_id": pl.Int64,
                "start_line": pl.Int64,
                "end_line": pl.Int64,
                "n_lines": pl.Int64,
                "text": pl.String,
            },
        )

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            ScanResult(found=True, element_id=1, reasoning="early preamble candidate"),
            Mock(
                correct=True,
                adjusted_element_id=None,
                reasoning="candidate accepted",
            ),
        ]

        result = find_code_start(mock_client, elements)

        assert result.element_id == 610
        assert result.start_line == line_number

    def test_verify_prompt_treats_toc_as_navigation_not_boundary(self):
        """Verification prompt should explicitly reject TOC/index boundaries."""
        import polars as pl

        from legiscope.parse.find_code_start import _verify_code_start

        elements = pl.DataFrame(
            [
                {
                    "element_id": 0,
                    "start_line": 1,
                    "end_line": 1,
                    "n_lines": 1,
                    "text": "TABLE OF CONTENTS",
                },
                {
                    "element_id": 1,
                    "start_line": 2,
                    "end_line": 2,
                    "n_lines": 1,
                    "text": "ARTICLE I GENERAL PROVISIONS",
                },
                {
                    "element_id": 2,
                    "start_line": 3,
                    "end_line": 5,
                    "n_lines": 3,
                    "text": (
                        "§ 1-100. Purpose.\n"
                        "This chapter establishes the general provisions of the code.\n"
                        "Additional body text follows here."
                    ),
                },
            ],
            schema={
                "element_id": pl.Int64,
                "start_line": pl.Int64,
                "end_line": pl.Int64,
                "n_lines": pl.Int64,
                "text": pl.String,
            },
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            correct=True,
            adjusted_element_id=None,
            reasoning="candidate accepted",
        )

        _verify_code_start(mock_client, elements, candidate_id=2)

        system_prompt = mock_client.chat.completions.create.call_args.kwargs[
            "messages"
        ][0]["content"]
        assert "navigation, not the boundary" in system_prompt

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

    def test_scan_legal_text_reduces_sample_after_timeout(self):
        """Timeout retries should shrink the representative sample before retrying."""
        sample_blocks = []
        for index in range(1, 261):
            sample_blocks.append(
                f"CHAPTER {index}\n"
                "This chapter contains enough substantive text to remain a distinct "
                "element during scan testing and retry backoff validation."
            )
        sample_text = "\n\n".join(sample_blocks)

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
                        regex_pattern=r"^CHAPTER\s+[A-Z0-9IVXLCDM.-]+(?:\s+.*)?$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1",
                        type_label="chapter",
                    )
                ],
                total_levels=1,
                file_sample_size=260,
            )

            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = [
                ScanResult(found=True, element_id=0, reasoning="Start of document"),
                TimeoutError("request timed out"),
                mock_response,
            ]

            with patch("legiscope.parse.scan.score_structure", return_value=(0.95, [])):
                scan_legal_text(mock_client, test_file)

            first_prompt = mock_client.chat.completions.create.call_args_list[1].kwargs[
                "messages"
            ][1]["content"]
            second_call = mock_client.chat.completions.create.call_args_list[2]
            second_prompt = second_call.kwargs["messages"][1]["content"]

            assert "These are 200 representative elements" in first_prompt
            assert "These are 150 representative elements" in second_prompt
            assert second_call.kwargs["messages"][0]["content"] == SCAN_SYSTEM_PROMPT
            assert second_call.kwargs["max_retries"] == 3
            assert "PREVIOUS ATTEMPT HAD THESE ISSUES" not in second_prompt
            assert "RETRY_FEEDBACK:" not in second_prompt

        finally:
            os.unlink(test_file)

    def test_scan_legal_text_reduces_sample_after_length_limit(self):
        """Length-limited retries should keep the same prompt shape and shrink sample size."""
        sample_blocks = []
        for index in range(1, 261):
            sample_blocks.append(
                f"CHAPTER {index}\n"
                "This chapter contains enough substantive text to remain a distinct "
                "element during scan testing and output-length retry validation."
            )
        sample_text = "\n\n".join(sample_blocks)

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
                        regex_pattern=r"^CHAPTER\s+[A-Z0-9IVXLCDM.-]+(?:\s+.*)?$",
                        example_heading="CHAPTER 1",
                        type_label="chapter",
                    )
                ],
            )

            completion = Mock()
            completion.choices = [Mock(finish_reason="length")]
            retry_error = InstructorRetryException(
                "The output is incomplete due to a max_tokens length limit.",
                n_attempts=1,
                total_usage=0,
                failed_attempts=[
                    FailedAttempt(
                        attempt_number=1,
                        exception=ValueError("structured output truncated"),
                        completion=completion,
                    )
                ],
            )

            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = [
                ScanResult(found=True, element_id=0, reasoning="Start of document"),
                retry_error,
                mock_response,
            ]

            with patch("legiscope.parse.scan.score_structure", return_value=(0.95, [])):
                scan_legal_text(mock_client, test_file)

            first_prompt = mock_client.chat.completions.create.call_args_list[1].kwargs[
                "messages"
            ][1]["content"]
            second_call = mock_client.chat.completions.create.call_args_list[2]
            second_prompt = second_call.kwargs["messages"][1]["content"]

            assert "These are 200 representative elements" in first_prompt
            assert "These are 140 representative elements" in second_prompt
            assert second_call.kwargs["messages"][0]["content"] == SCAN_SYSTEM_PROMPT
            assert second_call.kwargs["max_retries"] == 3
            assert "PREVIOUS ATTEMPT HAD THESE ISSUES" not in second_prompt
            assert "RETRY_FEEDBACK:" not in second_prompt

        finally:
            os.unlink(test_file)

    def test_scan_legal_text_adds_retry_feedback_only_after_scored_failure(self):
        """Scored retries should carry compact feedback; generation retries should not."""
        sample_text = """CHAPTER 1   TEST

1-100   Proper heading

Some body text here."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(sample_text)
            test_file = f.name

        try:
            first_response = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^\d+(?:-\d+)+\s+.*$",
                        markdown_prefix="#",
                        example_heading="1-100   Proper heading",
                        type_label="section",
                        number_regex=r"\d+(?:-\d+)+",
                    )
                ],
                total_levels=1,
                file_sample_size=3,
            )
            second_response = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+(?:\s+.*)?$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1   TEST",
                        type_label="chapter",
                        number_regex=r"\d+",
                    )
                ],
                total_levels=1,
                file_sample_size=3,
            )

            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = [
                ScanResult(found=True, element_id=0, reasoning="Start of document"),
                first_response,
                second_response,
            ]

            with patch(
                "legiscope.parse.scan.score_structure",
                side_effect=[
                    (
                        0.45,
                        [
                            "Low recall: patterns matched 1 of 3 heading-like elements (33%)",
                            "Low structural precision at level 1: 1 delimiter mismatches and 1 body-like matches across 2 matched elements (score 50%)",
                        ],
                    ),
                    (0.95, []),
                ],
            ):
                scan_legal_text(mock_client, test_file)

            first_prompt = mock_client.chat.completions.create.call_args_list[1].kwargs[
                "messages"
            ][1]["content"]
            second_prompt = mock_client.chat.completions.create.call_args_list[2].kwargs[
                "messages"
            ][1]["content"]

            assert "RETRY_FEEDBACK:" not in first_prompt
            assert "RETRY_FEEDBACK:" in second_prompt
            assert "- low_recall:" in second_prompt
            assert "- structural_precision:" in second_prompt

        finally:
            os.unlink(test_file)

    def test_scan_legal_text_writes_debug_artifact_on_total_generation_failure(self):
        """Scan debug artifact should persist timeout details even when all iterations fail."""
        sample_text = """CHAPTER 1 TEST

Some body text here."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(sample_text)
            test_file = f.name

        debug_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name

        try:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = [
                ScanResult(found=True, element_id=0, reasoning="Start of document"),
                TimeoutError("request timed out"),
                TimeoutError("request timed out"),
                TimeoutError("request timed out"),
                TimeoutError("request timed out"),
                TimeoutError("request timed out"),
            ]

            with pytest.raises(RuntimeError, match="Failed to generate heading structure"):
                scan_legal_text(
                    mock_client,
                    test_file,
                    debug_output_path=debug_path,
                )

            payload = json.loads(Path(debug_path).read_text())
            assert payload["best_iteration"] == 0
            assert len(payload["iterations"]) == 5
            first_failure = payload["iterations"][0]
            assert first_failure["status"] == "generation_error"
            assert first_failure["exception_debug"]["is_timeout"] is True
            assert first_failure["exception_debug"]["type"] == "TimeoutError"

        finally:
            os.unlink(test_file)
            if os.path.exists(debug_path):
                os.unlink(debug_path)

    def test_format_exception_debug_summary_includes_finish_reason(self):
        """Exception summaries should surface finish reasons inline for quick stderr review."""
        summary = _format_exception_debug_summary(
            {
                "type": "InstructorRetryException",
                "is_timeout": False,
                "is_context_length": False,
                "last_completion": {"finish_reasons": ["length"]},
                "message": "Structured output ended early.",
            }
        )

        assert "type=InstructorRetryException" in summary
        assert "timeout=False" in summary
        assert "finish_reason=length" in summary

    def test_scan_legal_text_logs_exception_debug_summary(self):
        """Generation failures should emit a one-line exception summary to stderr/logging."""
        sample_text = """CHAPTER 1 TEST

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
                        regex_pattern=r"^CHAPTER\s+\d+\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1 TEST",
                    )
                ],
                total_levels=1,
                file_sample_size=2,
            )
            completion = Mock()
            completion.choices = [Mock(finish_reason="length")]
            retry_error = InstructorRetryException(
                "retry failed",
                n_attempts=2,
                total_usage=0,
                failed_attempts=[
                    FailedAttempt(
                        attempt_number=1,
                        exception=ValueError("structured output truncated"),
                        completion=completion,
                    )
                ],
            )

            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = [
                ScanResult(found=True, element_id=0, reasoning="Start of document"),
                retry_error,
                mock_response,
            ]

            with (
                patch("loguru.logger.warning") as mock_warning,
                patch("legiscope.parse.scan.score_structure", return_value=(0.95, [])),
            ):
                scan_legal_text(mock_client, test_file)

            summary_calls = [
                call
                for call in mock_warning.call_args_list
                if call.args and call.args[0] == "Iteration {} exception_debug: {}"
            ]
            assert summary_calls
            assert "finish_reason=length" in summary_calls[0].args[2]

        finally:
            os.unlink(test_file)

    @patch(
        "legiscope.parse.scan.load_params",
        return_value={
            "convert": {"scan_max_lines": 200},
            "llm": {"temperature": 0.0, "max_retries": 3, "timeout": 300},
            "parse": {
                "scan": {
                    "initial_sample_count": 120,
                    "max_iterations": 2,
                    "score_threshold": 0.7,
                    "max_retries": 7,
                    "timeout": 480,
                    "max_tokens": 1400,
                }
            },
        },
    )
    def test_scan_legal_text_uses_configured_scan_params(self, _mock_load_params):
        """Scan stage should honor parse.scan overrides from params.yaml."""
        sample_blocks = []
        for index in range(1, 181):
            sample_blocks.append(
                f"CHAPTER {index}\n"
                "This chapter contains enough substantive text to remain a distinct "
                "element during scan param override testing."
            )
        sample_text = "\n\n".join(sample_blocks)

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
                        regex_pattern=r"^CHAPTER\s+[A-Z0-9IVXLCDM.-]+(?:\s+.*)?$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1",
                        type_label="chapter",
                    )
                ],
                total_levels=1,
                file_sample_size=180,
            )

            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = [
                ScanResult(found=True, element_id=0, reasoning="Start of document"),
                mock_response,
            ]

            with patch("legiscope.parse.scan.score_structure", return_value=(0.95, [])):
                scan_legal_text(mock_client, test_file)

            scan_call = mock_client.chat.completions.create.call_args_list[1]
            prompt = scan_call.kwargs["messages"][1]["content"]

            assert "These are 120 representative elements" in prompt
            assert scan_call.kwargs["max_retries"] == 7
            assert scan_call.kwargs["timeout"] == 480
            assert scan_call.kwargs["max_tokens"] == 1400

        finally:
            os.unlink(test_file)

    def test_scan_legal_text_prompt_calls_for_regex_variants(self):
        """Scan prompt should explicitly ask for multiple regexes per logical level."""
        sample_text = """ARTICLE III GENERAL POWERS

Article B-1.0 Adoption of the Building Code

CHAPTER 2 COUNCIL PROCEDURE

Chapter 2-100 City-County Consolidation

§ 1-100. The City's Powers Defined.

A-100 Certain Existing Departments

Body text here."""

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
                        regex_pattern=r"^ARTICLE\s+.+$",
                        markdown_prefix="#",
                        example_heading="ARTICLE III GENERAL POWERS",
                        type_label="article",
                    )
                ],
                total_levels=1,
                file_sample_size=7,
            )

            mock_client = _make_mock_client(mock_response)

            with patch("legiscope.parse.scan.score_structure", return_value=(0.95, [])):
                scan_legal_text(mock_client, test_file)

            prompt = mock_client.chat.completions.create.call_args_list[-1].kwargs[
                "messages"
            ][1]["content"]

            assert "multiple entries in `regex_patterns`" in prompt
            assert "FORMAT VARIANTS SEEN IN SAMPLE:" in prompt
            assert "- article:" in prompt
            assert "- chapter:" in prompt
            assert "- section:" in prompt

        finally:
            os.unlink(test_file)

    def test_scan_legal_text_writes_heading_scan_debug_artifact(self):
        """Scan debug output should preserve per-iteration generated structures and scores."""
        sample_text = """CHAPTER 1: GENERAL PROVISIONS

This chapter contains general provisions.
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(sample_text)
            test_file = f.name

        try:
            debug_path = f"{test_file}.heading_scan_debug.json"
            mock_response = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^CHAPTER\s+\d+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="CHAPTER 1: GENERAL PROVISIONS",
                    ),
                ],
                total_levels=1,
                file_sample_size=10,
            )

            mock_client = _make_mock_client(mock_response)
            with patch("legiscope.parse.scan.score_structure", return_value=(0.95, [])):
                result = scan_legal_text(
                    mock_client,
                    test_file,
                    debug_output_path=debug_path,
                )

            payload = json.loads(open(debug_path, encoding="utf-8").read())

            assert result.iterations == 1
            assert payload["best_iteration"] == 1
            assert payload["best_score"] == 0.95
            assert len(payload["iterations"]) == 1
            assert payload["iterations"][0]["status"] == "scored"
            assert payload["iterations"][0]["generated_structure"]["heading_levels"][0][
                "example_heading"
            ] == "CHAPTER 1: GENERAL PROVISIONS"
        finally:
            os.unlink(test_file)
            if os.path.exists(debug_path):
                os.unlink(debug_path)

    def test_sample_diagnostics_avoids_double_counting_heading_like_rows(self):
        """Heading-like diagnostics should count each sampled row only once."""
        import polars as pl

        sample = pl.DataFrame(
            [
                {"element_id": 0, "text": "TITLE I GENERAL PROVISIONS", "n_lines": 1},
                {"element_id": 1, "text": "CHAPTER 1", "n_lines": 1},
                {"element_id": 2, "text": "MISCELLANEOUS PROVISIONS", "n_lines": 1},
                {"element_id": 3, "text": "This is ordinary body text.", "n_lines": 1},
            ]
        )

        diagnostics = _sample_diagnostics(sample)

        assert diagnostics["title"] == 1
        assert diagnostics["chapter"] == 1
        assert diagnostics["heading_like"] == 3

    def test_example_refinement_supports_hyphenated_chapter_identifiers(self):
        """Chapter refinement should preserve hyphenated chapter numbers."""
        level = HeadingLevel(
            level=1,
            regex_pattern=r"^CHAPTER\s+.+$",
            markdown_prefix="#",
            example_heading="CHAPTER 9-600 DRUG PARAPHERNALIA",
            type_label="chapter",
        )

        _apply_example_based_pattern_refinement(level)

        assert re.match(level.regex_pattern, "CHAPTER 9-600 DRUG PARAPHERNALIA")
        assert level.number_regex == r"[A-Z0-9]+(?:[-.][A-Z0-9]+)+"

    def test_example_refinement_supports_letter_prefixed_sections(self):
        """Section refinement should support letter-prefixed identifiers generically."""
        level = HeadingLevel(
            level=1,
            regex_pattern=r"^§\s*.+$",
            markdown_prefix="#",
            example_heading="§ A-100. Certain Existing Departments.",
            type_label="section",
        )

        _apply_example_based_pattern_refinement(level)

        assert re.match(level.regex_pattern, "§ A-100. Certain Existing Departments.")
        assert re.match(level.regex_pattern, "A-100 Certain Existing Departments")
        assert level.number_regex == r"[A-Z0-9]+(?:[-.][A-Z0-9]+)+"

    def test_example_refinement_preserves_existing_regex_variants(self):
        """Example-based refinement should not discard other regex variants from the LLM."""
        variant_pattern = r"^ARTICLE\s+[A-Z0-9]+(?:[-.][A-Z0-9]+)+(?:\s+.*)?$"
        level = HeadingLevel(
            level=1,
            regex_patterns=[
                r"^ARTICLE\s+[IVXLCDM]+(?:\s+.*)?$",
                variant_pattern,
            ],
            markdown_prefix="#",
            example_heading="ARTICLE III GENERAL POWERS",
            type_label="article",
        )

        _apply_example_based_pattern_refinement(level)

        assert variant_pattern in level.regex_patterns
        assert re.match(level.regex_pattern, "ARTICLE III GENERAL POWERS")

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

    @staticmethod
    def _render_regions(input_text: str, structure: HeadingStructure) -> pl.DataFrame:
        """Convert synthetic input text and return the generated regions table."""
        import polars as pl

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(input_text)
            input_file = f.name

        try:
            with tempfile.TemporaryDirectory() as output_dir:
                output_file = os.path.join(output_dir, "code.md")
                text2md(structure, input_file, output_file, "PA", "Philadelphia")
                return pl.read_parquet(os.path.join(output_dir, "regions.parquet"))
        finally:
            os.unlink(input_file)

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

    def test_text2md_persists_code_start_metadata(self):
        """code_start metadata should be preserved in frontmatter when available."""
        input_text = """Published by Example Press

ARTICLE I: TEST

This charter is adopted pursuant to state law."""

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
                        regex_pattern=r"^ARTICLE\s+[IVXLCDM]+:\s+.+$",
                        markdown_prefix="#",
                        example_heading="ARTICLE I: TEST",
                        type_label="article",
                    )
                ],
                total_levels=1,
                file_sample_size=3,
                code_start_element_id=1,
                code_start_line=3,
            )

            text2md(structure, input_file, output_file, "PA", "Philadelphia")

            with open(output_file, "r", encoding="utf-8") as f:
                output_content = f.read()

            output_lines = output_content.splitlines()
            article_line = output_lines.index("# ARTICLE I: TEST") + 1

            frontmatter_start = output_content.find("---")
            frontmatter_end = output_content.find("---", frontmatter_start + 3)
            frontmatter_yaml = output_content[frontmatter_start + 3 : frontmatter_end]
            parsed_data = yaml.safe_load(frontmatter_yaml)

            assert parsed_data["code_start"]["element_id"] == 1
            assert parsed_data["code_start"]["source_line"] == 3
            assert parsed_data["code_start"]["output_line"] == article_line
        finally:
            os.unlink(input_file)
            os.unlink(output_file)

    def test_text2md_writes_regions_parquet_with_roles(self):
        """Converted output should include deterministic region roles."""
        import polars as pl

        input_text = """Published by: Example Press

TABLE OF CONTENTS

ARTICLE I GENERAL PROVISIONS

1-100 Purpose

PREAMBLE
This charter is adopted pursuant to state law and becomes effective January 1, 2025.

ARTICLE I GENERAL PROVISIONS

1-100 Purpose

This section establishes the purpose of the code and applies generally to the jurisdiction.

ANNOTATION
Notes about enactment history effective January 1, 2025."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(input_text)
            input_file = f.name

        with tempfile.TemporaryDirectory() as output_dir:
            output_file = os.path.join(output_dir, "code.md")

            structure = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^ARTICLE\s+[IVXLCDM]+(?:\s+.+)?$",
                        markdown_prefix="#",
                        example_heading="ARTICLE I GENERAL PROVISIONS",
                        type_label="article",
                        number_regex=r"[IVXLCDM]+",
                    ),
                    HeadingLevel(
                        level=2,
                        regex_pattern=r"^\d+(?:-\d+)+(?:\s+.+)?$",
                        markdown_prefix="##",
                        example_heading="1-100 Purpose",
                        type_label="section",
                        number_regex=r"\d+(?:-\d+)+",
                    ),
                ],
                total_levels=2,
                file_sample_size=10,
                code_start_element_id=2,
                code_start_line=5,
            )

            text2md(structure, input_file, output_file, "PA", "Philadelphia")

            regions_path = os.path.join(output_dir, "regions.parquet")
            assert os.path.exists(regions_path)

            regions_df = pl.read_parquet(regions_path)
            for col, dtype in REGIONS_SCHEMA.items():
                assert col in regions_df.columns
                assert regions_df.schema[col] == dtype

            roles = regions_df["region_role"].to_list()
            assert roles == [
                "publisher_boilerplate",
                "toc",
                "legal_intro",
                "main_body",
                "annotation",
            ]

            role_flags = {
                row["region_role"]: (
                    row["include_in_canonical_sections"],
                    row["include_in_default_chunks"],
                )
                for row in regions_df.to_dicts()
            }
            assert role_flags["toc"] == (False, False)
            assert role_flags["legal_intro"] == (False, True)
            assert role_flags["main_body"] == (True, True)
            assert role_flags["annotation"] == (False, True)

        os.unlink(input_file)

    def test_text2md_treats_toc_listings_as_toc_even_with_early_code_start(self):
        """TOC-style entries before substantive prose should stay out of main body."""
        import polars as pl

        input_text = """ARTICLE I GENERAL PROVISIONS

1-100 Purpose

1-200 Definitions

PREAMBLE
This charter is adopted pursuant to state law and becomes effective January 1, 2025.

1-100. Purpose

This section establishes the purpose of the code and applies generally to the jurisdiction."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(input_text)
            input_file = f.name

        with tempfile.TemporaryDirectory() as output_dir:
            output_file = os.path.join(output_dir, "code.md")

            structure = HeadingStructure(
                levels=[
                    HeadingLevel(
                        level=1,
                        regex_pattern=r"^ARTICLE\s+[IVXLCDM]+(?:\s+.+)?$",
                        markdown_prefix="#",
                        example_heading="ARTICLE I GENERAL PROVISIONS",
                        type_label="article",
                        number_regex=r"[IVXLCDM]+",
                    ),
                    HeadingLevel(
                        level=2,
                        regex_pattern=r"^\d+(?:-\d+)+\.\s+.+$",
                        markdown_prefix="##",
                        example_heading="1-100. Purpose",
                        type_label="section",
                        number_regex=r"\d+(?:-\d+)+",
                    ),
                ],
                total_levels=2,
                file_sample_size=7,
                code_start_element_id=0,
                code_start_line=1,
            )

            text2md(structure, input_file, output_file, "PA", "Philadelphia")

            regions_df = pl.read_parquet(os.path.join(output_dir, "regions.parquet"))
            roles = regions_df["region_role"].to_list()
            assert roles == ["toc", "legal_intro", "main_body"]

            toc_region = regions_df.to_dicts()[0]
            assert toc_region["include_in_canonical_sections"] is False
            assert toc_region["include_in_default_chunks"] is False
            assert (
                "toc-like" in toc_region["reason"]
                or "navigation" in toc_region["reason"]
            )

        os.unlink(input_file)

    def test_text2md_merges_annotation_runs_across_sources_purposes_and_notes(self):
        """Adjacent annotation-like elements should collapse into one annotation region."""
        structure = HeadingStructure(
            levels=[
                HeadingLevel(
                    level=1,
                    regex_pattern=r"^ARTICLE\s+[IVXLCDM]+(?:\s+.+)?$",
                    markdown_prefix="#",
                    example_heading="ARTICLE I MUNICIPAL AUTHORITY",
                    type_label="article",
                    number_regex=r"[IVXLCDM]+",
                ),
                HeadingLevel(
                    level=2,
                    regex_pattern=r"^CHAPTER\s+\d+(?:\s+.+)?$",
                    markdown_prefix="##",
                    example_heading="CHAPTER 1",
                    type_label="chapter",
                    number_regex=r"\d+",
                ),
                HeadingLevel(
                    level=3,
                    regex_pattern=r"^\d+(?:-\d+)+\.\s+.+$",
                    markdown_prefix="###",
                    example_heading="1-101. General Corporate Powers.",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                ),
            ],
            total_levels=3,
            file_sample_size=13,
            code_start_element_id=0,
            code_start_line=1,
        )

        input_text = """FOREWORD
This charter establishes a streamlined municipal framework and is adopted under state law.

ARTICLE I MUNICIPAL AUTHORITY

1-101. General Corporate Powers.

The city may adopt local measures and administer municipal affairs consistent with state law.

ANNOTATION

Sources: Charter enabling statute, Section 8.

Purposes: 1. This provision states the broad delegation of municipal authority.

Law Department Note (2024): Editorial references were standardized in this reprint.

Notes
Approved by the voters at the 2024 general election.

ARTICLE II COUNCIL

CHAPTER 1

2-100. Composition of the Council.

The council consists of district and at-large members serving staggered terms."""

        regions_df = self._render_regions(input_text, structure)
        roles = regions_df["region_role"].to_list()
        assert roles == ["legal_intro", "main_body", "annotation", "main_body"]

        annotation_region = regions_df.to_dicts()[2]
        assert annotation_region["element_count"] == 5
        assert annotation_region["include_in_canonical_sections"] is False
        assert annotation_region["include_in_default_chunks"] is True

    def test_text2md_detects_unlabeled_structural_toc_with_dotted_leaders(self):
        """Compact structural listings should be treated as TOC before body prose appears."""
        structure = HeadingStructure(
            levels=[
                HeadingLevel(
                    level=1,
                    regex_pattern=r"^TITLE\s+[A-Z0-9IVXLCDM]+(?:\s+.+)?$",
                    markdown_prefix="#",
                    example_heading="TITLE I ORGANIZATION",
                    type_label="title",
                    number_regex=r"[A-Z0-9IVXLCDM]+",
                ),
                HeadingLevel(
                    level=2,
                    regex_pattern=r"^CHAPTER\s+\d+(?:\s+.+)?$",
                    markdown_prefix="##",
                    example_heading="CHAPTER 1 COUNCIL",
                    type_label="chapter",
                    number_regex=r"\d+",
                ),
                HeadingLevel(
                    level=3,
                    regex_pattern=r"^\d+(?:-\d+)+\.\s+.+$",
                    markdown_prefix="###",
                    example_heading="1-100. Membership.",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                ),
            ],
            total_levels=3,
            file_sample_size=9,
            code_start_element_id=0,
            code_start_line=1,
        )

        input_text = """TITLE I ORGANIZATION

CHAPTER 1 Council ........ 1

1-100 Membership ........ 3

1-200 Procedure ........ 5

INTRODUCTION
This charter restates the organization of municipal government and clarifies local powers.

TITLE I

CHAPTER 1

1-100. Membership.

The council contains nine members chosen by district and citywide vote."""

        regions_df = self._render_regions(input_text, structure)
        roles = regions_df["region_role"].to_list()
        assert roles == ["toc", "legal_intro", "main_body"]

        toc_region = regions_df.to_dicts()[0]
        assert toc_region["element_count"] == 4
        assert toc_region["include_in_canonical_sections"] is False
        assert toc_region["include_in_default_chunks"] is False

    def test_text2md_keeps_structural_heading_chain_before_first_prose(self):
        """A genuine heading chain before the first prose block should stay main body."""
        structure = HeadingStructure(
            levels=[
                HeadingLevel(
                    level=1,
                    regex_pattern=r"^TITLE\s+[A-Z0-9IVXLCDM]+(?:\s+.+)?$",
                    markdown_prefix="#",
                    example_heading="TITLE I ORGANIZATION",
                    type_label="title",
                    number_regex=r"[A-Z0-9IVXLCDM]+",
                ),
                HeadingLevel(
                    level=2,
                    regex_pattern=r"^CHAPTER\s+\d+(?:\s+.+)?$",
                    markdown_prefix="##",
                    example_heading="CHAPTER 1 COUNCIL",
                    type_label="chapter",
                    number_regex=r"\d+",
                ),
                HeadingLevel(
                    level=3,
                    regex_pattern=r"^\d+(?:-\d+)+\.\s+.+$",
                    markdown_prefix="###",
                    example_heading="1-100. Membership.",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                ),
            ],
            total_levels=3,
            file_sample_size=6,
            code_start_element_id=0,
            code_start_line=1,
        )

        input_text = """TITLE I ORGANIZATION

CHAPTER 1 COUNCIL

1-100. Membership.

The council contains nine members chosen by district and citywide vote."""

        regions_df = self._render_regions(input_text, structure)
        roles = regions_df["region_role"].to_list()
        assert roles == ["main_body"]

        region = regions_df.to_dicts()[0]
        assert region["include_in_canonical_sections"] is True
        assert region["include_in_default_chunks"] is True

    def test_text2md_treats_heading_only_preamble_before_prose_as_legal_intro(self):
        """A heading-only PREAMBLE element before prose should not be forced into TOC."""
        structure = HeadingStructure(
            levels=[
                HeadingLevel(
                    level=1,
                    regex_pattern=r"^ARTICLE\s+[IVXLCDM]+(?:\s+.+)?$",
                    markdown_prefix="#",
                    example_heading="ARTICLE I GENERAL PROVISIONS",
                    type_label="article",
                    number_regex=r"[IVXLCDM]+",
                ),
                HeadingLevel(
                    level=2,
                    regex_pattern=r"^PREAMBLE$",
                    markdown_prefix="##",
                    example_heading="PREAMBLE",
                    type_label="preamble",
                    number_regex=None,
                ),
                HeadingLevel(
                    level=3,
                    regex_pattern=r"^\d+(?:-\d+)+\.\s+.+$",
                    markdown_prefix="###",
                    example_heading="1-100. Purpose.",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                ),
            ],
            total_levels=3,
            file_sample_size=6,
            code_start_element_id=0,
            code_start_line=1,
        )

        input_text = """ARTICLE I GENERAL PROVISIONS

PREAMBLE

This charter is adopted pursuant to state law and becomes effective January 1, 2025.

1-100. Purpose.

This section establishes the purpose of the code."""

        regions_df = self._render_regions(input_text, structure)
        roles = regions_df["region_role"].to_list()
        assert roles == ["main_body", "legal_intro", "main_body"]

        intro_region = regions_df.to_dicts()[1]
        assert intro_region["include_in_canonical_sections"] is False
        assert intro_region["include_in_default_chunks"] is True

    def test_text2md_handles_mixed_publisher_toc_intro_annotation_transitions(self):
        """Mixed pre-body and annotation transitions should produce stable region boundaries."""
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
            file_sample_size=13,
            code_start_element_id=0,
            code_start_line=1,
        )

        input_text = """Current through Ordinance 24-11.

Online edition maintained by Civic Publishing.

ARTICLE I ADMINISTRATION

1-100 Executive Branch ........ 2

PREFACE
This compilation reorganizes prior enactments and restates the charter in contemporary order.

ARTICLE I ADMINISTRATION

1-100. Executive Branch.

The executive authority is vested in a mayor and administrative departments.

Law Department Note (2025): Cross-reference numbering was adjusted for clarity.

Sources: Editorial compilation dated 2025.

ARTICLE II AUDITING

2-100. Fiscal Review.

An independent auditor reviews annual accounts."""

        regions_df = self._render_regions(input_text, structure)
        roles = regions_df["region_role"].to_list()
        assert roles == [
            "publisher_boilerplate",
            "toc",
            "legal_intro",
            "main_body",
            "annotation",
            "main_body",
        ]

        rows = regions_df.to_dicts()
        assert rows[0]["include_in_default_chunks"] is False
        assert rows[1]["include_in_canonical_sections"] is False
        assert rows[3]["include_in_canonical_sections"] is True
        assert rows[4]["include_in_default_chunks"] is True

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

    def test_mixed_identifier_families_have_comparable_sort_keys(self):
        """Sibling ordering checks should not crash on mixed numeric and alpha ids."""
        from legiscope.parse.scan import _identifier_sort_key

        alpha_key = _identifier_sort_key("A-10")
        numeric_key = _identifier_sort_key("1-10")

        assert alpha_key is not None
        assert numeric_key is not None
        assert (alpha_key < numeric_key) is False

    def test_outline_scope_argument_is_ignored_without_outline_contract(self):
        """Score calculation should not depend on scan-time outline ids or sample scope."""
        lines = [
            "1-100   Proper heading",
            "1-100 body text that should not be a heading",
            "2-100   Another heading",
            "2-100 more body text that should not be a heading",
        ]
        elements = self._make_elements(lines)
        sample = elements.head(2)

        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=1,
                    regex_patterns=[r"^\d+(?:-\d+)+\s+.*$"],
                    markdown_prefix="# ",
                    example_heading="1-100   Proper heading",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                ),
            ],
            total_levels=1,
            file_sample_size=len(lines),
        )

        full_score, full_errors = score_structure(elements, structure)
        scoped_score, scoped_errors = score_structure(
            elements,
            structure,
            outline_elements_df=sample,
        )

        assert full_score == scoped_score
        assert full_errors == scoped_errors

    def test_structural_precision_penalizes_body_like_overmatch(self):
        """Broad heading regexes should be penalized when they overmatch body-like lines."""
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
                ),
            ],
            total_levels=1,
            file_sample_size=len(lines),
        )

        score, errors = score_structure(elements, structure)

        assert score < 0.8
        assert any("low structural precision" in error.lower() for error in errors)

    def test_multiline_heading_elements_are_scored_as_matches(self):
        """Score evaluation should match multiline headings the same way conversion does."""
        import polars as pl

        elements = pl.DataFrame(
            [
                {
                    "element_id": 0,
                    "start_line": 1,
                    "end_line": 2,
                    "n_lines": 2,
                    "text": "CHAPTER 1\nTHE COUNCIL",
                },
                {
                    "element_id": 1,
                    "start_line": 3,
                    "end_line": 3,
                    "n_lines": 1,
                    "text": "§ 1-100. The City's Powers Defined.",
                },
                {
                    "element_id": 2,
                    "start_line": 4,
                    "end_line": 4,
                    "n_lines": 1,
                    "text": "Body text that is not a heading.",
                },
            ],
            schema={
                "element_id": pl.Int64,
                "start_line": pl.Int64,
                "end_line": pl.Int64,
                "n_lines": pl.Int64,
                "text": pl.String,
            },
        )

        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=1,
                    regex_patterns=[r"^CHAPTER\s+\d+\s+.*$"],
                    markdown_prefix="# ",
                    example_heading="CHAPTER 1 THE COUNCIL",
                    type_label="chapter",
                    number_regex=r"\d+",
                ),
                HeadingLevel(
                    level=2,
                    regex_patterns=[r"^§\s*\d+(?:-\d+)+\.\s*.*$"],
                    markdown_prefix="## ",
                    example_heading="§ 1-100. The City's Powers Defined.",
                    type_label="section",
                    number_regex=r"\d+(?:-\d+)+",
                ),
            ],
            total_levels=2,
            file_sample_size=elements.height,
        )

        score, errors = score_structure(elements, structure)

        assert score >= 0.8
        assert not any("low recall" in error.lower() for error in errors)

    def test_sibling_ordering_resets_after_higher_level_heading(self):
        """Chapter numbering should be allowed to restart under a new article."""
        lines = [
            "ARTICLE I   FIRST ARTICLE",
            "CHAPTER 10   LAST CHAPTER IN ARTICLE I",
            "ARTICLE II   SECOND ARTICLE",
            "CHAPTER 1   FIRST CHAPTER IN ARTICLE II",
        ]
        elements = self._make_elements(lines)

        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=1,
                    regex_patterns=[r"^ARTICLE\s+[IVXLCDM]+(?:\s+.*)?$"],
                    markdown_prefix="# ",
                    example_heading="ARTICLE I   FIRST ARTICLE",
                    type_label="article",
                    number_regex=r"[IVXLCDM]+",
                ),
                HeadingLevel(
                    level=2,
                    regex_patterns=[r"^CHAPTER\s+\d+(?:\s+.*)?$"],
                    markdown_prefix="## ",
                    example_heading="CHAPTER 10   LAST CHAPTER IN ARTICLE I",
                    type_label="chapter",
                    number_regex=r"\d+",
                ),
            ],
            total_levels=2,
            file_sample_size=len(lines),
        )

        score, errors = score_structure(elements, structure)

        assert score >= 0.8
        assert not any("out-of-order siblings" in error.lower() for error in errors)

    def test_recall_excludes_non_structural_enumerators_and_annotations(self):
        """Recall denominator should ignore common non-structural heading-like lines."""
        lines = [
            "CHAPTER 1 GENERAL PROVISIONS",
            "(1) This is a numbered clause.",
            "2. This is a numeric bullet.",
            "ANNOTATION",
            "NOTES",
            "---------------",
            "Ordinary body text.",
        ]
        elements = self._make_elements(lines)

        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=1,
                    regex_patterns=[r"^CHAPTER\s+\d+(?:\s+.*)?$"],
                    markdown_prefix="# ",
                    example_heading="CHAPTER 1 GENERAL PROVISIONS",
                    type_label="chapter",
                    number_regex=r"\d+",
                )
            ],
            total_levels=1,
            file_sample_size=len(lines),
        )

        score, errors = score_structure(elements, structure)

        assert score >= 0.9
        assert not any("low recall" in error.lower() for error in errors)


class TestScanSampling:
    """Tests for representative scan sampling."""

    @staticmethod
    def _make_elements(lines: list[str]) -> pl.DataFrame:
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

    def test_representative_sampling_includes_later_structural_headings(self):
        """Sampling should include later TITLE/CHAPTER exemplars, not just the front block."""
        lines = [f"§ 1-{i:03d}. Early section heading." for i in range(180)]
        lines.extend(
            [
                "TITLE 9 LATE TITLE HEADING",
                "CHAPTER 9 LATE CHAPTER HEADING",
                "ARTICLE IX LATE ARTICLE HEADING",
            ]
        )
        lines.extend(f"Body text {i}." for i in range(40))
        elements = self._make_elements(lines)

        sample = _select_scan_sample(elements, 120)
        sampled_lines = sample["text"].to_list()

        assert any(line.startswith("TITLE 9") for line in sampled_lines)
        assert any(line.startswith("CHAPTER 9") for line in sampled_lines)
        assert any(line.startswith("ARTICLE IX") for line in sampled_lines)

    def test_representative_sampling_includes_diverse_structural_families(self):
        """Sampling should include later appendix/preamble/compound-id exemplars."""
        lines = [f"§ 1-{i:03d}. Early section heading." for i in range(180)]
        lines.extend(
            [
                "APPENDIX A SUPPLEMENTAL RULES",
                "PREAMBLE",
                "A-100 Certain Existing Departments",
                "ARTICLE B-1.0 Adoption of the Building Code",
            ]
        )
        lines.extend(f"Body text {i}." for i in range(40))
        elements = self._make_elements(lines)

        sample = _select_scan_sample(elements, 120)
        sampled_lines = sample["text"].to_list()

        assert any(line.startswith("APPENDIX A") for line in sampled_lines)
        assert any(line.startswith("PREAMBLE") for line in sampled_lines)
        assert any(line.startswith("A-100") for line in sampled_lines)
        assert any(line.startswith("ARTICLE B-1.0") for line in sampled_lines)


class TestScanNormalization:
    """Tests for scan-time normalization of heading structures."""

    def test_normalization_sorts_by_declared_level_and_resets_markdown_prefixes(self):
        """Markdown prefixes should follow normalized declared level order, not stale LLM output."""
        from legiscope.parse.scan import _normalize_scanned_structure

        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=2,
                    regex_patterns=[r"^ARTICLE\s+[IVXLCDM]+\s+.*$"],
                    markdown_prefix="#",
                    example_heading="ARTICLE I   POWERS OF THE CITY",
                    type_label="article",
                ),
                HeadingLevel(
                    level=3,
                    regex_patterns=[r"^\d+(?:-\d+)\s+.*$"],
                    markdown_prefix="##",
                    example_heading="1-100   The City's Powers Defined",
                    type_label="section",
                ),
                HeadingLevel(
                    level=1,
                    regex_patterns=[r"^CHAPTER\s+\d+\s+.*$"],
                    markdown_prefix="###",
                    example_heading="CHAPTER 1   THE COUNCIL",
                    type_label="chapter",
                ),
            ],
            total_levels=3,
            file_sample_size=200,
        )

        normalized = _normalize_scanned_structure(structure)

        assert normalized.levels[0].type_label == "chapter"
        assert normalized.levels[0].markdown_prefix == "#"
        assert normalized.levels[1].type_label == "article"
        assert normalized.levels[1].markdown_prefix == "##"
        assert normalized.levels[2].type_label == "section"
        assert normalized.levels[2].markdown_prefix == "###"

    def test_section_refinement_matches_toc_and_body_variants(self):
        """Normalized section regexes should cover both TOC and body heading formats."""
        from legiscope.parse.scan import _normalize_scanned_structure

        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=1,
                    regex_patterns=[r"^\d+(?:-\d+)\s+.*$"],
                    markdown_prefix="#",
                    example_heading="1-100   The City's Powers Defined",
                    type_label="section",
                )
            ],
            total_levels=1,
            file_sample_size=10,
        )

        normalized = _normalize_scanned_structure(structure)
        section_pattern = re.compile(normalized.levels[0].regex_pattern, re.IGNORECASE)

        assert section_pattern.match("1-100   The City's Powers Defined")
        assert section_pattern.match("§ 1-100. The City's Powers Defined.")
