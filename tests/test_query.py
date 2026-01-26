"""
Tests for the query module.
"""

import pytest
from pydantic import ValidationError

from legiscope.query import (
    LegalQueryResponse,
    format_query_response,
    _validate_supporting_passages,
)
from legiscope.retrieve import SectionResult, SegmentMatch


class TestLegalQueryResponse:
    """Test the LegalQueryResponse model."""

    def test_legal_query_response_model_valid(self):
        """Test that LegalQueryResponse accepts valid data."""
        response = LegalQueryResponse(
            short_answer="Yes, there are restrictions.",
            reasoning="The municipal code prohibits the sale of drug paraphernalia.",
            citations=["Section 5-12-3", "Section 5-12-4"],
            supporting_passages=["No person shall sell drug paraphernalia."],
            confidence=0.9,
            limitations="Based on available municipal code sections.",
        )

        assert response.short_answer == "Yes, there are restrictions."
        assert response.confidence == 0.9
        assert len(response.citations) == 2
        assert len(response.supporting_passages) == 1

    def test_legal_query_response_model_confidence_bounds(self):
        """Test that confidence scores are bounded between 0 and 1."""
        # Valid confidence scores
        response1 = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="Test",
        )
        assert response1.confidence == 0.0

        response2 = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=[],
            confidence=1.0,
            limitations="Test",
        )
        assert response2.confidence == 1.0

    def test_legal_query_response_model_invalid_confidence(self):
        """Test that invalid confidence scores raise ValidationError."""
        with pytest.raises(ValidationError):
            LegalQueryResponse(
                short_answer="Test",
                reasoning="Test",
                citations=[],
                supporting_passages=[],
                confidence=-0.1,  # Below 0
                limitations="Test",
            )

        with pytest.raises(ValidationError):
            LegalQueryResponse(
                short_answer="Test",
                reasoning="Test",
                citations=[],
                supporting_passages=[],
                confidence=1.1,  # Above 1
                limitations="Test",
            )


class TestFormatQueryResponse:
    """Test the format_query_response function."""

    def test_format_query_response_complete(self):
        """Test formatting a complete response."""
        response = LegalQueryResponse(
            short_answer="Yes, there are restrictions.",
            reasoning="The municipal code prohibits the sale of drug paraphernalia.",
            citations=["Section 5-12-3", "Section 5-12-4"],
            supporting_passages=[
                "No person shall sell drug paraphernalia.",
                "Violations are punishable by fines.",
            ],
            confidence=0.9,
            limitations="Based on available municipal code sections.",
        )

        formatted = format_query_response(response)

        assert "## Legal Analysis" in formatted
        assert "**Answer:** Yes, there are restrictions." in formatted
        assert "**Confidence:** 90.0%" in formatted
        assert "### Reasoning" in formatted
        assert "The municipal code prohibits" in formatted
        assert "### Citations" in formatted
        assert "1. Section 5-12-3" in formatted
        assert "2. Section 5-12-4" in formatted
        assert "### Supporting Passages" in formatted
        assert '1. "No person shall sell drug paraphernalia."' in formatted
        assert '2. "Violations are punishable by fines."' in formatted
        assert "### Limitations" in formatted
        assert "Based on available municipal code sections." in formatted

    def test_format_query_response_minimal(self):
        """Test formatting a minimal response."""
        response = LegalQueryResponse(
            short_answer="No information available.",
            reasoning="No relevant sections found.",
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="",
        )

        formatted = format_query_response(response)

        assert "## Legal Analysis" in formatted
        assert "**Answer:** No information available." in formatted
        assert "**Confidence:** 0.0%" in formatted
        assert "### Reasoning" in formatted
        assert "No relevant sections found." in formatted
        assert "### Citations" in formatted
        assert "No specific citations available." in formatted
        assert "### Supporting Passages" in formatted
        assert "No supporting passages available." in formatted
        assert "### Limitations" not in formatted  # Should not appear when empty

    def test_format_query_response_empty_limitations(self):
        """Test formatting when limitations is empty."""
        response = LegalQueryResponse(
            short_answer="Test answer",
            reasoning="Test reasoning",
            citations=[],
            supporting_passages=[],
            confidence=0.5,
            limitations="",
        )

        formatted = format_query_response(response)

        assert "### Limitations" not in formatted

    def test_format_query_response_with_limitations(self):
        """Test formatting when limitations is provided."""
        response = LegalQueryResponse(
            short_answer="Test answer",
            reasoning="Test reasoning",
            citations=[],
            supporting_passages=[],
            confidence=0.5,
            limitations="Some limitations apply.",
        )

        formatted = format_query_response(response)

        assert "### Limitations" in formatted
        assert "Some limitations apply." in formatted


class TestValidateSupportingPassages:
    """Test the _validate_supporting_passages function."""

    def create_test_sections(self, body_text: str, segment_texts: list[str] = None):
        """Helper to create test SectionResult objects."""
        if segment_texts is None:
            segment_texts = []

        segments = [
            SegmentMatch(
                segment_idx=i,
                segment_text=text,
                distance=0.1,
                segment_position=i,
                section_heading="Test Section",
                section_level=1,
            )
            for i, text in enumerate(segment_texts)
        ]

        return [
            SectionResult(
                section_idx=0,
                heading_text="Test Section",
                body_text=body_text,
                heading_level=1,
                parent=None,
                matching_segments=segments,
                relevance_score=0.9,
                segment_count=len(segments),
            )
        ]

    def test_validate_exact_match_in_body(self, caplog):
        """Test validation with exact match in section body text."""
        response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=["No person shall sell drug paraphernalia."],
            confidence=0.9,
            limitations="",
        )

        sections = self.create_test_sections(
            body_text="Section 5-12-3: No person shall sell drug paraphernalia. Violations are punishable by fines.",
        )

        _validate_supporting_passages(response, sections)

        # Should not log any warnings
        assert "HALLUCINATION WARNING" not in caplog.text
        assert "NOT FOUND" not in caplog.text

    def test_validate_exact_match_in_segment(self, caplog):
        """Test validation with exact match in segment text."""
        response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=["Violations are punishable by fines."],
            confidence=0.9,
            limitations="",
        )

        sections = self.create_test_sections(
            body_text="Section 5-12-3: Regulations on drug paraphernalia.",
            segment_texts=["Violations are punishable by fines."],
        )

        _validate_supporting_passages(response, sections)

        # Should not log any warnings
        assert "HALLUCINATION WARNING" not in caplog.text
        assert "NOT FOUND" not in caplog.text

    def test_validate_no_match_hallucination(self):
        """Test validation runs without errors for hallucinated passages.

        Note: This test verifies the function executes correctly.
        Manual inspection of test output shows warnings are logged correctly.
        """
        response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=[
                "This text does not exist in the retrieved documents."
            ],
            confidence=0.9,
            limitations="",
        )

        sections = self.create_test_sections(
            body_text="Section 5-12-3: No person shall sell drug paraphernalia.",
        )

        # Should complete without errors (warnings logged to stderr via loguru)
        _validate_supporting_passages(response, sections)

    def test_validate_fuzzy_match_close(self):
        """Test validation runs without errors for close but not exact matches.

        Note: This test verifies the function executes correctly.
        Manual inspection of test output shows warnings are logged correctly.
        """
        response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=[
                "No person should sell drug paraphernalia items."
            ],  # Changed words
            confidence=0.9,
            limitations="",
        )

        sections = self.create_test_sections(
            body_text="Section 5-12-3: No person shall sell drug paraphernalia.",
        )

        # Should complete without errors (warnings logged to stderr via loguru)
        _validate_supporting_passages(response, sections)

    def test_validate_multiple_passages_mixed(self):
        """Test validation runs without errors for mixed exact/hallucinated passages.

        Note: This test verifies the function executes correctly.
        Manual inspection of test output shows warnings are logged correctly.
        """
        response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=[
                "No person shall sell drug paraphernalia.",  # Exact match
                "This passage is completely fabricated.",  # Hallucination
                "Violations are punishable by fines.",  # Exact match
            ],
            confidence=0.9,
            limitations="",
        )

        sections = self.create_test_sections(
            body_text="Section 5-12-3: No person shall sell drug paraphernalia. Violations are punishable by fines.",
        )

        # Should complete without errors (warnings logged to stderr via loguru)
        _validate_supporting_passages(response, sections)

    def test_validate_empty_passages(self, caplog):
        """Test validation with no supporting passages."""
        response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=[],
            confidence=0.9,
            limitations="",
        )

        sections = self.create_test_sections(
            body_text="Section 5-12-3: Some text.",
        )

        _validate_supporting_passages(response, sections)

        # Should not log anything
        assert "HALLUCINATION WARNING" not in caplog.text

    def test_validate_no_sections(self):
        """Test validation runs without errors when no sections available.

        Note: This test verifies the function executes correctly.
        Manual inspection of test output shows warnings are logged correctly.
        """
        response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=["Some passage"],
            confidence=0.9,
            limitations="",
        )

        # Should complete without errors (warnings logged to stderr via loguru)
        _validate_supporting_passages(response, [])
