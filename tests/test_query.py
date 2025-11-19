"""
Tests for the query module.
"""

import pytest
from pydantic import ValidationError

from legiscope.query import (
    LegalQueryResponse,
    format_query_response,
)


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


