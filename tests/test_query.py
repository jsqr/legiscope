"""
Tests for the query module.
"""

import pytest
from unittest.mock import Mock, patch
from pydantic import ValidationError
import polars as pl

from legiscope.query import (
    LegalQueryResponse,
    query_legal_documents,
    format_query_response,
    run_queries,
)
from legiscope.model_config import Config


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


class TestQueryLegalDocuments:
    """Test the query_legal_documents function."""

    def test_query_legal_documents_no_client(self):
        """Test that missing client raises ValueError."""
        with pytest.raises(ValueError, match="Client is required"):
            query_legal_documents(
                client=None, query="Test query", retrieval_results={"sections": []}
            )

    def test_query_legal_documents_empty_query(self):
        """Test that empty query raises ValueError."""
        client = Mock()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            query_legal_documents(
                client=client, query="", retrieval_results={"sections": []}
            )

    def test_query_legal_documents_no_results(self):
        """Test that missing retrieval results raises ValueError."""
        client = Mock()

        with pytest.raises(ValueError, match="Retrieval results are required"):
            query_legal_documents(
                client=client, query="Test query", retrieval_results=None
            )

    def test_query_legal_documents_no_sections(self):
        """Test handling of empty sections in results."""
        client = Mock()

        response = query_legal_documents(
            client=client, query="Test query", retrieval_results={"sections": []}
        )

        assert (
            response.short_answer
            == "I cannot answer your question as no relevant legal provisions were found."
        )
        assert response.confidence == 0.0
        assert len(response.citations) == 0
        assert len(response.supporting_passages) == 0

    @patch("legiscope.query.ask")
    def test_query_legal_documents_success(self, mock_ask):
        """Test successful query processing."""
        # Setup mock response
        mock_response = LegalQueryResponse(
            short_answer="Yes, there are restrictions.",
            reasoning="The municipal code prohibits the sale of drug paraphernalia.",
            citations=["Section 5-12-3"],
            supporting_passages=["No person shall sell drug paraphernalia."],
            confidence=0.9,
            limitations="Based on available sections.",
        )
        mock_ask.return_value = mock_response

        # Setup test data
        client = Mock()
        retrieval_results = {
            "sections": [
                {
                    "section_idx": 1,
                    "heading_text": "Drug Paraphernalia",
                    "body_text": "No person shall sell drug paraphernalia.",
                    "relevance_score": 0.1,
                    "matching_segments": [
                        {
                            "segment_text": "No person shall sell drug paraphernalia.",
                            "distance": 0.1,
                        }
                    ],
                }
            ]
        }

        # Call function
        response = query_legal_documents(
            client=client,
            query="Are there restrictions on drug paraphernalia sales?",
            retrieval_results=retrieval_results,
            model=Config.get_powerful_model(),
            temperature=0.2,
            max_retries=5,
        )

        # Verify response
        assert response.short_answer == "Yes, there are restrictions."
        assert response.confidence == 0.9
        assert len(response.citations) == 1

        # Verify mock was called correctly
        mock_ask.assert_called_once()
        call_args = mock_ask.call_args
        assert call_args[1]["model"] == Config.get_powerful_model()
        assert call_args[1]["temperature"] == 0.2
        assert call_args[1]["max_retries"] == 5

    @patch("legiscope.query.ask")
    def test_query_legal_documents_llm_failure(self, mock_ask):
        """Test handling of LLM call failure."""
        mock_ask.side_effect = Exception("LLM API error")

        client = Mock()
        retrieval_results = {
            "sections": [
                {
                    "section_idx": 1,
                    "heading_text": "Test Section",
                    "body_text": "Test content",
                    "relevance_score": 0.1,
                    "matching_segments": [],
                }
            ]
        }

        with pytest.raises(Exception, match="LLM API error"):
            query_legal_documents(
                client=client, query="Test query", retrieval_results=retrieval_results
            )

    @patch("legiscope.query.ask")
    def test_query_legal_documents_context_preparation(self, mock_ask):
        """Test that context is prepared correctly for the LLM."""
        # Setup mock response
        mock_response = LegalQueryResponse(
            short_answer="Test answer",
            reasoning="Test reasoning",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="Test limitations",
        )
        mock_ask.return_value = mock_response

        # Setup test data
        client = Mock()
        retrieval_results = {
            "sections": [
                {
                    "section_idx": 1,
                    "heading_text": "Test Heading",
                    "body_text": "Test content",
                    "relevance_score": 0.1,
                    "matching_segments": [
                        {"segment_text": "Test segment", "distance": 0.1}
                    ],
                }
            ]
        }

        # Call function
        query_legal_documents(
            client=client, query="Test Query", retrieval_results=retrieval_results
        )

        # Verify that ask was called with correct prompt content
        mock_ask.assert_called_once()
        call_args = mock_ask.call_args
        prompt = call_args[1]["prompt"]  # prompt is passed as keyword argument

        # Verify that prompt contains expected elements
        assert "Test Query" in prompt
        assert "Section 1: Test Heading" in prompt
        assert "Test content" in prompt
        assert "Matching Segments:" in prompt


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


class TestRunQueries:
    """Test the run_queries function."""

    def test_run_queries_no_client(self):
        """Test that missing client raises ValueError."""
        with pytest.raises(ValueError, match="Client is required"):
            run_queries(
                client=None,
                queries=["Test query"],
                jurisdiction_id="IL-WindyCity",
                sections_parquet_path="test.parquet",
                collection=Mock(),
            )

    def test_run_queries_no_queries(self):
        """Test that missing queries raises ValueError."""
        client = Mock()

        with pytest.raises(ValueError, match="Queries must be a non-empty list"):
            run_queries(
                client=client,
                queries=[],
                jurisdiction_id="IL-WindyCity",
                sections_parquet_path="test.parquet",
                collection=Mock(),
            )

    def test_run_queries_no_jurisdiction(self):
        """Test that missing jurisdiction raises ValueError."""
        client = Mock()

        with pytest.raises(ValueError, match="Jurisdiction ID is required"):
            run_queries(
                client=client,
                queries=["Test query"],
                jurisdiction_id="",
                sections_parquet_path="test.parquet",
                collection=Mock(),
            )

    def test_run_queries_no_collection(self):
        """Test that missing collection raises ValueError."""
        client = Mock()

        with pytest.raises(ValueError, match="ChromaDB collection is required"):
            run_queries(
                client=client,
                queries=["Test query"],
                jurisdiction_id="IL-WindyCity",
                sections_parquet_path="test.parquet",
                collection=None,
            )

    @patch("legiscope.query.retrieve_sections")
    @patch("legiscope.query.query_legal_documents")
    def test_run_queries_success(self, mock_query_legal, mock_retrieve):
        """Test successful processing of multiple queries."""
        # Setup mocks
        mock_retrieval_results = {
            "sections": [
                {
                    "section_idx": 1,
                    "heading_text": "Test Section",
                    "body_text": "Test content",
                    "relevance_score": 0.1,
                    "matching_segments": [],
                }
            ],
            "query_info": {"total_segments_found": 2, "unique_sections": 1},
        }
        mock_retrieve.return_value = mock_retrieval_results

        mock_response = LegalQueryResponse(
            short_answer="Test answer",
            reasoning="Test reasoning",
            citations=["Test citation"],
            supporting_passages=["Test passage"],
            confidence=0.8,
            limitations="Test limitations",
        )
        mock_query_legal.return_value = mock_response

        # Setup test data
        client = Mock()
        collection = Mock()
        queries = ["Query 1", "Query 2"]

        # Call function
        result_df = run_queries(
            client=client,
            queries=queries,
            jurisdiction_id="IL-WindyCity",
            sections_parquet_path="test.parquet",
            collection=collection,
            model=Config.get_powerful_model(),
            n_results=5,
        )

        # Verify DataFrame structure
        assert isinstance(result_df, pl.DataFrame)
        assert len(result_df) == 2
        assert "query" in result_df.columns
        assert "short_answer" in result_df.columns
        assert "confidence" in result_df.columns
        assert "processing_time" in result_df.columns

        # Verify content
        assert result_df[0, "query"] == "Query 1"
        assert result_df[0, "short_answer"] == "Test answer"
        assert result_df[0, "confidence"] == 0.8
        assert result_df[1, "query"] == "Query 2"

        # Verify function calls
        assert mock_retrieve.call_count == 2
        assert mock_query_legal.call_count == 2

    @patch("legiscope.query.retrieve_sections")
    @patch("legiscope.query.query_legal_documents")
    def test_run_queries_with_hyde(self, mock_query_legal, mock_retrieve):
        """Test run_queries with HYDE rewriting enabled."""
        # Setup mocks
        mock_retrieval_results = {
            "sections": [],
            "query_info": {"total_segments_found": 0, "unique_sections": 0},
        }
        mock_retrieve.return_value = mock_retrieval_results

        mock_response = LegalQueryResponse(
            short_answer="No information",
            reasoning="No sections found",
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="No relevant information",
        )
        mock_query_legal.return_value = mock_response

        # Setup test data
        client = Mock()
        collection = Mock()
        queries = ["Test query"]

        # Call function with HYDE enabled
        run_queries(
            client=client,
            queries=queries,
            jurisdiction_id="IL-WindyCity",
            sections_parquet_path="test.parquet",
            collection=collection,
            use_hyde=True,
            model=Config.get_powerful_model(),
        )

        # Verify retrieve_sections was called with HYDE parameters
        mock_retrieve.assert_called_once()
        call_args = mock_retrieve.call_args
        assert call_args[1]["rewrite"] is True
        assert call_args[1]["client"] == client
        assert call_args[1]["model"] == Config.get_powerful_model()

    @patch("legiscope.query.retrieve_sections")
    @patch("legiscope.query.query_legal_documents")
    def test_run_queries_retrieval_failure(self, mock_query_legal, mock_retrieve):
        """Test handling of retrieval failures."""
        # Setup mock to raise exception
        mock_retrieve.side_effect = Exception("Retrieval failed")

        # Setup test data
        client = Mock()
        collection = Mock()
        queries = ["Test query"]

        # Call function
        result_df = run_queries(
            client=client,
            queries=queries,
            jurisdiction_id="IL-WindyCity",
            sections_parquet_path="test.parquet",
            collection=collection,
        )

        # Verify error handling
        assert len(result_df) == 1
        assert result_df[0, "query"] == "Test query"
        assert "Error:" in result_df[0, "short_answer"]
        assert result_df[0, "confidence"] == 0.0
        assert "Retrieval failed" in result_df[0, "reasoning"]

    @patch("legiscope.query.retrieve_sections")
    @patch("legiscope.query.query_legal_documents")
    def test_run_queries_query_failure(self, mock_query_legal, mock_retrieve):
        """Test handling of query processing failures."""
        # Setup mocks
        mock_retrieval_results = {
            "sections": [{"section_idx": 1, "heading_text": "Test"}],
            "query_info": {"total_segments_found": 1, "unique_sections": 1},
        }
        mock_retrieve.return_value = mock_retrieval_results

        # Setup mock to raise exception during query processing
        mock_query_legal.side_effect = Exception("Query processing failed")

        # Setup test data
        client = Mock()
        collection = Mock()
        queries = ["Test query"]

        # Call function
        result_df = run_queries(
            client=client,
            queries=queries,
            jurisdiction_id="IL-WindyCity",
            sections_parquet_path="test.parquet",
            collection=collection,
        )

        # Verify error handling
        assert len(result_df) == 1
        assert result_df[0, "query"] == "Test query"
        assert "Error:" in result_df[0, "short_answer"]
        assert result_df[0, "confidence"] == 0.0
        assert "Query processing failed" in result_df[0, "reasoning"]

    @patch("legiscope.query.retrieve_sections")
    @patch("legiscope.query.query_legal_documents")
    def test_run_queries_empty_query_handling(self, mock_query_legal, mock_retrieve):
        """Test handling of empty queries in the list."""
        # Setup mocks
        mock_retrieval_results = {
            "sections": [],
            "query_info": {"total_segments_found": 0, "unique_sections": 0},
        }
        mock_retrieve.return_value = mock_retrieval_results

        mock_response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=[],
            confidence=0.5,
            limitations="Test",
        )
        mock_query_legal.return_value = mock_response

        # Setup test data
        client = Mock()
        collection = Mock()
        queries = ["Valid query", "", "Another valid query", None]

        # Call function
        result_df = run_queries(
            client=client,
            queries=queries,
            jurisdiction_id="IL-WindyCity",
            sections_parquet_path="test.parquet",
            collection=collection,
        )

        # Verify only valid queries were processed
        assert len(result_df) == 2  # Only 2 valid queries
        assert result_df[0, "query"] == "Valid query"
        assert result_df[1, "query"] == "Another valid query"

        # Verify retrieve_sections was called only for valid queries
        assert mock_retrieve.call_count == 2

    @patch("legiscope.query.retrieve_sections")
    @patch("legiscope.query.query_legal_documents")
    def test_run_queries_dataframe_schema(self, mock_query_legal, mock_retrieve):
        """Test that returned DataFrame has correct schema."""
        # Setup mocks
        mock_retrieval_results = {
            "sections": [],
            "query_info": {"total_segments_found": 0, "unique_sections": 0},
        }
        mock_retrieve.return_value = mock_retrieval_results

        mock_response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=[],
            confidence=0.5,
            limitations="Test",
        )
        mock_query_legal.return_value = mock_response

        # Setup test data
        client = Mock()
        collection = Mock()
        queries = ["Test query"]

        # Call function
        result_df = run_queries(
            client=client,
            queries=queries,
            jurisdiction_id="IL-WindyCity",
            sections_parquet_path="test.parquet",
            collection=collection,
        )

        # Verify DataFrame schema
        expected_columns = {
            "query",
            "short_answer",
            "reasoning",
            "citations",
            "supporting_passages",
            "confidence",
            "limitations",
            "sections_found",
            "segments_found",
            "processing_time",
        }
        assert set(result_df.columns) == expected_columns

        # Verify data types
        assert result_df["query"].dtype == pl.Utf8
        assert result_df["confidence"].dtype == pl.Float64
        assert result_df["sections_found"].dtype == pl.Int64
        assert result_df["processing_time"].dtype == pl.Float64
