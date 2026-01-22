"""
Tests for query functions using new config-based API.

This test file covers the refactored query_legal_documents() and run_queries()
functions that use QueryConfig and BatchQueryConfig.
"""

import polars as pl
from unittest.mock import Mock, patch
from instructor import Instructor

from legiscope.utils import LLMConfig
from legiscope.query import (
    QuerySettings,
    BatchQuerySettings,
    LegalQueryResponse,
    query_legal_documents,
    run_queries,
)
from legiscope.retrieve import (
    SectionCollection,
    SectionResult,
    QueryInfo,
)


class TestQueryConfigBasics:
    """Test QuerySettings-based query_legal_documents function."""

    def test_query_legal_documents_with_config(self):
        """Test basic query_legal_documents with settings object."""
        mock_client = Mock(spec=Instructor)

        # Mock retrieval results
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_idx=0,
                    heading_text="# Parking Regulations",
                    body_text="No parking between 2am and 6am",
                    heading_level=1,
                    parent=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="parking rules",
                total_segments_found=1,
                unique_sections=1,
            ),
        )

        # Mock LLM response
        mock_response = LegalQueryResponse(
            short_answer="Parking prohibited 2am-6am",
            reasoning="Municipal code restricts overnight parking",
            citations=["Parking Regulations, Section 1"],
            supporting_passages=["No parking between 2am and 6am"],
            confidence=0.9,
            limitations="None",
        )

        with patch("legiscope.query.ask", return_value=mock_response):
            llm_config = LLMConfig(client=mock_client, model="test-model")
            settings = QuerySettings(llm=llm_config)

            response, similarity_scores = query_legal_documents(
                retrieval_results, "What are the parking rules?", settings
            )

            assert response.short_answer == "Parking prohibited 2am-6am"
            assert response.confidence == 0.9
            assert len(response.citations) == 1

    def test_query_with_relevance_filtering(self):
        """Test query with relevance filtering enabled."""
        mock_client = Mock(spec=Instructor)

        section = SectionResult(
            section_idx=0,
            heading_text="# Test Section",
            body_text="Test content",
            heading_level=1,
            parent=None,
            matching_segments=[],
            relevance_score=0.1,
            segment_count=1,
        )

        retrieval_results = SectionCollection(
            sections=[section], query_info=QueryInfo(original_query="test query")
        )

        mock_response = LegalQueryResponse(
            short_answer="Test answer",
            reasoning="Test reasoning",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        with patch("legiscope.query.filter_sections") as mock_filter:
            with patch("legiscope.query.ask", return_value=mock_response):
                mock_filter.return_value = {"sections": [section]}

                llm_config = LLMConfig(client=mock_client, model="test-model")
                settings = QuerySettings(
                    llm=llm_config, filter_relevance=True, relevance_threshold=0.7
                )

                response, similarity_scores = query_legal_documents(
                    retrieval_results, "test query", settings
                )

                assert response.short_answer == "Test answer"
                mock_filter.assert_called_once()


class TestBatchQueryConfigBasics:
    """Test BatchQuerySettings-based run_queries function."""

    def test_run_queries_with_minimal_config(self, tmp_path):
        """Test run_queries with minimal configuration."""

        # Create test sections parquet
        sections_data = {
            "section_idx": [0],
            "heading_text": ["# Test"],
            "body_text": ["Content"],
            "heading_level": [1],
            "parent": [None],
        }
        sections_df = pl.DataFrame(sections_data)
        sections_path = tmp_path / "sections.parquet"
        sections_df.write_parquet(sections_path)

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "ids": [["0"]],
            "documents": [["doc"]],
            "metadatas": [
                [
                    {
                        "section_ref": 0,
                        "segment_position": 0,
                        "section_heading": "# Test",
                        "section_level": 1,
                    }
                ]
            ],
            "distances": [[0.1]],
        }

        mock_llm_response = LegalQueryResponse(
            short_answer="Test answer",
            reasoning="Test reasoning",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        with patch("legiscope.query.retrieve_sections") as mock_retrieve:
            with patch(
                "legiscope.query.query_legal_documents", return_value=mock_llm_response
            ):
                mock_retrieve.return_value = SectionCollection(
                    sections=[],
                    query_info=QueryInfo(
                        original_query="", total_segments_found=0, unique_sections=0
                    ),
                )

                mock_client = Mock(spec=Instructor)
                llm_config = LLMConfig(client=mock_client, model="test-model")

                settings = BatchQuerySettings(llm=llm_config)

                results_df = run_queries(
                    collection=mock_collection,
                    sections_parquet_path=str(sections_path),
                    queries=["query1", "query2"],
                    jurisdiction_id="IL-WindyCity",
                    settings=settings,
                )

                assert isinstance(results_df, pl.DataFrame)
                assert len(results_df) == 2
                assert "query" in results_df.columns
                assert "short_answer" in results_df.columns

    def test_batch_query_creates_default_llm(self, tmp_path):
        """Test that BatchQuerySettings creates default LLM if not provided."""
        sections_path = tmp_path / "sections.parquet"
        sections_path.write_text("")  # Create empty file

        with patch("legiscope.llm_config.Config.get_fast_client") as mock_get_client:
            mock_client = Mock(spec=Instructor)
            mock_get_client.return_value = mock_client

            settings = BatchQuerySettings()
            # No llm provided - should use default

            assert settings.llm is not None
            assert settings.llm.client is mock_client
