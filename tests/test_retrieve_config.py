"""
Tests for retrieve functions using new config-based API.

This test file covers the refactored retrieve_segments() and retrieve_sections()
functions that use RetrievalConfig and SectionRetrievalConfig.
"""

import pytest
from unittest.mock import Mock, patch
from chromadb.api.models.Collection import Collection
from instructor import Instructor

from legiscope.retrieve import (
    RetrievalSettings,
    retrieve_segments,
    retrieve_sections,
)


class TestRetrievalConfigBasics:
    """Test RetrievalSettings-based retrieve_segments function."""

    def test_retrieve_segments_with_config_basic(self):
        """Test basic retrieve_segments with settings object."""
        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["1", "2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.2]],
        }

        with patch("legiscope.retrieve.get_embedding_client"):
            with patch("legiscope.retrieve.get_embeddings") as mock_get_embeddings:
                mock_get_embeddings.return_value = [[0.1, 0.2, 0.3]]

                results = retrieve_segments(mock_collection, "test query")

                assert len(results.ids[0]) == 2
                assert results.documents[0] == ["doc1", "doc2"]
                mock_collection.query.assert_called_once()

    def test_retrieve_segments_with_jurisdiction_filter(self):
        """Test retrieve_segments with jurisdiction filtering."""
        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"jurisdiction_id": "IL-WindyCity"}]],
            "distances": [[0.1]],
        }

        with patch("legiscope.retrieve.get_embedding_client"):
            with patch("legiscope.retrieve.get_embeddings") as mock_get_embeddings:
                mock_get_embeddings.return_value = [[0.1, 0.2, 0.3]]

                settings = RetrievalSettings(
                    jurisdiction_id="IL-WindyCity", n_results=5
                )

                retrieve_segments(mock_collection, "test query", settings)

                # Check that query was called with where filter
                call_kwargs = mock_collection.query.call_args.kwargs
                assert "where" in call_kwargs
                assert call_kwargs["where"] == {"jurisdiction_id": "IL-WindyCity"}

    def test_retrieve_segments_hyde_requires_client(self):
        """Test that use_hyde=True requires hyde_client."""
        with pytest.raises(ValueError, match="hyde_client required"):
            RetrievalSettings(
                use_hyde=True  # Missing hyde_client
            )

    def test_retrieve_segments_with_hyde(self):
        """Test retrieve_segments with HYDE rewriting enabled."""
        from legiscope.retrieve import HydeRewrite

        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["1", "2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.2]],
        }

        mock_hyde_result = HydeRewrite(
            rewritten_query="Municipal code parking regulations",
            confidence=0.9,
            reasoning="Rewritten",
            query_type="parking",
        )

        with patch("legiscope.retrieve.hyde_rewriter", return_value=mock_hyde_result):
            with patch("legiscope.retrieve.get_embedding_client"):
                with patch("legiscope.retrieve.get_embeddings") as mock_embeddings:
                    mock_embeddings.return_value = [[0.1, 0.2, 0.3]]

                    mock_client = Mock(spec=Instructor)
                    settings = RetrievalSettings(use_hyde=True, hyde_client=mock_client)

                    results = retrieve_segments(
                        mock_collection, "where can I park", settings
                    )

                    assert len(results.ids[0]) == 2


class TestSectionRetrievalConfigBasics:
    """Test SectionRetrievalSettings-based retrieve_sections function."""

    def test_retrieve_sections_requires_parquet_path(self):
        """Test that retrieve_sections requires sections_parquet_path parameter."""
        mock_collection = Mock(spec=Collection)

        # This should raise because sections_parquet_path is a required parameter
        with pytest.raises(TypeError):
            retrieve_sections(
                mock_collection,
                # Missing sections_parquet_path parameter
                query_text="test query",
            )

    def test_retrieve_sections_with_config(self, tmp_path):
        """Test retrieve_sections with settings object."""
        import polars as pl

        # Create test sections parquet file
        sections_data = {
            "section_idx": [0, 1],
            "heading_text": ["# Section 1", "## Section 2"],
            "body_text": ["Content 1", "Content 2"],
            "heading_level": [1, 2],
            "parent": [None, 0],
        }
        sections_df = pl.DataFrame(sections_data)
        sections_path = tmp_path / "sections.parquet"
        sections_df.write_parquet(sections_path)

        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["0", "1"]],
            "documents": [["seg1", "seg2"]],
            "metadatas": [
                [
                    {
                        "section_ref": 0,
                        "segment_position": 0,
                        "section_heading": "# Section 1",
                        "section_level": 1,
                    },
                    {
                        "section_ref": 1,
                        "segment_position": 0,
                        "section_heading": "## Section 2",
                        "section_level": 2,
                    },
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        with patch("legiscope.retrieve.get_embedding_client"):
            with patch("legiscope.retrieve.get_embeddings") as mock_embeddings:
                mock_embeddings.return_value = [[0.1, 0.2, 0.3]]

                results = retrieve_sections(
                    mock_collection, str(sections_path), "test query"
                )

                assert hasattr(results, "sections")
                assert hasattr(results, "query_info")
                assert len(results.sections) == 2
