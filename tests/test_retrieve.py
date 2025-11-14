"""
Tests for the retrieve module, including HYDE functionality.
"""

import pytest
from unittest.mock import Mock, patch
from instructor import Instructor
from chromadb import Collection

from legiscope.retrieve import (
    hyde_rewriter,
    retrieve_embeddings,
    retrieve_sections,
    is_relevant,
    filter_results,
    HydeRewrite,
    RelevanceAssessment,
)


class TestHydeRewrite:
    """Test the HydeRewrite Pydantic model."""

    def test_hyde_rewrite_model_valid(self):
        """Test creating a valid HydeRewrite instance."""
        rewrite = HydeRewrite(
            rewritten_query="The following provisions regulate parking within municipal boundaries.",
            confidence=0.85,
            reasoning="Transformed natural language query into formal municipal code style",
            query_type="parking",
        )

        assert (
            rewrite.rewritten_query
            == "The following provisions regulate parking within municipal boundaries."
        )
        assert rewrite.confidence == 0.85
        assert (
            rewrite.reasoning
            == "Transformed natural language query into formal municipal code style"
        )
        assert rewrite.query_type == "parking"

    def test_hyde_rewrite_model_confidence_bounds(self):
        """Test confidence score bounds validation."""
        # Valid confidence scores
        rewrite1 = HydeRewrite(
            rewritten_query="Test query",
            confidence=0.0,
            reasoning="Test",
            query_type="test",
        )
        assert rewrite1.confidence == 0.0

        rewrite2 = HydeRewrite(
            rewritten_query="Test query",
            confidence=1.0,
            reasoning="Test",
            query_type="test",
        )
        assert rewrite2.confidence == 1.0

    def test_hyde_rewrite_model_invalid_confidence(self):
        """Test that invalid confidence scores are rejected."""
        with pytest.raises(ValueError):
            HydeRewrite(
                rewritten_query="Test query",
                confidence=-0.1,
                reasoning="Test",
                query_type="test",
            )

        with pytest.raises(ValueError):
            HydeRewrite(
                rewritten_query="Test query",
                confidence=1.1,
                reasoning="Test",
                query_type="test",
            )


class TestHydeRewriter:
    """Test the LLM-powered hyde_rewriter_llm function."""

    def test_hyde_rewriter_llm_success(self):
        """Test successful LLM rewrite."""
        # Set environment variable for consistent testing
        import os

        os.environ["LEGISCOPE_LLM_PROVIDER"] = "openai"

        # Mock the ask function
        mock_result = HydeRewrite(
            rewritten_query="The following provisions regulate vehicle parking within municipal boundaries.",
            confidence=0.92,
            reasoning="Transformed informal query into formal municipal code language",
            query_type="parking",
        )

        with patch("legiscope.retrieve.ask", return_value=mock_result) as mock_ask:
            mock_client = Mock(spec=Instructor)

            result = hyde_rewriter("where can I park my car", mock_client)

            assert isinstance(result, HydeRewrite)
            assert (
                result.rewritten_query
                == "The following provisions regulate vehicle parking within municipal boundaries."
            )
            assert result.confidence == 0.92
            assert result.query_type == "parking"

            # Verify ask was called correctly
            mock_ask.assert_called_once()
            call_args = mock_ask.call_args
            assert call_args[1]["client"] == mock_client
            assert "where can I park my car" in call_args[1]["prompt"]
            assert call_args[1]["response_model"] == HydeRewrite
            assert call_args[1]["model"] == "gpt-4.1-mini"

    def test_hyde_rewriter_llm_custom_model(self):
        """Test LLM rewrite with custom model."""
        mock_result = HydeRewrite(
            rewritten_query="Test query",
            confidence=0.8,
            reasoning="Test",
            query_type="test",
        )

        with patch("legiscope.retrieve.ask", return_value=mock_result) as mock_ask:
            mock_client = Mock(spec=Instructor)

            hyde_rewriter("test query", mock_client, model="gpt-4")

            # Verify custom model was used
            mock_ask.assert_called_once()
            call_args = mock_ask.call_args
            assert call_args[1]["model"] == "gpt-4"

    def test_hyde_rewriter_llm_empty_query(self):
        """Test that empty query raises ValueError."""
        mock_client = Mock(spec=Instructor)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            hyde_rewriter("", mock_client)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            hyde_rewriter("   ", mock_client)

    def test_hyde_rewriter_llm_api_failure(self):
        """Test handling of LLM API failures."""
        with patch("legiscope.retrieve.ask", side_effect=Exception("API Error")):
            mock_client = Mock(spec=Instructor)

            with pytest.raises(Exception, match="API Error"):
                hyde_rewriter("test query", mock_client)


class TestHydeRewriterIntegrated:
    """Test the integrated hyde_rewriter function with both modes."""

    def test_hyde_rewriter_with_llm_success(self):
        """Test hyde_rewriter with successful LLM call."""
        mock_result = HydeRewrite(
            rewritten_query="The following provisions regulate parking within municipal boundaries.",
            confidence=0.9,
            reasoning="Good rewrite",
            query_type="parking",
        )

        with patch("legiscope.retrieve.ask", return_value=mock_result):
            mock_client = Mock(spec=Instructor)

            result = hyde_rewriter("where can I park", client=mock_client)

            assert (
                result.rewritten_query
                == "The following provisions regulate parking within municipal boundaries."
            )

    def test_hyde_rewriter_client_validation(self):
        """Test hyde_rewriter validates client parameter."""
        mock_client = Mock(spec=Instructor)
        mock_result = HydeRewrite(
            rewritten_query="The following provisions regulate parking within municipal boundaries.",
            confidence=0.9,
            reasoning="Test rewrite",
            query_type="parking",
        )

        with patch("legiscope.retrieve.ask", return_value=mock_result):
            # Should work with valid client
            result = hyde_rewriter("where can I park", client=mock_client)
            assert isinstance(result, HydeRewrite)
            assert (
                result.rewritten_query
                == "The following provisions regulate parking within municipal boundaries."
            )


class TestRetrieveEmbeddings:
    """Test the retrieve_embeddings function with HYDE integration."""

    def test_retrieve_embeddings_hyde_requires_client(self):
        """Test retrieve_embeddings requires client for HYDE rewriting."""
        mock_collection = Mock(spec=Collection)

        with pytest.raises(ValueError, match="Client is required"):
            retrieve_embeddings(
                collection=mock_collection, query_text="where can I park", rewrite=True
            )

        # Verify query was NOT called since validation happens first
        mock_collection.query.assert_not_called()

    def test_retrieve_embeddings_with_hyde_llm(self):
        """Test retrieve_embeddings with LLM-powered HYDE."""
        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["1", "2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"jurisdiction_id": "test"}]],
            "distances": [[0.1, 0.2]],
        }

        mock_result = HydeRewrite(
            rewritten_query="The following provisions regulate vehicle parking within municipal boundaries.",
            confidence=0.9,
            reasoning="Good rewrite",
            query_type="parking",
        )

        with patch("legiscope.retrieve.hyde_rewriter", return_value=mock_result):
            with patch("legiscope.retrieve.get_embeddings") as mock_get_embeddings:
                mock_get_embeddings.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding
                mock_client = Mock(spec=Instructor)

                retrieve_embeddings(
                    collection=mock_collection,
                    query_text="where can I park",
                    rewrite=True,
                    rewrite_client=mock_client,
                )

                # Verify embeddings were generated with rewritten text
                mock_get_embeddings.assert_called_once()
                embedding_call_args = mock_get_embeddings.call_args[0]
                query_text_passed = embedding_call_args[1][
                    0
                ]  # Second arg is list of texts
                assert (
                    query_text_passed
                    == "The following provisions regulate vehicle parking within municipal boundaries."
                )

                # Verify query was called with embeddings
                mock_collection.query.assert_called_once()
                call_args = mock_collection.query.call_args
                assert "query_embeddings" in call_args[1]
                assert call_args[1]["query_embeddings"] == [[0.1, 0.2, 0.3]]

    def test_retrieve_embeddings_without_hyde(self):
        """Test retrieve_embeddings without HYDE rewriting."""
        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["1", "2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"jurisdiction_id": "test"}]],
            "distances": [[0.1, 0.2]],
        }

        with patch("legiscope.retrieve.get_embeddings") as mock_get_embeddings:
            mock_get_embeddings.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding

            retrieve_embeddings(
                collection=mock_collection, query_text="where can I park", rewrite=False
            )

            # Verify embeddings were generated with original text
            mock_get_embeddings.assert_called_once()
            embedding_call_args = mock_get_embeddings.call_args[0]
            query_text_passed = embedding_call_args[1][0]  # Second arg is list of texts
            assert query_text_passed == "where can I park"

            # Verify query was called with embeddings
            mock_collection.query.assert_called_once()
            call_args = mock_collection.query.call_args
            assert "query_embeddings" in call_args[1]
            assert call_args[1]["query_embeddings"] == [[0.1, 0.2, 0.3]]

    def test_retrieve_embeddings_with_jurisdiction_filter(self):
        """Test retrieve_embeddings with jurisdiction filtering."""
        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"jurisdiction_id": "IL-WindyCity"}]],
            "distances": [[0.1]],
        }

        retrieve_embeddings(
            collection=mock_collection,
            query_text="parking regulations",
            jurisdiction_id="IL-WindyCity",
        )

        # Verify jurisdiction filter was applied
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args
        where_filter = call_args[1]["where"]
        assert where_filter == {"jurisdiction_id": "IL-WindyCity"}

    def test_retrieve_embeddings_with_custom_model(self):
        """Test retrieve_embeddings passes custom model to HYDE."""
        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"jurisdiction_id": "test"}]],
            "distances": [[0.1]],
        }

        with patch("legiscope.retrieve.hyde_rewriter") as mock_hyde:
            with patch("legiscope.retrieve.get_embeddings") as mock_get_embeddings:
                mock_get_embeddings.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding
                mock_client = Mock(spec=Instructor)

                retrieve_embeddings(
                    collection=mock_collection,
                    query_text="test query",
                    rewrite=True,
                    rewrite_client=mock_client,
                    rewrite_model="gpt-4",
                )

                # Verify custom model was passed to hyde_rewriter
                mock_hyde.assert_called_once_with("test query", mock_client, "gpt-4")


class TestRetrieveSections:
    """Test the retrieve_sections function."""

    def test_retrieve_sections_basic(self):
        """Test basic section retrieval functionality."""
        from unittest.mock import patch
        import tempfile
        import polars as pl

        # Mock segment retrieval results
        mock_segment_results = {
            "ids": [["1", "2", "3"]],
            "documents": [["segment1", "segment2", "segment3"]],
            "distances": [[0.1, 0.2, 0.3]],
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
                    {
                        "section_ref": 0,
                        "segment_position": 1,
                        "section_heading": "# Section 1",
                        "section_level": 1,
                    },
                ]
            ],
        }

        # Create mock sections DataFrame
        sections_df = pl.DataFrame(
            {
                "section_idx": [0, 1],
                "heading_text": ["# Section 1", "## Section 2"],
                "body_text": ["Content of section 1", "Content of section 2"],
                "heading_level": [1, 2],
                "parent": [None, 0],
            }
        )

        with patch(
            "legiscope.retrieve.retrieve_embeddings", return_value=mock_segment_results
        ):
            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as tmp_file:
                sections_df.write_parquet(tmp_file.name)

                mock_collection = Mock()

                result = retrieve_sections(
                    collection=mock_collection,
                    query_text="test query",
                    sections_parquet_path=tmp_file.name,
                )

                # Verify structure
                assert "sections" in result
                assert "query_info" in result

                # Verify query info
                query_info = result["query_info"]
                assert query_info["original_query"] == "test query"
                assert query_info["rewritten_query"] is None
                assert query_info["total_segments_found"] == 3
                assert query_info["unique_sections"] == 2

                # Verify sections
                sections = result["sections"]
                assert len(sections) == 2

                # Find section 0 (should have 2 segments)
                section_0 = next(s for s in sections if s["section_idx"] == 0)
                assert section_0["heading_text"] == "# Section 1"
                assert section_0["body_text"] == "Content of section 1"
                assert section_0["segment_count"] == 2
                assert len(section_0["matching_segments"]) == 2
                assert section_0["relevance_score"] == 0.1  # Best segment score

                # Find section 1 (should have 1 segment)
                section_1 = next(s for s in sections if s["section_idx"] == 1)
                assert section_1["heading_text"] == "## Section 2"
                assert section_1["body_text"] == "Content of section 2"
                assert section_1["segment_count"] == 1
                assert len(section_1["matching_segments"]) == 1
                assert section_1["relevance_score"] == 0.2

    def test_retrieve_sections_with_hyde(self):
        """Test section retrieval with HYDE rewriting."""
        from unittest.mock import patch
        import tempfile
        import polars as pl

        # Mock segment retrieval results with HYDE
        mock_segment_results = {
            "ids": [["1"]],
            "documents": [["segment1"]],
            "distances": [[0.1]],
            "metadatas": [[{"section_ref": 0, "segment_position": 0}]],
        }

        sections_df = pl.DataFrame(
            {
                "section_idx": [0],
                "heading_text": ["# Section 1"],
                "body_text": ["Content"],
                "heading_level": [1],
                "parent": [None],
            }
        )

        with patch(
            "legiscope.retrieve.retrieve_embeddings", return_value=mock_segment_results
        ) as mock_retrieve:
            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as tmp_file:
                sections_df.write_parquet(tmp_file.name)

                mock_collection = Mock()

                retrieve_sections(
                    collection=mock_collection,
                    query_text="where can I park",
                    sections_parquet_path=tmp_file.name,
                    rewrite=True,
                    rewrite_client=Mock(),
                    rewrite_model="gpt-4",
                )

                # Verify retrieve_embeddings was called with correct parameters
                mock_retrieve.assert_called_once()
                call_args = mock_retrieve.call_args
                assert call_args[1]["collection"] == mock_collection
                assert call_args[1]["query_text"] == "where can I park"
                assert call_args[1]["rewrite"]
                assert call_args[1]["rewrite_model"] == "gpt-4"

    def test_retrieve_sections_no_results(self):
        """Test section retrieval with no segment results."""
        from unittest.mock import patch
        import tempfile
        import polars as pl

        # Mock empty segment results
        mock_segment_results = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        sections_df = pl.DataFrame(
            {
                "section_idx": [0],
                "heading_text": ["# Section 1"],
                "body_text": ["Content"],
                "heading_level": [1],
                "parent": [None],
            }
        )

        with patch(
            "legiscope.retrieve.retrieve_embeddings", return_value=mock_segment_results
        ):
            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as tmp_file:
                sections_df.write_parquet(tmp_file.name)

                mock_collection = Mock()

                result = retrieve_sections(
                    collection=mock_collection,
                    query_text="no results query",
                    sections_parquet_path=tmp_file.name,
                )

                assert result["sections"] == []
                assert result["query_info"]["total_segments_found"] == 0
                assert result["query_info"]["unique_sections"] == 0

    def test_retrieve_sections_missing_file(self):
        """Test section retrieval with missing parquet file."""
        mock_collection = Mock()

        with pytest.raises(FileNotFoundError, match="Sections parquet file not found"):
            retrieve_sections(
                collection=mock_collection,
                query_text="test query",
                sections_parquet_path="/nonexistent/path.parquet",
            )

    def test_retrieve_sections_missing_columns(self):
        """Test section retrieval with missing required columns in parquet."""
        from unittest.mock import patch
        import tempfile
        import polars as pl

        # Mock segment results
        mock_segment_results = {
            "ids": [["1"]],
            "documents": [["segment1"]],
            "distances": [[0.1]],
            "metadatas": [[{"section_ref": 0}]],
        }

        # Create sections DataFrame missing required columns
        sections_df = pl.DataFrame(
            {
                "section_idx": [0],
                "heading_text": ["# Section 1"],
                # Missing body_text, heading_level, parent
            }
        )

        with patch(
            "legiscope.retrieve.retrieve_embeddings", return_value=mock_segment_results
        ):
            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as tmp_file:
                sections_df.write_parquet(tmp_file.name)

                mock_collection = Mock()

                with pytest.raises(
                    ValueError, match="Sections parquet missing required columns"
                ):
                    retrieve_sections(
                        collection=mock_collection,
                        query_text="test query",
                        sections_parquet_path=tmp_file.name,
                    )

    def test_retrieve_sections_missing_section_ref(self):
        """Test section retrieval with segments missing section_ref metadata."""
        from unittest.mock import patch
        import tempfile
        import polars as pl

        # Mock segment results with missing section_ref
        mock_segment_results = {
            "ids": [["1"]],
            "documents": [["segment1"]],
            "distances": [[0.1]],
            "metadatas": [[{"segment_position": 0}]],  # Missing section_ref
        }

        sections_df = pl.DataFrame(
            {
                "section_idx": [0],
                "heading_text": ["# Section 1"],
                "body_text": ["Content"],
                "heading_level": [1],
                "parent": [None],
            }
        )

        with patch(
            "legiscope.retrieve.retrieve_embeddings", return_value=mock_segment_results
        ):
            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as tmp_file:
                sections_df.write_parquet(tmp_file.name)

                mock_collection = Mock()

                result = retrieve_sections(
                    collection=mock_collection,
                    query_text="test query",
                    sections_parquet_path=tmp_file.name,
                )

                # Should return empty results since no valid section references
                assert result["sections"] == []
                assert result["query_info"]["total_segments_found"] == 1
                assert result["query_info"]["unique_sections"] == 0

    def test_retrieve_sections_jurisdiction_filter(self):
        """Test section retrieval with jurisdiction filtering."""
        from unittest.mock import patch
        import tempfile
        import polars as pl

        mock_segment_results = {
            "ids": [["1"]],
            "documents": [["segment1"]],
            "distances": [[0.1]],
            "metadatas": [[{"section_ref": 0}]],
        }

        sections_df = pl.DataFrame(
            {
                "section_idx": [0],
                "heading_text": ["# Section 1"],
                "body_text": ["Content"],
                "heading_level": [1],
                "parent": [None],
            }
        )

        with patch(
            "legiscope.retrieve.retrieve_embeddings", return_value=mock_segment_results
        ) as mock_retrieve:
            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as tmp_file:
                sections_df.write_parquet(tmp_file.name)

                mock_collection = Mock()

                retrieve_sections(
                    collection=mock_collection,
                    query_text="test query",
                    sections_parquet_path=tmp_file.name,
                    jurisdiction_id="IL-WindyCity",
                )

                # Verify retrieve_embeddings was called with jurisdiction filter
                mock_retrieve.assert_called_once()
                call_args = mock_retrieve.call_args
                assert call_args[1]["jurisdiction_id"] == "IL-WindyCity"

    def test_retrieve_sections_segment_ordering(self):
        """Test that segments within sections are ordered by relevance."""
        from unittest.mock import patch
        import tempfile
        import polars as pl

        # Mock segment results with varying distances
        mock_segment_results = {
            "ids": [["1", "2", "3"]],
            "documents": [["segment1", "segment2", "segment3"]],
            "distances": [[0.3, 0.1, 0.2]],  # Different relevance scores
            "metadatas": [
                [
                    {"section_ref": 0, "segment_position": 0},
                    {"section_ref": 0, "segment_position": 1},
                    {"section_ref": 0, "segment_position": 2},
                ]
            ],
        }

        sections_df = pl.DataFrame(
            {
                "section_idx": [0],
                "heading_text": ["# Section 1"],
                "body_text": ["Content"],
                "heading_level": [1],
                "parent": [None],
            }
        )

        with patch(
            "legiscope.retrieve.retrieve_embeddings", return_value=mock_segment_results
        ):
            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as tmp_file:
                sections_df.write_parquet(tmp_file.name)

                mock_collection = Mock()

                result = retrieve_sections(
                    collection=mock_collection,
                    query_text="test query",
                    sections_parquet_path=tmp_file.name,
                )

                sections = result["sections"]
                assert len(sections) == 1

                section = sections[0]
                assert section["segment_count"] == 3

                # Verify segments are ordered by distance (relevance)
                segments = section["matching_segments"]
                assert len(segments) == 3
                assert segments[0]["distance"] == 0.1  # Most relevant
                assert segments[1]["distance"] == 0.2
                assert segments[2]["distance"] == 0.3  # Least relevant

                # Verify relevance score is the best segment score
                assert section["relevance_score"] == 0.1


class TestRelevanceAssessment:
    """Test the RelevanceAssessment Pydantic model."""

    def test_relevance_assessment_model_valid(self):
        """Test creating a valid RelevanceAssessment instance."""
        assessment = RelevanceAssessment(
            is_relevant=True,
            confidence=0.85,
            reasoning="The text directly addresses parking regulations with specific rules",
        )

        assert assessment.is_relevant is True
        assert assessment.confidence == 0.85
        assert (
            assessment.reasoning
            == "The text directly addresses parking regulations with specific rules"
        )

    def test_relevance_assessment_model_confidence_bounds(self):
        """Test confidence score bounds validation."""
        # Valid confidence scores
        assessment1 = RelevanceAssessment(
            is_relevant=True,
            confidence=0.0,
            reasoning="Test",
        )
        assert assessment1.confidence == 0.0

        assessment2 = RelevanceAssessment(
            is_relevant=False,
            confidence=1.0,
            reasoning="Test",
        )
        assert assessment2.confidence == 1.0

    def test_relevance_assessment_model_invalid_confidence(self):
        """Test that invalid confidence scores are rejected."""
        with pytest.raises(ValueError):
            RelevanceAssessment(
                is_relevant=True,
                confidence=-0.1,
                reasoning="Test",
            )

        with pytest.raises(ValueError):
            RelevanceAssessment(
                is_relevant=False,
                confidence=1.1,
                reasoning="Test",
            )


class TestIsRelevant:
    """Test the is_relevant function."""

    def test_is_relevant_success(self):
        """Test successful relevance assessment."""
        # Set environment variable for consistent testing
        import os

        os.environ["LEGISCOPE_LLM_PROVIDER"] = "openai"

        mock_result = RelevanceAssessment(
            is_relevant=True,
            confidence=0.9,
            reasoning="The text contains specific parking regulations that directly answer the query",
        )

        with patch("legiscope.retrieve.ask", return_value=mock_result) as mock_ask:
            mock_client = Mock(spec=Instructor)

            result = is_relevant(
                mock_client,
                "parking regulations",
                "No vehicle shall be parked on any street between 2 AM and 6 AM",
            )

            assert isinstance(result, RelevanceAssessment)
            assert result.is_relevant is True
            assert result.confidence == 0.9
            assert "parking regulations" in result.reasoning

            # Verify ask was called correctly
            mock_ask.assert_called_once()
            call_args = mock_ask.call_args
            assert call_args[1]["client"] == mock_client
            assert "parking regulations" in call_args[1]["prompt"]
            assert call_args[1]["response_model"] == RelevanceAssessment
            assert call_args[1]["model"] == "gpt-4.1-mini"

    def test_is_relevant_custom_model(self):
        """Test relevance assessment with custom model."""
        mock_result = RelevanceAssessment(
            is_relevant=False,
            confidence=0.8,
            reasoning="The text discusses unrelated topics",
        )

        with patch("legiscope.retrieve.ask", return_value=mock_result) as mock_ask:
            mock_client = Mock(spec=Instructor)

            is_relevant(mock_client, "test query", "test text", model="gpt-4")

            # Verify custom model was used
            mock_ask.assert_called_once()
            call_args = mock_ask.call_args
            assert call_args[1]["model"] == "gpt-4"

    def test_is_relevant_empty_query(self):
        """Test that empty query raises ValueError."""
        mock_client = Mock(spec=Instructor)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            is_relevant(mock_client, "", "some text")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            is_relevant(mock_client, "   ", "some text")

    def test_is_relevant_empty_text(self):
        """Test that empty text raises ValueError."""
        mock_client = Mock(spec=Instructor)

        with pytest.raises(ValueError, match="Text cannot be empty"):
            is_relevant(mock_client, "some query", "")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            is_relevant(mock_client, "some query", "   ")

    def test_is_relevant_no_client(self):
        """Test that missing client raises ValueError."""
        with pytest.raises(ValueError, match="Client is required"):
            is_relevant(None, "query", "text")

    def test_is_relevant_api_failure(self):
        """Test handling of LLM API failures."""
        with patch("legiscope.retrieve.ask", side_effect=Exception("API Error")):
            mock_client = Mock(spec=Instructor)

            with pytest.raises(Exception, match="API Error"):
                is_relevant(mock_client, "test query", "test text")


class TestFilterResults:
    """Test the filter_results function."""

    def test_filter_results_basic(self):
        """Test basic result filtering."""
        # Mock relevance assessments
        mock_assessments = [
            RelevanceAssessment(is_relevant=True, confidence=0.9, reasoning="Relevant"),
            RelevanceAssessment(
                is_relevant=False, confidence=0.8, reasoning="Not relevant"
            ),
            RelevanceAssessment(is_relevant=True, confidence=0.7, reasoning="Relevant"),
        ]

        # Mock input results
        input_results = {
            "ids": [["1", "2", "3"]],
            "documents": [["doc1", "doc2", "doc3"]],
            "distances": [[0.1, 0.2, 0.3]],
            "metadatas": [[{"meta": "1"}, {"meta": "2"}, {"meta": "3"}]],
        }

        with patch("legiscope.retrieve.is_relevant", side_effect=mock_assessments):
            mock_client = Mock(spec=Instructor)

            result = filter_results(
                mock_client,
                input_results,
                "test query",
                threshold=0.5,
            )

            # Verify structure
            assert "ids" in result
            assert "documents" in result
            assert "distances" in result
            assert "metadatas" in result
            assert "filtering_metadata" in result

            # Verify filtering results
            assert len(result["ids"][0]) == 2  # Only relevant documents
            assert result["ids"][0] == ["1", "3"]
            assert result["documents"][0] == ["doc1", "doc3"]
            assert result["distances"][0] == [0.1, 0.3]
            assert result["metadatas"][0] == [{"meta": "1"}, {"meta": "3"}]

            # Verify metadata
            metadata = result["filtering_metadata"]
            assert metadata["original_count"] == 3
            assert metadata["filtered_count"] == 2
            assert metadata["threshold"] == 0.5
            assert len(metadata["assessments"]) == 3

    def test_filter_results_with_threshold(self):
        """Test filtering with confidence threshold."""
        # Mock relevance assessments with varying confidence
        mock_assessments = [
            RelevanceAssessment(
                is_relevant=True, confidence=0.9, reasoning="High confidence"
            ),
            RelevanceAssessment(
                is_relevant=True, confidence=0.3, reasoning="Low confidence"
            ),
            RelevanceAssessment(
                is_relevant=True, confidence=0.7, reasoning="Medium confidence"
            ),
        ]

        input_results = {
            "ids": [["1", "2", "3"]],
            "documents": [["doc1", "doc2", "doc3"]],
            "distances": [[0.1, 0.2, 0.3]],
            "metadatas": [[None, None, None]],
        }

        with patch("legiscope.retrieve.is_relevant", side_effect=mock_assessments):
            mock_client = Mock(spec=Instructor)

            result = filter_results(
                mock_client,
                input_results,
                "test query",
            )

            # Only documents 1 and 3 should pass threshold
            assert len(result["ids"][0]) == 2
            assert result["ids"][0] == ["1", "3"]

    def test_filter_results_no_client(self):
        """Test that missing client raises ValueError."""
        input_results = {
            "ids": [["1"]],
            "documents": [["doc1"]],
            "distances": [[0.1]],
        }

        with pytest.raises(ValueError, match="Client is required"):
            filter_results(None, input_results, "query")

    def test_filter_results_invalid_structure(self):
        """Test handling of invalid results structure."""
        mock_client = Mock(spec=Instructor)

        # Empty results
        with pytest.raises(ValueError, match="Invalid results structure"):
            filter_results(mock_client, None, "query")

        # Missing required keys
        with pytest.raises(ValueError, match="Results missing required keys"):
            filter_results(mock_client, {"wrong": "structure"}, "query")

    def test_filter_results_empty_results(self):
        """Test filtering with empty results."""
        empty_results = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
        }

        mock_client = Mock(spec=Instructor)

        result = filter_results(mock_client, empty_results, "query")

        assert result["filtering_metadata"]["original_count"] == 0
        assert result["filtering_metadata"]["filtered_count"] == 0
        assert len(result["filtering_metadata"]["assessments"]) == 0

    def test_filter_results_assessment_failure(self):
        """Test handling of assessment failures."""

        def failing_assessment(client, query, text, model):
            if text == "doc2":
                raise Exception("Assessment failed")
            return RelevanceAssessment(
                is_relevant=True, confidence=0.9, reasoning="Good"
            )

        input_results = {
            "ids": [["1", "2", "3"]],
            "documents": [["doc1", "doc2", "doc3"]],
            "distances": [[0.1, 0.2, 0.3]],
            "metadatas": [[None, None, None]],
        }

        with patch("legiscope.retrieve.is_relevant", side_effect=failing_assessment):
            mock_client = Mock(spec=Instructor)

            result = filter_results(mock_client, input_results, "query")

            # Should still work, with failed assessment marked as not relevant
            assert len(result["ids"][0]) == 2  # doc1 and doc3
            assert result["ids"][0] == ["1", "3"]

            # Check assessment metadata
            assessments = result["filtering_metadata"]["assessments"]
            assert len(assessments) == 3

            # Find failed assessment
            failed_assessment = next(a for a in assessments if a["index"] == 1)
            assert failed_assessment["is_relevant"] is False
            assert failed_assessment["confidence"] == 0.0
            assert "Assessment failed" in failed_assessment["reasoning"]

    def test_filter_results_preserves_extra_keys(self):
        """Test that extra keys in results are preserved."""
        input_results = {
            "ids": [["1"]],
            "documents": [["doc1"]],
            "distances": [[0.1]],
            "extra_key": "extra_value",
            "another_key": {"nested": "data"},
        }

        mock_assessment = RelevanceAssessment(
            is_relevant=True, confidence=0.9, reasoning="Relevant"
        )

        with patch("legiscope.retrieve.is_relevant", return_value=mock_assessment):
            mock_client = Mock(spec=Instructor)

            result = filter_results(mock_client, input_results, "query")

            # Extra keys should be preserved
            assert result["extra_key"] == "extra_value"
            assert result["another_key"] == {"nested": "data"}

    def test_filter_results_no_metadatas(self):
        """Test filtering when metadatas are missing."""
        input_results = {
            "ids": [["1", "2"]],
            "documents": [["doc1", "doc2"]],
            "distances": [[0.1, 0.2]],
            # No metadatas key
        }

        mock_assessments = [
            RelevanceAssessment(is_relevant=True, confidence=0.9, reasoning="Relevant"),
            RelevanceAssessment(
                is_relevant=False, confidence=0.8, reasoning="Not relevant"
            ),
        ]

        with patch("legiscope.retrieve.is_relevant", side_effect=mock_assessments):
            mock_client = Mock(spec=Instructor)

            result = filter_results(mock_client, input_results, "query")

            # Should work without metadatas
            assert len(result["ids"][0]) == 1
            assert result["ids"][0] == ["1"]
            assert result["metadatas"] == [
                [None]
            ]  # Should be [[None]] when no original metadatas
