"""
Tests for the retrieve module, including HYDE functionality.
"""

from unittest.mock import Mock, patch

import pytest
from instructor import Instructor

from legiscope.retrieve import (
    HydeRewrite,
    RelevanceAssessment,
    SegmentCollection,
    filter_results,
    hyde_rewriter,
    is_relevant,
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

            result = hyde_rewriter(mock_client, "where can I park my car")

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

            hyde_rewriter(mock_client, "test query", model="gpt-4")

            # Verify custom model was used
            mock_ask.assert_called_once()
            call_args = mock_ask.call_args
            assert call_args[1]["model"] == "gpt-4"

    def test_hyde_rewriter_llm_empty_query(self):
        """Test that empty query raises ValueError."""
        mock_client = Mock(spec=Instructor)

        with pytest.raises(ValueError, match="query cannot be empty"):
            hyde_rewriter(mock_client, "")

        with pytest.raises(ValueError, match="query cannot be empty"):
            hyde_rewriter(mock_client, "   ")

    def test_hyde_rewriter_llm_api_failure(self):
        """Test handling of LLM API failures."""
        with patch("legiscope.retrieve.ask", side_effect=Exception("API Error")):
            mock_client = Mock(spec=Instructor)

            with pytest.raises(Exception, match="API Error"):
                hyde_rewriter(mock_client, "test query")


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

            result = hyde_rewriter(mock_client, "where can I park")

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
            result = hyde_rewriter(mock_client, "where can I park")
            assert isinstance(result, HydeRewrite)
            assert (
                result.rewritten_query
                == "The following provisions regulate parking within municipal boundaries."
            )


class TestRelevanceAssessment:
    """Test the RelevanceAssessment Pydantic model."""

    def test_relevance_assessment_model_valid(self):
        """Test creating a valid RelevanceAssessment instance."""
        assessment = RelevanceAssessment(
            is_relevant=True,
            relevance_score=0.75,
            confidence=0.85,
            reasoning="The text directly addresses parking regulations with specific rules",
        )

        assert assessment.is_relevant is True
        assert assessment.relevance_score == 0.75
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
            relevance_score=0.5,
            confidence=0.0,
            reasoning="Test",
        )
        assert assessment1.confidence == 0.0

        assessment2 = RelevanceAssessment(
            is_relevant=False,
            relevance_score=0.1,
            confidence=1.0,
            reasoning="Test",
        )
        assert assessment2.confidence == 1.0

    def test_relevance_assessment_model_invalid_confidence(self):
        """Test that invalid confidence scores are rejected."""
        with pytest.raises(ValueError):
            RelevanceAssessment(
                is_relevant=True,
                relevance_score=0.5,
                confidence=-0.1,
                reasoning="Test",
            )

        with pytest.raises(ValueError):
            RelevanceAssessment(
                is_relevant=False,
                relevance_score=0.1,
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
            relevance_score=0.85,
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
            relevance_score=0.2,
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

        with pytest.raises(ValueError, match="query cannot be empty"):
            is_relevant(mock_client, "", "some text")

        with pytest.raises(ValueError, match="query cannot be empty"):
            is_relevant(mock_client, "   ", "some text")

    def test_is_relevant_empty_text(self):
        """Test that empty text raises ValueError."""
        mock_client = Mock(spec=Instructor)

        with pytest.raises(ValueError, match="text cannot be empty"):
            is_relevant(mock_client, "some query", "")

        with pytest.raises(ValueError, match="text cannot be empty"):
            is_relevant(mock_client, "some query", "   ")

    def test_is_relevant_no_client(self):
        """Test that missing client raises ValueError."""
        with pytest.raises(ValueError, match="client is required"):
            is_relevant(None, "query", "text")  # type: ignore

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
            RelevanceAssessment(
                is_relevant=True,
                relevance_score=0.85,
                confidence=0.9,
                reasoning="Relevant",
            ),
            RelevanceAssessment(
                is_relevant=False,
                relevance_score=0.2,
                confidence=0.8,
                reasoning="Not relevant",
            ),
            RelevanceAssessment(
                is_relevant=True,
                relevance_score=0.7,
                confidence=0.7,
                reasoning="Relevant",
            ),
        ]

        # Mock input results using dataclass
        input_results = SegmentCollection(
            ids=[["1", "2", "3"]],
            documents=[["doc1", "doc2", "doc3"]],
            distances=[[0.1, 0.2, 0.3]],
            metadatas=[[{"meta": "1"}, {"meta": "2"}, {"meta": "3"}]],
        )

        with patch("legiscope.retrieve.is_relevant", side_effect=mock_assessments):
            mock_client = Mock(spec=Instructor)

            result = filter_results(
                mock_client,
                input_results,
                "test query",
                threshold=0.5,
            )

            # Verify structure using dataclass attributes
            assert hasattr(result, "ids")
            assert hasattr(result, "documents")
            assert hasattr(result, "distances")
            assert hasattr(result, "metadatas")
            assert hasattr(result, "filtering_metadata")

            # Verify filtering results
            assert len(result.ids[0]) == 2  # Only relevant documents
            assert result.ids[0] == ["1", "3"]
            assert result.documents[0] == ["doc1", "doc3"]
            assert result.distances[0] == [0.1, 0.3]
            assert result.metadatas[0] == [{"meta": "1"}, {"meta": "3"}]

            # Verify metadata
            metadata = result.filtering_metadata
            assert metadata.original_count == 3
            assert metadata.filtered_count == 2
            assert metadata.threshold == 0.5
            assert len(metadata.assessments) == 3

    def test_filter_results_with_threshold(self):
        """Test filtering with confidence threshold."""
        # Mock relevance assessments with varying confidence
        mock_assessments = [
            RelevanceAssessment(
                is_relevant=True,
                relevance_score=0.9,
                confidence=0.9,
                reasoning="High confidence",
            ),
            RelevanceAssessment(
                is_relevant=True,
                relevance_score=0.8,
                confidence=0.3,
                reasoning="Low confidence",
            ),
            RelevanceAssessment(
                is_relevant=True,
                relevance_score=0.75,
                confidence=0.7,
                reasoning="Medium confidence",
            ),
        ]

        input_results = SegmentCollection(
            ids=[["1", "2", "3"]],
            documents=[["doc1", "doc2", "doc3"]],
            distances=[[0.1, 0.2, 0.3]],
            metadatas=[[None, None, None]],
        )

        with patch("legiscope.retrieve.is_relevant", side_effect=mock_assessments):
            mock_client = Mock(spec=Instructor)

            result = filter_results(
                mock_client,
                input_results,
                "test query",
            )

            # Only documents 1 and 3 should pass threshold
            assert len(result.ids[0]) == 2
            assert result.ids[0] == ["1", "3"]

    def test_filter_results_no_client(self):
        """Test that missing client raises ValueError."""
        input_results = SegmentCollection(
            ids=[["1"]],
            documents=[["doc1"]],
            distances=[[0.1]],
        )

        with pytest.raises(ValueError, match="client is required"):
            filter_results(None, input_results, "query")  # type: ignore

    def test_filter_results_invalid_structure(self):
        """Test handling of invalid results structure."""
        mock_client = Mock(spec=Instructor)

        # Empty results
        with pytest.raises(ValueError, match="results cannot be None"):
            filter_results(mock_client, None, "query")  # type: ignore

    def test_filter_results_empty_results(self):
        """Test filtering with empty results."""
        empty_results = SegmentCollection(
            ids=[[]],
            documents=[[]],
            distances=[[]],
        )

        mock_client = Mock(spec=Instructor)

        result = filter_results(mock_client, empty_results, "query")

        assert result.filtering_metadata.original_count == 0
        assert result.filtering_metadata.filtered_count == 0
        assert len(result.filtering_metadata.assessments) == 0

    def test_filter_results_assessment_failure(self):
        """Test handling of assessment failures."""

        def failing_assessment(client, query, text, model):
            if text == "doc2":
                raise Exception("Assessment failed")
            return RelevanceAssessment(
                is_relevant=True, relevance_score=0.85, confidence=0.9, reasoning="Good"
            )

        input_results = SegmentCollection(
            ids=[["1", "2", "3"]],
            documents=[["doc1", "doc2", "doc3"]],
            distances=[[0.1, 0.2, 0.3]],
            metadatas=[[None, None, None]],
        )

        with patch("legiscope.retrieve.is_relevant", side_effect=failing_assessment):
            mock_client = Mock(spec=Instructor)

            result = filter_results(mock_client, input_results, "query")

            # Should still work, with failed assessment marked as not relevant
            assert len(result.ids[0]) == 2  # doc1 and doc3
            assert result.ids[0] == ["1", "3"]

            # Check assessment metadata
            assessments = result.filtering_metadata.assessments
            assert len(assessments) == 3

            # Find failed assessment
            failed_assessment = next(a for a in assessments if a["index"] == 1)
            assert failed_assessment["is_relevant"] is False
            assert failed_assessment["confidence"] == 0.0
            assert "Assessment failed" in failed_assessment["reasoning"]

    def test_filter_results_no_metadatas(self):
        """Test filtering when metadatas are missing."""
        input_results = SegmentCollection(
            ids=[["1", "2"]],
            documents=[["doc1", "doc2"]],
            distances=[[0.1, 0.2]],
            metadatas=None,  # Explicitly None
        )

        mock_assessments = [
            RelevanceAssessment(
                is_relevant=True,
                relevance_score=0.85,
                confidence=0.9,
                reasoning="Relevant",
            ),
            RelevanceAssessment(
                is_relevant=False,
                relevance_score=0.2,
                confidence=0.8,
                reasoning="Not relevant",
            ),
        ]

        with patch("legiscope.retrieve.is_relevant", side_effect=mock_assessments):
            mock_client = Mock(spec=Instructor)

            result = filter_results(mock_client, input_results, "query")

            # Should work without metadatas
            assert len(result.ids[0]) == 1
            assert result.ids[0] == ["1"]
            assert result.metadatas is None or result.metadatas == [[None]]


class TestRetrievalConfig:
    """Test RetrievalSettings dataclass."""

    def test_minimal_config(self):
        """Test creating settings with defaults."""
        from legiscope.retrieve import RetrievalSettings

        settings = RetrievalSettings()

        assert settings.n_results == 10  # Default
        assert settings.jurisdiction_id is None
        assert settings.use_hyde is False

    def test_with_jurisdiction(self):
        """Test settings with jurisdiction filter."""
        from legiscope.retrieve import RetrievalSettings

        settings = RetrievalSettings(jurisdiction_id="IL-WindyCity")

        assert settings.jurisdiction_id == "IL-WindyCity"

    def test_with_hyde(self):
        """Test settings with HYDE rewriting enabled."""
        from legiscope.retrieve import RetrievalSettings
        from unittest.mock import Mock

        mock_client = Mock()
        settings = RetrievalSettings(use_hyde=True, hyde_client=mock_client)

        assert settings.use_hyde is True
        assert settings.hyde_client is mock_client

    def test_hyde_without_client_raises_error(self):
        """Test that use_hyde=True without hyde_client raises error."""
        from legiscope.retrieve import RetrievalSettings

        with pytest.raises(ValueError, match="hyde_client required"):
            RetrievalSettings(use_hyde=True)

    def test_empty_query_text_raises_error(self):
        """Test that empty query_text is validated at function call."""
        from legiscope.retrieve import retrieve_segments
        from unittest.mock import Mock

        # query_text validation moved to function, not settings
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            retrieve_segments(
                Mock(),
                "",  # Empty query_text
            )

    def test_invalid_n_results_raises_error(self):
        """Test that invalid n_results raises error."""
        from legiscope.retrieve import RetrievalSettings

        with pytest.raises(ValueError, match="n_results must be positive"):
            RetrievalSettings(n_results=0)


class TestSectionRetrievalConfig:
    """Test SectionRetrievalSettings dataclass."""

    def test_minimal_config(self):
        """Test creating settings with defaults."""
        from legiscope.retrieve import SectionRetrievalSettings

        settings = SectionRetrievalSettings()

        # All inherited from RetrievalSettings
        assert settings.n_results == 10
        assert settings.use_hyde is False

    def test_missing_parquet_path_raises_error(self):
        """Test that sections_parquet_path is now a function parameter."""
        from legiscope.retrieve import retrieve_sections
        from unittest.mock import Mock

        # sections_parquet_path is now a required function parameter
        with pytest.raises(TypeError):
            retrieve_sections(
                Mock(),
                # Missing sections_parquet_path
                "test query",
            )

    def test_inherits_from_retrieval_config(self):
        """Test that SectionRetrievalSettings inherits RetrievalSettings attributes."""
        from legiscope.retrieve import SectionRetrievalSettings
        from unittest.mock import Mock

        mock_client = Mock()
        settings = SectionRetrievalSettings(
            jurisdiction_id="IL-WindyCity",
            n_results=20,
            use_hyde=True,
            hyde_client=mock_client,
        )

        # Check inherited attributes work
        assert settings.jurisdiction_id == "IL-WindyCity"
        assert settings.n_results == 20
        assert settings.use_hyde is True
        assert settings.hyde_client is mock_client


class TestRetrievalConfigBasics:
    """Test RetrievalSettings-based retrieve_segments function."""

    def test_retrieve_segments_with_config_basic(self):
        """Test basic retrieve_segments with settings object."""
        from legiscope.retrieve import retrieve_segments
        from chromadb.api.models.Collection import Collection
        from unittest.mock import Mock, patch

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
        from legiscope.retrieve import RetrievalSettings, retrieve_segments
        from chromadb.api.models.Collection import Collection
        from unittest.mock import Mock, patch

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
        from legiscope.retrieve import RetrievalSettings
        
        with pytest.raises(ValueError, match="hyde_client required"):
            RetrievalSettings(
                use_hyde=True  # Missing hyde_client
            )

    def test_retrieve_segments_with_hyde(self):
        """Test retrieve_segments with HYDE rewriting enabled."""
        from legiscope.retrieve import HydeRewrite, RetrievalSettings, retrieve_segments
        from chromadb.api.models.Collection import Collection
        from unittest.mock import Mock, patch

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
        from legiscope.retrieve import retrieve_sections
        from chromadb.api.models.Collection import Collection
        from unittest.mock import Mock
        
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
        from legiscope.retrieve import retrieve_sections
        from chromadb.api.models.Collection import Collection
        from unittest.mock import Mock, patch

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


class TestGetJurisdictionStats:
    """Test the get_jurisdiction_stats function."""

    def test_stats_with_data(self):
        """Test stats calculation with populated collection."""
        from legiscope.retrieve import get_jurisdiction_stats
        from unittest.mock import Mock

        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"jurisdiction_id": "J1", "state": "S1", "municipality": "M1"},
                {"jurisdiction_id": "J1", "state": "S1", "municipality": "M2"},
                {"jurisdiction_id": "J2", "state": "S2", "municipality": "M3"},
                None,  # Should be handled
                {},  # Should be handled
            ]
        }

        stats = get_jurisdiction_stats(mock_collection)

        assert stats.total_documents == 5
        assert stats.jurisdictions == {"J1": 2, "J2": 1}
        assert stats.states == {"S1": 2, "S2": 1}
        assert stats.municipalities == {"M1": 1, "M2": 1, "M3": 1}

    def test_stats_empty_collection(self):
        """Test stats with empty collection."""
        from legiscope.retrieve import get_jurisdiction_stats
        from unittest.mock import Mock

        mock_collection = Mock()
        mock_collection.get.return_value = {"metadatas": []}

        stats = get_jurisdiction_stats(mock_collection)

        assert stats.total_documents == 0
        assert stats.jurisdictions == {}

    def test_stats_error_handling(self):
        """Test that errors return empty stats."""
        from legiscope.retrieve import get_jurisdiction_stats
        from unittest.mock import Mock

        mock_collection = Mock()
        mock_collection.get.side_effect = Exception("DB Error")

        stats = get_jurisdiction_stats(mock_collection)

        assert stats.total_documents == 0
        assert stats.jurisdictions == {}


class TestFilterSections:
    """Test the filter_sections function."""

    def test_filter_sections_basic(self):
        """Test basic section filtering."""
        from legiscope.retrieve import (
            SectionCollection,
            SectionResult,
            QueryInfo,
            RelevanceAssessment,
            filter_sections,
        )
        from unittest.mock import Mock, patch

        # Mock sections
        sections = [
            SectionResult(
                section_idx=1,
                heading_text="H1",
                body_text="B1",
                heading_level=1,
                parent=None,
                matching_segments=[],
                relevance_score=0.1,
                segment_count=1,
            ),
            SectionResult(
                section_idx=2,
                heading_text="H2",
                body_text="B2",
                heading_level=1,
                parent=None,
                matching_segments=[],
                relevance_score=0.2,
                segment_count=1,
            ),
        ]

        input_results = SectionCollection(
            sections=sections, query_info=QueryInfo(original_query="query")
        )

        # Mock assessments
        mock_assessments = [
            RelevanceAssessment(
                is_relevant=True,
                relevance_score=0.9,
                confidence=0.9,
                reasoning="Good",
            ),
            RelevanceAssessment(
                is_relevant=False,
                relevance_score=0.1,
                confidence=0.9,
                reasoning="Bad",
            ),
        ]

        with patch("legiscope.retrieve.is_relevant", side_effect=mock_assessments):
            # Patch Path.mkdir to avoid filesystem side effects from debug logging
            with patch(
                "pathlib.Path.mkdir"
            ), patch("legiscope.retrieve.pl.DataFrame.write_csv"):
                mock_client = Mock(spec=Instructor)

                result = filter_sections(
                    mock_client, input_results, "query", confidence_threshold=0.5
                )

                assert len(result.sections) == 1
                assert result.sections[0].section_idx == 1
                assert result.sections[0].relevance_score == 0.9  # Updated from LLM
                assert result.sections[0].llm_assessed is True
                assert result.filtering_metadata.filtered_count == 1

    def test_filter_sections_sorting(self):
        """Test that results are sorted by LLM relevance score."""
        from legiscope.retrieve import (
            SectionCollection,
            SectionResult,
            QueryInfo,
            RelevanceAssessment,
            filter_sections,
        )
        from unittest.mock import Mock, patch

        sections = [
            SectionResult(
                section_idx=1,
                heading_text="H1",
                body_text="B1",
                heading_level=1,
                parent=None,
                matching_segments=[],
                relevance_score=0,
                segment_count=1,
            ),
            SectionResult(
                section_idx=2,
                heading_text="H2",
                body_text="B2",
                heading_level=1,
                parent=None,
                matching_segments=[],
                relevance_score=0,
                segment_count=1,
            ),
        ]

        input_results = SectionCollection(
            sections=sections, query_info=QueryInfo(original_query="query")
        )

        mock_assessments = [
            RelevanceAssessment(
                is_relevant=True,
                relevance_score=0.7,
                confidence=0.9,
                reasoning="Okay",
            ),
            RelevanceAssessment(
                is_relevant=True,
                relevance_score=0.9,
                confidence=0.9,
                reasoning="Great",
            ),
        ]

        with patch("legiscope.retrieve.is_relevant", side_effect=mock_assessments):
            with patch(
                "pathlib.Path.mkdir"
            ), patch("legiscope.retrieve.pl.DataFrame.write_csv"):
                mock_client = Mock(spec=Instructor)
                result = filter_sections(mock_client, input_results, "query")

                # Should be sorted by score descending (Great > Okay)
                assert len(result.sections) == 2
                assert result.sections[0].section_idx == 2  # Score 0.9
                assert result.sections[1].section_idx == 1  # Score 0.7

    def test_filter_sections_validation(self):
        """Test input validation."""
        from legiscope.retrieve import filter_sections
        from unittest.mock import Mock

        with pytest.raises(ValueError, match="sections_results cannot be None"):
            filter_sections(Mock(), None, "query")  # type: ignore

        with pytest.raises(ValueError, match="client is required"):
            filter_sections(None, Mock(), "query")  # type: ignore



