"""
Tests for the retrieve module, including HYDE functionality.
"""

from unittest.mock import Mock, patch

import pytest
from instructor import Instructor

from legiscope.llm_config import Config
from legiscope.retrieval_guidance import RetrievalGuidance
from legiscope.retrieve import (
    HydeRewrite,
    RelevanceAssessment,
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
            assert call_args[1]["model"] == Config.get_fast_model()

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
            relevance_score=0.75,
            reasoning="The text directly addresses parking regulations with specific rules",
        )

        assert assessment.relevance_score == 0.75
        assert (
            assessment.reasoning
            == "The text directly addresses parking regulations with specific rules"
        )

    def test_relevance_assessment_model_relevance_score_bounds(self):
        """Test relevance score bounds validation."""
        assessment1 = RelevanceAssessment(
            relevance_score=0.0,
            reasoning="Test",
        )
        assert assessment1.relevance_score == 0.0

        assessment2 = RelevanceAssessment(
            relevance_score=1.0,
            reasoning="Test",
        )
        assert assessment2.relevance_score == 1.0

    def test_relevance_assessment_model_invalid_relevance_score(self):
        """Test that invalid relevance scores are rejected."""
        with pytest.raises(ValueError):
            RelevanceAssessment(
                relevance_score=-0.1,
                reasoning="Test",
            )

        with pytest.raises(ValueError):
            RelevanceAssessment(
                relevance_score=1.1,
                reasoning="Test",
            )


class TestIsRelevant:
    """Test the is_relevant function."""

    def test_is_relevant_success(self):
        """Test successful relevance assessment."""
        mock_result = RelevanceAssessment(
            relevance_score=0.85,
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
            assert result.relevance_score == 0.85
            assert "parking regulations" in result.reasoning

            # Verify ask was called correctly
            mock_ask.assert_called_once()
            call_args = mock_ask.call_args
            assert call_args[1]["client"] == mock_client
            assert "parking regulations" in call_args[1]["prompt"]
            assert call_args[1]["response_model"] == RelevanceAssessment
            assert call_args[1]["model"] == Config.get_fast_model()

    def test_is_relevant_custom_model(self):
        """Test relevance assessment with custom model."""
        mock_result = RelevanceAssessment(
            relevance_score=0.2,
            reasoning="The text discusses unrelated topics",
        )

        with patch("legiscope.retrieve.ask", return_value=mock_result) as mock_ask:
            mock_client = Mock(spec=Instructor)

            is_relevant(mock_client, "test query", "test text", model="gpt-4")

            # Verify custom model was used
            mock_ask.assert_called_once()
            call_args = mock_ask.call_args
            assert call_args[1]["model"] == "gpt-4"

    def test_is_relevant_includes_retrieval_guidance(self):
        """Project-provided retrieval guidance should be injected into the prompt."""
        mock_result = RelevanceAssessment(
            relevance_score=0.9,
            reasoning="The text directly answers the date query.",
        )

        guidance = RetrievalGuidance(
            guidance_topic="date",
            shared_context=(
                "This query concerns a local municipal ordinance regulating "
                "drug paraphernalia-related activities."
            ),
            retrieval_instructions=(
                "Retrieve effective-date clauses tied to the ordinance."
            ),
            relevance_instructions="Prefer enactment and effective-date language.",
            anchor_terms=["effective", "enacted"],
        )

        with patch("legiscope.retrieve.ask", return_value=mock_result) as mock_ask:
            mock_client = Mock(spec=Instructor)

            is_relevant(
                mock_client,
                "When did this ordinance take effect?",
                "This ordinance shall take effect 30 days after becoming law.",
                retrieval_guidance=guidance,
            )

            call_args = mock_ask.call_args
            assert "Topic focus for this query: date." in call_args[1]["system"]
            assert (
                "Query context: This query concerns a local municipal ordinance regulating drug paraphernalia-related activities."
                in call_args[1]["system"]
            )
            assert (
                "Prefer enactment and effective-date language."
                in call_args[1]["system"]
            )
            assert "effective, enacted" in call_args[1]["system"]

    def test_is_relevant_includes_threshold_and_date_metadata_retention_guidance(self):
        """Date and metadata queries should explain the keep threshold in the prompt."""
        mock_result = RelevanceAssessment(
            relevance_score=0.9,
            reasoning="The text contains effective-date metadata.",
        )

        with patch("legiscope.retrieve.ask", return_value=mock_result) as mock_ask:
            mock_client = Mock(spec=Instructor)

            is_relevant(
                mock_client,
                "On which date did the ordinance go into effect?",
                "This ordinance shall take effect 30 days after adoption.",
                relevance_threshold=0.7,
            )

            call_args = mock_ask.call_args
            assert "The keep threshold for this run is 0.70." in call_args[1]["system"]
            assert "should usually score at or above 0.70" in call_args[1]["system"]
            assert "Retention threshold: 0.70" in call_args[1]["prompt"]

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


class TestRetrievalConfig:
    """Test RetrievalSettings dataclass."""

    def test_minimal_config(self):
        """Test creating settings with defaults."""
        from legiscope.retrieve import (
            RetrievalSettings,
            DEFAULT_N_RESULTS,
            DEFAULT_HYDE_ENABLED,
        )

        settings = RetrievalSettings()

        assert settings.n_results == DEFAULT_N_RESULTS
        assert settings.jurisdiction_id is None
        assert settings.use_hyde == DEFAULT_HYDE_ENABLED

    def test_with_jurisdiction(self):
        """Test settings with jurisdiction filter."""
        from legiscope.retrieve import RetrievalSettings

        settings = RetrievalSettings(jurisdiction_id="IL-WindyTown")

        assert settings.jurisdiction_id == "IL-WindyTown"

    def test_with_hyde(self):
        """Test settings with HYDE rewriting enabled."""
        from unittest.mock import Mock

        from legiscope.retrieve import RetrievalSettings

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
        from unittest.mock import Mock

        from legiscope.retrieve import retrieve_segments

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
        from legiscope.retrieve import (
            SectionRetrievalSettings,
            DEFAULT_N_RESULTS,
            DEFAULT_HYDE_ENABLED,
            DEFAULT_LEXICAL_RERANKING_ENABLED,
        )

        settings = SectionRetrievalSettings()

        # All inherited from RetrievalSettings
        assert settings.n_results == DEFAULT_N_RESULTS
        assert settings.use_hyde == DEFAULT_HYDE_ENABLED
        assert settings.use_lexical_reranking == DEFAULT_LEXICAL_RERANKING_ENABLED

    def test_missing_parquet_path_raises_error(self):
        """Test that sections_parquet_path is now a function parameter."""
        from unittest.mock import Mock

        from legiscope.retrieve import retrieve_sections

        # sections_parquet_path is now a required function parameter
        with pytest.raises(TypeError):
            retrieve_sections(
                Mock(),
                # Missing sections_parquet_path
                "test query",
            )

    def test_inherits_from_retrieval_config(self):
        """Test that SectionRetrievalSettings inherits RetrievalSettings attributes."""
        from unittest.mock import Mock

        from legiscope.retrieve import SectionRetrievalSettings

        mock_client = Mock()
        settings = SectionRetrievalSettings(
            jurisdiction_id="IL-WindyTown",
            n_results=20,
            use_hyde=True,
            hyde_client=mock_client,
        )

        # Check inherited attributes work
        assert settings.jurisdiction_id == "IL-WindyTown"
        assert settings.n_results == 20
        assert settings.use_hyde is True
        assert settings.hyde_client is mock_client

    def test_can_enable_lexical_reranking_explicitly(self):
        """Lexical reranking remains available behind an explicit settings flag."""
        from legiscope.retrieve import SectionRetrievalSettings

        settings = SectionRetrievalSettings(use_lexical_reranking=True)

        assert settings.use_lexical_reranking is True


class TestRetrievalConfigBasics:
    """Test RetrievalSettings-based retrieve_segments function."""

    def test_retrieve_segments_with_config_basic(self):
        """Test basic retrieve_segments with settings object."""
        from unittest.mock import Mock, patch

        from chromadb.api.models.Collection import Collection

        from legiscope.retrieve import retrieve_segments

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
        from unittest.mock import Mock, patch

        from chromadb.api.models.Collection import Collection

        from legiscope.retrieve import RetrievalSettings, retrieve_segments

        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"jurisdiction_id": "IL-WindyTown"}]],
            "distances": [[0.1]],
        }

        with patch("legiscope.retrieve.get_embedding_client"):
            with patch("legiscope.retrieve.get_embeddings") as mock_get_embeddings:
                mock_get_embeddings.return_value = [[0.1, 0.2, 0.3]]

                settings = RetrievalSettings(
                    jurisdiction_id="IL-WindyTown", n_results=5
                )

                retrieve_segments(mock_collection, "test query", settings)

                # Check that query was called with where filter
                call_kwargs = mock_collection.query.call_args.kwargs
                assert "where" in call_kwargs
                assert call_kwargs["where"] == {"jurisdiction_id": "IL-WindyTown"}

    def test_retrieve_segments_hyde_requires_client(self):
        """Test that use_hyde=True requires hyde_client."""
        from legiscope.retrieve import RetrievalSettings

        with pytest.raises(ValueError, match="hyde_client required"):
            RetrievalSettings(
                use_hyde=True  # Missing hyde_client
            )

    def test_retrieve_segments_with_hyde(self):
        """Test retrieve_segments with HYDE rewriting enabled."""
        from unittest.mock import Mock, patch

        from chromadb.api.models.Collection import Collection

        from legiscope.retrieve import HydeRewrite, RetrievalSettings, retrieve_segments

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
        from unittest.mock import Mock

        from chromadb.api.models.Collection import Collection

        from legiscope.retrieve import retrieve_sections

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
        from unittest.mock import Mock, patch

        import polars as pl
        from chromadb.api.models.Collection import Collection

        from legiscope.retrieve import retrieve_sections

        # Create test sections parquet file
        sections_data = {
            "section_ordinal": [0, 1],
            "heading_text": ["# Section 1", "## Section 2"],
            "body_text": ["Content 1", "Content 2"],
            "heading_level": [1, 2],
            "parent_id": [None, "s0"],
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
                        "section_ordinal": 0,
                        "segment_position": 0,
                        "section_heading": "# Section 1",
                        "section_level": 1,
                    },
                    {
                        "section_ordinal": 1,
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

    def test_retrieve_sections_prefers_chunks_when_metadata_available(self, tmp_path):
        """Sibling chunks.parquet should drive retrieval context when chunk IDs exist."""
        from unittest.mock import Mock, patch

        import polars as pl
        from chromadb.api.models.Collection import Collection

        from legiscope.retrieve import retrieve_sections

        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Section 1"],
                "body_text": ["Full section content that should not be returned."],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(tmp_path / "sections.parquet")

        pl.DataFrame(
            {
                "chunk_ordinal": [0],
                "chunk_id": ["chunk-0"],
                "section_ordinal": [0],
                "section_id": ["s0"],
                "heading_text": ["# Section 1"],
                "body_text": ["Chunk-scoped content only."],
                "heading_level": [1],
                "parent_id": [None],
                "line_number": [1],
                "context_path": ["Section 1"],
                "source_kind": ["section_subtree"],
                "region_role": ["main_body"],
                "retrieval_priority": [3],
                "chunk_part": [1],
                "chunk_count": [1],
                "section_type": ["section"],
                "section_number": ["1"],
                "token_count": [12],
            }
        ).write_parquet(tmp_path / "chunks.parquet")

        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["0"]],
            "documents": [["seg1"]],
            "metadatas": [
                [
                    {
                        "chunk_id": "chunk-0",
                        "chunk_ordinal": 0,
                        "section_ordinal": 0,
                        "segment_position": 0,
                        "section_heading": "# Section 1",
                        "section_level": 1,
                    }
                ]
            ],
            "distances": [[0.1]],
        }

        with patch("legiscope.retrieve.get_embedding_client"):
            with patch("legiscope.retrieve.get_embeddings") as mock_embeddings:
                mock_embeddings.return_value = [[0.1, 0.2, 0.3]]

                results = retrieve_sections(
                    mock_collection,
                    str(tmp_path / "sections.parquet"),
                    "test query",
                )

        assert len(results.sections) == 1
        assert results.sections[0].section_id == "chunk-0"
        assert results.sections[0].chunk_id == "chunk-0"
        assert results.sections[0].body_text == "Chunk-scoped content only."

    def test_retrieve_sections_keeps_chunk_heading_for_chunk_match(self, tmp_path):
        """Chunk-backed retrieval should keep the chunk's stored heading text."""
        from unittest.mock import Mock, patch

        import polars as pl
        from chromadb.api.models.Collection import Collection

        from legiscope.retrieve import retrieve_sections

        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["### CHAPTER 39"],
                "body_text": ["Chapter intro."],
                "heading_level": [3],
                "parent_id": [None],
            }
        ).write_parquet(tmp_path / "sections.parquet")

        pl.DataFrame(
            {
                "chunk_ordinal": [0],
                "chunk_id": ["chunk-0"],
                "section_ordinal": [0],
                "section_id": ["s0"],
                "heading_text": ["### CHAPTER 39"],
                "body_text": [
                    "#### § 6-301. Purpose.\n\nPurpose text.\n\n#### § 6-302. Scope.\n\nScope text."
                ],
                "heading_level": [3],
                "parent_id": [None],
                "line_number": [1],
                "context_path": ["TITLE 6 > CHAPTER 39"],
                "source_kind": ["section_neighborhood"],
                "region_role": ["main_body"],
                "retrieval_priority": [4],
                "chunk_part": [1],
                "chunk_count": [1],
                "section_type": ["chapter"],
                "section_number": ["39"],
                "token_count": [40],
            }
        ).write_parquet(tmp_path / "chunks.parquet")

        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["0"]],
            "documents": [["#### § 6-302. Scope.\n\nScope text."]],
            "metadatas": [
                [
                    {
                        "chunk_id": "chunk-0",
                        "chunk_ordinal": 0,
                        "section_ordinal": 0,
                        "segment_position": 1,
                        "section_heading": "### CHAPTER 39",
                        "section_level": 3,
                    }
                ]
            ],
            "distances": [[0.1]],
        }

        with patch("legiscope.retrieve.get_embedding_client"):
            with patch("legiscope.retrieve.get_embeddings") as mock_embeddings:
                mock_embeddings.return_value = [[0.1, 0.2, 0.3]]

                results = retrieve_sections(
                    mock_collection,
                    str(tmp_path / "sections.parquet"),
                    "scope query",
                )

        assert len(results.sections) == 1
        assert results.sections[0].heading_text == "### CHAPTER 39"
        assert results.sections[0].heading_level == 3

    def test_retrieve_sections_keeps_semantic_order_when_lexical_hints_present(
        self, tmp_path
    ):
        """Lexical hints should not change ordering when lexical reranking is disabled."""
        from unittest.mock import Mock, patch

        import polars as pl
        from chromadb.api.models.Collection import Collection

        from legiscope.retrieve import SectionRetrievalSettings, retrieve_sections

        pl.DataFrame(
            {
                "section_ordinal": [0, 1],
                "section_id": ["s0", "s1"],
                "heading_text": ["# General zoning", "# Drug paraphernalia"],
                "body_text": [
                    "Accessory-use regulations for commercial zoning districts.",
                    "It is unlawful to offer for sale or sell drug paraphernalia.",
                ],
                "heading_level": [1, 1],
                "parent_id": [None, None],
                "context_path": ["Title 14 > Zoning", "Title 9 > Chapter 9-600"],
            }
        ).write_parquet(tmp_path / "sections.parquet")

        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["0", "1"]],
            "documents": [["seg zoning", "seg paraphernalia"]],
            "metadatas": [
                [
                    {
                        "section_ordinal": 0,
                        "segment_position": 0,
                        "section_heading": "# General zoning",
                        "section_level": 1,
                    },
                    {
                        "section_ordinal": 1,
                        "segment_position": 0,
                        "section_heading": "# Drug paraphernalia",
                        "section_level": 1,
                    },
                ]
            ],
            "distances": [[0.05, 0.2]],
        }

        settings = SectionRetrievalSettings(
            n_results=1,
            lexical_query_text="drug paraphernalia offer for sale",
            anchor_terms=["drug paraphernalia", "offer for sale"],
            use_lexical_reranking=False,
        )

        with patch("legiscope.retrieve.get_embedding_client"):
            with patch("legiscope.retrieve.get_embeddings") as mock_embeddings:
                mock_embeddings.return_value = [[0.1, 0.2, 0.3]]

                results = retrieve_sections(
                    mock_collection,
                    str(tmp_path / "sections.parquet"),
                    "semantic query text",
                    settings,
                )

        assert len(results.sections) == 1
        assert results.sections[0].section_id == "s0"
        assert mock_collection.query.call_args.kwargs["n_results"] == 1

    def test_retrieve_sections_caps_overfetched_retrieval_units(self, tmp_path):
        """Lexical hints should not trigger overfetch when lexical reranking is disabled."""
        from unittest.mock import Mock, patch

        import polars as pl
        from chromadb.api.models.Collection import Collection

        from legiscope.retrieve import SectionRetrievalSettings, retrieve_sections

        pl.DataFrame(
            {
                "section_ordinal": [0, 1, 2, 3, 4, 5],
                "section_id": [f"s{i}" for i in range(6)],
                "heading_text": [f"# Section {i}" for i in range(6)],
                "body_text": [f"Content {i}" for i in range(6)],
                "heading_level": [1, 1, 1, 1, 1, 1],
                "parent_id": [None, None, None, None, None, None],
            }
        ).write_parquet(tmp_path / "sections.parquet")

        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [[str(i) for i in range(6)]],
            "documents": [[f"seg{i}" for i in range(6)]],
            "metadatas": [
                [
                    {
                        "section_ordinal": i,
                        "segment_position": 0,
                        "section_heading": f"# Section {i}",
                        "section_level": 1,
                    }
                    for i in range(6)
                ]
            ],
            "distances": [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]],
        }

        settings = SectionRetrievalSettings(
            n_results=2,
            lexical_query_text="section content",
            anchor_terms=["content"],
            use_lexical_reranking=False,
        )

        with patch("legiscope.retrieve.get_embedding_client"):
            with patch("legiscope.retrieve.get_embeddings") as mock_embeddings:
                mock_embeddings.return_value = [[0.1, 0.2, 0.3]]

                results = retrieve_sections(
                    mock_collection,
                    str(tmp_path / "sections.parquet"),
                    "semantic query text",
                    settings,
                )

        assert len(results.sections) == 2
        assert results.query_info.total_segments_found == 6
        assert results.query_info.unique_sections == 2
        assert [section.section_id for section in results.sections] == ["s0", "s1"]
        assert mock_collection.query.call_args.kwargs["n_results"] == 2

    def test_retrieve_sections_lexically_reranks_when_enabled(self, tmp_path):
        """Enabled lexical reranking should overfetch, rerank, and keep the top semantic+lexical result."""
        from unittest.mock import Mock, patch

        import polars as pl
        from chromadb.api.models.Collection import Collection

        from legiscope.retrieve import SectionRetrievalSettings, retrieve_sections

        pl.DataFrame(
            {
                "section_ordinal": [0, 1, 2],
                "section_id": ["s0", "s1", "s2"],
                "heading_text": [
                    "# General zoning",
                    "# Drug paraphernalia",
                    "# Licensing",
                ],
                "body_text": [
                    "Accessory-use regulations for commercial zoning districts.",
                    "It is unlawful to offer for sale or sell drug paraphernalia.",
                    "General licensing provisions.",
                ],
                "heading_level": [1, 1, 1],
                "parent_id": [None, None, None],
                "context_path": [
                    "Title 14 > Zoning",
                    "Title 9 > Chapter 9-600",
                    "Title 3 > Licensing",
                ],
            }
        ).write_parquet(tmp_path / "sections.parquet")

        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["0", "1", "2"]],
            "documents": [["seg zoning", "seg paraphernalia", "seg licensing"]],
            "metadatas": [
                [
                    {
                        "section_ordinal": 0,
                        "segment_position": 0,
                        "section_heading": "# General zoning",
                        "section_level": 1,
                    },
                    {
                        "section_ordinal": 1,
                        "segment_position": 0,
                        "section_heading": "# Drug paraphernalia",
                        "section_level": 1,
                    },
                    {
                        "section_ordinal": 2,
                        "segment_position": 0,
                        "section_heading": "# Licensing",
                        "section_level": 1,
                    },
                ]
            ],
            "distances": [[0.05, 0.2, 0.4]],
        }

        settings = SectionRetrievalSettings(
            n_results=1,
            lexical_query_text="drug paraphernalia offer for sale",
            anchor_terms=["drug paraphernalia", "offer for sale"],
            use_lexical_reranking=True,
        )

        with patch("legiscope.retrieve.get_embedding_client"):
            with patch("legiscope.retrieve.get_embeddings") as mock_embeddings:
                mock_embeddings.return_value = [[0.1, 0.2, 0.3]]

                results = retrieve_sections(
                    mock_collection,
                    str(tmp_path / "sections.parquet"),
                    "semantic query text",
                    settings,
                )

        assert len(results.sections) == 1
        assert results.sections[0].section_id == "s1"
        assert mock_collection.query.call_args.kwargs["n_results"] == 3

    def test_retrieve_sections_lexically_reranks_ssp_synonyms_when_enabled(
        self, tmp_path
    ):
        """SSP lexical anchors should lift ordinance-specific program synonyms over generic health text."""
        from unittest.mock import Mock, patch

        import polars as pl
        from chromadb.api.models.Collection import Collection

        from legiscope.retrieve import SectionRetrievalSettings, retrieve_sections

        pl.DataFrame(
            {
                "section_ordinal": [0, 1, 2],
                "section_id": ["s0", "s1", "s2"],
                "heading_text": [
                    "# Public health findings",
                    "# Syringe exchange facility",
                    "# General permits",
                ],
                "body_text": [
                    "Harm reduction programs may improve public health outcomes and reduce disease transmission.",
                    "A sterile needle and needle exchange program may operate only with a permit for the syringe exchange facility.",
                    "Permit applications must include the applicant name and address.",
                ],
                "heading_level": [1, 1, 1],
                "parent_id": [None, None, None],
                "context_path": [
                    "Title 9 > Health",
                    "Title 9 > Syringe Exchange Facility Location",
                    "Title 3 > Licensing",
                ],
            }
        ).write_parquet(tmp_path / "sections.parquet")

        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["0", "1", "2"]],
            "documents": [["seg health", "seg ssp", "seg permit"]],
            "metadatas": [
                [
                    {
                        "section_ordinal": 0,
                        "segment_position": 0,
                        "section_heading": "# Public health findings",
                        "section_level": 1,
                    },
                    {
                        "section_ordinal": 1,
                        "segment_position": 0,
                        "section_heading": "# Syringe exchange facility",
                        "section_level": 1,
                    },
                    {
                        "section_ordinal": 2,
                        "segment_position": 0,
                        "section_heading": "# General permits",
                        "section_level": 1,
                    },
                ]
            ],
            "distances": [[0.1, 0.14, 0.25]],
        }

        settings = SectionRetrievalSettings(
            n_results=1,
            lexical_query_text=(
                "Does the jurisdiction have a law that authorizes, prohibits, or limits syringe service programs (SSPs)?"
            ),
            anchor_terms=[
                "syringe service program",
                "syringe exchange facility",
                "syringe exchange program",
                "needle exchange program",
                "sterile needle",
            ],
            use_lexical_reranking=True,
        )

        with patch("legiscope.retrieve.get_embedding_client"):
            with patch("legiscope.retrieve.get_embeddings") as mock_embeddings:
                mock_embeddings.return_value = [[0.1, 0.2, 0.3]]

                results = retrieve_sections(
                    mock_collection,
                    str(tmp_path / "sections.parquet"),
                    "semantic query text",
                    settings,
                )

        assert len(results.sections) == 1
        assert results.sections[0].section_id == "s1"
        assert mock_collection.query.call_args.kwargs["n_results"] == 3


class TestGetJurisdictionStats:
    """Test the get_jurisdiction_stats function."""

    def test_stats_with_data(self):
        """Test stats calculation with populated collection."""
        from unittest.mock import Mock

        from legiscope.retrieve import get_jurisdiction_stats

        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"jurisdiction_id": "J1", "state": "S1", "locality": "M1"},
                {"jurisdiction_id": "J1", "state": "S1", "locality": "M2"},
                {"jurisdiction_id": "J2", "state": "S2", "locality": "M3"},
                None,  # Should be handled
                {},  # Should be handled
            ]
        }

        stats = get_jurisdiction_stats(mock_collection)

        assert stats.total_documents == 5
        assert stats.jurisdictions == {"J1": 2, "J2": 1}
        assert stats.states == {"S1": 2, "S2": 1}
        assert stats.localities == {"M1": 1, "M2": 1, "M3": 1}

    def test_stats_empty_collection(self):
        """Test stats with empty collection."""
        from unittest.mock import Mock

        from legiscope.retrieve import get_jurisdiction_stats

        mock_collection = Mock()
        mock_collection.get.return_value = {"metadatas": []}

        stats = get_jurisdiction_stats(mock_collection)

        assert stats.total_documents == 0
        assert stats.jurisdictions == {}

    def test_stats_error_handling(self):
        """Test that errors return empty stats."""
        from unittest.mock import Mock

        from legiscope.retrieve import get_jurisdiction_stats

        mock_collection = Mock()
        mock_collection.get.side_effect = Exception("DB Error")

        stats = get_jurisdiction_stats(mock_collection)

        assert stats.total_documents == 0
        assert stats.jurisdictions == {}


class TestFilterSections:
    """Test the filter_sections function."""

    def test_filter_sections_basic(self):
        """Test basic section filtering."""
        from unittest.mock import Mock, patch

        from legiscope.retrieve import (
            QueryInfo,
            RelevanceAssessment,
            SectionCollection,
            SectionResult,
            filter_sections,
        )

        # Mock sections
        sections = [
            SectionResult(
                section_id="s1",
                heading_text="H1",
                body_text="B1",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.1,
                segment_count=1,
            ),
            SectionResult(
                section_id="s2",
                heading_text="H2",
                body_text="B2",
                heading_level=1,
                parent_id=None,
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
                relevance_score=0.9,
                reasoning="Good",
            ),
            RelevanceAssessment(
                relevance_score=0.1,
                reasoning="Bad",
            ),
        ]

        with patch("legiscope.retrieve.is_relevant", side_effect=mock_assessments):
            # Patch Path.mkdir to avoid filesystem side effects from debug logging
            with (
                patch("pathlib.Path.mkdir"),
                patch("legiscope.retrieve.pl.DataFrame.write_csv"),
            ):
                mock_client = Mock(spec=Instructor)

                result = filter_sections(
                    mock_client, input_results, "query", relevance_threshold=0.5
                )

                assert len(result.sections) == 1
                assert result.sections[0].section_id == "s1"
                assert result.sections[0].relevance_score == 0.9  # Updated from LLM
                assert result.sections[0].llm_assessed is True
                assert result.filtering_metadata.filtered_count == 1

    def test_filter_sections_backfills_borderline_sections_when_score_just_misses_threshold(
        self,
    ):
        """A borderline score should survive through backfill when it just misses threshold."""
        from unittest.mock import Mock, patch

        from legiscope.retrieve import (
            QueryInfo,
            RelevanceAssessment,
            SectionCollection,
            SectionResult,
            filter_sections,
        )

        sections = [
            SectionResult(
                section_id="s1",
                heading_text="H1",
                body_text="B1",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.1,
                segment_count=1,
            )
        ]

        input_results = SectionCollection(
            sections=sections, query_info=QueryInfo(original_query="query")
        )

        mock_assessment = RelevanceAssessment(
            relevance_score=0.65,
            reasoning="Related but not specific enough",
        )

        with patch("legiscope.retrieve.is_relevant", return_value=mock_assessment):
            with (
                patch("pathlib.Path.mkdir"),
                patch("legiscope.retrieve.pl.DataFrame.write_csv"),
            ):
                mock_client = Mock(spec=Instructor)
                result = filter_sections(
                    mock_client, input_results, "query", relevance_threshold=0.7
                )

                assert len(result.sections) == 1
                assert result.filtering_metadata.filtered_count == 1
                assert (
                    result.filtering_metadata.assessments[0]["keep_reason"]
                    == "backfill"
                )

    def test_filter_sections_keeps_relevant_sections_when_score_clears_threshold(self):
        """A high-score section should survive when the score clears threshold."""
        from unittest.mock import Mock, patch

        from legiscope.retrieve import (
            QueryInfo,
            RelevanceAssessment,
            SectionCollection,
            SectionResult,
            filter_sections,
        )

        sections = [
            SectionResult(
                section_id="s1",
                heading_text="H1",
                body_text="B1",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.1,
                segment_count=1,
            )
        ]

        input_results = SectionCollection(
            sections=sections, query_info=QueryInfo(original_query="query")
        )

        mock_assessment = RelevanceAssessment(
            relevance_score=0.95,
            reasoning="Specific but uncertain",
        )

        with patch("legiscope.retrieve.is_relevant", return_value=mock_assessment):
            with (
                patch("pathlib.Path.mkdir"),
                patch("legiscope.retrieve.pl.DataFrame.write_csv"),
            ):
                mock_client = Mock(spec=Instructor)
                result = filter_sections(
                    mock_client, input_results, "query", relevance_threshold=0.7
                )

                assert len(result.sections) == 1
                assert result.filtering_metadata.filtered_count == 1
                assert (
                    result.filtering_metadata.assessments[0]["keep_reason"]
                    == "threshold"
                )

    def test_filter_sections_backfills_relevant_sections_when_scores_are_borderline(
        self,
    ):
        """Soft filtering should retain a small evidence set when all relevant scores are borderline."""
        from unittest.mock import Mock, patch

        from legiscope.retrieve import (
            QueryInfo,
            RelevanceAssessment,
            SectionCollection,
            SectionResult,
            filter_sections,
        )

        sections = [
            SectionResult(
                section_id="s1",
                heading_text="H1",
                body_text="B1",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.1,
                segment_count=1,
            ),
            SectionResult(
                section_id="s2",
                heading_text="H2",
                body_text="B2",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.2,
                segment_count=1,
            ),
        ]

        input_results = SectionCollection(
            sections=sections, query_info=QueryInfo(original_query="query")
        )

        mock_assessments = [
            RelevanceAssessment(
                relevance_score=0.64,
                reasoning="Useful but not confidently above threshold",
            ),
            RelevanceAssessment(
                relevance_score=0.61,
                reasoning="Also useful but borderline",
            ),
        ]

        with patch("legiscope.retrieve.is_relevant", side_effect=mock_assessments):
            with (
                patch("pathlib.Path.mkdir"),
                patch("legiscope.retrieve.pl.DataFrame.write_csv"),
            ):
                mock_client = Mock(spec=Instructor)
                result = filter_sections(
                    mock_client, input_results, "query", relevance_threshold=0.7
                )

                assert len(result.sections) == 2
                assert result.filtering_metadata.filtered_count == 2
                keep_reasons = {
                    row["section_id"]: row["keep_reason"]
                    for row in result.filtering_metadata.assessments
                }
                assert keep_reasons == {"s1": "backfill", "s2": "backfill"}

    def test_filter_sections_respects_guidance_that_disables_backfill(self):
        """Family-specific guidance should be able to prevent borderline rescue."""
        from unittest.mock import Mock, patch

        from legiscope.retrieve import (
            QueryInfo,
            RelevanceAssessment,
            SectionCollection,
            SectionResult,
            filter_sections,
        )
        from legiscope.retrieval_guidance import RetrievalGuidance

        sections = [
            SectionResult(
                section_id="s1",
                heading_text="H1",
                body_text="B1",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.1,
                segment_count=1,
            )
        ]

        input_results = SectionCollection(
            sections=sections, query_info=QueryInfo(original_query="query")
        )

        mock_assessment = RelevanceAssessment(
            relevance_score=0.65,
            reasoning="Borderline only",
        )

        with patch("legiscope.retrieve.is_relevant", return_value=mock_assessment):
            with (
                patch("pathlib.Path.mkdir"),
                patch("legiscope.retrieve.pl.DataFrame.write_csv"),
            ):
                mock_client = Mock(spec=Instructor)
                result = filter_sections(
                    mock_client,
                    input_results,
                    "query",
                    relevance_threshold=0.7,
                    retrieval_guidance=RetrievalGuidance(
                        guidance_topic="ssp_scope",
                        enable_relevance_backfill=False,
                    ),
                )

                assert len(result.sections) == 0
                assert result.filtering_metadata.filtered_count == 0
                assert (
                    result.filtering_metadata.assessments[0]["keep_reason"]
                    == "below_threshold"
                )

    def test_filter_sections_sorting(self):
        """Test that results are sorted by LLM relevance score."""
        from unittest.mock import Mock, patch

        from legiscope.retrieve import (
            QueryInfo,
            RelevanceAssessment,
            SectionCollection,
            SectionResult,
            filter_sections,
        )

        sections = [
            SectionResult(
                section_id="s1",
                heading_text="H1",
                body_text="B1",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0,
                segment_count=1,
            ),
            SectionResult(
                section_id="s2",
                heading_text="H2",
                body_text="B2",
                heading_level=1,
                parent_id=None,
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
                relevance_score=0.7,
                reasoning="Okay",
            ),
            RelevanceAssessment(
                relevance_score=0.9,
                reasoning="Great",
            ),
        ]

        with patch("legiscope.retrieve.is_relevant", side_effect=mock_assessments):
            with (
                patch("pathlib.Path.mkdir"),
                patch("legiscope.retrieve.pl.DataFrame.write_csv"),
            ):
                mock_client = Mock(spec=Instructor)
                result = filter_sections(mock_client, input_results, "query")

                # Should be sorted by score descending (Great > Okay)
                assert len(result.sections) == 2
                assert result.sections[0].section_id == "s2"  # Score 0.9
                assert result.sections[1].section_id == "s1"  # Score 0.7

    def test_filter_sections_validation(self):
        """Test input validation."""
        from unittest.mock import Mock

        from legiscope.retrieve import filter_sections

        with pytest.raises(ValueError, match="sections_results cannot be None"):
            filter_sections(Mock(), None, "query")  # type: ignore

        with pytest.raises(ValueError, match="client is required"):
            filter_sections(None, Mock(), "query")  # type: ignore

    def test_relevance_filter_worker_count_drops_for_large_prompts(self, monkeypatch):
        """Large prompt payloads should reduce the allowed parallelism."""
        from legiscope.retrieve import (
            SectionResult,
            _determine_relevance_filter_worker_count,
        )

        monkeypatch.setattr(
            "legiscope.retrieve.DEFAULT_RELEVANCE_FILTER_TARGET_CONCURRENT_TOKENS",
            1000,
        )

        sections = [
            SectionResult(
                section_id="s1",
                heading_text="H1",
                body_text="long " * 300,
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.0,
                segment_count=1,
            ),
            SectionResult(
                section_id="s2",
                heading_text="H2",
                body_text="short body",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.0,
                segment_count=1,
            ),
        ]

        worker_count = _determine_relevance_filter_worker_count(
            sections,
            query="query",
            retrieval_guidance=None,
            requested_max_concurrency=4,
        )

        assert worker_count == 1

    def test_filter_sections_uses_client_factory_for_concurrent_assessment(
        self, monkeypatch
    ):
        """Concurrent filtering should obtain per-thread clients from the supplied factory."""
        from legiscope.retrieve import (
            QueryInfo,
            RelevanceAssessment,
            SectionCollection,
            SectionResult,
            filter_sections,
        )

        monkeypatch.setattr(
            "legiscope.retrieve.DEFAULT_RELEVANCE_FILTER_TARGET_CONCURRENT_TOKENS",
            20000,
        )

        sections = [
            SectionResult(
                section_id="s1",
                heading_text="H1",
                body_text="B1",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.0,
                segment_count=1,
            ),
            SectionResult(
                section_id="s2",
                heading_text="H2",
                body_text="B2",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.0,
                segment_count=1,
            ),
        ]

        input_results = SectionCollection(
            sections=sections,
            query_info=QueryInfo(original_query="query"),
        )

        created_clients: list[object] = []
        seen_client_ids: set[int] = set()

        def client_factory() -> Instructor:
            client = Mock(spec=Instructor)
            created_clients.append(client)
            return client

        def fake_is_relevant(
            client: Instructor,
            query: str,
            text: str,
            model: str | None = None,
            retrieval_guidance: RetrievalGuidance | None = None,
            relevance_threshold: float = 0.5,
        ) -> RelevanceAssessment:
            seen_client_ids.add(id(client))
            return RelevanceAssessment(
                relevance_score=0.9 if "B1" in text else 0.2,
                reasoning="ok",
            )

        with patch("legiscope.retrieve.is_relevant", side_effect=fake_is_relevant):
            result = filter_sections(
                Mock(spec=Instructor),
                input_results,
                "query",
                relevance_threshold=0.5,
                max_concurrency=2,
                client_factory=client_factory,
            )

        assert len(created_clients) >= 1
        assert seen_client_ids == {id(client) for client in created_clients}
        assert [section.section_id for section in result.sections] == ["s1"]

    def test_resolve_relevance_filter_client_factory_requires_self_hosted_source(self):
        """Only self-hosted configs with an explicit factory should enable concurrency."""
        from legiscope.retrieve import resolve_relevance_filter_client_factory
        from legiscope.utils import LLMConfig

        local_factory = Mock(spec=Instructor)

        external_llm = LLMConfig(
            client=Mock(spec=Instructor),
            model="test-model",
            source="external",
            client_factory=local_factory,
        )
        self_hosted_llm = LLMConfig(
            client=Mock(spec=Instructor),
            model="test-model",
            source="self_hosted",
            client_factory=local_factory,
        )

        assert resolve_relevance_filter_client_factory(external_llm) is None
        assert resolve_relevance_filter_client_factory(self_hosted_llm) is local_factory
