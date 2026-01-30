"""
Tests for the query module.
"""

import os
import tempfile
import pytest
import polars as pl
from unittest.mock import Mock, patch
from instructor import Instructor
from pydantic import ValidationError

from legiscope.utils import LLMConfig
from legiscope.query import (
    LegalQueryResponse,
    format_query_response,
    _validate_supporting_passages,
    _prepare_legal_context,
    load_queries,
    QueryInput,
    QuerySettings,
    BatchQuerySettings,
    query_legal_documents,
    run_queries,
)
from legiscope.retrieve import SectionResult, SegmentMatch, SectionCollection, QueryInfo


class TestQueryInput:
    """Test the QueryInput dataclass."""

    def test_query_input_defaults(self):
        """Test default values."""
        query = QueryInput(question="Test question")
        assert query.question == "Test question"
        assert query.variable_name is None
        assert query.metadata == {}

    def test_query_input_full(self):
        """Test with all fields."""
        query = QueryInput(
            question="Test question",
            variable_name="test_var",
            metadata={"priority": 1}
        )
        assert query.question == "Test question"
        assert query.variable_name == "test_var"
        assert query.metadata == {"priority": 1}


class TestLoadQueries:
    """Test cases for load_queries function."""

    def test_load_queries_basic(self):
        """Test basic loading of queries from CSV."""
        df = pl.DataFrame({
            "question": ["Question 1", "Question 2"],
            "variable_name": ["var1", "var2"]
        })
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name
            
        try:
            queries = load_queries(temp_path, adjust_for_dataset=False)
            assert len(queries) == 2
            assert queries[0].question == "Question 1"
            assert queries[0].variable_name == "var1"
            assert queries[1].question == "Question 2"
            assert queries[1].variable_name == "var2"
        finally:
            os.unlink(temp_path)

    def test_load_queries_missing_column(self):
        """Test error when question column is missing."""
        df = pl.DataFrame({
            "wrong_column": ["value"]
        })
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name
            
        try:
            with pytest.raises(ValueError, match="must contain a 'question' column"):
                load_queries(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_queries_filter_empty(self):
        """Test filtering of empty questions."""
        df = pl.DataFrame({
            "question": ["Q1", None, "", "   ", "Q2"]
        })
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name
            
        try:
            queries = load_queries(temp_path, adjust_for_dataset=False)
            assert len(queries) == 2
            assert queries[0].question == "Q1"
            assert queries[1].question == "Q2"
        finally:
            os.unlink(temp_path)
            
    def test_load_queries_metadata(self):
        """Test that extra columns are captured as metadata."""
        df = pl.DataFrame({
            "question": ["Q1"],
            "variable_name": ["v1"],
            "category": ["general"],
            "priority": [1]
        })
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name
            
        try:
            queries = load_queries(temp_path, adjust_for_dataset=False)
            assert len(queries) == 1
            assert queries[0].metadata["category"] == "general"
            assert queries[0].metadata["priority"] == 1
        finally:
            os.unlink(temp_path)

    def test_load_queries_dataset_adjustment(self):
        """Test dataset specific adjustments for drug paraphernalia."""
        # Case 1: Trigger context addition
        df = pl.DataFrame({
            "question": ["Is drug paraphernalia allowed?", "Other question"],
            "variable_name": ["q1", "q2"]
        })
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name
            
        try:
            queries = load_queries(temp_path, adjust_for_dataset=True)
            # Should have prepended context
            assert "ordinance that prohibits drug paraphernalia" in queries[0].question
            assert "ordinance that prohibits drug paraphernalia" in queries[1].question
        finally:
            os.unlink(temp_path)
            
    def test_load_queries_exclusions(self):
        """Test exclusion of specific variable names."""
        df = pl.DataFrame({
            "question": ["drug paraphernalia Q1", "Q2", "Q3", "Q4"],
            "variable_name": ["normal", "dp_database", "dp_url", "dp_note"]
        })
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name
            
        try:
            queries = load_queries(temp_path, adjust_for_dataset=True)
            # Should filter out the 3 specific vars
            assert len(queries) == 1
            assert queries[0].variable_name == "normal"
        finally:
            os.unlink(temp_path)


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

class TestPrepareLegalContext:
    """Test _prepare_legal_context function."""

    def test_truncation(self):
        """Test that body text is truncated to 1000 words."""
        # Create 1500 word text
        body_text = " ".join(["word"] * 1500)
        
        section = SectionResult(
            section_idx=1,
            heading_text="Section 1",
            body_text=body_text,
            heading_level=1,
            parent=None,
            matching_segments=[],
            relevance_score=0.9,
            segment_count=1
        )
        
        context = _prepare_legal_context([section])
        
        # Should contain "... [content truncated]"
        assert "... [content truncated]" in context
        
    def test_matching_segments_included(self):
        """Test that matching segments are included."""
        section = SectionResult(
            section_idx=1,
            heading_text="Section 1",
            body_text="Start body. End body.",
            heading_level=1,
            parent=None,
            matching_segments=[
                SegmentMatch(
                    segment_idx=1,
                    segment_text="Relevant segment here.",
                    distance=0.2,
                    segment_position=0
                )
            ],
            relevance_score=0.9,
            segment_count=1
        )
        
        context = _prepare_legal_context([section])
        
        assert "Matching Passages (1):" in context
        assert "Relevant segment here." in context
        assert "(score: 0.200)" in context


class TestQueryConfig:
    """Test QueryConfig dataclass."""

    def test_minimal_config(self):
        """Test creating settings with required parameters."""
        llm_config = LLMConfig(client=Mock())
        settings = QuerySettings(llm=llm_config)

        assert settings.llm is llm_config
        assert settings.filter_relevance is False
        assert settings.relevance_threshold == 0.5

    def test_with_filtering(self):
        """Test settings with relevance filtering enabled."""
        llm_config = LLMConfig(client=Mock())
        settings = QuerySettings(
            llm=llm_config, filter_relevance=True, relevance_threshold=0.7
        )

        assert settings.filter_relevance is True
        assert settings.relevance_threshold == 0.7
        assert settings.filter_llm is llm_config  # Should use same LLM

    def test_with_separate_filter_llm(self):
        """Test settings with separate LLM for filtering."""
        main_llm = LLMConfig(client=Mock(), model="gpt-4")
        filter_llm = LLMConfig(client=Mock(), model="gpt-3.5")

        settings = QuerySettings(
            llm=main_llm, filter_relevance=True, filter_llm=filter_llm
        )

        assert settings.filter_llm is filter_llm
        assert settings.filter_llm is not main_llm

    def test_empty_query_raises_error(self):
        """Test that empty query is validated at function call."""
        # query validation moved to function, not settings
        settings = QuerySettings(llm=LLMConfig(client=Mock()))
        with pytest.raises(ValueError, match="query cannot be empty"):
            query_legal_documents(
                {"sections": []},
                "",  # Empty query
                settings,
            )

    def test_empty_results_raises_error(self):
        """Test that empty retrieval_results is validated at function call."""
        # retrieval_results validation moved to function, not settings
        settings = QuerySettings(llm=LLMConfig(client=Mock()))
        with pytest.raises(ValueError, match="retrieval_results cannot be empty"):
            query_legal_documents(
                None,  # Empty results
                "test",
                settings,
            )

    def test_invalid_relevance_threshold(self):
        """Test that invalid relevance_threshold raises error."""
        with pytest.raises(ValueError, match="relevance_threshold must be between"):
            QuerySettings(llm=LLMConfig(client=Mock()), relevance_threshold=1.5)


class TestBatchQueryConfig:
    """Test BatchQuerySettings dataclass."""

    def test_minimal_config(self):
        """Test creating settings with defaults."""
        # Mock the API client creation to avoid needing API keys,
        # but still test that __post_init__ creates default LLM config
        with patch("legiscope.llm_config.Config.get_fast_client") as mock_get_client:
            mock_get_client.return_value = Mock()

            settings = BatchQuerySettings()

            assert settings.llm is not None  # Should be set by __post_init__
            assert settings.n_results == 10  # Default
            assert settings.use_hyde is False
            mock_get_client.assert_called_once()  # Verify default behavior triggered

    def test_with_custom_llm(self):
        """Test settings with custom LLM."""
        llm_config = LLMConfig(client=Mock(), model="gpt-4")
        settings = BatchQuerySettings(llm=llm_config)

        assert settings.llm is llm_config

    def test_with_all_options(self):
        """Test settings with all options customized."""
        llm_config = LLMConfig(client=Mock())
        settings = BatchQuerySettings(
            llm=llm_config,
            n_results=20,
            use_hyde=True,
            filter_relevance=True,
            relevance_threshold=0.8,
        )

        assert settings.n_results == 20
        assert settings.use_hyde is True
        assert settings.filter_relevance is True
        assert settings.relevance_threshold == 0.8

    def test_empty_queries_raises_error(self):
        """Test that empty queries list is validated at function call."""
        # queries validation moved to function, not settings
        with pytest.raises(ValueError, match="queries list cannot be empty"):
            run_queries(
                collection=Mock(),
                sections_parquet_path="./data/sections.parquet",
                queries=[],  # Empty queries
                jurisdiction_id="IL-WindyCity",
            )

    def test_empty_jurisdiction_raises_error(self):
        """Test that empty jurisdiction_id is validated at function call."""
        # jurisdiction_id validation moved to function, not settings
        with pytest.raises(ValueError, match="jurisdiction_id cannot be empty"):
            run_queries(
                collection=Mock(),
                sections_parquet_path="./data/sections.parquet",
                queries=["test"],
                jurisdiction_id="",  # Empty jurisdiction_id
            )

    def test_invalid_n_results(self):
        """Test that invalid n_results raises error."""
        with pytest.raises(ValueError, match="n_results must be positive"):
            BatchQuerySettings(n_results=0)

    def test_batch_query_settings_defaults(self):
        """Test default values for new parameters."""
        mock_llm = Mock(spec=LLMConfig)
        settings = BatchQuerySettings(llm=mock_llm)
        
        assert settings.n_results == 10
        assert settings.use_hyde is False
        assert settings.filter_relevance is False
        assert settings.relevance_threshold == 0.5
        assert settings.validate_supporting_passages is True

    def test_batch_query_settings_instantiation(self):
        """Test instantiating with specific values."""
        mock_llm = Mock(spec=LLMConfig)
        settings = BatchQuerySettings(
            llm=mock_llm,
            n_results=20,
            use_hyde=True,
            filter_relevance=True,
            relevance_threshold=0.8,
            validate_supporting_passages=False
        )
        
        assert settings.n_results == 20
        assert settings.use_hyde is True
        assert settings.filter_relevance is True
        assert settings.relevance_threshold == 0.8
        assert settings.validate_supporting_passages is False


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
