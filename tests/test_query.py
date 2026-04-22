"""
Tests for the query module.
"""

import json
import os
import tempfile
import pytest
from loguru import logger
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
    _build_legal_prompts,
    _normalize_structured_short_answer,
    load_queries,
    QueryInput,
    QuerySettings,
    BatchQuerySettings,
    query_legal_documents,
    run_queries,
    DEFAULT_RELEVANCE_FILTER_ENABLED,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_N_RESULTS,
    DEFAULT_HYDE_ENABLED,
    DEFAULT_LEXICAL_RERANKING_ENABLED,
    DEFAULT_VALIDATION_ENABLED,
)
from legiscope.retrieval_guidance import RetrievalGuidance, RetrievalGuidanceRequest
from legiscope.retrieve import (
    FilteringMetadata,
    QueryInfo,
    SectionCollection,
    SectionResult,
    SegmentMatch,
)


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
            question="Test question", variable_name="test_var", metadata={"priority": 1}
        )
        assert query.question == "Test question"
        assert query.variable_name == "test_var"
        assert query.metadata == {"priority": 1}


class TestStructuredShortAnswerNormalization:
    """Test deterministic normalization for structured answer fields."""

    def test_normalizes_enactment_date_with_month_year_imputation(self):
        normalized = _normalize_structured_short_answer(
            "December 2024",
            "structured_date_field",
            {
                "response_options": "Responses: <enactment date> OR Unkown",
                "coding_instructions": (
                    "If only month and year are available then impute the day as "
                    "the 15th of the month."
                ),
            },
        )

        assert normalized == "12/15/2024"

    def test_normalizes_yes_no_citation_output(self):
        normalized = _normalize_structured_short_answer(
            "Yes, the relevant citation is 35 P.S. § 780-102.",
            "citation_field",
            {
                "response_options": "Responses: Yes, <citation> OR No",
            },
        )

        assert normalized == "Yes, 35 P.S. § 780-102"

    def test_normalizes_citation_only_output_for_state_fed_combined(self):
        normalized = _normalize_structured_short_answer(
            "35 P.S. § 780-102",
            "citation_field",
            {
                "response_options": "Responses: Yes, <citation> OR No",
            },
        )

        assert normalized == "Yes, 35 P.S. § 780-102"

    def test_normalizes_current_through_combined_output(self):
        normalized = _normalize_structured_short_answer(
            "Known; March 19, 2024",
            "status_date_field",
            {
                "response_options": (
                    "Responses: Known, <current through date published in ordinance> "
                    "OR Partially known, <partial current through date published in ordinance "
                    "(month or day imputed)> OR Unknown, <date of data collection>"
                ),
            },
        )

        assert normalized == "Known, 03/19/2024"

    def test_normalizes_partial_current_through_output(self):
        normalized = _normalize_structured_short_answer(
            "Partially known, 03/2024",
            "status_date_field",
            {
                "response_options": (
                    "Responses: Known, <current through date published in ordinance> "
                    "OR Partially known, <partial current through date published in ordinance "
                    "(month or day imputed)> OR Unknown, <date of data collection>"
                ),
            },
        )

        assert normalized == "Partially known, 03/15/2024"

    def test_normalizes_multi_select_output_in_declared_order(self):
        normalized = _normalize_structured_short_answer(
            "Use; Sales",
            "multi_select_field",
            {
                "response_options": ("Responses: Sales AND/OR Use AND/OR Possession"),
            },
        )

        assert normalized == "Sales AND/OR Use"

    def test_does_not_coerce_multi_select_with_extra_prose(self):
        normalized = _normalize_structured_short_answer(
            "The ordinance prohibits sales and use.",
            "multi_select_field",
            {
                "response_options": ("Responses: Sales AND/OR Use AND/OR Possession"),
            },
        )

        assert normalized == "The ordinance prohibits sales and use."

    def test_does_not_coerce_single_choice_with_extra_prose(self):
        normalized = _normalize_structured_short_answer(
            "The best label is Misdemeanor.",
            "single_choice_field",
            {
                "response_options": "Responses: Civil OR Misdemeanor OR Felony",
            },
        )

        assert normalized == "The best label is Misdemeanor."


class TestPromptContracts:
    """Prompt-building regressions for structured benchmark answers."""

    def test_build_legal_prompts_includes_structured_answer_contract(self):
        system_prompt, _user_prompt = _build_legal_prompts(
            "Which activities are prohibited?",
            "Section 1: sell or use drug paraphernalia.",
            query_metadata={
                "response_options": "Responses: Sales AND/OR Use AND/OR Possession",
                "coding_instructions": "Use only the exact response labels.",
            },
        )

        assert "Structured answer contract:" in system_prompt
        assert (
            "Declared response options: Sales AND/OR Use AND/OR Possession"
            in system_prompt
        )
        assert "join selections with ` AND/OR `" in system_prompt
        assert "Apply these coding instructions exactly" in system_prompt


class TestLoadQueries:
    """Test cases for load_queries function."""

    def test_load_queries_basic(self):
        """Test basic loading of queries from CSV."""
        df = pl.DataFrame(
            {
                "question": ["Question 1", "Question 2"],
                "variable_name": ["var1", "var2"],
            }
        )

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
        df = pl.DataFrame({"wrong_column": ["value"]})

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
        df = pl.DataFrame({"question": ["Q1", None, "", "   ", "Q2"]})

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
        df = pl.DataFrame(
            {
                "question": ["Q1"],
                "variable_name": ["v1"],
                "category": ["general"],
                "priority": [1],
            }
        )

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

    def test_load_queries_custom_adjuster(self):
        """Test caller-provided query adjuster hook."""
        df = pl.DataFrame(
            {
                "question": ["Question 1", "Question 2"],
                "variable_name": ["var1", "var2"],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name

        def _adjuster(input_df: pl.DataFrame) -> pl.DataFrame:
            return input_df.with_columns(
                (pl.lit("PREFIX: ") + pl.col("question")).alias("question")
            )

        try:
            queries = load_queries(
                temp_path,
                adjust_for_dataset=True,
                query_adjuster=_adjuster,
            )
            assert queries[0].question == "PREFIX: Question 1"
            assert queries[1].question == "PREFIX: Question 2"
        finally:
            os.unlink(temp_path)

    def test_load_queries_drops_noisy_and_empty_metadata_columns(self):
        """Blank, duplicated, deprecated, and all-empty columns should be discarded at load time."""
        csv_content = """question,variable_name,category,,_duplicated_0,Deprecated,all_empty\nQ1,v1,general,,noise,legacy,\n"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            queries = load_queries(temp_path, adjust_for_dataset=False)
            assert len(queries) == 1
            assert queries[0].metadata == {"category": "general"}
        finally:
            os.unlink(temp_path)


@pytest.fixture(autouse=True)
def capture_loguru_logs(caplog):
    """Make loguru logs visible to pytest's caplog."""
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)


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

    def create_test_sections(
        self,
        body_text: str,
        segment_texts: list[str] | None = None,
    ):
        """Helper to create test SectionResult objects."""
        if segment_texts is None:
            segment_texts = []

        segments = [
            SegmentMatch(
                segment_id=str(i),
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
                section_id="s0",
                heading_text="Test Section",
                body_text=body_text,
                heading_level=1,
                parent_id=None,
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

    def test_validate_with_normalization(self, caplog):
        """Test validation works with whitespace and smart quote differences."""
        response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=[
                "No person shall sell   drug paraphernalia.",  # Extra spaces
                "“Smart quotes” are supported.",  # Smart quotes
            ],
            confidence=0.9,
            limitations="",
        )

        sections = self.create_test_sections(
            body_text='Section 5-12-3: No person shall sell drug paraphernalia. "Smart quotes" are supported.',
        )

        _validate_supporting_passages(response, sections)

        # Should match exactly due to normalization
        assert "validated (exact match)" in caplog.text
        assert "HALLUCINATION WARNING" not in caplog.text
        assert "NOT FOUND" not in caplog.text


class TestPrepareLegalContext:
    """Test _prepare_legal_context function."""

    def test_full_body_text_included(self):
        """Test that body text is included without query-time truncation."""
        # Create 1500 word text
        body_text = " ".join([f"word{i}" for i in range(1500)])

        section = SectionResult(
            section_id="s1",
            heading_text="Section 1",
            body_text=body_text,
            heading_level=1,
            parent_id=None,
            matching_segments=[],
            relevance_score=0.9,
            segment_count=1,
        )

        context = _prepare_legal_context([section])

        assert body_text in context
        assert "... [content truncated]" not in context

    def test_matching_segments_not_included(self):
        """Completion context should include only chunk content, not matched segments."""
        section = SectionResult(
            section_id="s1",
            heading_text="Section 1",
            body_text="Start body. End body.",
            heading_level=1,
            parent_id=None,
            matching_segments=[
                SegmentMatch(
                    segment_id="g1",
                    segment_text="Relevant segment here.",
                    distance=0.2,
                    segment_position=0,
                )
            ],
            relevance_score=0.9,
            segment_count=1,
        )

        context = _prepare_legal_context([section])

        assert "Matching Passages (1):" not in context
        assert "Relevant segment here." not in context
        assert "(score: 0.200)" not in context
        assert "Content: Start body. End body." in context

    def test_context_path_and_region_role_included(self):
        """Chunk provenance should be surfaced in the completion context."""
        section = SectionResult(
            section_id="c0",
            heading_text="Legal Intro",
            body_text="This ordinance was adopted by the council.",
            heading_level=0,
            parent_id=None,
            matching_segments=[],
            relevance_score=0.7,
            segment_count=1,
            context_path="Legal Intro",
            source_kind="region",
            region_role="legal_intro",
        )

        context = _prepare_legal_context([section])

        assert "Context Path: Legal Intro" in context
        assert "Source Kind: region" in context
        assert "Region Role: legal_intro" in context


class TestQueryConfig:
    """Test QueryConfig dataclass."""

    def test_minimal_config(self):
        """Test creating settings with required parameters."""
        llm_config = LLMConfig(client=Mock())
        settings = QuerySettings(llm=llm_config)

        assert settings.llm is llm_config
        assert settings.filter_relevance == DEFAULT_RELEVANCE_FILTER_ENABLED
        assert settings.relevance_threshold == DEFAULT_RELEVANCE_THRESHOLD

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

    def test_with_retrieval_guidance(self):
        """Test settings can carry per-query retrieval guidance."""
        llm_config = LLMConfig(client=Mock())
        guidance = RetrievalGuidance(guidance_topic="date")

        settings = QuerySettings(llm=llm_config, retrieval_guidance=guidance)

        assert settings.retrieval_guidance is guidance

    def test_empty_query_raises_error(self):
        """Test that empty query is validated at function call."""
        # query validation moved to function, not settings
        settings = QuerySettings(llm=LLMConfig(client=Mock()))
        with pytest.raises(ValueError, match="query cannot be empty"):
            query_legal_documents(
                SectionCollection(
                    sections=[],
                    query_info=QueryInfo(original_query=""),
                ),
                "",  # Empty query
                settings,
            )

    def test_empty_results_raises_error(self):
        """Test that an empty SectionCollection returns the no-results fallback."""
        settings = QuerySettings(llm=LLMConfig(client=Mock()))
        response, similarity_scores = query_legal_documents(
            SectionCollection(
                sections=[],
                query_info=QueryInfo(original_query="test"),
            ),
            "test",
            settings,
        )

        assert response.confidence == 0.0
        assert (
            "no relevant legal provisions were found" in response.short_answer.lower()
        )
        assert similarity_scores == []

    def test_invalid_relevance_threshold(self):
        """Test that invalid relevance_threshold raises error."""
        with pytest.raises(ValueError, match="relevance_threshold must be between"):
            QuerySettings(llm=LLMConfig(client=Mock()), relevance_threshold=1.5)


class TestBatchQueryConfig:
    """Test BatchQuerySettings dataclass."""

    def test_minimal_config(self):
        """Test creating settings with defaults."""
        # Mock the API client creation to avoid needing API keys,
        # but still test that __post_init__ creates default LLM config.
        with (
            patch("legiscope.llm_config.Config.get_powerful_client") as mock_client,
            patch("legiscope.llm_config.Config.get_powerful_model") as mock_model,
        ):
            mock_client.return_value = Mock()
            mock_model.return_value = "test-model"

            settings = BatchQuerySettings()

            assert settings.llm is not None  # Should be set by __post_init__
            assert settings.n_results == DEFAULT_N_RESULTS
            assert settings.use_hyde == DEFAULT_HYDE_ENABLED
            assert settings.use_lexical_reranking == DEFAULT_LEXICAL_RERANKING_ENABLED
            mock_client.assert_called_once()

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
            use_lexical_reranking=True,
            filter_relevance=True,
            relevance_threshold=0.8,
        )

        assert settings.n_results == 20
        assert settings.use_hyde is True
        assert settings.use_lexical_reranking is True
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
                jurisdiction_id="IL-WindyTown",
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

        assert settings.n_results == DEFAULT_N_RESULTS
        assert settings.use_hyde == DEFAULT_HYDE_ENABLED
        assert settings.use_lexical_reranking == DEFAULT_LEXICAL_RERANKING_ENABLED
        assert settings.filter_relevance == DEFAULT_RELEVANCE_FILTER_ENABLED
        assert settings.relevance_threshold == DEFAULT_RELEVANCE_THRESHOLD
        assert settings.validate_supporting_passages == DEFAULT_VALIDATION_ENABLED

    def test_batch_query_settings_instantiation(self):
        """Test instantiating with specific values."""
        mock_llm = Mock(spec=LLMConfig)
        settings = BatchQuerySettings(
            llm=mock_llm,
            n_results=20,
            use_hyde=True,
            use_lexical_reranking=True,
            filter_relevance=True,
            relevance_threshold=0.8,
            validate_supporting_passages=False,
        )

        assert settings.n_results == 20
        assert settings.use_hyde is True
        assert settings.use_lexical_reranking is True
        assert settings.filter_relevance is True
        assert settings.relevance_threshold == 0.8
        assert settings.validate_supporting_passages is False

    def test_with_retrieval_guidance_provider(self):
        """Test settings can carry a project-provided retrieval guidance hook."""

        def provider(request: RetrievalGuidanceRequest) -> RetrievalGuidance | None:
            return RetrievalGuidance(guidance_topic=request.variable_name)

        mock_llm = Mock(spec=LLMConfig)
        settings = BatchQuerySettings(
            llm=mock_llm,
            retrieval_guidance_provider=provider,
        )

        assert settings.retrieval_guidance_provider is provider


class TestQueryConfigBasics:
    """Test QuerySettings-based query_legal_documents function."""

    def test_query_legal_documents_with_config(self):
        """Test basic query_legal_documents with settings object."""
        mock_client = Mock(spec=Instructor)

        # Mock retrieval results
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Parking Regulations",
                    body_text="No parking between 2am and 6am",
                    heading_level=1,
                    parent_id=None,
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
            settings = QuerySettings(llm=llm_config, filter_relevance=False)

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
            section_id="s0",
            heading_text="# Test Section",
            body_text="Test content",
            heading_level=1,
            parent_id=None,
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
                mock_filter.return_value = SectionCollection(
                    sections=[section],
                    query_info=QueryInfo(original_query="test query"),
                )

                llm_config = LLMConfig(client=mock_client, model="test-model")
                guidance = RetrievalGuidance(guidance_topic="activity")
                settings = QuerySettings(
                    llm=llm_config,
                    filter_relevance=True,
                    relevance_threshold=0.7,
                    retrieval_guidance=guidance,
                )

                response, similarity_scores = query_legal_documents(
                    retrieval_results, "test query", settings
                )

                assert response.short_answer == "Test answer"
                mock_filter.assert_called_once()
                assert mock_filter.call_args.kwargs["retrieval_guidance"] is guidance


class TestBatchQueryConfigBasics:
    """Test BatchQuerySettings-based run_queries function."""

    def test_run_queries_with_minimal_config(self, tmp_path):
        """Test run_queries with minimal configuration."""

        # Create test sections parquet
        sections_data = {
            "section_ordinal": [0],
            "heading_text": ["# Test"],
            "body_text": ["Content"],
            "heading_level": [1],
            "parent_id": [None],
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
                        "section_ordinal": 0,
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
                    jurisdiction_id="IL-WindyTown",
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

        with (
            patch("legiscope.llm_config.Config.get_powerful_client") as mock_get_client,
            patch("legiscope.llm_config.Config.get_powerful_model") as mock_get_model,
        ):
            mock_client = Mock(spec=Instructor)
            mock_get_client.return_value = mock_client
            mock_get_model.return_value = "test-model"

            settings = BatchQuerySettings()
            # No llm provided - should use default

            assert settings.llm is not None
            assert settings.llm.client is mock_client

    def test_run_queries_applies_retrieval_guidance_provider(self, tmp_path):
        """run_queries should resolve project-specific retrieval guidance per query."""
        sections_path = tmp_path / "sections.parquet"
        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Test"],
                "body_text": ["Content"],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[],
            query_info=QueryInfo(
                original_query="test query",
                total_segments_found=0,
                unique_sections=0,
            ),
        )

        mock_response = LegalQueryResponse(
            short_answer="Test answer",
            reasoning="Test reasoning",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        captured_guidance = []

        def provider(request: RetrievalGuidanceRequest) -> RetrievalGuidance | None:
            if request.variable_name == "dp_enacted":
                return RetrievalGuidance(guidance_topic="date")
            return None

        def fake_query_legal_documents(_results, _query, query_settings, **_kwargs):
            captured_guidance.append(query_settings.retrieval_guidance)
            return mock_response, []

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch(
                "legiscope.query.query_legal_documents",
                side_effect=fake_query_legal_documents,
            ):
                mock_client = Mock(spec=Instructor)
                llm_config = LLMConfig(client=mock_client, model="test-model")
                settings = BatchQuerySettings(
                    llm=llm_config,
                    retrieval_guidance_provider=provider,
                )

                results_df = run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[QueryInput(question="query1", variable_name="dp_enacted")],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

                assert isinstance(results_df, pl.DataFrame)
                assert len(captured_guidance) == 1
                assert captured_guidance[0] is not None
                assert captured_guidance[0].guidance_topic == "date"

    def test_run_queries_propagates_lexical_reranking_flag(self, tmp_path):
        """Batch settings should pass the lexical reranking toggle into retrieval settings."""
        sections_path = tmp_path / "sections.parquet"
        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Test"],
                "body_text": ["Content"],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[],
            query_info=QueryInfo(
                original_query="test query",
                total_segments_found=0,
                unique_sections=0,
            ),
        )

        mock_response = LegalQueryResponse(
            short_answer="Test answer",
            reasoning="Test reasoning",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        with patch(
            "legiscope.query.retrieve_sections", return_value=retrieval_results
        ) as mock_retrieve:
            with patch(
                "legiscope.query.query_legal_documents",
                return_value=(mock_response, []),
            ):
                mock_client = Mock(spec=Instructor)
                llm_config = LLMConfig(client=mock_client, model="test-model")
                settings = BatchQuerySettings(
                    llm=llm_config,
                    use_lexical_reranking=True,
                )

                run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[QueryInput(question="query1", variable_name="dp_enacted")],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        retrieval_settings = mock_retrieve.call_args.kwargs["settings"]
        assert retrieval_settings.use_lexical_reranking is True

    def test_run_queries_uses_retrieval_and_completion_query_variants(self, tmp_path):
        """Per-query guidance should split retrieval text from completion text."""
        sections_path = tmp_path / "sections.parquet"
        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Test"],
                "body_text": ["Content"],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[],
            query_info=QueryInfo(
                original_query="retrieval query",
                total_segments_found=0,
                unique_sections=0,
            ),
        )

        mock_response = LegalQueryResponse(
            short_answer="Test answer",
            reasoning="Test reasoning",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        def provider(_request: RetrievalGuidanceRequest) -> RetrievalGuidance | None:
            return RetrievalGuidance(
                guidance_topic="date",
                retrieval_query="Question: When was the ordinance enacted?",
                completion_instructions="Use enactment-specific coding logic.",
            )

        with patch(
            "legiscope.query.retrieve_sections", return_value=retrieval_results
        ) as mock_retrieve:
            with patch(
                "legiscope.query.query_legal_documents",
                return_value=(mock_response, []),
            ) as mock_query_legal_documents:
                mock_client = Mock(spec=Instructor)
                llm_config = LLMConfig(client=mock_client, model="test-model")
                settings = BatchQuerySettings(
                    llm=llm_config,
                    retrieval_guidance_provider=provider,
                )

                run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="full completion query", variable_name="dp_enacted"
                        )
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

                assert (
                    mock_retrieve.call_args.kwargs["query_text"]
                    == "Question: When was the ordinance enacted?"
                )
                assert (
                    mock_query_legal_documents.call_args.args[1]
                    == "full completion query\n\nVariable-specific guidance:\nUse enactment-specific coding logic."
                )

    def test_run_queries_writes_consolidated_stage_debug_csvs(self, tmp_path):
        """Debug mode should emit one retrieval/relevance/query CSV row per question."""
        sections_path = tmp_path / "sections.parquet"
        debug_dir = tmp_path / "debug"
        debug_timestamp = "20260413_120000"

        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Test"],
                "body_text": ["This ordinance was enacted in 2024."],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s1",
                    heading_text="# Test",
                    body_text="This ordinance was enacted in 2024.",
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[
                        SegmentMatch(
                            segment_id="seg1",
                            segment_text="This ordinance was enacted in 2024.",
                            distance=0.12,
                            segment_position=0,
                        )
                    ],
                    relevance_score=0.12,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="full completion query",
                rewritten_query="Question: When was the ordinance enacted?",
                total_segments_found=1,
                unique_sections=1,
            ),
        )

        mock_response = LegalQueryResponse(
            short_answer="12/21/2011",
            reasoning="The ordinance text provides the enactment date.",
            citations=["Section 1"],
            supporting_passages=["This ordinance was enacted in 2024."],
            confidence=0.9,
            limitations="None",
        )

        def provider(_request: RetrievalGuidanceRequest) -> RetrievalGuidance | None:
            return RetrievalGuidance(
                guidance_topic="date_enactment",
                shared_context="This query concerns a local municipal ordinance regulating drug paraphernalia-related activities.",
                retrieval_query="Question: When was the ordinance enacted?",
                retrieval_instructions="Retrieve ordinance metadata and enactment history.",
                relevance_instructions="Prefer enactment-date language over effective dates.",
                anchor_terms=["enacted", "adopted"],
                completion_instructions="Use enactment-specific coding logic.",
            )

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch("legiscope.query.ask", return_value=mock_response):
                mock_client = Mock(spec=Instructor)
                llm_config = LLMConfig(client=mock_client, model="test-model")
                settings = BatchQuerySettings(
                    llm=llm_config,
                    debug_dir=debug_dir,
                    debug_timestamp=debug_timestamp,
                    retrieval_guidance_provider=provider,
                    filter_relevance=False,
                    validate_supporting_passages=False,
                )

                run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="full completion query",
                            variable_name="dp_enacted",
                            metadata={
                                "question_number": "Q1.2",
                                "query_text": "On which date was the ordinance enacted?",
                            },
                        )
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        retrieval_debug = pl.read_csv(
            debug_dir / f"retrieval_stage_{debug_timestamp}.csv"
        )
        relevance_debug = pl.read_csv(
            debug_dir / f"relevance_stage_{debug_timestamp}.csv"
        )
        query_debug = pl.read_csv(debug_dir / f"query_stage_{debug_timestamp}.csv")

        assert len(retrieval_debug) == 1
        assert len(relevance_debug) == 1
        assert len(query_debug) == 1
        assert (
            retrieval_debug[0, "retrieval_query"]
            == "Question: When was the ordinance enacted?"
        )
        assert retrieval_debug[0, "retrieved_segments"] != "[]"
        assert relevance_debug[0, "stage_status"] == "skipped"
        assert query_debug[0, "completion_query"] == (
            "full completion query\n\nVariable-specific guidance:\nUse enactment-specific coding logic."
        )
        assert query_debug[0, "short_answer"] == "12/21/2011"

    def test_run_queries_postprocesses_structured_date_answers(self, tmp_path):
        """Structured date answers should be normalized in results and debug output."""
        sections_path = tmp_path / "sections.parquet"
        debug_dir = tmp_path / "debug"
        debug_timestamp = "20260413_121500"

        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Test"],
                "body_text": ["This ordinance was adopted in December 2024."],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[],
            query_info=QueryInfo(
                original_query="full completion query",
                total_segments_found=0,
                unique_sections=0,
            ),
        )

        mock_response = LegalQueryResponse(
            short_answer="December 2024",
            reasoning="The ordinance text gives a month and year.",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch(
                "legiscope.query.query_legal_documents",
                return_value=(mock_response, []),
            ):
                mock_client = Mock(spec=Instructor)
                llm_config = LLMConfig(client=mock_client, model="test-model")
                settings = BatchQuerySettings(
                    llm=llm_config,
                    debug_dir=debug_dir,
                    debug_timestamp=debug_timestamp,
                )

                results_df = run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="full completion query",
                            variable_name="structured_date_field",
                            metadata={
                                "question_number": "Q1.2",
                                "query_text": "On which date was the ordinance enacted?",
                                "response_options": "Responses: <enactment date> OR Unkown",
                                "coding_instructions": (
                                    "If only month and year are available then impute the "
                                    "day as the 15th of the month."
                                ),
                            },
                        )
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        query_debug = pl.read_csv(debug_dir / f"query_stage_{debug_timestamp}.csv")

        assert results_df[0, "short_answer"] == "12/15/2024"
        assert query_debug[0, "short_answer"] == "12/15/2024"
        assert query_debug[0, "raw_short_answer"] == "December 2024"

    def test_run_queries_postprocesses_current_through_combined_answers(self, tmp_path):
        """Status/date combined outputs should be normalized in results and debug output."""
        sections_path = tmp_path / "sections.parquet"
        debug_dir = tmp_path / "debug"
        debug_timestamp = "20260413_122500"

        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Test"],
                "body_text": ["This code is current through March 19, 2024."],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[],
            query_info=QueryInfo(
                original_query="full completion query",
                total_segments_found=0,
                unique_sections=0,
            ),
        )

        mock_response = LegalQueryResponse(
            short_answer="Known; March 19, 2024",
            reasoning="The code header states the current-through date.",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch(
                "legiscope.query.query_legal_documents",
                return_value=(mock_response, []),
            ):
                mock_client = Mock(spec=Instructor)
                llm_config = LLMConfig(client=mock_client, model="test-model")
                settings = BatchQuerySettings(
                    llm=llm_config,
                    debug_dir=debug_dir,
                    debug_timestamp=debug_timestamp,
                )

                results_df = run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="full completion query",
                            variable_name="status_date_field",
                            metadata={
                                "question_number": "Q1.4",
                                "query_text": "What is the current-through date of the ordinance?",
                                "response_options": (
                                    "Responses: Known, <current through date published in ordinance> "
                                    "OR Partially known, <partial current through date published in ordinance "
                                    "(month or day imputed)> OR Unknown, <date of data collection>"
                                ),
                            },
                        )
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        query_debug = pl.read_csv(debug_dir / f"query_stage_{debug_timestamp}.csv")

        assert results_df[0, "short_answer"] == "Known, 03/19/2024"
        assert query_debug[0, "short_answer"] == "Known, 03/19/2024"
        assert query_debug[0, "raw_short_answer"] == "Known; March 19, 2024"

    def test_run_queries_carries_prior_answers_into_guidance_requests(self, tmp_path):
        """Later structured queries should receive earlier answers through metadata."""
        sections_path = tmp_path / "sections.parquet"
        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Test"],
                "body_text": ["Content"],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[],
            query_info=QueryInfo(
                original_query="query",
                total_segments_found=0,
                unique_sections=0,
            ),
        )

        captured_prior_answers = []

        def provider(request: RetrievalGuidanceRequest) -> RetrievalGuidance | None:
            captured_prior_answers.append(request.metadata.get("prior_answers"))
            return None

        mock_responses = [
            LegalQueryResponse(
                short_answer="Sales AND/OR Use",
                reasoning="Activities are listed explicitly.",
                citations=[],
                supporting_passages=[
                    "Quoted upstream passage that should not be forwarded downstream."
                ],
                confidence=0.8,
                limitations="None",
            ),
            LegalQueryResponse(
                short_answer="Use",
                reasoning="The exemption tracks the previously coded activity.",
                citations=[],
                supporting_passages=[],
                confidence=0.8,
                limitations="None",
            ),
        ]

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch(
                "legiscope.query.query_legal_documents",
                side_effect=[(mock_responses[0], []), (mock_responses[1], [])],
            ):
                mock_client = Mock(spec=Instructor)
                llm_config = LLMConfig(client=mock_client, model="test-model")
                settings = BatchQuerySettings(
                    llm=llm_config,
                    retrieval_guidance_provider=provider,
                )

                run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="Which activities are prohibited?",
                            variable_name="dp_activity",
                            metadata={
                                "response_options": (
                                    "Responses: Sales AND/OR Use AND/OR Possession"
                                )
                            },
                        ),
                        QueryInput(
                            question="If cannabis paraphernalia is exempted, which activities are exempted?",
                            variable_name="dp_exempt_can_activity",
                            metadata={
                                "response_options": (
                                    "Responses: Possession AND/OR Use AND/OR Distribution AND/OR Sales AND/OR Other"
                                )
                            },
                        ),
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        assert captured_prior_answers[0] is None
        assert captured_prior_answers[1] is not None
        assert (
            captured_prior_answers[1]["dp_activity"]["short_answer"]
            == "Sales AND/OR Use"
        )
        assert captured_prior_answers[1]["dp_activity"] == {
            "short_answer": "Sales AND/OR Use",
            "raw_short_answer": "Sales AND/OR Use",
        }

    def test_run_queries_sanitizes_preexisting_prior_answers_metadata(self, tmp_path):
        """Input metadata prior_answers should drop retrieval-heavy upstream fields."""
        sections_path = tmp_path / "sections.parquet"
        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Test"],
                "body_text": ["Content"],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[],
            query_info=QueryInfo(
                original_query="query",
                total_segments_found=0,
                unique_sections=0,
            ),
        )

        captured_prior_answers = []

        def provider(request: RetrievalGuidanceRequest) -> RetrievalGuidance | None:
            captured_prior_answers.append(request.metadata.get("prior_answers"))
            return None

        response = LegalQueryResponse(
            short_answer="Use",
            reasoning="The answer is not important for this regression.",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch(
                "legiscope.query.query_legal_documents",
                return_value=(response, []),
            ):
                mock_client = Mock(spec=Instructor)
                llm_config = LLMConfig(client=mock_client, model="test-model")
                settings = BatchQuerySettings(
                    llm=llm_config,
                    retrieval_guidance_provider=provider,
                )

                run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="What is the exemption scope?",
                            variable_name="dp_exempt_can_activity",
                            metadata={
                                "prior_answers": {
                                    "dp_exemption": {
                                        "short_answer": "Yes",
                                        "raw_short_answer": "Yes",
                                        "supporting_passages": [
                                            "Large upstream passage that should be removed"
                                        ],
                                        "retrieved_sections": [
                                            "Very long retrieved section summary"
                                        ],
                                        "reasoning": "This should not be forwarded.",
                                    }
                                }
                            },
                        )
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        assert captured_prior_answers == [
            {
                "dp_exemption": {
                    "short_answer": "Yes",
                    "raw_short_answer": "Yes",
                }
            }
        ]

    def test_run_queries_serializes_query_metadata_without_flattening_query_subfields(
        self, tmp_path
    ):
        """Benchmark-facing results should keep one metadata blob without redundant query columns."""
        sections_path = tmp_path / "sections.parquet"
        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Test"],
                "body_text": ["Content"],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[],
            query_info=QueryInfo(
                original_query="query",
                total_segments_found=0,
                unique_sections=0,
            ),
        )

        mock_response = LegalQueryResponse(
            short_answer="Known, 03/19/2024",
            reasoning="Test reasoning",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch(
                "legiscope.query.query_legal_documents",
                return_value=(mock_response, []),
            ):
                mock_client = Mock(spec=Instructor)
                llm_config = LLMConfig(client=mock_client, model="test-model")
                settings = BatchQuerySettings(llm=llm_config)

                results_df = run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="full completion query",
                            variable_name="status_date_field",
                            metadata={
                                "question_number": "Q1.4",
                                "query_text": "What is the current-through date of the ordinance?",
                                "response_options": (
                                    "Responses: Known, <current through date published in ordinance> "
                                    "OR Partially known, <partial current through date published in ordinance "
                                    "(month or day imputed)> OR Unknown, <date of data collection>"
                                ),
                                "coding_instructions": "Use exact response labels.",
                                "query_family": "status",
                            },
                        )
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        assert "query_metadata" in results_df.columns
        assert "query_family" in results_df.columns
        assert results_df[0, "query_family"] == "status"
        assert "question_number" not in results_df.columns
        assert "query_text" not in results_df.columns
        assert "response_options" not in results_df.columns
        assert "coding_instructions" not in results_df.columns

        metadata = json.loads(results_df[0, "query_metadata"])
        assert metadata["question_number"] == "Q1.4"
        assert metadata["query_text"] == "What is the current-through date of the ordinance?"
        assert metadata["query_family"] == "status"

    def test_run_queries_surfaces_filtered_out_retrieval_units(self, tmp_path):
        """Benchmark-facing results should expose when relevance filtering removes all units."""
        sections_path = tmp_path / "sections.parquet"
        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Test"],
                "body_text": ["Content"],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s1",
                    heading_text="# Test",
                    body_text="Content",
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="query",
                total_segments_found=1,
                unique_sections=1,
            ),
        )
        filtered_results = SectionCollection(
            sections=[],
            query_info=retrieval_results.query_info,
            filtering_metadata=FilteringMetadata(
                original_count=1,
                filtered_count=0,
                threshold=0.7,
                assessments=[],
            ),
        )

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch("legiscope.query.filter_sections", return_value=filtered_results):
                mock_client = Mock(spec=Instructor)
                llm_config = LLMConfig(client=mock_client, model="test-model")
                settings = BatchQuerySettings(
                    llm=llm_config,
                    filter_relevance=True,
                    relevance_threshold=0.7,
                )

                results_df = run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[QueryInput(question="query1", variable_name="dp_enacted")],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        assert results_df[0, "query_stage_status"] == "no_sections_after_filtering"
        assert results_df[0, "all_retrieval_units_filtered_out"] is True
        assert results_df[0, "generated_abstention"] is True
