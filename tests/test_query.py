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
import legiscope.query as query_module
from legiscope.query import (
    LegalQueryResponse,
    ResponseOptionEvidence,
    format_query_response,
    _build_no_sections_response,
    _repair_supporting_passages,
    _validate_supporting_passages,
    _prepare_legal_context,
    _build_legal_prompts,
    _normalize_option_text,
    _normalize_structured_short_answer,
    combine_query_input_batches,
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
from legiscope.query_hierarchy import (
    LabelBlockerRule,
    QueryHierarchy,
    REQUIRES_DATA_COLUMN,
    REQUIRES_LABELS_COLUMN,
    REQUIRES_YES_COLUMN,
    hierarchy_to_metadata,
)
from legiscope.retrieval_guidance import (
    RetrievalGuidance,
    RetrievalGuidanceRequest,
)
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

    def test_normalizes_scalar_citation_answer_to_single_best_unit(self):
        normalized = _normalize_structured_short_answer(
            "Relevant citation: Sections 30-31-1 et seq. NMSA 1978; § 30-31-2 NMSA 1978.",
            "citation_field",
            {
                "response_options": "Responses: <citation> OR Unknown",
            },
        )

        assert normalized == "§ 30-31-2"


class TestOptionPatternMap:
    def test_exemption_patterns_cover_dallas_medical_and_religious_aliases(self):
        patterns = query_module._option_pattern_map("exemption_presence")

        medical_key = _normalize_option_text(
            "Other paraphernalia for approved medical use"
        )
        other_key = _normalize_option_text("Other")

        assert r"\bauthorized to prescribe\b" in patterns[medical_key]
        assert r"\breligious ritual\b" in patterns[other_key]
        assert r"\bbona fide religious\b" in patterns[other_key]


class TestAuthoritativeOptionEvidenceGate:
    def test_promotes_supported_penalties_over_unlawful_only(self):
        response = LegalQueryResponse(
            short_answer='"Unlawful" only',
            reasoning="Initial answer undercalled the penalties.",
            citations=["§ 10.99"],
            supporting_passages=["A violation is punishable by a fine and imprisonment."],
            confidence=0.4,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option='"Unlawful" only', selected=True),
                ResponseOptionEvidence(option="Unspecified Fine", selected=False),
                ResponseOptionEvidence(option="Incarceration", selected=False),
            ],
        )
        sections = [
            SectionResult(
                section_id="s1",
                heading_text="# Penalty",
                body_text=(
                    "A violation is punishable by a fine not to exceed $500 or imprisonment for up to 60 days."
                ),
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        gated = query_module._apply_authoritative_option_evidence_gate(
            response,
            sections,
            {
                "guidance_topic": "penalty",
                "response_options": '"Unlawful" only AND/OR Unspecified Fine AND/OR Incarceration',
            },
        )

        assert gated.short_answer == "Unspecified Fine AND/OR Incarceration"
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "Unspecified Fine",
            "Incarceration",
        ]

    def test_rewrites_unsupported_sales_to_display_only(self):
        response = LegalQueryResponse(
            short_answer="Sales, possession with intent to sell, offer for sale",
            reasoning="Initial answer overcalled sales.",
            citations=["§ 12-1"],
            supporting_passages=["It is unlawful to display drug paraphernalia for advertising purposes."],
            confidence=0.5,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Sales, possession with intent to sell, offer for sale",
                    selected=True,
                ),
                ResponseOptionEvidence(option="Advertising, display", selected=False),
                ResponseOptionEvidence(option="Not specified", selected=False),
            ],
        )
        sections = [
            SectionResult(
                section_id="s2",
                heading_text="# Activity",
                body_text="It is unlawful to display or advertise drug paraphernalia.",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        gated = query_module._apply_authoritative_option_evidence_gate(
            response,
            sections,
            {
                "guidance_topic": "prohibited_activity",
                "response_options": (
                    "Sales, possession with intent to sell, offer for sale AND/OR "
                    "Advertising, display AND/OR Not specified"
                ),
            },
        )

        assert gated.short_answer == "Advertising, display"
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "Advertising, display"
        ]

    def test_promotes_supported_exemption_over_none(self):
        response = LegalQueryResponse(
            short_answer="None",
            reasoning="Initial answer missed the exemption.",
            citations=["§ 5-10"],
            supporting_passages=["Nothing in this section shall apply to cannabis paraphernalia."],
            confidence=0.4,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option="None", selected=True),
                ResponseOptionEvidence(
                    option="Paraphernalia for consumption of cannabis, generally or medical use",
                    selected=False,
                ),
            ],
        )
        sections = [
            SectionResult(
                section_id="s3",
                heading_text="# Exemptions",
                body_text="Nothing in this section shall apply to cannabis paraphernalia or medical marijuana accessories.",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        gated = query_module._apply_authoritative_option_evidence_gate(
            response,
            sections,
            {
                "guidance_topic": "exemption_presence",
                "response_options": (
                    "None AND/OR Paraphernalia for consumption of cannabis, generally or medical use"
                ),
            },
        )

        assert (
            gated.short_answer
            == "Paraphernalia for consumption of cannabis, generally or medical use"
        )
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "Paraphernalia for consumption of cannabis, generally or medical use"
        ]

    def test_defaults_binary_scope_question_to_no_when_yes_lacks_direct_support(self):
        response = LegalQueryResponse(
            short_answer="Yes",
            reasoning="Initial answer inferred an SSP law from nearby public-health text.",
            citations=[],
            supporting_passages=["A local public health emergency may be declared for communicable disease control."],
            confidence=0.35,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option="Yes", selected=True),
                ResponseOptionEvidence(option="No", selected=False),
            ],
        )

        gated = query_module._apply_authoritative_option_evidence_gate(
            response,
            [],
            {
                "guidance_topic": "ssp_scope",
                "response_options": "Yes OR No",
            },
        )

        assert gated.short_answer == "No"
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "No"
        ]

    def test_defaults_ssp_restriction_multi_select_to_no_restrictions_without_option_support(self):
        response = LegalQueryResponse(
            short_answer="Permit or license required for operation AND/OR Restrictions on mobile sites",
            reasoning="Initial answer inferred multiple restrictions from general administrative text.",
            citations=[],
            supporting_passages=["The program is recognized during a declared emergency."],
            confidence=0.3,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Permit or license required for operation",
                    selected=True,
                ),
                ResponseOptionEvidence(
                    option="Restrictions on mobile sites",
                    selected=True,
                ),
                ResponseOptionEvidence(option="No restrictions listed", selected=False),
            ],
        )

        gated = query_module._apply_authoritative_option_evidence_gate(
            response,
            [],
            {
                "guidance_topic": "ssp_restriction",
                "response_options": (
                    "Permit or license required for operation AND/OR Restrictions on mobile sites AND/OR No restrictions listed"
                ),
            },
        )

        assert gated.short_answer == "No restrictions listed"
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "No restrictions listed"
        ]


class TestTimeoutExecution:
    """Test timeout behavior for wrapped LLM calls."""

    def test_run_with_timeout_does_not_wait_for_timed_out_worker(self):
        fake_future = Mock()
        fake_future.result.side_effect = query_module.FutureTimeoutError()
        fake_executor = Mock()
        fake_executor.submit.return_value = fake_future

        with patch.object(
            query_module,
            "ThreadPoolExecutor",
            return_value=fake_executor,
        ):
            with pytest.raises(query_module.FutureTimeoutError):
                query_module._run_with_timeout(lambda: None, 0.01)

        fake_future.cancel.assert_called_once_with()
        fake_executor.shutdown.assert_called_once_with(
            wait=False,
            cancel_futures=True,
        )

    def test_query_legal_documents_records_initial_timeout_once(self):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Test Section",
                    body_text="Current through Ordinance 24-11.",
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="current through",
                total_segments_found=1,
                unique_sections=1,
            ),
        )
        debug_capture = {"query": {}}

        with patch.object(
            query_module,
            "_run_with_timeout",
            side_effect=query_module.FutureTimeoutError(),
        ):
            settings = QuerySettings(
                llm=LLMConfig(client=mock_client, model="test-model"),
                filter_relevance=False,
                validate_supporting_passages=False,
            )

            response, _similarity_scores = query_legal_documents(
                retrieval_results,
                "What is the current-through date of the ordinance?",
                settings,
                query_metadata={
                    "response_options": "Responses: <current-through date>"
                },
                debug_capture=debug_capture,
            )

        assert response.short_answer == "Error: LLM call timed out."
        assert debug_capture["query"]["stage_status"] == "timeout"
        assert (
            debug_capture["query"]["query_attempts"].count('"attempt_type": "initial"')
            == 1
        )

    def test_query_legal_documents_retries_on_timeout_by_dropping_last_chunk(self):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id=f"s{i}",
                    heading_text=f"# Section {i}",
                    body_text="Short supporting text.",
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
                for i in range(3)
            ],
            query_info=QueryInfo(
                original_query="timeout retry",
                total_segments_found=3,
                unique_sections=3,
            ),
        )
        debug_capture = {"query": {}}
        execution_capture: dict[str, object] = {}
        recovered_response = LegalQueryResponse(
            short_answer="Recovered answer",
            reasoning="Recovered after shrinking context.",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )
        
        with patch.object(
            query_module,
            "_run_with_timeout",
            side_effect=[query_module.FutureTimeoutError(), recovered_response],
        ):
            settings = QuerySettings(
                llm=LLMConfig(client=mock_client, model="test-model"),
                filter_relevance=False,
                validate_supporting_passages=False,
            )
            
            response, similarity_scores = query_legal_documents(
                retrieval_results,
                "What does the ordinance say?",
                settings,
                debug_capture=debug_capture,
                execution_capture=execution_capture,
            )
        
        assert response.short_answer == "Recovered answer"
        assert similarity_scores == []
        assert len(execution_capture["completion_sections"]) == 2
        assert execution_capture["completion_budgeting"]["overflow_retry_count"] == 1
        assert json.loads(
            debug_capture["query"]["overflow_retry_dropped_chunk_ids"]
        ) == ["s2"]

    def test_query_legal_documents_does_not_add_extra_initial_attempt_on_review_timeout(
        self,
    ):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Penalty",
                    body_text=(
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ),
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="penalty",
                total_segments_found=1,
                unique_sections=1,
            ),
        )
        first_response = LegalQueryResponse(
            reasoning="The text includes a fine and imprisonment.",
            citations=["§ 10.99"],
            supporting_passages=[
                "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
            ],
            confidence=0.73,
            limitations="None",
            option_evidence=[
                ResponseOptionEvidence(
                    option='"Unlawful" only',
                    selected=False,
                    confidence=0.05,
                    citations=[],
                    supporting_passages=[],
                ),
                ResponseOptionEvidence(
                    option="Unspecified Fine",
                    selected=True,
                    confidence=0.84,
                    citations=["§ 10.99"],
                    supporting_passages=[
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Incarceration",
                    selected=True,
                    confidence=0.81,
                    citations=["§ 10.99"],
                    supporting_passages=[
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ],
                ),
            ],
            short_answer='"Unlawful" only',
        )
        debug_capture = {"query": {}}

        with patch.object(
            query_module,
            "_run_with_timeout",
            side_effect=[first_response, query_module.FutureTimeoutError()],
        ):
            settings = QuerySettings(
                llm=LLMConfig(client=mock_client, model="test-model"),
                filter_relevance=False,
                validate_supporting_passages=False,
            )

            response, _similarity_scores = query_legal_documents(
                retrieval_results,
                "What penalties apply?",
                settings,
                query_metadata={
                    "response_options": 'Responses: "Unlawful" only AND/OR Infraction AND/OR Misdemeanor AND/OR Felony AND/OR Civil Fine AND/OR Criminal Fine AND/OR Unspecified Fine AND/OR Incarceration AND/OR Forfeiture/Seizure AND/OR Other',
                },
                debug_capture=debug_capture,
            )

        assert response.short_answer == "Error: LLM call timed out."
        assert (
            debug_capture["query"]["query_attempts"].count('"attempt_type": "initial"')
            == 1
        )
        assert (
            debug_capture["query"]["query_attempts"].count('"attempt_type": "review"')
            == 1
        )

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

    def test_normalize_option_text_ignores_new_suffix_annotations(self):
        assert _normalize_option_text("Civil Fine (NEW)") == "civil fine"
        assert (
            _normalize_option_text(
                "Sales, possession with intent to sell, offer for sale (NEW)"
            )
            == "sales possession with intent to sell offer for sale"
        )


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
        assert "copy-paste exact verbatim quotes" in system_prompt
        assert "join selections with ` AND/OR `" in system_prompt
        assert "Apply these coding instructions exactly" in system_prompt

    def test_build_legal_prompts_adds_option_evidence_and_none_other_rules(self):
        system_prompt, _user_prompt = _build_legal_prompts(
            "Which exemptions apply?",
            "Section 1: exception language.",
            query_metadata={
                "response_options": "Responses: None AND/OR Cannabis AND/OR Other",
            },
        )

        assert (
            "fill `option_evidence` with one entry per declared response option"
            in system_prompt
        )
        assert (
            "Treat `short_answer` as the final authoritative coded answer"
            in system_prompt
        )
        assert "Select `None` only if no specific option is supported" in system_prompt
        assert (
            "Select `Other` only when the legal text clearly supports an answer not captured"
            in system_prompt
        )
        assert (
            "mark it as selected=false rather than inferring it from nearby or loosely related text"
            in system_prompt
        )


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

    def test_load_queries_parses_hierarchy_metadata_with_pipe_delimiters(self):
        """Enriched CSV hierarchy columns should normalize into structured metadata."""
        df = pl.DataFrame(
            {
                "question_number": ["Q1", "Q1.1"],
                "question": ["Parent question", "Child question"],
                "variable_name": ["parent_var", "child_var"],
                REQUIRES_YES_COLUMN: ["", "Q1 || Q0"],
                REQUIRES_DATA_COLUMN: ["", "Q1"],
                REQUIRES_LABELS_COLUMN: [
                    "",
                    "Q1 => Syringes, generally || Pipes/smoking equipment, generally",
                ],
                "response_options": [
                    "Responses: Yes OR No",
                    'Responses: "Option, with comma" OR No',
                ],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)
            temp_path = f.name

        try:
            queries = load_queries(temp_path, adjust_for_dataset=False)
            assert queries[1].query_id == "Q1.1"
            assert queries[1].metadata["query_id"] == "Q1.1"
            assert (
                queries[1].metadata["response_options"]
                == 'Responses: "Option, with comma" OR No'
            )
            assert queries[1].metadata["hierarchy"] == {
                "query_id": "Q1.1",
                "parent_ids": ["Q1", "Q0"],
                "boolean_parent_ids": ["Q1", "Q0"],
                "context_parent_ids": ["Q1"],
                "label_blockers": [
                    {
                        "parent_query_id": "Q1",
                        "blocker_labels": [
                            "Syringes, generally",
                            "Pipes/smoking equipment, generally",
                        ],
                    }
                ],
                "pass_parent_question": True,
                "pass_parent_short_answer": True,
                "inherit_parent_retrieval": True,
            }
        finally:
            os.unlink(temp_path)


class TestCombineQueryInputBatches:
    def test_rekeys_duplicate_question_numbers_to_variable_names(self):
        hierarchy = hierarchy_to_metadata(QueryHierarchy(query_id="Q1"))

        combined = combine_query_input_batches(
            [
                [
                    QueryInput(
                        question="Drug paraphernalia law?",
                        variable_name="dp_law",
                        metadata={
                            "question_number": "Q1",
                            "query_id": "Q1",
                            "hierarchy": hierarchy,
                        },
                        query_id="Q1",
                    )
                ],
                [
                    QueryInput(
                        question="SSP law?",
                        variable_name="ssp_law",
                        metadata={
                            "question_number": "Q1",
                            "query_id": "Q1",
                            "hierarchy": hierarchy,
                        },
                        query_id="Q1",
                    )
                ],
            ]
        )

        assert [query.query_id for query in combined] == ["dp_law", "ssp_law"]
        assert combined[0].metadata["query_id"] == "dp_law"
        assert combined[1].metadata["query_id"] == "ssp_law"
        assert combined[0].metadata["hierarchy"]["query_id"] == "dp_law"
        assert combined[1].metadata["hierarchy"]["query_id"] == "ssp_law"

    def test_rejects_duplicate_variable_names_across_batches(self):
        with pytest.raises(ValueError, match="Duplicate variable_name values"):
            combine_query_input_batches(
                [
                    [QueryInput(question="Q1", variable_name="shared_var")],
                    [QueryInput(question="Q2", variable_name="shared_var")],
                ]
            )


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

        result = _validate_supporting_passages(response, sections)

        # Should match exactly due to normalization
        assert result.match_types == ["exact", "exact"]
        assert "validated (exact match)" in caplog.text
        assert "HALLUCINATION WARNING" not in caplog.text
        assert "NOT FOUND" not in caplog.text

    def test_validate_normalizes_section_prefixes_before_matching(self, caplog):
        response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=["No person shall sell drug paraphernalia."],
            confidence=0.9,
            limitations="",
        )

        sections = self.create_test_sections(
            body_text="§ 5-12-3: No person shall sell drug paraphernalia.",
        )

        result = _validate_supporting_passages(response, sections)

        assert result.match_types == ["exact"]
        assert result.similarity_scores == [1.0]
        assert "HALLUCINATION WARNING" not in caplog.text

    def test_validate_separates_near_exact_drift_from_true_not_found(self, caplog):
        response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=[
                "No person should sell drug paraphernalia items.",
                "This passage is completely fabricated.",
            ],
            confidence=0.9,
            limitations="",
        )

        sections = self.create_test_sections(
            body_text="Section 5-12-3: No person shall sell drug paraphernalia.",
        )

        result = _validate_supporting_passages(response, sections)

        assert result.match_types == ["near_exact", "not_found"]
        assert "DRIFT WARNING" in caplog.text
        assert "HALLUCINATION WARNING" in caplog.text

    def test_repair_supporting_passages_snaps_near_exact_match_to_source_text(self):
        response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=["No person should sell drug paraphernalia items."],
            confidence=0.9,
            limitations="",
        )

        sections = self.create_test_sections(
            body_text="Section 5-12-3: General rules.",
            segment_texts=["No person shall sell drug paraphernalia."],
        )

        validation_result = _validate_supporting_passages(response, sections)
        repaired_response, repaired_validation, repaired = _repair_supporting_passages(
            response,
            validation_result,
        )

        assert repaired is True
        assert repaired_response.supporting_passages == [
            "No person shall sell drug paraphernalia."
        ]
        assert repaired_validation.match_types == ["exact"]
        assert repaired_validation.similarity_scores == [1.0]

    def test_repair_supporting_passages_does_not_change_true_not_found_text(self):
        response = LegalQueryResponse(
            short_answer="Test",
            reasoning="Test",
            citations=[],
            supporting_passages=["This passage is completely fabricated."],
            confidence=0.9,
            limitations="",
        )

        sections = self.create_test_sections(
            body_text="Section 5-12-3: No person shall sell drug paraphernalia.",
        )

        validation_result = _validate_supporting_passages(response, sections)
        repaired_response, repaired_validation, repaired = _repair_supporting_passages(
            response,
            validation_result,
        )

        assert repaired is False
        assert repaired_response.supporting_passages == response.supporting_passages
        assert repaired_validation.match_types == ["not_found"]


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

    def test_query_legal_documents_preflights_completion_budget_and_records_drops(
        self, monkeypatch
    ):
        mock_client = Mock(spec=Instructor)
        sections = [
            SectionResult(
                section_id=f"s{i}",
                heading_text=f"# Section {i}",
                body_text=" ".join(["word"] * 80),
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.1,
                segment_count=1,
            )
            for i in range(3)
        ]
        retrieval_results = SectionCollection(
            sections=sections,
            query_info=QueryInfo(
                original_query="test query",
                total_segments_found=3,
                unique_sections=3,
            ),
        )
        mock_response = LegalQueryResponse(
            short_answer="Budgeted answer",
            reasoning="Budgeted reasoning",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )
        debug_capture = {"query": {}}
        execution_capture: dict[str, object] = {}

        monkeypatch.setattr("legiscope.query.DEFAULT_COMPLETION_CONTEXT_LIMIT", 4200)

        with patch("legiscope.query.ask", return_value=mock_response):
            settings = QuerySettings(
                llm=LLMConfig(client=mock_client, model="test-model"),
                filter_relevance=False,
                validate_supporting_passages=False,
            )

            response, similarity_scores = query_legal_documents(
                retrieval_results,
                "What does the ordinance say?",
                settings,
                debug_capture=debug_capture,
                execution_capture=execution_capture,
            )

        assert response.short_answer == "Budgeted answer"
        assert similarity_scores == []
        assert len(execution_capture["completion_sections"]) < len(sections)
        budget_metadata = execution_capture["completion_budgeting"]
        assert budget_metadata["preflight_dropped_count"] > 0
        assert (
            json.loads(debug_capture["query"]["completion_total_dropped_chunk_ids"])
            == budget_metadata["total_dropped_chunk_ids"]
        )

    def test_query_legal_documents_retries_on_context_overflow_by_dropping_last_chunk(
        self, monkeypatch
    ):
        mock_client = Mock(spec=Instructor)
        sections = [
            SectionResult(
                section_id=f"s{i}",
                heading_text=f"# Section {i}",
                body_text="Short supporting text.",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.1,
                segment_count=1,
            )
            for i in range(3)
        ]
        retrieval_results = SectionCollection(
            sections=sections,
            query_info=QueryInfo(
                original_query="test query",
                total_segments_found=3,
                unique_sections=3,
            ),
        )
        debug_capture = {"query": {}}
        execution_capture: dict[str, object] = {}
        mock_response = LegalQueryResponse(
            short_answer="Recovered answer",
            reasoning="Recovered reasoning",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )
        call_count = 0

        def fake_ask(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError(
                    "This model's maximum context length is 32768 tokens. However, your prompt contains at least 40000 tokens."
                )
            return mock_response

        monkeypatch.setattr("legiscope.query.DEFAULT_COMPLETION_CONTEXT_LIMIT", 32768)

        with patch("legiscope.query.ask", side_effect=fake_ask):
            settings = QuerySettings(
                llm=LLMConfig(client=mock_client, model="test-model"),
                filter_relevance=False,
                validate_supporting_passages=False,
            )

            response, similarity_scores = query_legal_documents(
                retrieval_results,
                "What does the ordinance say?",
                settings,
                debug_capture=debug_capture,
                execution_capture=execution_capture,
            )

        assert response.short_answer == "Recovered answer"
        assert similarity_scores == []
        assert call_count == 2
        assert len(execution_capture["completion_sections"]) == 2
        assert execution_capture["completion_budgeting"]["overflow_retry_count"] == 1
        assert json.loads(
            debug_capture["query"]["overflow_retry_dropped_chunk_ids"]
        ) == ["s2"]

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

    def test_query_legal_documents_runs_targeted_review_for_suspicious_penalty_answer(
        self,
    ):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Penalty",
                    body_text=(
                        "Penalty, see Section 10.99. A violation is punishable by a fine "
                        "not to exceed $500 or imprisonment for a period not to exceed 60 days, "
                        "or both such fine and imprisonment."
                    ),
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="penalty",
                total_segments_found=1,
                unique_sections=1,
            ),
        )
        first_response = LegalQueryResponse(
            short_answer='"Unlawful" only',
            reasoning="No explicit penalty located.",
            citations=["§ 134.28"],
            supporting_passages=["It is unlawful for any person..."],
            confidence=0.62,
            limitations="None",
        )
        second_response = LegalQueryResponse(
            short_answer="Unspecified Fine AND/OR Incarceration",
            reasoning="The cited penalty section provides both a fine and imprisonment.",
            citations=["§ 10.99"],
            supporting_passages=[
                "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days"
            ],
            confidence=0.88,
            limitations="None",
        )
        debug_capture = {"query": {}}
        prompts: list[str] = []

        def fake_ask(*args, **kwargs):
            prompts.append(kwargs["prompt"])
            if len(prompts) == 1:
                return first_response
            return second_response

        with patch("legiscope.query.ask", side_effect=fake_ask):
            settings = QuerySettings(
                llm=LLMConfig(client=mock_client, model="test-model"),
                filter_relevance=False,
                validate_supporting_passages=False,
            )

            response, _similarity_scores = query_legal_documents(
                retrieval_results,
                "What penalties apply?",
                settings,
                query_metadata={
                    "response_options": 'Responses: "Unlawful" only AND/OR Infraction AND/OR Misdemeanor AND/OR Felony AND/OR Civil Fine AND/OR Criminal Fine AND/OR Unspecified Fine AND/OR Incarceration AND/OR Forfeiture/Seizure AND/OR Other',
                    "guidance_topic": "penalty",
                },
                debug_capture=debug_capture,
            )

        assert len(prompts) == 2
        assert response.short_answer == "Unspecified Fine AND/OR Incarceration"
        assert debug_capture["query"]["review_rerun_triggered"] is True
        assert debug_capture["query"]["review_rerun_guidance_topic"] == "penalty"
        assert "Review request:" in prompts[1]
        assert 'Original short_answer: "Unlawful" only' in prompts[1]

    def test_augment_sections_with_same_text_cross_references_imports_local_section_once(
        self, tmp_path
    ):
        sections_path = tmp_path / "sections.parquet"
        pl.DataFrame(
            {
                "section_id": ["s0", "s_penalty"],
                "section_ordinal": [0, 1],
                "heading_text": [
                    "# Drug Paraphernalia",
                    "### SEC. 10.99. GENERAL PENALTY.",
                ],
                "body_text": [
                    "Penalty, see Section 10.99.",
                    "A violation is punishable by a fine not to exceed $500.",
                ],
                "heading_level": [1, 3],
                "parent_id": [None, None],
                "context_path": [None, "Chapter 10 > Section 10.99"],
            }
        ).write_parquet(sections_path)

        source_section = SectionResult(
            section_id="s0",
            heading_text="# Drug Paraphernalia",
            body_text="Penalty, see Section 10.99.",
            heading_level=1,
            parent_id=None,
            matching_segments=[],
            relevance_score=0.1,
            segment_count=1,
        )

        augmented = query_module._augment_sections_with_same_text_cross_references(
            [source_section],
            sections_parquet_path=str(sections_path),
            guidance_topic="penalty",
        )
        deduped = query_module._augment_sections_with_same_text_cross_references(
            augmented,
            sections_parquet_path=str(sections_path),
            guidance_topic="penalty",
        )

        assert [section.section_id for section in augmented] == ["s0", "s_penalty"]
        assert [section.section_id for section in deduped] == ["s0", "s_penalty"]

    def test_query_legal_documents_imports_same_text_penalty_cross_reference_into_completion_context(
        self, tmp_path
    ):
        sections_path = tmp_path / "sections.parquet"
        pl.DataFrame(
            {
                "section_id": ["s0", "s_penalty"],
                "section_ordinal": [0, 1],
                "heading_text": [
                    "# Drug Paraphernalia",
                    "### SEC. 10.99. GENERAL PENALTY.",
                ],
                "body_text": [
                    "Penalty, see Section 10.99.",
                    "A violation is punishable by a fine not to exceed $500 or imprisonment for up to 60 days.",
                ],
                "heading_level": [1, 3],
                "parent_id": [None, None],
                "context_path": [None, "Chapter 10 > Section 10.99"],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Drug Paraphernalia",
                    body_text="Penalty, see Section 10.99.",
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="penalty",
                total_segments_found=1,
                unique_sections=1,
            ),
        )
        mock_response = LegalQueryResponse(
            short_answer="Unspecified Fine AND/OR Incarceration",
            reasoning="The imported penalty section provides both a fine and imprisonment.",
            citations=["§ 10.99"],
            supporting_passages=[
                "A violation is punishable by a fine not to exceed $500 or imprisonment for up to 60 days."
            ],
            confidence=0.87,
            limitations="None",
        )
        prompt_texts: list[str] = []
        execution_capture: dict[str, object] = {}

        def fake_ask(*args, **kwargs):
            prompt_texts.append(kwargs["prompt"])
            return mock_response

        with patch("legiscope.query.ask", side_effect=fake_ask):
            settings = QuerySettings(
                llm=LLMConfig(client=Mock(spec=Instructor), model="test-model"),
                filter_relevance=False,
                retrieval_guidance=RetrievalGuidance(guidance_topic="penalty"),
                same_text_sections_parquet_path=str(sections_path),
                validate_supporting_passages=False,
            )

            response, _similarity_scores = query_legal_documents(
                retrieval_results,
                "What penalties apply?",
                settings,
                execution_capture=execution_capture,
            )

        assert response.short_answer == "Unspecified Fine AND/OR Incarceration"
        assert [
            section.section_id
            for section in execution_capture["completion_sections"]
        ] == ["s0", "s_penalty"]
        assert "SEC. 10.99. GENERAL PENALTY." in prompt_texts[0]
        assert "punishable by a fine not to exceed $500" in prompt_texts[0]

    def test_query_legal_documents_skips_review_when_supported_activity_answer_is_consistent(
        self,
    ):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Drug Paraphernalia",
                    body_text=(
                        "It is unlawful for any person to use or possess with intent to use drug paraphernalia. "
                        "It is unlawful for any person to place any advertisement to promote the sale of objects designed for use as drug paraphernalia."
                    ),
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="activity",
                total_segments_found=1,
                unique_sections=1,
            ),
        )
        response_payload = LegalQueryResponse(
            short_answer="Possession, possession with intent to use, keep AND/OR Use AND/OR Advertising, display",
            reasoning="The text expressly prohibits use/possess with intent to use and advertisement.",
            citations=["§ 134.28(A)", "§ 134.28(C)"],
            supporting_passages=[
                "It is unlawful for any person to use or possess with intent to use drug paraphernalia.",
                "It is unlawful for any person to place any advertisement to promote the sale of objects designed for use as drug paraphernalia.",
            ],
            confidence=0.84,
            limitations="None",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Sales, possession with intent to sell, offer for sale",
                    selected=False,
                    confidence=0.05,
                    citations=[],
                    supporting_passages=[],
                ),
                ResponseOptionEvidence(
                    option="Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange",
                    selected=False,
                    confidence=0.05,
                    citations=[],
                    supporting_passages=[],
                ),
                ResponseOptionEvidence(
                    option="Give away, give, gift, free distribution",
                    selected=False,
                    confidence=0.05,
                    citations=[],
                    supporting_passages=[],
                ),
                ResponseOptionEvidence(
                    option="Possession, possession with intent to use, keep",
                    selected=True,
                    confidence=0.88,
                    citations=["§ 134.28(A)"],
                    supporting_passages=[
                        "It is unlawful for any person to use or possess with intent to use drug paraphernalia."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Use",
                    selected=True,
                    confidence=0.88,
                    citations=["§ 134.28(A)"],
                    supporting_passages=[
                        "It is unlawful for any person to use or possess with intent to use drug paraphernalia."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Advertising, display",
                    selected=True,
                    confidence=0.86,
                    citations=["§ 134.28(C)"],
                    supporting_passages=[
                        "It is unlawful for any person to place any advertisement to promote the sale of objects designed for use as drug paraphernalia."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Manufacturing, manufacture with intent to deliver or sell",
                    selected=False,
                    confidence=0.05,
                    citations=[],
                    supporting_passages=[],
                ),
                ResponseOptionEvidence(
                    option="Other",
                    selected=False,
                    confidence=0.05,
                    citations=[],
                    supporting_passages=[],
                ),
                ResponseOptionEvidence(
                    option="Not specified",
                    selected=False,
                    confidence=0.05,
                    citations=[],
                    supporting_passages=[],
                ),
            ],
        )

        with patch("legiscope.query.ask", return_value=response_payload) as mock_ask:
            settings = QuerySettings(
                llm=LLMConfig(client=mock_client, model="test-model"),
                filter_relevance=False,
                validate_supporting_passages=False,
            )

            response, _similarity_scores = query_legal_documents(
                retrieval_results,
                "Which activities are prohibited?",
                settings,
                query_metadata={
                    "response_options": "Responses: Sales, possession with intent to sell, offer for sale AND/OR Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange AND/OR Give away, give, gift, free distribution AND/OR Possession, possession with intent to use, keep AND/OR Use AND/OR Advertising, display AND/OR Manufacturing, manufacture with intent to deliver or sell AND/OR Other AND/OR Not specified",
                    "guidance_topic": "prohibited_activity",
                },
            )

        assert mock_ask.call_count == 1
        assert response.short_answer == response_payload.short_answer

    def test_query_legal_documents_reruns_when_short_answer_conflicts_with_option_evidence(
        self,
    ):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Penalty",
                    body_text=(
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ),
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="penalty",
                total_segments_found=1,
                unique_sections=1,
            ),
        )
        first_response = LegalQueryResponse(
            reasoning="The text includes a fine and imprisonment.",
            citations=["§ 10.99"],
            supporting_passages=[
                "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
            ],
            confidence=0.73,
            limitations="None",
            option_evidence=[
                ResponseOptionEvidence(
                    option='"Unlawful" only',
                    selected=False,
                    confidence=0.05,
                    citations=[],
                    supporting_passages=[],
                ),
                ResponseOptionEvidence(
                    option="Unspecified Fine",
                    selected=True,
                    confidence=0.84,
                    citations=["§ 10.99"],
                    supporting_passages=[
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Incarceration",
                    selected=True,
                    confidence=0.81,
                    citations=["§ 10.99"],
                    supporting_passages=[
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ],
                ),
            ],
            short_answer='"Unlawful" only',
        )
        second_response = LegalQueryResponse(
            reasoning="The text expressly provides both a fine and imprisonment.",
            citations=["§ 10.99"],
            supporting_passages=[
                "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
            ],
            confidence=0.9,
            limitations="None",
            option_evidence=[
                ResponseOptionEvidence(
                    option='"Unlawful" only',
                    selected=False,
                    confidence=0.02,
                    citations=[],
                    supporting_passages=[],
                ),
                ResponseOptionEvidence(
                    option="Unspecified Fine",
                    selected=True,
                    confidence=0.9,
                    citations=["§ 10.99"],
                    supporting_passages=[
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Incarceration",
                    selected=True,
                    confidence=0.89,
                    citations=["§ 10.99"],
                    supporting_passages=[
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ],
                ),
            ],
            short_answer="Unspecified Fine AND/OR Incarceration",
        )
        debug_capture = {"query": {}}
        prompts: list[str] = []

        def fake_ask(*args, **kwargs):
            prompts.append(kwargs["prompt"])
            if len(prompts) == 1:
                return first_response
            return second_response

        with patch("legiscope.query.ask", side_effect=fake_ask):
            settings = QuerySettings(
                llm=LLMConfig(client=mock_client, model="test-model"),
                filter_relevance=False,
                validate_supporting_passages=False,
            )

            response, _similarity_scores = query_legal_documents(
                retrieval_results,
                "What penalties apply?",
                settings,
                query_metadata={
                    "response_options": 'Responses: "Unlawful" only AND/OR Infraction AND/OR Misdemeanor AND/OR Felony AND/OR Civil Fine AND/OR Criminal Fine AND/OR Unspecified Fine AND/OR Incarceration AND/OR Forfeiture/Seizure AND/OR Other',
                },
                debug_capture=debug_capture,
            )

        assert len(prompts) == 2
        assert response.short_answer == "Unspecified Fine AND/OR Incarceration"
        assert debug_capture["query"]["review_rerun_triggered"] is True
        assert (
            debug_capture["query"]["review_rerun_guidance_topic"]
            == "response_option_consistency"
        )
        assert (
            "short_answer_conflicts_with_option_evidence"
            in debug_capture["query"]["review_rerun_reasons"]
            or "incomplete_option_evidence"
            in debug_capture["query"]["review_rerun_reasons"]
        )
        assert '"attempt_type": "review"' in debug_capture["query"]["query_attempts"]

    def test_query_legal_documents_reruns_when_option_evidence_is_missing(self):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Exemptions",
                    body_text=(
                        "This article does not apply to syringes distributed through a syringe exchange program."
                    ),
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="exemption",
                total_segments_found=1,
                unique_sections=1,
            ),
        )
        first_response = LegalQueryResponse(
            short_answer="None",
            reasoning="No exemption is clear.",
            citations=["§ 1.23"],
            supporting_passages=[
                "This article does not apply to syringes distributed through a syringe exchange program."
            ],
            confidence=0.58,
            limitations="None",
            option_evidence=[],
        )
        second_response = LegalQueryResponse(
            short_answer="Syringes from syringe services, harm reduction programs, or supervised use sites",
            reasoning="The text expressly exempts syringes distributed through a syringe exchange program.",
            citations=["§ 1.23"],
            supporting_passages=[
                "This article does not apply to syringes distributed through a syringe exchange program."
            ],
            confidence=0.88,
            limitations="None",
            option_evidence=[
                ResponseOptionEvidence(
                    option="None",
                    selected=False,
                    confidence=0.05,
                    citations=[],
                    supporting_passages=[],
                ),
                ResponseOptionEvidence(
                    option="Syringes from syringe services, harm reduction programs, or supervised use sites",
                    selected=True,
                    confidence=0.88,
                    citations=["§ 1.23"],
                    supporting_passages=[
                        "This article does not apply to syringes distributed through a syringe exchange program."
                    ],
                ),
            ],
        )
        debug_capture = {"query": {}}
        prompts: list[str] = []

        def fake_ask(*args, **kwargs):
            prompts.append(kwargs["prompt"])
            if len(prompts) == 1:
                return first_response
            return second_response

        with patch("legiscope.query.ask", side_effect=fake_ask):
            settings = QuerySettings(
                llm=LLMConfig(client=mock_client, model="test-model"),
                filter_relevance=False,
                validate_supporting_passages=False,
            )

            response, _similarity_scores = query_legal_documents(
                retrieval_results,
                "Are there any exemptions?",
                settings,
                query_metadata={
                    "response_options": "Responses: None AND/OR Syringes from syringe services, harm reduction programs, or supervised use sites",
                },
                debug_capture=debug_capture,
            )

        assert len(prompts) == 2
        assert (
            response.short_answer
            == "Syringes from syringe services, harm reduction programs, or supervised use sites"
        )
        assert debug_capture["query"]["review_rerun_triggered"] is True
        assert (
            debug_capture["query"]["review_rerun_guidance_topic"]
            == "response_option_consistency"
        )
        assert (
            "missing_option_evidence" in debug_capture["query"]["review_rerun_reasons"]
        )

    def test_query_legal_documents_skips_review_for_scalar_date_placeholder(self):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Ordinance history",
                    body_text="(Ord. 96-1973; Am. Ord. 2-1981; Am. Ord. 2018-005; Am. Ord. 2022-009)",
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="enacted",
                total_segments_found=1,
                unique_sections=1,
            ),
        )
        response_payload = LegalQueryResponse(
            short_answer="07/15/2022",
            reasoning="The latest amendment date in range is 2022.",
            citations=["Ord. 2022-009"],
            supporting_passages=[
                "(Ord. 96-1973; Am. Ord. 2-1981; Am. Ord. 2018-005; Am. Ord. 2022-009)"
            ],
            confidence=0.85,
            limitations="None",
            option_evidence=[],
        )
        debug_capture = {"query": {}}

        with patch("legiscope.query.ask", return_value=response_payload) as mock_ask:
            settings = QuerySettings(
                llm=LLMConfig(client=mock_client, model="test-model"),
                filter_relevance=False,
                validate_supporting_passages=False,
            )

            response, _similarity_scores = query_legal_documents(
                retrieval_results,
                "On which date was the ordinance enacted?",
                settings,
                query_metadata={
                    "response_options": "Responses: <enactment date> OR Unknown",
                },
                debug_capture=debug_capture,
            )

        assert mock_ask.call_count == 1
        assert response.short_answer == "07/15/2022"
        assert debug_capture["query"].get("review_rerun_triggered") in (None, False)

    def test_query_legal_documents_skips_review_for_scalar_citation_placeholder(self):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# State law reference",
                    body_text="This section incorporates the State Controlled Substances Act, Sections 30-31-1 et seq. NMSA 1978.",
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="citation",
                total_segments_found=1,
                unique_sections=1,
            ),
        )
        response_payload = LegalQueryResponse(
            short_answer="Sections 30-31-1 et seq. NMSA 1978",
            reasoning="The state law citation is explicit.",
            citations=["§ 12-4-10"],
            supporting_passages=[
                "This section incorporates the State Controlled Substances Act, Sections 30-31-1 et seq. NMSA 1978."
            ],
            confidence=0.95,
            limitations="None",
            option_evidence=[
                ResponseOptionEvidence(
                    option="<citation>",
                    selected=True,
                    confidence=0.95,
                    citations=["§ 12-4-10"],
                    supporting_passages=[
                        "This section incorporates the State Controlled Substances Act, Sections 30-31-1 et seq. NMSA 1978."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Unknown",
                    selected=False,
                    confidence=0.0,
                    citations=[],
                    supporting_passages=[],
                ),
            ],
        )
        debug_capture = {"query": {}}

        with patch("legiscope.query.ask", return_value=response_payload) as mock_ask:
            settings = QuerySettings(
                llm=LLMConfig(client=mock_client, model="test-model"),
                filter_relevance=False,
                validate_supporting_passages=False,
            )

            response, _similarity_scores = query_legal_documents(
                retrieval_results,
                "If yes, what is the citation of the relevant law?",
                settings,
                query_metadata={
                    "response_options": "Responses: <citation> OR Unknown",
                },
                debug_capture=debug_capture,
            )

        assert mock_ask.call_count == 1
        assert response.short_answer == "§ 30-31-1"
        assert debug_capture["query"].get("review_rerun_triggered") in (None, False)

    def test_query_with_relevance_filtering(self):
        """Test query with relevance filtering enabled."""
        from legiscope.llm_config import Config

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

                llm_config = LLMConfig(
                    client=mock_client,
                    model=Config.get_fast_model(),
                    source="self_hosted",
                    client_factory=Config.get_fast_client,
                )
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
                assert (
                    mock_filter.call_args.kwargs["client_factory"].__func__
                    is Config.get_fast_client.__func__
                )

    def test_query_with_relevance_filtering_uses_no_client_factory_for_external_llm(
        self,
    ):
        """External LLM configs should not enable threaded relevance filtering."""
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

                llm_config = LLMConfig(
                    client=mock_client,
                    model="external-model",
                    source="external",
                )
                settings = QuerySettings(
                    llm=llm_config,
                    filter_relevance=True,
                    relevance_threshold=0.7,
                )

                response, _similarity_scores = query_legal_documents(
                    retrieval_results, "test query", settings
                )

                assert response.short_answer == "Test answer"
                mock_filter.assert_called_once()
                assert mock_filter.call_args.kwargs["client_factory"] is None


class TestBatchQueryConfigBasics:
    """Test BatchQuerySettings-based run_queries function."""

    def test_run_queries_records_completion_drop_metadata(self, tmp_path, monkeypatch):
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
                    section_id=f"s{i}",
                    heading_text=f"# Section {i}",
                    body_text=" ".join(["word"] * 80),
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
                for i in range(3)
            ],
            query_info=QueryInfo(
                original_query="test query",
                total_segments_found=3,
                unique_sections=3,
            ),
        )
        mock_response = LegalQueryResponse(
            short_answer="Budgeted answer",
            reasoning="Budgeted reasoning",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        monkeypatch.setattr("legiscope.query.DEFAULT_COMPLETION_CONTEXT_LIMIT", 4200)

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch("legiscope.query.ask", return_value=mock_response):
                settings = BatchQuerySettings(
                    llm=LLMConfig(client=Mock(spec=Instructor), model="test-model"),
                    filter_relevance=False,
                    validate_supporting_passages=False,
                )

                results_df = run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[QueryInput(question="query1", variable_name="dp_test")],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        assert results_df[0, "completion_total_dropped_count"] > 0
        assert json.loads(results_df[0, "completion_preflight_dropped_chunk_ids"]) != []

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

    def test_run_queries_writes_review_attempts_to_query_debug_csv(self, tmp_path):
        sections_path = tmp_path / "sections.parquet"
        debug_dir = tmp_path / "debug"
        debug_timestamp = "20260513_2015"

        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Penalty"],
                "body_text": [
                    "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                ],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s_penalty",
                    heading_text="# Penalty",
                    body_text=(
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ),
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="penalty",
                total_segments_found=1,
                unique_sections=1,
            ),
        )
        first_response = LegalQueryResponse(
            reasoning="The text includes a fine and imprisonment.",
            citations=["§ 10.99"],
            supporting_passages=[
                "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
            ],
            confidence=0.73,
            limitations="None",
            option_evidence=[
                ResponseOptionEvidence(
                    option='"Unlawful" only',
                    selected=False,
                    confidence=0.05,
                ),
                ResponseOptionEvidence(
                    option="Unspecified Fine",
                    selected=True,
                    confidence=0.84,
                    citations=["§ 10.99"],
                    supporting_passages=[
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Incarceration",
                    selected=True,
                    confidence=0.81,
                    citations=["§ 10.99"],
                    supporting_passages=[
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ],
                ),
            ],
            short_answer='"Unlawful" only',
        )
        second_response = LegalQueryResponse(
            reasoning="The text expressly provides both a fine and imprisonment.",
            citations=["§ 10.99"],
            supporting_passages=[
                "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
            ],
            confidence=0.9,
            limitations="None",
            option_evidence=[
                ResponseOptionEvidence(
                    option='"Unlawful" only',
                    selected=False,
                    confidence=0.02,
                ),
                ResponseOptionEvidence(
                    option="Unspecified Fine",
                    selected=True,
                    confidence=0.9,
                    citations=["§ 10.99"],
                    supporting_passages=[
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Incarceration",
                    selected=True,
                    confidence=0.89,
                    citations=["§ 10.99"],
                    supporting_passages=[
                        "A violation is punishable by a fine not to exceed $500 or imprisonment for a period not to exceed 60 days."
                    ],
                ),
            ],
            short_answer="Unspecified Fine AND/OR Incarceration",
        )

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch(
                "legiscope.query.ask", side_effect=[first_response, second_response]
            ):
                settings = BatchQuerySettings(
                    llm=LLMConfig(client=Mock(spec=Instructor), model="test-model"),
                    debug_dir=debug_dir,
                    debug_timestamp=debug_timestamp,
                    filter_relevance=False,
                    validate_supporting_passages=False,
                )

                run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="What penalties apply?",
                            variable_name="dp_penalties",
                            metadata={
                                "response_options": 'Responses: "Unlawful" only AND/OR Infraction AND/OR Misdemeanor AND/OR Felony AND/OR Civil Fine AND/OR Criminal Fine AND/OR Unspecified Fine AND/OR Incarceration AND/OR Forfeiture/Seizure AND/OR Other',
                            },
                        )
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        query_debug = pl.read_csv(debug_dir / f"query_stage_{debug_timestamp}.csv")
        assert len(query_debug) == 1
        assert query_debug[0, "review_rerun_triggered"] is True
        assert (
            query_debug[0, "review_rerun_guidance_topic"]
            == "response_option_consistency"
        )
        assert '"attempt_type": "initial"' in query_debug[0, "query_attempts"]
        assert '"attempt_type": "review"' in query_debug[0, "query_attempts"]

    def test_run_queries_writes_failed_attempts_to_query_debug_csv(self, tmp_path):
        sections_path = tmp_path / "sections.parquet"
        debug_dir = tmp_path / "debug"
        debug_timestamp = "20260513_230500"

        pl.DataFrame(
            {
                "section_ordinal": [0],
                "heading_text": ["# Test"],
                "body_text": ["It is unlawful to deliver drug paraphernalia."],
                "heading_level": [1],
                "parent_id": [None],
            }
        ).write_parquet(sections_path)

        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Test",
                    body_text="It is unlawful to deliver drug paraphernalia.",
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="activity",
                total_segments_found=1,
                unique_sections=1,
            ),
        )

        failure = Exception(
            "The output is incomplete due to a max_tokens length limit."
        )

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch("legiscope.query.ask", side_effect=failure):
                settings = BatchQuerySettings(
                    llm=LLMConfig(client=Mock(spec=Instructor), model="test-model"),
                    debug_dir=debug_dir,
                    debug_timestamp=debug_timestamp,
                    filter_relevance=False,
                    validate_supporting_passages=False,
                )

                run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="Which activities are prohibited?",
                            variable_name="dp_activity",
                            metadata={
                                "response_options": "Responses: Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange AND/OR Other",
                            },
                        )
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        query_debug = pl.read_csv(debug_dir / f"query_stage_{debug_timestamp}.csv")
        assert len(query_debug) == 1
        assert query_debug[0, "stage_status"] == "error"
        assert '"attempt_type": "initial"' in query_debug[0, "query_attempts"]
        assert '"status": "error"' in query_debug[0, "query_attempts"]
        assert '"max_token_limited": true' in query_debug[0, "query_attempts"]

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
        assert (
            metadata["query_text"]
            == "What is the current-through date of the ordinance?"
        )
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
            with patch(
                "legiscope.query.filter_sections", return_value=filtered_results
            ):
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

    def test_build_no_sections_response_uses_structured_ssp_fallback_when_no_anchor_text(self):
        response = _build_no_sections_response(
            "no_sections_after_filtering",
            query_metadata={
                "variable_name": "ssp_law",
                "response_options": "Yes OR No",
            },
            retrieval_guidance=RetrievalGuidance(
                guidance_topic="ssp_scope",
                anchor_terms=["syringe exchange facility", "needle exchange"],
                no_context_fallback_short_answer="No",
            ),
            original_sections=[
                SectionResult(
                    section_id="s1",
                    heading_text="### Retail uses",
                    body_text="Paraphernalia shop zoning text only.",
                    heading_level=3,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
        )

        assert response.short_answer == "No"
        assert not response.short_answer.startswith("I cannot answer your question")

    def test_build_no_sections_response_keeps_abstention_when_anchor_text_was_seen(self):
        response = _build_no_sections_response(
            "no_sections_after_filtering",
            query_metadata={
                "variable_name": "ssp_law",
                "response_options": "Yes OR No",
            },
            retrieval_guidance=RetrievalGuidance(
                guidance_topic="ssp_scope",
                anchor_terms=["syringe exchange facility", "needle exchange"],
                no_context_fallback_short_answer="No",
            ),
            original_sections=[
                SectionResult(
                    section_id="s1",
                    heading_text="## ARTICLE 15: SYRINGE EXCHANGE FACILITY LOCATION",
                    body_text="This article shall be known as the Syringe Exchange Facility Location Ordinance.",
                    heading_level=2,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
        )

        assert response.short_answer.startswith(
            "I cannot answer your question as no relevant legal provisions were found after filtering."
        )


class TestHierarchicalQueryExecution:
    """Focused regressions for parent/child execution behavior."""

    @staticmethod
    def _write_sections_parquet(tmp_path):
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
        return sections_path

    @staticmethod
    def _make_section(section_id: str, chunk_id: str | None = None) -> SectionResult:
        return SectionResult(
            section_id=section_id,
            heading_text=f"Section {section_id}",
            body_text=f"Body for {section_id}",
            heading_level=1,
            parent_id=None,
            matching_segments=[],
            relevance_score=0.1,
            segment_count=1,
            chunk_id=chunk_id,
        )

    def test_run_queries_skips_child_when_requires_yes_parent_is_explicit_no(
        self, tmp_path
    ):
        sections_path = self._write_sections_parquet(tmp_path)
        retrieval_results = SectionCollection(
            sections=[self._make_section("s1")],
            query_info=QueryInfo(
                original_query="query", total_segments_found=1, unique_sections=1
            ),
        )
        parent_response = LegalQueryResponse(
            short_answer="No",
            reasoning="No exemption exists.",
            citations=[],
            supporting_passages=[],
            confidence=0.9,
            limitations="None",
        )

        child_hierarchy = QueryHierarchy(
            query_id="Q1.1",
            parent_ids=("parent_var",),
            boolean_parent_ids=("parent_var",),
        )
        parent_hierarchy = QueryHierarchy(query_id="Q1")

        with patch(
            "legiscope.query.retrieve_sections", return_value=retrieval_results
        ) as mock_retrieve:
            with patch(
                "legiscope.query.query_legal_documents",
                side_effect=[(parent_response, [])],
            ) as mock_query:
                mock_client = Mock(spec=Instructor)
                settings = BatchQuerySettings(
                    llm=LLMConfig(client=mock_client, model="test-model"),
                )

                results_df = run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="Parent",
                            variable_name="parent_var",
                            metadata={
                                "response_options": "Responses: Yes OR No",
                                "hierarchy": hierarchy_to_metadata(parent_hierarchy),
                            },
                            query_id="Q1",
                        ),
                        QueryInput(
                            question="Child",
                            variable_name="child_var",
                            metadata={
                                "hierarchy": hierarchy_to_metadata(child_hierarchy),
                            },
                            query_id="Q1.1",
                        ),
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        assert mock_query.call_count == 1
        assert mock_retrieve.call_count == 1
        child_row = results_df.filter(pl.col("query_id") == "Q1.1")
        assert child_row[0, "query_status"] == "skipped"
        assert child_row[0, "query_stage_status"] == "skipped"
        assert child_row[0, "skip_reason"] == "requires_yes_not_satisfied"
        assert child_row[0, "blocking_parent_query_id"] == "Q1"

    def test_run_queries_executes_child_when_required_parent_is_missing(self, tmp_path):
        sections_path = self._write_sections_parquet(tmp_path)
        retrieval_results = SectionCollection(
            sections=[self._make_section("s1")],
            query_info=QueryInfo(
                original_query="query", total_segments_found=1, unique_sections=1
            ),
        )
        response = LegalQueryResponse(
            short_answer="Yes",
            reasoning="Executed despite missing parent.",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        child_hierarchy = QueryHierarchy(
            query_id="Q1.1",
            parent_ids=("Q0",),
            boolean_parent_ids=("Q0",),
        )

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch(
                "legiscope.query.query_legal_documents",
                return_value=(response, []),
            ) as mock_query:
                mock_client = Mock(spec=Instructor)
                settings = BatchQuerySettings(
                    llm=LLMConfig(client=mock_client, model="test-model"),
                )

                results_df = run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="Child",
                            variable_name="child_var",
                            metadata={
                                "response_options": "Responses: Yes OR No",
                                "hierarchy": hierarchy_to_metadata(child_hierarchy),
                            },
                            query_id="Q1.1",
                        )
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        assert mock_query.call_count == 1
        assert results_df[0, "query_status"] == "completed"
        assert results_df[0, "skip_reason"] is None
        assert results_df[0, "blocking_parent_query_id"] is None
        assert results_df[0, "executed_despite_missing_parent"] is True
        assert results_df[0, "missing_parent_ids"] == '["Q0"]'

    def _run_label_blocker_case(
        self,
        tmp_path,
        *,
        parent_short_answer: str,
        response_options: str,
        blocker_labels: tuple[str, ...],
        parent_confidence: float = 0.9,
        dependency_skip_confidence_threshold: float | None = None,
    ):
        sections_path = self._write_sections_parquet(tmp_path)
        retrieval_results = SectionCollection(
            sections=[self._make_section("s1")],
            query_info=QueryInfo(
                original_query="query", total_segments_found=1, unique_sections=1
            ),
        )
        parent_response = LegalQueryResponse(
            short_answer=parent_short_answer,
            reasoning="Parent answer.",
            citations=[],
            supporting_passages=[],
            confidence=parent_confidence,
            limitations="None",
        )
        child_response = LegalQueryResponse(
            short_answer="Allowed uses",
            reasoning="Child answer.",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        parent_hierarchy = QueryHierarchy(query_id="Q1")
        child_hierarchy = QueryHierarchy(
            query_id="Q1.1",
            parent_ids=("parent_var",),
            label_blockers=(
                LabelBlockerRule(
                    parent_query_id="parent_var",
                    blocker_labels=blocker_labels,
                ),
            ),
        )

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch(
                "legiscope.query.query_legal_documents",
                side_effect=[(parent_response, []), (child_response, [])],
            ) as mock_query:
                mock_client = Mock(spec=Instructor)
                settings = BatchQuerySettings(
                    llm=LLMConfig(client=mock_client, model="test-model"),
                    dependency_skip_confidence_threshold=dependency_skip_confidence_threshold,
                )

                results_df = run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="Parent",
                            variable_name="parent_var",
                            metadata={
                                "response_options": response_options,
                                "hierarchy": hierarchy_to_metadata(parent_hierarchy),
                            },
                            query_id="Q1",
                        ),
                        QueryInput(
                            question="Child",
                            variable_name="child_var",
                            metadata={
                                "hierarchy": hierarchy_to_metadata(child_hierarchy),
                            },
                            query_id="Q1.1",
                        ),
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        return results_df, mock_query.call_count

    def test_run_queries_executes_child_on_exact_normalized_label_match(self, tmp_path):
        results_df, query_call_count = self._run_label_blocker_case(
            tmp_path,
            parent_short_answer="Pipes/Smoking Equipment, Generally",
            response_options=(
                "Pipes/Smoking Equipment, Generally OR Syringes, generally"
            ),
            blocker_labels=("pipes smoking equipment generally",),
        )

        child_row = results_df.filter(pl.col("query_id") == "Q1.1")
        assert query_call_count == 2
        assert child_row[0, "query_status"] == "completed"
        assert child_row[0, "label_match_method"] == "exact_normalized"
        assert child_row[0, "label_match_score"] == 100.0
        assert child_row[0, "label_match_ambiguous"] is False

    def test_run_queries_executes_child_on_fuzzy_unique_label_match(self, tmp_path):
        results_df, query_call_count = self._run_label_blocker_case(
            tmp_path,
            parent_short_answer="Pipes and smoking equipment, generally",
            response_options=(
                "Pipes and smoking equipment, generally OR Syringes, generally"
            ),
            blocker_labels=("Pipes/smoking equipment, generally",),
        )

        child_row = results_df.filter(pl.col("query_id") == "Q1.1")
        assert query_call_count == 2
        assert child_row[0, "query_status"] == "completed"
        assert child_row[0, "label_match_method"] == "fuzzy_unique"
        assert child_row[0, "label_match_score"] >= 90.0
        assert child_row[0, "label_match_ambiguous"] is False

    def test_run_queries_executes_child_when_label_match_is_ambiguous_fuzzy(
        self, tmp_path
    ):
        results_df, query_call_count = self._run_label_blocker_case(
            tmp_path,
            parent_short_answer=(
                "Pipes smoking equipment generally AND/OR Pipe smoking equipment generally"
            ),
            response_options=(
                "Pipes smoking equipment generally AND/OR Pipe smoking equipment generally"
            ),
            blocker_labels=("Pipes and smoking equipment generally",),
        )

        child_row = results_df.filter(pl.col("query_id") == "Q1.1")
        assert query_call_count == 2
        assert child_row[0, "query_status"] == "completed"
        assert child_row[0, "label_match_method"] == "ambiguous_fuzzy"
        assert child_row[0, "label_match_ambiguous"] is True

    def test_run_queries_skips_child_when_label_blocker_is_not_satisfied(
        self, tmp_path
    ):
        results_df, query_call_count = self._run_label_blocker_case(
            tmp_path,
            parent_short_answer="Syringes, generally",
            response_options=(
                "Pipes/smoking equipment, generally OR Syringes, generally"
            ),
            blocker_labels=("Pipes/smoking equipment, generally",),
        )

        child_row = results_df.filter(pl.col("query_id") == "Q1.1")
        assert query_call_count == 1
        assert child_row[0, "query_status"] == "skipped"
        assert child_row[0, "skip_reason"] == "label_blocker_not_satisfied"
        assert child_row[0, "label_match_method"] == "no_confident_match"

    def test_run_queries_executes_child_when_label_blocker_parent_confidence_is_low(
        self, tmp_path
    ):
        results_df, query_call_count = self._run_label_blocker_case(
            tmp_path,
            parent_short_answer="Syringes, generally",
            response_options=(
                "Pipes/smoking equipment, generally OR Syringes, generally"
            ),
            blocker_labels=("Pipes/smoking equipment, generally",),
            parent_confidence=0.2,
            dependency_skip_confidence_threshold=0.5,
        )

        child_row = results_df.filter(pl.col("query_id") == "Q1.1")
        assert query_call_count == 2
        assert child_row[0, "query_status"] == "completed"
        assert child_row[0, "label_match_method"] == "no_confident_match"
        assert child_row[0, "dependency_override_applied"] is True
        assert (
            child_row[0, "dependency_override_reason"]
            == "low_confidence_parent_label_blocker"
        )
        assert child_row[0, "dependency_override_parent_query_id"] == "Q1"

    def test_run_queries_passes_only_parent_question_and_short_answer_context(
        self, tmp_path
    ):
        sections_path = self._write_sections_parquet(tmp_path)
        retrieval_results = SectionCollection(
            sections=[self._make_section("s1")],
            query_info=QueryInfo(
                original_query="query", total_segments_found=1, unique_sections=1
            ),
        )
        parent_response = LegalQueryResponse(
            short_answer="Sales AND/OR Use",
            reasoning="Long explanation that should not flow downstream.",
            citations=["Section 1"],
            supporting_passages=["Large upstream passage"],
            confidence=0.9,
            limitations="None",
        )
        child_response = LegalQueryResponse(
            short_answer="Use",
            reasoning="Child answer.",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        parent_hierarchy = QueryHierarchy(query_id="Q1")
        child_hierarchy = QueryHierarchy(
            query_id="Q1.1",
            parent_ids=("dp_activity",),
            context_parent_ids=("dp_activity",),
            inherit_parent_retrieval=True,
        )

        captured_parent_contexts: list[list[dict[str, str | None]]] = []

        def fake_query_legal_documents(
            retrieval_results,
            _query,
            _settings,
            *,
            query_metadata=None,
            preselected_sections=None,
            execution_capture=None,
            **_kwargs,
        ):
            if execution_capture is not None:
                execution_capture["completion_sections"] = list(
                    preselected_sections or retrieval_results.sections
                )
            if query_metadata and query_metadata.get("query_id") == "Q1.1":
                captured_parent_contexts.append(query_metadata.get("parent_contexts"))
                return child_response, []
            return parent_response, []

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch(
                "legiscope.query.query_legal_documents",
                side_effect=fake_query_legal_documents,
            ):
                mock_client = Mock(spec=Instructor)
                settings = BatchQuerySettings(
                    llm=LLMConfig(client=mock_client, model="test-model"),
                    filter_relevance=False,
                )

                run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="Question: Which activities are prohibited?",
                            variable_name="dp_activity",
                            metadata={
                                "query_text": "Which activities are prohibited?",
                                "response_options": "Responses: Sales AND/OR Use",
                                "hierarchy": hierarchy_to_metadata(parent_hierarchy),
                            },
                            query_id="Q1",
                        ),
                        QueryInput(
                            question="Question: If exempted, which activities stay allowed?",
                            variable_name="dp_exempt_can_activity",
                            metadata={
                                "query_text": "If exempted, which activities stay allowed?",
                                "hierarchy": hierarchy_to_metadata(child_hierarchy),
                            },
                            query_id="Q1.1",
                        ),
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        assert captured_parent_contexts == [
            [
                {
                    "query_id": "Q1",
                    "question": "Which activities are prohibited?",
                    "short_answer": "Sales AND/OR Use",
                    "raw_short_answer": "Sales AND/OR Use",
                    "variable_name": "dp_activity",
                }
            ]
        ]

    def test_run_queries_carries_parent_option_evidence_into_child_context(
        self, tmp_path
    ):
        sections_path = self._write_sections_parquet(tmp_path)
        retrieval_results = SectionCollection(
            sections=[self._make_section("s1")],
            query_info=QueryInfo(
                original_query="query", total_segments_found=1, unique_sections=1
            ),
        )
        parent_response = LegalQueryResponse(
            reasoning="Parent answer.",
            citations=["§ 10.99"],
            supporting_passages=["A violation is punishable by a fine."],
            confidence=0.9,
            limitations="None",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Unspecified Fine",
                    selected=True,
                    confidence=0.9,
                    citations=["§ 10.99"],
                    supporting_passages=["A violation is punishable by a fine."],
                ),
                ResponseOptionEvidence(
                    option="Incarceration",
                    selected=False,
                    confidence=0.1,
                ),
            ],
            short_answer="Unspecified Fine",
        )
        child_response = LegalQueryResponse(
            reasoning="Child answer.",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
            short_answer="Allowed uses",
        )

        parent_hierarchy = QueryHierarchy(query_id="Q1")
        child_hierarchy = QueryHierarchy(
            query_id="Q1.1",
            parent_ids=("dp_penalties",),
            context_parent_ids=("dp_penalties",),
        )

        captured_parent_contexts: list[list[dict[str, object]]] = []

        def fake_query_legal_documents(
            retrieval_results,
            _query,
            _settings,
            *,
            query_metadata=None,
            preselected_sections=None,
            execution_capture=None,
            **_kwargs,
        ):
            if execution_capture is not None:
                execution_capture["completion_sections"] = list(
                    preselected_sections or retrieval_results.sections
                )
            if query_metadata and query_metadata.get("query_id") == "Q1.1":
                captured_parent_contexts.append(query_metadata.get("parent_contexts"))
                return child_response, []
            return parent_response, []

        with patch("legiscope.query.retrieve_sections", return_value=retrieval_results):
            with patch(
                "legiscope.query.query_legal_documents",
                side_effect=fake_query_legal_documents,
            ):
                settings = BatchQuerySettings(
                    llm=LLMConfig(client=Mock(spec=Instructor), model="test-model"),
                    filter_relevance=False,
                )

                run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="Question: Which penalties apply?",
                            variable_name="dp_penalties",
                            metadata={
                                "query_text": "Which penalties apply?",
                                "response_options": "Responses: Unspecified Fine AND/OR Incarceration",
                                "hierarchy": hierarchy_to_metadata(parent_hierarchy),
                            },
                            query_id="Q1",
                        ),
                        QueryInput(
                            question="Question: If exempted, which activities stay allowed?",
                            variable_name="dp_exempt_can_activity",
                            metadata={
                                "query_text": "If exempted, which activities stay allowed?",
                                "hierarchy": hierarchy_to_metadata(child_hierarchy),
                            },
                            query_id="Q1.1",
                        ),
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        assert len(captured_parent_contexts) == 1
        assert len(captured_parent_contexts[0]) == 1
        parent_context = captured_parent_contexts[0][0]
        assert parent_context["short_answer"] == "Unspecified Fine"
        assert (
            parent_context["response_options"]
            == "Unspecified Fine AND/OR Incarceration"
        )
        assert parent_context["confidence"] == 0.9
        assert parent_context["option_evidence"] == [
            {
                "option": "Unspecified Fine",
                "selected": True,
                "confidence": 0.9,
                "citations": ["§ 10.99"],
                "supporting_passages": ["A violation is punishable by a fine."],
                "anchor_terms": [],
            },
            {
                "option": "Incarceration",
                "selected": False,
                "confidence": 0.1,
                "citations": [],
                "supporting_passages": [],
                "anchor_terms": [],
            },
        ]

    def test_run_queries_uses_only_child_retrieval_units_for_completion(self, tmp_path):
        sections_path = self._write_sections_parquet(tmp_path)
        parent_results = SectionCollection(
            sections=[
                self._make_section("s_parent_only", chunk_id="chunk_parent_only"),
                self._make_section("s_parent", chunk_id="chunk_parent"),
            ],
            query_info=QueryInfo(
                original_query="parent query", total_segments_found=2, unique_sections=2
            ),
        )
        child_results = SectionCollection(
            sections=[
                self._make_section("s_parent_dup", chunk_id="chunk_parent"),
                self._make_section("s_child", chunk_id="chunk_child"),
            ],
            query_info=QueryInfo(
                original_query="child query", total_segments_found=2, unique_sections=2
            ),
        )
        parent_response = LegalQueryResponse(
            short_answer="Yes",
            reasoning="Parent answer.",
            citations=[],
            supporting_passages=[],
            confidence=0.9,
            limitations="None",
        )
        child_response = LegalQueryResponse(
            short_answer="Use",
            reasoning="Child answer.",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        parent_hierarchy = QueryHierarchy(query_id="Q1")
        child_hierarchy = QueryHierarchy(
            query_id="Q1.1",
            parent_ids=("Q1",),
            context_parent_ids=("Q1",),
            inherit_parent_retrieval=True,
        )

        merged_section_ids: list[list[str]] = []

        def fake_query_legal_documents(
            retrieval_results,
            _query,
            _settings,
            *,
            preselected_sections=None,
            execution_capture=None,
            **_kwargs,
        ):
            if execution_capture is not None:
                execution_capture["completion_sections"] = list(
                    preselected_sections or retrieval_results.sections
                )
            merged_sections = preselected_sections or retrieval_results.sections
            merged_section_ids.append(
                [section.chunk_id or section.section_id for section in merged_sections]
            )
            if len(merged_section_ids) == 1:
                return parent_response, []
            return child_response, []

        with patch(
            "legiscope.query.retrieve_sections",
            side_effect=[parent_results, child_results],
        ):
            with patch(
                "legiscope.query.query_legal_documents",
                side_effect=fake_query_legal_documents,
            ):
                mock_client = Mock(spec=Instructor)
                settings = BatchQuerySettings(
                    llm=LLMConfig(client=mock_client, model="test-model"),
                    filter_relevance=False,
                )

                results_df = run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="Parent",
                            variable_name="parent_var",
                            metadata={
                                "hierarchy": hierarchy_to_metadata(parent_hierarchy)
                            },
                            query_id="Q1",
                        ),
                        QueryInput(
                            question="Child",
                            variable_name="child_var",
                            metadata={
                                "hierarchy": hierarchy_to_metadata(child_hierarchy)
                            },
                            query_id="Q1.1",
                        ),
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        assert merged_section_ids[0] == ["chunk_parent_only", "chunk_parent"]
        assert merged_section_ids[1] == ["chunk_parent", "chunk_child"]
        child_row = results_df.filter(pl.col("query_id") == "Q1.1")
        assert (
            child_row[0, "inherited_chunk_ids"]
            == '["chunk_parent_only", "chunk_parent"]'
        )
        assert child_row[0, "new_chunk_ids"] == '["chunk_parent", "chunk_child"]'
        assert child_row[0, "merged_chunk_ids"] == '["chunk_parent", "chunk_child"]'
        assert child_row[0, "coalesced_duplicate_chunk_ids"] == "[]"

    def test_run_queries_appends_parent_retrieval_prompt_to_child_retrieval_query(
        self, tmp_path
    ):
        sections_path = self._write_sections_parquet(tmp_path)
        retrieval_results = SectionCollection(
            sections=[self._make_section("s1")],
            query_info=QueryInfo(
                original_query="query", total_segments_found=1, unique_sections=1
            ),
        )
        parent_response = LegalQueryResponse(
            short_answer="Yes",
            reasoning="Parent answer.",
            citations=[],
            supporting_passages=[],
            confidence=0.9,
            limitations="None",
        )
        child_response = LegalQueryResponse(
            short_answer="Use",
            reasoning="Child answer.",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        parent_hierarchy = QueryHierarchy(query_id="Q1")
        child_hierarchy = QueryHierarchy(
            query_id="Q1.1",
            parent_ids=("Q1",),
            context_parent_ids=("Q1",),
            inherit_parent_retrieval=True,
        )

        with patch(
            "legiscope.query.retrieve_sections",
            side_effect=[retrieval_results, retrieval_results],
        ) as mock_retrieve:
            with patch(
                "legiscope.query.query_legal_documents",
                side_effect=[(parent_response, []), (child_response, [])],
            ):
                mock_client = Mock(spec=Instructor)
                settings = BatchQuerySettings(
                    llm=LLMConfig(client=mock_client, model="test-model"),
                    filter_relevance=False,
                )

                run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="Parent retrieval prompt",
                            variable_name="parent_var",
                            metadata={
                                "hierarchy": hierarchy_to_metadata(parent_hierarchy)
                            },
                            query_id="Q1",
                        ),
                        QueryInput(
                            question="Child retrieval prompt",
                            variable_name="child_var",
                            metadata={
                                "hierarchy": hierarchy_to_metadata(child_hierarchy)
                            },
                            query_id="Q1.1",
                        ),
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        assert mock_retrieve.call_args_list[1].kwargs["query_text"] == (
            "Upstream retrieval context from Q1:\nParent retrieval prompt\n\n"
            "Child retrieval prompt"
        )

    def test_run_queries_can_disable_inherited_retrieval_while_preserving_parent_context(
        self, tmp_path
    ):
        sections_path = self._write_sections_parquet(tmp_path)
        parent_results = SectionCollection(
            sections=[self._make_section("s_parent", chunk_id="chunk_parent")],
            query_info=QueryInfo(
                original_query="parent query", total_segments_found=1, unique_sections=1
            ),
        )
        child_results = SectionCollection(
            sections=[self._make_section("s_child", chunk_id="chunk_child")],
            query_info=QueryInfo(
                original_query="child query", total_segments_found=1, unique_sections=1
            ),
        )
        parent_response = LegalQueryResponse(
            short_answer="Pipes AND/OR Other",
            reasoning="Parent answer.",
            citations=[],
            supporting_passages=[],
            confidence=0.9,
            limitations="None",
        )
        child_response = LegalQueryResponse(
            short_answer="None",
            reasoning="Child answer.",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        parent_hierarchy = QueryHierarchy(query_id="Q1")
        child_hierarchy = QueryHierarchy(
            query_id="Q1.1",
            parent_ids=("Q1",),
            context_parent_ids=("Q1",),
            inherit_parent_retrieval=True,
        )

        merged_section_ids: list[list[str]] = []
        captured_parent_contexts: list[list[dict[str, str | None]]] = []

        def fake_query_legal_documents(
            retrieval_results,
            _query,
            _settings,
            *,
            query_metadata=None,
            preselected_sections=None,
            execution_capture=None,
            **_kwargs,
        ):
            if execution_capture is not None:
                execution_capture["completion_sections"] = list(
                    preselected_sections or retrieval_results.sections
                )
            merged_sections = preselected_sections or retrieval_results.sections
            merged_section_ids.append(
                [section.chunk_id or section.section_id for section in merged_sections]
            )
            if query_metadata and query_metadata.get("query_id") == "Q1.1":
                captured_parent_contexts.append(query_metadata.get("parent_contexts"))
                return child_response, []
            return parent_response, []

        with patch(
            "legiscope.query.retrieve_sections",
            side_effect=[parent_results, child_results],
        ) as mock_retrieve:
            with patch(
                "legiscope.query.query_legal_documents",
                side_effect=fake_query_legal_documents,
            ):
                mock_client = Mock(spec=Instructor)
                settings = BatchQuerySettings(
                    llm=LLMConfig(client=mock_client, model="test-model"),
                    filter_relevance=False,
                )

                results_df = run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="Parent retrieval prompt",
                            variable_name="dp_type",
                            metadata={
                                "hierarchy": hierarchy_to_metadata(parent_hierarchy)
                            },
                            query_id="Q1",
                        ),
                        QueryInput(
                            question="Child retrieval prompt",
                            variable_name="dp_exemption",
                            metadata={
                                "hierarchy": hierarchy_to_metadata(child_hierarchy),
                                "disable_inherited_retrieval_from": "dp_type",
                            },
                            query_id="Q1.1",
                        ),
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        assert mock_retrieve.call_args_list[1].kwargs["query_text"] == (
            "Child retrieval prompt"
        )
        assert merged_section_ids[0] == ["chunk_parent"]
        assert merged_section_ids[1] == ["chunk_child"]
        assert len(captured_parent_contexts) == 1
        assert len(captured_parent_contexts[0]) == 1
        parent_context = captured_parent_contexts[0][0]
        assert parent_context["query_id"] == "Q1"
        assert parent_context["question"] == "Parent retrieval prompt"
        assert parent_context["short_answer"] == "Pipes AND/OR Other"
        assert parent_context["variable_name"] == "dp_type"
        child_row = results_df.filter(pl.col("query_id") == "Q1.1")
        assert child_row[0, "inherited_retrieval_prompt_sources"] == "[]"
        assert child_row[0, "inherited_chunk_ids"] == "[]"
        assert child_row[0, "merged_chunk_ids"] == "[]"
