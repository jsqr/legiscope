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
            supporting_passages=[
                "A violation is punishable by a fine and imprisonment."
            ],
            confidence=0.4,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option='"Unlawful" only', selected=True),
                ResponseOptionEvidence(option="Criminal Fine", selected=False),
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
                "response_options": '"Unlawful" only AND/OR Criminal Fine AND/OR Unspecified Fine AND/OR Incarceration',
            },
        )

        assert gated.short_answer == "Unspecified Fine AND/OR Incarceration"
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "Unspecified Fine",
            "Incarceration",
        ]

    def test_penalty_gate_drops_unspecified_fine_when_item_has_no_fine_signal(self):
        response = LegalQueryResponse(
            short_answer="Misdemeanor AND/OR Unspecified Fine",
            reasoning="Initial answer overcalled unspecified fine from generic penalty context.",
            citations=["§ 607.18 Penalty"],
            supporting_passages=[
                "Whoever violates this section is guilty of a misdemeanor of the second degree."
            ],
            confidence=0.53,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option='"Unlawful" only', selected=False),
                ResponseOptionEvidence(
                    option="Misdemeanor",
                    selected=True,
                    citations=["§ 607.18 Penalty"],
                    supporting_passages=[
                        "Whoever violates this section is guilty of a misdemeanor of the second degree."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Unspecified Fine",
                    selected=True,
                    citations=["§ 607.18 Penalty"],
                    supporting_passages=[
                        "Whoever violates this section is guilty of a misdemeanor of the second degree."
                    ],
                ),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-oh-penalty",
                heading_text="# Penalty",
                body_text="Whoever violates this section is guilty of a misdemeanor of the second degree.",
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
                "response_options": '"Unlawful" only AND/OR Misdemeanor AND/OR Unspecified Fine',
            },
        )

        assert gated.short_answer == "Misdemeanor"
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "Misdemeanor",
        ]

    def test_penalty_gate_promotes_civil_penalty_over_unlawful_only(self):
        response = LegalQueryResponse(
            short_answer='"Unlawful" only',
            reasoning="The ordinance imposes a civil penalty for violations.",
            citations=["§ 9-629(4)(a)"],
            supporting_passages=[
                "Any person violating this section shall be subject to a civil penalty of $150."
            ],
            confidence=0.58,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option='"Unlawful" only', selected=True),
                ResponseOptionEvidence(option="Civil Fine", selected=False),
                ResponseOptionEvidence(option="Unspecified Fine", selected=False),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-phl-penalty",
                heading_text="# Penalty",
                body_text=(
                    "Any person violating this section shall be subject to a civil penalty of $150."
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
                "response_options": '"Unlawful" only AND/OR Civil Fine AND/OR Unspecified Fine',
            },
        )

        assert gated.short_answer == "Civil Fine"
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "Civil Fine"
        ]

    def test_penalty_gate_keeps_conviction_fine_as_unspecified_without_offense_class(
        self,
    ):
        response = LegalQueryResponse(
            short_answer="Criminal Fine AND/OR Unspecified Fine",
            reasoning="The section says the violation is punishable by a fine upon conviction.",
            citations=["SEC. 31-32.1(d)"],
            supporting_passages=[
                "A person violating a provision of this section is, upon conviction, punishable by a fine not to exceed $2,000."
            ],
            confidence=0.71,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option='"Unlawful" only', selected=False),
                ResponseOptionEvidence(option="Criminal Fine", selected=True),
                ResponseOptionEvidence(option="Unspecified Fine", selected=True),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-dallas-penalty",
                heading_text="# Penalty",
                body_text=(
                    "A person violating a provision of this section is, upon conviction, punishable by a fine not to exceed $2,000."
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
                "response_options": '"Unlawful" only AND/OR Criminal Fine AND/OR Unspecified Fine',
            },
        )

        assert gated.short_answer == "Unspecified Fine"
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "Unspecified Fine"
        ]

    def test_ssp_restriction_gate_does_not_treat_exchange_only_basis_as_quantity_limit(
        self,
    ):
        response = LegalQueryResponse(
            short_answer=(
                "Programs may not operate within certain distance of schools or childcare facilities "
                "AND/OR Programs may not operate within certain distance of parks or other public spaces "
                "AND/OR Restrictions on quantity of syringes that may be provided or exchanged"
            ),
            reasoning="Initial answer overcalled quantity restriction from exchange-only operation text.",
            citations=["§ 91.83(C)", "§ 91.87(D)"],
            supporting_passages=[
                "No SSP facility or mobile or pop-up exchange program will be allowed to operate within 750 feet of any playground, library, or state-licensed daycare facility; and no SSP facility or mobile or pop-up exchange program shall be located within a drug-free school zone.",
                "The operation of an SSP, and mobile or pop-up exchange programs in a city park is prohibited.",
                "An SSP shall operate to an exchange-only basis, whereby a participant receives sterile needles only by providing a used one.",
            ],
            confidence=0.58,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Programs may not operate within certain distance of schools or childcare facilities",
                    selected=True,
                    citations=["§ 91.83(C)"],
                    supporting_passages=[
                        "No SSP facility or mobile or pop-up exchange program will be allowed to operate within 750 feet of any playground, library, or state-licensed daycare facility; and no SSP facility or mobile or pop-up exchange program shall be located within a drug-free school zone."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Restrictions on quantity of syringes that may be provided or exchanged",
                    selected=True,
                    citations=["§ 91.87(D)"],
                    supporting_passages=[
                        "An SSP shall operate to an exchange-only basis, whereby a participant receives sterile needles only by providing a used one."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Programs may not operate within certain distance of parks or other public spaces",
                    selected=True,
                    citations=["§ 96.06(J)"],
                    supporting_passages=[
                        "The operation of an SSP, and mobile or pop-up exchange programs in a city park is prohibited."
                    ],
                ),
                ResponseOptionEvidence(option="No restrictions listed", selected=False),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-nh-ssp-restrict",
                heading_text="# SSP restrictions",
                body_text=(
                    "No SSP facility or mobile or pop-up exchange program will be allowed to operate within 750 feet of any playground, library, or state-licensed daycare facility; and no SSP facility or mobile or pop-up exchange program shall be located within a drug-free school zone. "
                    "The operation of an SSP, and mobile or pop-up exchange programs in a city park is prohibited. "
                    "An SSP shall operate to an exchange-only basis, whereby a participant receives sterile needles only by providing a used one."
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
                "guidance_topic": "ssp_restriction",
                "response_options": (
                    "Programs may not operate within certain distance of schools or childcare facilities AND/OR "
                    "Programs may not operate within certain distance of parks or other public spaces AND/OR "
                    "Restrictions on quantity of syringes that may be provided or exchanged AND/OR No restrictions listed"
                ),
            },
        )

        selected = [item.option for item in gated.option_evidence if item.selected]
        assert (
            "Restrictions on quantity of syringes that may be provided or exchanged"
            not in selected
        )
        assert "No restrictions listed" not in selected
        assert (
            "Programs may not operate within certain distance of parks or other public spaces"
            in selected
        )

    def test_rewrites_unsupported_sales_to_display_only(self):
        response = LegalQueryResponse(
            short_answer="Sales, possession with intent to sell, offer for sale",
            reasoning="Initial answer overcalled sales.",
            citations=["§ 12-1"],
            supporting_passages=[
                "It is unlawful to display drug paraphernalia for advertising purposes."
            ],
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

    def test_prohibited_activity_gate_drops_product_only_sales_and_delivery_noise(self):
        response = LegalQueryResponse(
            short_answer=(
                "Sales, possession with intent to sell, offer for sale AND/OR "
                "Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange"
            ),
            reasoning="Initial answer treated illegal smoking product clauses as paraphernalia activity support.",
            citations=["§ 41A-13.11"],
            supporting_passages=[
                "A person commits an offense if the person sells any illegal smoking product.",
                "A person commits an offense if the person delivers any illegal smoking product.",
            ],
            confidence=0.61,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Sales, possession with intent to sell, offer for sale",
                    selected=True,
                    citations=["§ 41A-13.11(b)(1)"],
                    supporting_passages=[
                        "A person commits an offense if the person sells any illegal smoking product."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange",
                    selected=True,
                    citations=["§ 41A-13.11(b)(2)"],
                    supporting_passages=[
                        "A person commits an offense if the person delivers any illegal smoking product."
                    ],
                ),
                ResponseOptionEvidence(option="Not specified", selected=False),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-dallas-product-only",
                heading_text="# Product-only offense",
                body_text=(
                    "A person commits an offense if the person sells any illegal smoking product. "
                    "A person commits an offense if the person delivers any illegal smoking product."
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
                "guidance_topic": "prohibited_activity",
                "response_options": (
                    "Sales, possession with intent to sell, offer for sale AND/OR "
                    "Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange AND/OR Not specified"
                ),
            },
        )

        assert gated.short_answer == "Not specified"

    def test_definition_type_gate_keeps_explicit_pipe_support_in_product_only_smoking_definition(self):
        response = LegalQueryResponse(
            short_answer="Not specified",
            reasoning="Initial answer missed the explicit paraphernalia definition.",
            citations=["§ 41A-13.1"],
            supporting_passages=[
                "ILLEGAL SMOKING PARAPHERNALIA means any equipment, device, or utensil that is used or intended to be used in ingesting, inhaling, or otherwise introducing into the human body an illegal smoking product, which paraphernalia includes but is not limited to: a pipe, a water pipe, an electric pipe, a chillum, or a bong."
            ],
            confidence=0.52,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Syringes, hypodermic needles, other inject/ion/ing equipment/instrument/supplies",
                    selected=False,
                ),
                ResponseOptionEvidence(
                    option="Pipes, other smoke/ing or inhal/ing/ation equipment or supplies",
                    selected=False,
                ),
                ResponseOptionEvidence(
                    option="Drug test/ing or check/ing equipment or supplies",
                    selected=False,
                ),
                ResponseOptionEvidence(option="Other", selected=False),
                ResponseOptionEvidence(option="Not specified", selected=True),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-dallas-type",
                heading_text="# Definition",
                body_text=(
                    "ILLEGAL SMOKING PARAPHERNALIA means any equipment, device, or utensil that is used or intended to be used in ingesting, inhaling, or otherwise introducing into the human body an illegal smoking product, which paraphernalia includes but is not limited to: a pipe, a water pipe, an electric pipe, a chillum, or a bong."
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
                "guidance_topic": "definition_type",
                "response_options": (
                    "Syringes, hypodermic needles, other inject/ion/ing equipment/instrument/supplies AND/OR "
                    "Pipes, other smoke/ing or inhal/ing/ation equipment or supplies AND/OR "
                    "Drug test/ing or check/ing equipment or supplies AND/OR Other AND/OR Not specified"
                ),
            },
        )

        assert (
            gated.short_answer
            == "Pipes, other smoke/ing or inhal/ing/ation equipment or supplies AND/OR Other"
        )

    def test_definition_type_gate_recovers_guidance_topic_from_variable_name(self):
        response = LegalQueryResponse(
            short_answer="Not specified",
            reasoning="Initial answer missed the explicit paraphernalia definition.",
            citations=["§ 31-32.1"],
            supporting_passages=[
                "ILLEGAL SMOKING PARAPHERNALIA means any equipment, device, or utensil that is used or intended to be used in ingesting, inhaling, or otherwise introducing into the human body an illegal smoking product, which paraphernalia includes but is not limited to: a pipe, a water pipe, an electric pipe, a chillum, or a bong."
            ],
            confidence=0.52,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Not specified",
                    selected=True,
                ),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-dallas-type-missing-topic",
                heading_text="# Definition",
                body_text=(
                    "ILLEGAL SMOKING PARAPHERNALIA means any equipment, device, or utensil that is used or intended to be used in ingesting, inhaling, or otherwise introducing into the human body an illegal smoking product, which paraphernalia includes but is not limited to: a pipe, a water pipe, an electric pipe, a chillum, or a bong."
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
                "variable_name": "dp_type",
                "response_options": (
                    "Syringes, hypodermic needles, other inject/ion/ing equipment/instrument/supplies AND/OR "
                    "Pipes, other smoke/ing or inhal/ing/ation equipment or supplies AND/OR "
                    "Drug test/ing or check/ing equipment or supplies AND/OR Other AND/OR Not specified"
                ),
            },
        )

        assert (
            gated.short_answer
            == "Pipes, other smoke/ing or inhal/ing/ation equipment or supplies AND/OR Other"
        )

    def test_definition_type_gate_keeps_explicit_drug_paraphernalia_pipe_support(self):
        response = LegalQueryResponse(
            short_answer="Not specified",
            reasoning="Initial answer missed the explicit drug paraphernalia definition.",
            citations=["§ 12-34"],
            supporting_passages=[
                "Drug paraphernalia means any equipment, product, or material used with controlled substances, including a pipe, water pipe, bong, roach clip, spoon, or straw."
            ],
            confidence=0.54,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Pipes, other smoke/ing or inhal/ing/ation equipment or supplies",
                    selected=False,
                ),
                ResponseOptionEvidence(option="Other", selected=False),
                ResponseOptionEvidence(option="Not specified", selected=True),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-drug-paraphernalia-type",
                heading_text="# Definition",
                body_text=(
                    "Drug paraphernalia means any equipment, product, or material used with controlled substances, including a pipe, water pipe, bong, roach clip, spoon, or straw."
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
                "guidance_topic": "definition_type",
                "response_options": (
                    "Pipes, other smoke/ing or inhal/ing/ation equipment or supplies AND/OR "
                    "Other AND/OR Not specified"
                ),
            },
        )

        assert gated.short_answer == (
            "Pipes, other smoke/ing or inhal/ing/ation equipment or supplies AND/OR Other"
        )

    def test_keeps_use_when_selected_option_evidence_explicitly_supports_use(self):
        response = LegalQueryResponse(
            short_answer="Use AND/OR Possession, possession with intent to use, keep",
            reasoning="Initial answer includes direct use prohibition language.",
            citations=["§ 21-10"],
            supporting_passages=[
                "It shall be unlawful for any person to use or possess with intent to use drug paraphernalia."
            ],
            confidence=0.61,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Use",
                    selected=True,
                    citations=["§ 21-10"],
                    supporting_passages=[
                        "It shall be unlawful for any person to use or possess with intent to use drug paraphernalia."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Possession, possession with intent to use, keep",
                    selected=True,
                    citations=["§ 21-10"],
                    supporting_passages=[
                        "It shall be unlawful for any person to use or possess with intent to use drug paraphernalia."
                    ],
                ),
                ResponseOptionEvidence(option="Not specified", selected=False),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-use",
                heading_text="# Prohibited activities",
                body_text="It shall be unlawful for any person to use or possess with intent to use drug paraphernalia.",
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
                    "Use AND/OR Possession, possession with intent to use, keep AND/OR Not specified"
                ),
            },
        )

        assert "Use" in [item.option for item in gated.option_evidence if item.selected]

    def test_promotes_supported_exemption_over_none(self):
        response = LegalQueryResponse(
            short_answer="None",
            reasoning="Initial answer missed the exemption.",
            citations=["§ 5-10"],
            supporting_passages=[
                "Nothing in this section shall apply to cannabis paraphernalia."
            ],
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

    def test_drops_other_restrictions_when_permit_requirement_fully_covers_it(self):
        response = LegalQueryResponse(
            short_answer=(
                "Permit or license required for operation AND/OR Other restrictions"
            ),
            reasoning="Initial answer treated permit administration details as residual restrictions.",
            citations=["§ 6-4-2"],
            supporting_passages=[
                "An SSP operator shall obtain a permit from the mayor, renew it annually, and file the required application materials."
            ],
            confidence=0.6,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Permit or license required for operation",
                    selected=True,
                    citations=["§ 6-4-2"],
                    supporting_passages=[
                        "An SSP operator shall obtain a permit from the mayor, renew it annually, and file the required application materials."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Other restrictions",
                    selected=True,
                    citations=["§ 6-4-2"],
                    supporting_passages=[
                        "An SSP operator shall obtain a permit from the mayor, renew it annually, and file the required application materials."
                    ],
                ),
            ],
        )
        sections = [
            SectionResult(
                section_id="s4",
                heading_text="# SSP permit",
                body_text=(
                    "An SSP operator shall obtain a permit from the mayor, renew it annually, and file the required application materials."
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
                "guidance_topic": "ssp_restriction",
                "response_options": (
                    "Permit or license required for operation AND/OR Other restrictions"
                ),
            },
        )

        assert gated.short_answer == "Permit or license required for operation"
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "Permit or license required for operation"
        ]

    def test_crosswalks_unspecified_fine_to_criminal_fine_when_jail_is_present(self):
        response = LegalQueryResponse(
            short_answer="Unspecified Fine AND/OR Incarceration",
            reasoning="Initial answer used the generic fine label.",
            citations=["§ 10.99"],
            supporting_passages=[
                "Any person convicted under this section is guilty of a misdemeanor and shall be punished by a fine of up to $500 or imprisonment for up to 60 days."
            ],
            confidence=0.7,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option='"Unlawful" only', selected=False),
                ResponseOptionEvidence(option="Civil Fine", selected=False),
                ResponseOptionEvidence(option="Criminal Fine", selected=False),
                ResponseOptionEvidence(option="Unspecified Fine", selected=True),
                ResponseOptionEvidence(option="Incarceration", selected=True),
            ],
        )
        sections = [
            SectionResult(
                section_id="s5",
                heading_text="# Penalty",
                body_text=(
                    "Any person convicted under this section is guilty of a misdemeanor and shall be punished by a fine of up to $500 or imprisonment for up to 60 days."
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
                "response_options": (
                    '"Unlawful" only AND/OR Civil Fine AND/OR Criminal Fine AND/OR '
                    "Unspecified Fine AND/OR Incarceration"
                ),
            },
        )

        assert gated.short_answer == "Criminal Fine AND/OR Incarceration"
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "Criminal Fine",
            "Incarceration",
        ]

    def test_limits_cannabis_exemption_activity_scope_to_explicitly_supported_labels(
        self,
    ):
        response = LegalQueryResponse(
            short_answer="Possession AND/OR Use AND/OR Distribution AND/OR Sales",
            reasoning="Initial answer expanded cannabis use or commerce into every activity label.",
            citations=["§ 12-4-10(C)(3)"],
            supporting_passages=[
                "Nothing in this section shall be construed to establish a criminal penalty for possession of paraphernalia for the exclusive purpose of cannabis use, or for any activities associated with cannabis use or commerce."
            ],
            confidence=0.7,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Possession",
                    selected=True,
                    citations=["§ 12-4-10(C)(3)"],
                    supporting_passages=[
                        "Nothing in this section shall be construed to establish a criminal penalty for possession of paraphernalia for the exclusive purpose of cannabis use, or for any activities associated with cannabis use or commerce."
                    ],
                ),
                ResponseOptionEvidence(option="Use", selected=True),
                ResponseOptionEvidence(option="Distribution", selected=True),
                ResponseOptionEvidence(option="Sales", selected=True),
            ],
        )
        sections = [
            SectionResult(
                section_id="s6",
                heading_text="# Cannabis exemption",
                body_text=(
                    "Nothing in this section shall be construed to establish a criminal penalty for possession of paraphernalia for the exclusive purpose of cannabis use, or for any activities associated with cannabis use or commerce."
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
                "guidance_topic": "exemption_activity_scope",
                "response_options": "Possession AND/OR Use AND/OR Distribution AND/OR Sales",
            },
        )

        assert gated.short_answer == "Possession"

    def test_exemption_activity_scope_requires_direct_quote_per_selected_label(self):
        response = LegalQueryResponse(
            short_answer="Possession AND/OR Use",
            reasoning="Initial answer selected Use without direct option evidence.",
            citations=["§ 12-4-10(C)(3)"],
            supporting_passages=[
                "Nothing in this section shall be construed to establish a criminal penalty for possession of paraphernalia for the exclusive purpose of cannabis use."
            ],
            confidence=0.65,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Possession",
                    selected=True,
                    citations=["§ 12-4-10(C)(3)"],
                    supporting_passages=[
                        "Nothing in this section shall be construed to establish a criminal penalty for possession of paraphernalia for the exclusive purpose of cannabis use."
                    ],
                ),
                ResponseOptionEvidence(option="Use", selected=True),
            ],
        )

        gated = query_module._apply_authoritative_option_evidence_gate(
            response,
            [],
            {
                "guidance_topic": "exemption_activity_scope",
                "response_options": "Possession AND/OR Use",
            },
        )

        assert gated.short_answer == "Possession"

    def test_defaults_binary_scope_question_to_no_when_yes_lacks_direct_support(self):
        response = LegalQueryResponse(
            short_answer="Yes",
            reasoning="Initial answer inferred an SSP law from nearby public-health text.",
            citations=[],
            supporting_passages=[
                "A local public health emergency may be declared for communicable disease control."
            ],
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

    def test_defaults_ssp_restriction_multi_select_to_no_restrictions_without_option_support(
        self,
    ):
        response = LegalQueryResponse(
            short_answer="Permit or license required for operation AND/OR Restrictions on mobile sites",
            reasoning="Initial answer inferred multiple restrictions from general administrative text.",
            citations=[],
            supporting_passages=[
                "The program is recognized during a declared emergency."
            ],
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


class TestSecondStageStructuredValidators:
    def test_prohibited_activity_gate_drops_illegal_smoking_product_sales_noise(self):
        response = LegalQueryResponse(
            short_answer=(
                "Sales, possession with intent to sell, offer for sale AND/OR "
                "Possession, possession with intent to use, keep"
            ),
            reasoning="Initial answer mixed illegal smoking product sales with paraphernalia possession.",
            citations=["§ 31-32.1"],
            supporting_passages=[
                "A person commits an offense if the person sells any illegal smoking product. A person commits an offense if the person uses or possesses with intent to use any illegal smoking paraphernalia."
            ],
            confidence=0.62,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Sales, possession with intent to sell, offer for sale",
                    selected=True,
                    citations=["§ 31-32.1"],
                    supporting_passages=[
                        "A person commits an offense if the person sells any illegal smoking product."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Possession, possession with intent to use, keep",
                    selected=True,
                    citations=["§ 31-32.1"],
                    supporting_passages=[
                        "A person commits an offense if the person uses or possesses with intent to use any illegal smoking paraphernalia."
                    ],
                ),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-dallas",
                heading_text="# Illegal smoking products and paraphernalia",
                body_text=(
                    "A person commits an offense if the person sells any illegal smoking product. "
                    "A person commits an offense if the person uses or possesses with intent to use any illegal smoking paraphernalia."
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
                "guidance_topic": "prohibited_activity",
                "response_options": (
                    "Sales, possession with intent to sell, offer for sale AND/OR "
                    "Possession, possession with intent to use, keep AND/OR Not specified"
                ),
            },
        )

        assert gated.short_answer == "Possession, possession with intent to use, keep"

    def test_reference_necessity_forces_no_for_definition_only_support(self):
        response = LegalQueryResponse(
            short_answer="Yes",
            reasoning="Initial answer treated a controlled-substance definition as operative outside-law support.",
            citations=["Health and Safety Code § 11054"],
            supporting_passages=[
                "Controlled substance means a drug listed in Schedule I through V of the Health and Safety Code."
            ],
            confidence=0.6,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Yes",
                    selected=True,
                    citations=["Health and Safety Code § 11054"],
                    supporting_passages=[
                        "Controlled substance means a drug listed in Schedule I through V of the Health and Safety Code."
                    ],
                ),
                ResponseOptionEvidence(option="No", selected=False),
            ],
        )

        validated = query_module._apply_reference_necessity_validator(
            response,
            [],
            {
                "variable_name": "dp_state_fed_reference",
                "guidance_topic": "reference_necessity",
                "response_options": "Yes OR No",
            },
        )

        assert validated.short_answer == "No"
        assert [item.option for item in validated.option_evidence if item.selected] == [
            "No"
        ]

    def test_reference_necessity_forces_no_for_ssp_admin_only_state_reference(self):
        response = LegalQueryResponse(
            short_answer="Yes",
            reasoning="Initial answer treated RSA authorization background as a necessary outside-law dependency.",
            citations=["R.S.A. 318-B:43"],
            supporting_passages=[
                "Syringe service program means a program authorized by R.S.A. 318-B:43 and coordinated with local health officials."
            ],
            confidence=0.58,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Yes",
                    selected=True,
                    citations=["R.S.A. 318-B:43"],
                    supporting_passages=[
                        "Syringe service program means a program authorized by R.S.A. 318-B:43 and coordinated with local health officials."
                    ],
                ),
                ResponseOptionEvidence(option="No", selected=False),
            ],
        )

        validated = query_module._apply_reference_necessity_validator(
            response,
            [],
            {
                "variable_name": "ssp_state_fed_reference",
                "guidance_topic": "reference_necessity",
                "response_options": "Yes OR No",
            },
        )

        assert validated.short_answer == "No"
        assert [item.option for item in validated.option_evidence if item.selected] == [
            "No"
        ]

    def test_prohibited_activity_gate_drops_product_only_sales_and_delivery_noise(self):
        response = LegalQueryResponse(
            short_answer=(
                "Sales, possession with intent to sell, offer for sale AND/OR "
                "Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange"
            ),
            reasoning="Initial answer treated illegal smoking product clauses as paraphernalia activity support.",
            citations=["§ 31-32.1"],
            supporting_passages=[
                "A person commits an offense if the person sells any illegal smoking product.",
                "A person commits an offense if the person delivers any illegal smoking product.",
            ],
            confidence=0.62,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Sales, possession with intent to sell, offer for sale",
                    selected=True,
                    citations=["§ 31-32.1"],
                    supporting_passages=[
                        "A person commits an offense if the person sells any illegal smoking product."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange",
                    selected=True,
                    citations=["§ 31-32.1"],
                    supporting_passages=[
                        "A person commits an offense if the person delivers any illegal smoking product."
                    ],
                ),
                ResponseOptionEvidence(option="Not specified", selected=False),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-dallas-product-only",
                heading_text="# Product-only offense",
                body_text=(
                    "A person commits an offense if the person sells any illegal smoking product. "
                    "A person commits an offense if the person delivers any illegal smoking product."
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
                "guidance_topic": "prohibited_activity",
                "response_options": (
                    "Sales, possession with intent to sell, offer for sale AND/OR "
                    "Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange AND/OR Not specified"
                ),
            },
        )

        assert gated.short_answer == "Not specified"

    def test_reference_citation_validator_prefers_parent_aligned_citation_family(self):
        parent_contexts = [
            query_module.ParentQueryContext(
                query_id="parent-1",
                question="Does the local law require state-law review?",
                short_answer="Yes",
                variable_name="dp_state_fed_reference",
                response_options="Yes OR No",
                option_evidence=[
                    query_module.ParentOptionEvidence(
                        option="Yes",
                        selected=True,
                        citations=["Sections 30-31-1 et seq. NMSA 1978"],
                        supporting_passages=[
                            "The ordinance relies on the Controlled Substances Act, Sections 30-31-1 et seq. NMSA 1978."
                        ],
                    )
                ],
            )
        ]
        response = LegalQueryResponse(
            short_answer="§ 26-2B-1",
            reasoning="Initial answer picked a local citation instead of the cited state-law family.",
            citations=["§ 26-2B-1"],
            supporting_passages=[
                "The state controlled substances act, Sections 30-31-1 et seq. NMSA 1978, governs controlled substances."
            ],
            confidence=0.51,
            limitations="",
            option_evidence=[],
        )
        sections = [
            SectionResult(
                section_id="s-citation",
                heading_text="# Cross references",
                body_text=(
                    "Local citation § 26-2B-1 appears elsewhere. The state controlled substances act, "
                    "Sections 30-31-1 et seq. NMSA 1978, governs controlled substances."
                ),
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        validated = query_module._apply_reference_citation_validator(
            response,
            sections,
            {
                "variable_name": "dp_state_fed_citation",
                "response_options": "<citation>",
                "parent_contexts": query_module._serialize_parent_contexts(
                    parent_contexts
                ),
            },
        )

        assert validated.short_answer == "§ 30-31-1"
        assert validated.citations == ["§ 30-31-1"]

    def test_penalty_validator_suppresses_inferred_labels_from_default_penalty_text(
        self,
    ):
        response = LegalQueryResponse(
            short_answer="Criminal Fine AND/OR Incarceration",
            reasoning="Initial answer inferred benchmark sanctions from a generic default-penalty cross-reference.",
            citations=["§ 1-8-1"],
            supporting_passages=[
                "A violation is a class A offense and punishable as provided in the general penalty section."
            ],
            confidence=0.55,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option='"Unlawful" only', selected=False),
                ResponseOptionEvidence(option="Criminal Fine", selected=True),
                ResponseOptionEvidence(option="Incarceration", selected=True),
            ],
        )

        validated = query_module._apply_penalty_specificity_validator(
            response,
            [],
            {
                "guidance_topic": "penalty",
                "response_options": '"Unlawful" only AND/OR Criminal Fine AND/OR Incarceration',
            },
        )

        assert validated.short_answer == '"Unlawful" only'
        assert [item.option for item in validated.option_evidence if item.selected] == [
            '"Unlawful" only'
        ]

    def test_penalty_validator_promotes_stronger_labels_over_unlawful_only_without_unlawful_text(
        self,
    ):
        response = LegalQueryResponse(
            short_answer='"Unlawful" only',
            reasoning="Initial answer stopped at fallback despite stronger sanctions.",
            citations=["§ 1-8-1"],
            supporting_passages=[
                "A violation is punishable by a fine and imprisonment for not more than 60 days."
            ],
            confidence=0.57,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option='"Unlawful" only', selected=True),
                ResponseOptionEvidence(option="Criminal Fine", selected=False),
                ResponseOptionEvidence(option="Unspecified Fine", selected=False),
                ResponseOptionEvidence(option="Incarceration", selected=False),
            ],
        )

        validated = query_module._apply_penalty_specificity_validator(
            response,
            [],
            {
                "guidance_topic": "penalty",
                "response_options": '"Unlawful" only AND/OR Criminal Fine AND/OR Unspecified Fine AND/OR Incarceration',
            },
        )

        selected = [item.option for item in validated.option_evidence if item.selected]
        assert '"Unlawful" only' not in selected
        assert "Incarceration" in selected

    def test_penalty_validator_suppresses_unlawful_only_when_forfeiture_is_explicit(
        self,
    ):
        response = LegalQueryResponse(
            short_answer='"Unlawful" only',
            reasoning="Initial answer stopped at fallback despite explicit forfeiture.",
            citations=["§ 8-12"],
            supporting_passages=[
                "Any drug paraphernalia involved in the violation is subject to forfeiture and seizure.",
            ],
            confidence=0.58,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option='"Unlawful" only', selected=True),
                ResponseOptionEvidence(option="Forfeiture/Seizure", selected=False),
            ],
        )

        validated = query_module._apply_penalty_specificity_validator(
            response,
            [],
            {
                "guidance_topic": "penalty",
                "response_options": '"Unlawful" only AND/OR Forfeiture/Seizure',
            },
        )

        assert validated.short_answer == "Forfeiture/Seizure"

    def test_ssp_restriction_validator_drops_labels_without_direct_option_support(self):
        response = LegalQueryResponse(
            short_answer=(
                "Programs may not operate within certain distance of schools or childcare facilities AND/OR "
                "Permit or license required for operation"
            ),
            reasoning="Initial answer overcalled permit from broad operational context.",
            citations=["§ 91.83(C)", "§ 91.87(D)"],
            supporting_passages=[
                "No SSP facility will be allowed to operate within 750 feet of any state-licensed daycare facility.",
                "A mobile or pop-up exchange SSP program proposed to be operated on public property shall require prior approval of the commissioner.",
            ],
            confidence=0.64,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Programs may not operate within certain distance of schools or childcare facilities",
                    selected=True,
                    citations=["§ 91.83(C)"],
                    supporting_passages=[
                        "No SSP facility will be allowed to operate within 750 feet of any state-licensed daycare facility."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Permit or license required for operation",
                    selected=True,
                    citations=["§ 91.87(D)"],
                    supporting_passages=[
                        "A mobile or pop-up exchange SSP program proposed to be operated on public property shall require prior approval of the commissioner."
                    ],
                ),
                ResponseOptionEvidence(option="No restrictions listed", selected=False),
            ],
        )

        validated = query_module._apply_ssp_restriction_consistency_validator(
            response,
            [],
            {
                "guidance_topic": "ssp_restriction",
                "response_options": (
                    "Programs may not operate within certain distance of schools or childcare facilities AND/OR "
                    "Permit or license required for operation AND/OR No restrictions listed"
                ),
            },
        )

        selected = [item.option for item in validated.option_evidence if item.selected]
        assert (
            "Programs may not operate within certain distance of schools or childcare facilities"
            in selected
        )
        assert "Permit or license required for operation" not in selected

    def test_exemption_gate_drops_cannabis_business_zoning_noise_without_carveout(self):
        response = LegalQueryResponse(
            short_answer="Paraphernalia for consumption of cannabis, generally or medical use",
            reasoning="Initial answer treated cannabis-business zoning text as an exemption.",
            citations=["§ 12-4-10"],
            supporting_passages=[
                "Cannabis retail businesses are a permitted use in this zoning district."
            ],
            confidence=0.6,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option="None", selected=False),
                ResponseOptionEvidence(
                    option="Paraphernalia for consumption of cannabis, generally or medical use",
                    selected=True,
                    citations=["§ 12-4-10"],
                    supporting_passages=[
                        "Cannabis retail businesses are a permitted use in this zoning district."
                    ],
                ),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-noise",
                heading_text="# Zoning",
                body_text="Cannabis retail businesses are a permitted use in this zoning district.",
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

        assert gated.short_answer == "None"
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "None"
        ]

    def test_ssp_permit_validator_forces_no_for_registration_only_regime(self):
        response = LegalQueryResponse(
            short_answer="Yes",
            reasoning="Initial answer treated registration and mayoral approval as explicit authorization.",
            citations=["R.S.A. 318-B:43"],
            supporting_passages=[
                "Each program shall register annually with the city health department and maintain complaint procedures approved by the mayor."
            ],
            confidence=0.57,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option="No", selected=False),
                ResponseOptionEvidence(option="Yes", selected=True),
                ResponseOptionEvidence(
                    option="Yes, only if a local public health emergency or disease outbreak has been declared",
                    selected=False,
                ),
            ],
        )

        validated = query_module._apply_ssp_permit_validator(
            response,
            [],
            {
                "variable_name": "ssp_permit",
                "response_options": (
                    "No OR Yes OR Yes, only if a local public health emergency or disease outbreak has been declared"
                ),
            },
        )

        assert validated.short_answer == "No"
        assert [item.option for item in validated.option_evidence if item.selected] == [
            "No"
        ]

    def test_ssp_permit_validator_promotes_no_to_conditional_yes_for_emergency_authorization(
        self,
    ):
        response = LegalQueryResponse(
            short_answer="No",
            reasoning="Initial answer missed the emergency-conditioned authorization.",
            citations=["Sec. 8.32(b)"],
            supporting_passages=[
                "The Mayor is hereby empowered to declare the existence of a Local Public Health Emergency when the Mayor finds that the authorization of clean needle and syringe exchange projects would abate the spread of HIV and AIDS."
            ],
            confidence=0.61,
            limitations="",
            option_evidence=[],
        )
        sections = [
            SectionResult(
                section_id="s-ssp-permit-emergency",
                heading_text="# Local Public Health Emergency",
                body_text=(
                    "The Mayor is hereby empowered to declare the existence of a Local Public Health Emergency when the Mayor finds that the authorization of clean needle and syringe exchange projects would abate the spread of HIV and AIDS."
                ),
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        validated = query_module._apply_ssp_permit_validator(
            response,
            sections,
            {
                "variable_name": "ssp_permit",
                "response_options": (
                    "No OR Yes OR Yes, only if a local public health emergency or disease outbreak has been declared"
                ),
            },
        )

        assert (
            validated.short_answer
            == "Yes, only if a local public health emergency or disease outbreak has been declared"
        )

    def test_ssp_restriction_gate_keeps_only_mobile_restriction_without_permit_signal(
        self,
    ):
        response = LegalQueryResponse(
            short_answer="Permit or license required for operation AND/OR Restrictions on mobile sites",
            reasoning="Initial answer treated site approval as an operating permit requirement.",
            citations=["§ 46-3"],
            supporting_passages=[
                "Mobile units may operate only at approved sites designated by the health department."
            ],
            confidence=0.55,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Permit or license required for operation",
                    selected=True,
                    citations=["§ 46-3"],
                    supporting_passages=[
                        "Mobile units may operate only at approved sites designated by the health department."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Restrictions on mobile sites",
                    selected=True,
                    citations=["§ 46-3"],
                    supporting_passages=[
                        "Mobile units may operate only at approved sites designated by the health department."
                    ],
                ),
                ResponseOptionEvidence(option="No restrictions listed", selected=False),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-ssp-restrict",
                heading_text="# Mobile units",
                body_text="Mobile units may operate only at approved sites designated by the health department.",
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
                "guidance_topic": "ssp_restriction",
                "response_options": (
                    "Permit or license required for operation AND/OR "
                    "Restrictions on mobile sites AND/OR No restrictions listed"
                ),
            },
        )

        assert gated.short_answer == "Restrictions on mobile sites"
        assert [item.option for item in gated.option_evidence if item.selected] == [
            "Restrictions on mobile sites",
        ]

    def test_ssp_restriction_consistency_promotes_permit_when_reasoning_and_evidence_require_it(
        self,
    ):
        response = LegalQueryResponse(
            short_answer="No restrictions listed",
            reasoning="The ordinance explicitly requires a permit to operate a syringe exchange facility.",
            citations=["§ 9-15-4"],
            supporting_passages=[
                "No person shall operate a syringe exchange facility without having a valid permit.",
            ],
            confidence=0.72,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Permit or license required for operation", selected=False
                ),
                ResponseOptionEvidence(option="No restrictions listed", selected=True),
            ],
        )

        validated = query_module._apply_ssp_restriction_consistency_validator(
            response,
            [],
            {
                "guidance_topic": "ssp_restriction",
                "response_options": (
                    "Permit or license required for operation AND/OR No restrictions listed"
                ),
            },
        )

        assert validated.short_answer == "Permit or license required for operation"

    def test_ssp_restriction_consistency_does_not_promote_registration_only_operation(
        self,
    ):
        response = LegalQueryResponse(
            short_answer="No restrictions listed",
            reasoning="Registration requirement governs SSP operation.",
            citations=["§ 9-15-9"],
            supporting_passages=[
                "A syringe exchange program registration is required for operation in the city.",
            ],
            confidence=0.71,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Permit or license required for operation", selected=False
                ),
                ResponseOptionEvidence(option="No restrictions listed", selected=True),
            ],
        )

        validated = query_module._apply_ssp_restriction_consistency_validator(
            response,
            [],
            {
                "guidance_topic": "ssp_restriction",
                "response_options": (
                    "Permit or license required for operation AND/OR No restrictions listed"
                ),
            },
        )

        assert validated.short_answer == "No restrictions listed"

    def test_exemption_crosswalk_maps_lawful_hypodermic_to_approved_medical_use(self):
        response = LegalQueryResponse(
            short_answer="Lawful use of hypodermic syringes",
            reasoning="The exemption allows lawful use of hypodermic syringes for approved medical use, including diabetes care.",
            citations=["§ 607.17(d)"],
            supporting_passages=[
                "The lawful use of hypodermic syringes for approved medical use, including diabetes treatment, is exempt.",
            ],
            confidence=0.77,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Syringes for approved medical use (i.e. diabetes)",
                    selected=False,
                ),
                ResponseOptionEvidence(
                    option="Lawful use of hypodermic syringes", selected=True
                ),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-exempt",
                heading_text="# Exemptions",
                body_text="The lawful use of hypodermic syringes for approved medical use, including diabetes treatment, is exempt.",
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
                    "Syringes for approved medical use (i.e. diabetes) AND/OR Lawful use of hypodermic syringes"
                ),
            },
        )

        assert gated.short_answer == "Syringes for approved medical use (i.e. diabetes)"

    def test_exemption_crosswalk_maps_prescription_carveout_to_other_approved_medical_use(
        self,
    ):
        response = LegalQueryResponse(
            short_answer=(
                "Professionals acting in their course of business [e.g. pharmacists, physicians, manufacturers]"
            ),
            reasoning="Initial answer omitted the medical-use carve-out tied to prescription authority.",
            citations=["§ 41A-13.12(c)"],
            supporting_passages=[
                "This section does not apply to paraphernalia possessed or used by a person under a prescription issued by a licensed physician or dentist authorized to prescribe controlled substances."
            ],
            confidence=0.71,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option="None", selected=False),
                ResponseOptionEvidence(
                    option="Other paraphernalia for approved medical use",
                    selected=False,
                ),
                ResponseOptionEvidence(
                    option="Professionals acting in their course of business [e.g. pharmacists, physicians, manufacturers]",
                    selected=True,
                    citations=["§ 41A-13.12(c)"],
                    supporting_passages=[
                        "This section does not apply to paraphernalia possessed or used by a person under a prescription issued by a licensed physician or dentist authorized to prescribe controlled substances."
                    ],
                ),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-prescription-exemption",
                heading_text="# Exemptions",
                body_text=(
                    "This section does not apply to paraphernalia possessed or used by a person under a prescription issued by a licensed physician or dentist authorized to prescribe controlled substances."
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
                "guidance_topic": "exemption_presence",
                "response_options": (
                    "None AND/OR Other paraphernalia for approved medical use AND/OR "
                    "Professionals acting in their course of business [e.g. pharmacists, physicians, manufacturers]"
                ),
            },
        )

        selected = [item.option for item in gated.option_evidence if item.selected]
        assert "Other paraphernalia for approved medical use" in selected

    def test_exemption_crosswalk_keeps_professionals_when_institute_support_coexists_with_prescription_medical_use(
        self,
    ):
        response = LegalQueryResponse(
            short_answer=(
                "Professionals acting in their course of business [e.g. pharmacists, physicians, manufacturers]"
            ),
            reasoning="The exemption includes both institute-based professional support and prescription-based medical use.",
            citations=["§ 31-32.1(c)"],
            supporting_passages=[
                "This section does not apply to paraphernalia possessed or used by a medical, educational, or research institute operating in compliance with all applicable city ordinances and state and federal laws.",
                "This section does not apply to paraphernalia possessed or used by a person under a prescription issued by a licensed physician or dentist authorized to prescribe controlled substances.",
            ],
            confidence=0.73,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option="None", selected=False),
                ResponseOptionEvidence(
                    option="Other paraphernalia for approved medical use",
                    selected=False,
                ),
                ResponseOptionEvidence(
                    option="Professionals acting in their course of business [e.g. pharmacists, physicians, manufacturers]",
                    selected=True,
                    citations=["§ 31-32.1(c)(3)"],
                    supporting_passages=[
                        "This section does not apply to paraphernalia possessed or used by a medical, educational, or research institute operating in compliance with all applicable city ordinances and state and federal laws."
                    ],
                ),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-dallas-exemption-mixed",
                heading_text="# Defenses",
                body_text=(
                    "This section does not apply to paraphernalia possessed or used by a medical, educational, or research institute operating in compliance with all applicable city ordinances and state and federal laws. This section does not apply to paraphernalia possessed or used by a person under a prescription issued by a licensed physician or dentist authorized to prescribe controlled substances."
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
                "guidance_topic": "exemption_presence",
                "response_options": (
                    "None AND/OR Other paraphernalia for approved medical use AND/OR "
                    "Professionals acting in their course of business [e.g. pharmacists, physicians, manufacturers]"
                ),
            },
        )

        selected = [item.option for item in gated.option_evidence if item.selected]
        assert "Other paraphernalia for approved medical use" in selected
        assert (
            "Professionals acting in their course of business [e.g. pharmacists, physicians, manufacturers]"
            in selected
        )

    def test_exemption_crosswalk_keeps_lawful_hypodermic_without_medical_scope(self):
        response = LegalQueryResponse(
            short_answer="Lawful use of hypodermic syringes",
            reasoning="Broad lawful carve-out with no medical qualifier.",
            citations=["§ 607.17(d)"],
            supporting_passages=[
                "The lawful use of hypodermic syringes is exempt.",
            ],
            confidence=0.73,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Syringes for approved medical use (i.e. diabetes)",
                    selected=False,
                ),
                ResponseOptionEvidence(
                    option="Lawful use of hypodermic syringes", selected=True
                ),
            ],
        )

        gated = query_module._apply_authoritative_option_evidence_gate(
            response,
            [],
            {
                "guidance_topic": "exemption_presence",
                "response_options": (
                    "Syringes for approved medical use (i.e. diabetes) AND/OR Lawful use of hypodermic syringes"
                ),
            },
        )

        assert gated.short_answer == "Lawful use of hypodermic syringes"

    def test_prohibited_activity_gate_drops_illegal_smoking_product_delivery_noise(
        self,
    ):
        response = LegalQueryResponse(
            short_answer=(
                "Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange AND/OR "
                "Possession, possession with intent to use, keep"
            ),
            reasoning="Initial answer mixed illegal smoking product delivery with paraphernalia possession.",
            citations=["§ 31-32.1"],
            supporting_passages=[
                "A person commits an offense if the person delivers any illegal smoking product. A person commits an offense if the person possesses with intent to use illegal smoking paraphernalia."
            ],
            confidence=0.66,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange",
                    selected=True,
                    citations=["§ 31-32.1"],
                    supporting_passages=[
                        "A person commits an offense if the person delivers any illegal smoking product."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Possession, possession with intent to use, keep",
                    selected=True,
                    citations=["§ 31-32.1"],
                    supporting_passages=[
                        "A person commits an offense if the person possesses with intent to use illegal smoking paraphernalia."
                    ],
                ),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-dallas-delivery",
                heading_text="# Illegal smoking products and paraphernalia",
                body_text=(
                    "A person commits an offense if the person delivers any illegal smoking product. "
                    "A person commits an offense if the person possesses with intent to use illegal smoking paraphernalia."
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
                "guidance_topic": "prohibited_activity",
                "response_options": (
                    "Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange AND/OR "
                    "Possession, possession with intent to use, keep AND/OR Not specified"
                ),
            },
        )

        assert gated.short_answer == "Possession, possession with intent to use, keep"

    def test_prohibited_activity_gate_drops_tobacco_retail_noise(self):
        response = LegalQueryResponse(
            short_answer=(
                "Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange AND/OR "
                "Give away, give, gift, free distribution"
            ),
            reasoning="Initial answer relied on tobacco-retailer compliance text.",
            citations=["§ 5.90.020(G)"],
            supporting_passages=[
                "It shall be a violation of this chapter for any person engaged in tobacco retailing or any of the tobacco retailer's agents or employees to violate any local, state, or federal law regulating controlled substances or drug paraphernalia."
            ],
            confidence=0.64,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange",
                    selected=True,
                    citations=["§ 5.90.020(G)"],
                    supporting_passages=[
                        "It shall be a violation of this chapter for any person engaged in tobacco retailing or any of the tobacco retailer's agents or employees to violate any local, state, or federal law regulating controlled substances or drug paraphernalia."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Give away, give, gift, free distribution",
                    selected=True,
                    citations=["§ 5.90.020(G)"],
                    supporting_passages=[
                        "It shall be a violation of this chapter for any person engaged in tobacco retailing or any of the tobacco retailer's agents or employees to violate any local, state, or federal law regulating controlled substances or drug paraphernalia."
                    ],
                ),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-alhambra-noise",
                heading_text="# Tobacco retail license",
                body_text=(
                    "It shall be a violation of this chapter for any person engaged in tobacco retailing or any of the tobacco retailer's agents or employees to violate any local, state, or federal law regulating controlled substances or drug paraphernalia."
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
                "guidance_topic": "prohibited_activity",
                "response_options": (
                    "Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange AND/OR "
                    "Give away, give, gift, free distribution AND/OR Not specified"
                ),
            },
        )

        assert gated.short_answer == "Not specified"

    def test_answer_review_decision_flags_dp_law_scope_noise_for_rerun(self):
        response = LegalQueryResponse(
            short_answer="Yes",
            reasoning="The local code references drug paraphernalia in a tobacco retail license chapter.",
            citations=["§ 5.90.020(G)"],
            supporting_passages=[
                "It shall be a violation of this chapter for any person engaged in tobacco retailing or any of the tobacco retailer's agents or employees to violate any local, state, or federal law regulating controlled substances or drug paraphernalia."
            ],
            confidence=0.72,
            limitations="",
        )
        sections = [
            SectionResult(
                section_id="s-alhambra-dp-law-noise",
                heading_text="# Tobacco retail license",
                body_text=(
                    "It shall be a violation of this chapter for any person engaged in tobacco retailing or any of the tobacco retailer's agents or employees to violate any local, state, or federal law regulating controlled substances or drug paraphernalia."
                ),
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        decision = query_module._build_answer_review_decision(
            response=response,
            sections=sections,
            query_metadata={
                "variable_name": "dp_law",
                "guidance_topic": "existence_scope",
                "response_options": "Yes OR No",
            },
            settings=QuerySettings(
                llm=LLMConfig(client=Mock()),
                enable_answer_review=True,
            ),
        )

        assert decision.should_rerun is True
        assert any(
            signal.issue == "dp_law_yes_may_rest_on_scope_noise_only"
            for signal in decision.reasons
        )

    def test_activity_gate_drops_selected_labels_without_option_specific_support(self):
        response = LegalQueryResponse(
            short_answer=(
                "Give away, give, gift, free distribution AND/OR Advertising, display AND/OR "
                "Manufacturing, manufacture with intent to deliver or sell"
            ),
            reasoning="Initial answer over-selected unsupported activity labels.",
            citations=["SEC. 31-32.1(b)(1)", "SEC. 31-32.1(b)(4)"],
            supporting_passages=[
                "(1) possesses, buys, sells, offers for sale, delivers, or transfers any illegal smoking product.",
                "(4) uses or possesses with the intent to use any illegal smoking paraphernalia.",
            ],
            confidence=0.64,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Give away, give, gift, free distribution",
                    selected=True,
                ),
                ResponseOptionEvidence(
                    option="Advertising, display",
                    selected=True,
                ),
                ResponseOptionEvidence(
                    option="Manufacturing, manufacture with intent to deliver or sell",
                    selected=True,
                ),
                ResponseOptionEvidence(option="Not specified", selected=False),
            ],
        )
        sections = [
            SectionResult(
                section_id="s-dallas-activity",
                heading_text="# Smoking products",
                body_text=(
                    "(1) possesses, buys, sells, offers for sale, delivers, or transfers any illegal smoking product. "
                    "(4) uses or possesses with the intent to use any illegal smoking paraphernalia."
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
                "guidance_topic": "prohibited_activity",
                "response_options": (
                    "Give away, give, gift, free distribution AND/OR Advertising, display AND/OR "
                    "Manufacturing, manufacture with intent to deliver or sell AND/OR Not specified"
                ),
            },
        )

        assert gated.short_answer == "Not specified"

    def test_answer_review_decision_flags_unselected_activity_option_with_strong_support(
        self,
    ):
        response = LegalQueryResponse(
            short_answer="Advertising, display AND/OR Manufacturing, manufacture with intent to deliver or sell",
            reasoning=(
                "The ordinance prohibits delivery, advertising, and manufacture with intent to deliver."
            ),
            citations=["§ 12-4-10(C)(1)", "§ 12-4-10(C)(2)"],
            supporting_passages=[
                "It is unlawful for any person to deliver, possess with intent to deliver, or manufacture with intent to deliver, drug paraphernalia.",
                "It is unlawful for any person to place in any newspaper, magazine, handbill, or other publication any advertisement for drug paraphernalia.",
            ],
            confidence=0.83,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange",
                    selected=False,
                    citations=["§ 12-4-10(C)(1)"],
                    supporting_passages=[
                        "It is unlawful for any person to deliver, possess with intent to deliver, or manufacture with intent to deliver, drug paraphernalia."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Advertising, display",
                    selected=True,
                    citations=["§ 12-4-10(C)(2)"],
                    supporting_passages=[
                        "It is unlawful for any person to place in any newspaper, magazine, handbill, or other publication any advertisement for drug paraphernalia."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Manufacturing, manufacture with intent to deliver or sell",
                    selected=True,
                    citations=["§ 12-4-10(C)(1)"],
                    supporting_passages=[
                        "It is unlawful for any person to deliver, possess with intent to deliver, or manufacture with intent to deliver, drug paraphernalia."
                    ],
                ),
            ],
        )

        decision = query_module._build_answer_review_decision(
            response=response,
            sections=[],
            query_metadata={
                "variable_name": "dp_activity",
                "guidance_topic": "prohibited_activity",
                "response_options": (
                    "Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange AND/OR "
                    "Advertising, display AND/OR Manufacturing, manufacture with intent to deliver or sell"
                ),
            },
            settings=QuerySettings(
                llm=LLMConfig(client=Mock()),
                enable_answer_review=True,
            ),
        )

        assert decision.should_rerun is True
        assert any(
            signal.issue == "unselected_option_has_strong_support"
            and signal.option
            == "Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange"
            for signal in decision.reasons
        )

    def test_answer_review_decision_flags_penalty_reasoning_that_mentions_civil_penalty(
        self,
    ):
        response = LegalQueryResponse(
            short_answer='"Unlawful" only',
            reasoning=(
                "Section 9-629(4) states that a violator is subject to a civil penalty of $150."
            ),
            citations=["§ 9-629(4)"],
            supporting_passages=[
                "A violator is subject to a civil penalty of $150."
            ],
            confidence=0.74,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option='"Unlawful" only', selected=True),
                ResponseOptionEvidence(option="Civil Fine", selected=False),
            ],
        )

        decision = query_module._build_answer_review_decision(
            response=response,
            sections=[],
            query_metadata={
                "variable_name": "dp_penalties",
                "guidance_topic": "penalty",
                "response_options": '"Unlawful" only AND/OR Civil Fine',
            },
            settings=QuerySettings(
                llm=LLMConfig(client=Mock()),
                enable_answer_review=True,
            ),
        )

        assert decision.should_rerun is True
        assert any(
            signal.issue == "reasoning_mentions_unselected_penalty_option"
            and signal.option == "Civil Fine"
            for signal in decision.reasons
        )

    def test_answer_review_decision_ignores_negated_penalty_mentions(self):
        response = LegalQueryResponse(
            short_answer='"Unlawful" only',
            reasoning=(
                "The penalty section text is not provided here. No other penalties "
                "such as incarceration or criminal fines are specified, and misdemeanor "
                "appears only in a reference note so it is not coded."
            ),
            citations=["§ 134.26"],
            supporting_passages=["It shall be unlawful for any person to use or possess with intent to use drug paraphernalia."],
            confidence=0.71,
            limitations="",
            option_evidence=[
                ResponseOptionEvidence(option='"Unlawful" only', selected=True),
                ResponseOptionEvidence(option="Misdemeanor", selected=False),
                ResponseOptionEvidence(option="Criminal Fine", selected=False),
                ResponseOptionEvidence(option="Incarceration", selected=False),
            ],
        )

        decision = query_module._build_answer_review_decision(
            response=response,
            sections=[],
            query_metadata={
                "variable_name": "dp_penalties",
                "guidance_topic": "penalty",
                "response_options": '"Unlawful" only AND/OR Misdemeanor AND/OR Criminal Fine AND/OR Incarceration',
            },
            settings=QuerySettings(
                llm=LLMConfig(client=Mock()),
                enable_answer_review=True,
            ),
        )

        assert not any(
            signal.issue == "reasoning_mentions_unselected_penalty_option"
            for signal in decision.reasons
        )


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

    def test_build_legal_prompts_allows_parent_context_for_reasoning_only(self):
        system_prompt, _user_prompt = _build_legal_prompts(
            "What is the current-through date?",
            "Retrieval Unit 1: Ordinance history\nContent: Current through Ordinance 2024-10 adopted March 19, 2024.",
            query_metadata={
                "response_options": (
                    "Responses: Known, <current through date published in ordinance> "
                    "OR Partially known, <partial current through date published in ordinance "
                    "(month or day imputed)> OR Unknown, <date of data collection>"
                ),
                "parent_contexts": [
                    {
                        "query_id": "Q1",
                        "question": "Does the code specify a current-through date?",
                        "short_answer": "Yes",
                    }
                ],
            },
        )

        assert "Dependency context from upstream questions:" in system_prompt
        assert (
            "You may use upstream dependency context to inform your reasoning"
            in system_prompt
        )
        assert (
            "do not copy parent-question text or parent-answer text into `supporting_passages`"
            in system_prompt
        )
        assert (
            "Every item in `supporting_passages` must be a verbatim quote from the retrieved Legal Context"
            in system_prompt
        )

    def test_build_legal_prompts_tells_citation_children_to_stay_with_parent_family(
        self,
    ):
        system_prompt, _user_prompt = _build_legal_prompts(
            "If yes, what is the citation of the relevant law?",
            "Retrieval Unit 1: Section 12-4-10 incorporates the State Controlled Substances Act.",
            query_metadata={
                "response_options": "Responses: <citation> OR Unknown",
                "parent_contexts": [
                    {
                        "query_id": "Q1",
                        "question": "Does local law require outside-law review?",
                        "short_answer": "Yes",
                        "option_evidence": [
                            {
                                "option": "Yes",
                                "selected": True,
                                "citations": ["Sections 30-31-1 et seq. NMSA 1978"],
                                "supporting_passages": [
                                    "This section incorporates the State Controlled Substances Act, Sections 30-31-1 et seq. NMSA 1978."
                                ],
                            }
                        ],
                    }
                ],
            },
        )

        assert "keep the chosen citation in that same family" in system_prompt


class TestCurrentThroughHelpers:
    def test_prefers_metadata_sections_for_current_through_completion(self):
        sections = [
            SectionResult(
                section_id="s0",
                heading_text="# Weeds and litter",
                body_text="No person shall deposit litter in the right-of-way.",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.5,
                segment_count=1,
            ),
            SectionResult(
                section_id="s1",
                heading_text="# Publisher's note",
                body_text="Current through Ordinance 2024-10 adopted March 19, 2024.",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.9,
                segment_count=1,
            ),
        ]

        preferred = query_module._prefer_current_through_metadata_sections(sections)

        assert [section.section_id for section in preferred] == ["s1"]

    def test_does_not_treat_bare_ordinance_history_as_current_through_metadata(self):
        sections = [
            SectionResult(
                section_id="s0",
                heading_text="# Offenses",
                body_text="(Ord. 2025-009)",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.5,
                segment_count=1,
            )
        ]

        preferred = query_module._prefer_current_through_metadata_sections(sections)

        assert [section.section_id for section in preferred] == ["s0"]
        assert (
            query_module._section_matches_current_through_metadata(sections[0]) is False
        )

    def test_date_surface_validator_uses_explicit_current_through_date(self):
        response = LegalQueryResponse(
            short_answer="Unknown",
            reasoning="No date found.",
            citations=["Legal Intro"],
            supporting_passages=[
                "Contains 2025 S-95, current through Ordinance 2025-026, passed 9-3-2025"
            ],
            confidence=0.6,
            limitations="",
            option_evidence=[],
        )
        sections = [
            SectionResult(
                section_id="s1",
                heading_text="Legal Intro",
                body_text="Contains 2025 S-95, current through Ordinance 2025-026, passed 9-3-2025",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        validated = query_module._apply_date_surface_validators(
            response,
            sections,
            {
                "query_id": "ssp_collected",
                "response_options": "Responses: <current-through date>",
            },
        )

        assert validated.short_answer == "09/03/2025"

    def test_date_surface_validator_filters_historical_amendment_dates_for_collected(
        self,
    ):
        response = LegalQueryResponse(
            short_answer="02/04/1987",
            reasoning="Using amendment history date.",
            citations=["Legal Intro"],
            supporting_passages=[
                "History: amended 02/04/1987. Current through Ordinance 2025-026 as of 03/26/2025."
            ],
            confidence=0.6,
            limitations="",
            option_evidence=[],
        )
        sections = [
            SectionResult(
                section_id="s1",
                heading_text="Legal Intro",
                body_text=(
                    "History: amended 02/04/1987. "
                    "Current through Ordinance 2025-026 as of 03/26/2025."
                ),
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        validated = query_module._apply_date_surface_validators(
            response,
            sections,
            {
                "query_id": "dp_collected",
                "response_options": "Responses: <current-through date>",
            },
        )

        assert validated.short_answer == "03/26/2025"

    def test_date_surface_validator_prefers_date_sentence_nearest_anchor(self):
        response = LegalQueryResponse(
            short_answer="Unknown",
            reasoning="No date found.",
            citations=["Legal Intro"],
            supporting_passages=[
                (
                    "Passed 01/01/1990 in early ordinance history. "
                    "Current through Ordinance 2025-026. "
                    "Passed 03/26/2025."
                )
            ],
            confidence=0.6,
            limitations="",
            option_evidence=[],
        )
        sections = [
            SectionResult(
                section_id="s1",
                heading_text="Legal Intro",
                body_text=(
                    "Passed 01/01/1990 in early ordinance history. "
                    "Current through Ordinance 2025-026. "
                    "Passed 03/26/2025."
                ),
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        validated = query_module._apply_date_surface_validators(
            response,
            sections,
            {
                "query_id": "ssp_collected",
                "response_options": "Responses: <current-through date>",
            },
        )

        assert validated.short_answer == "03/26/2025"

    def test_date_surface_validator_rejects_inferred_only_dates_to_unknown(self):
        response = LegalQueryResponse(
            short_answer="07/15/2001",
            reasoning="Inferred year-only effective date.",
            citations=["§ 4-1"],
            supporting_passages=["This ordinance became effective in 2001."],
            confidence=0.6,
            limitations="",
            option_evidence=[],
        )
        sections = [
            SectionResult(
                section_id="s1",
                heading_text="Effective date",
                body_text="This ordinance became effective in 2001.",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        validated = query_module._apply_date_surface_validators(
            response,
            sections,
            {
                "query_id": "dp_effective_dt",
                "response_options": "Responses: <effective date> OR Unknown",
            },
        )

        assert validated.short_answer == "Unknown"

    def test_date_surface_validator_uses_enacted_and_effective_scope_anchors(self):
        enacted_response = LegalQueryResponse(
            short_answer="Unknown",
            reasoning="No date found.",
            citations=["§ 7-1"],
            supporting_passages=[
                "This chapter was adopted by ordinance on 07/19/1989. Effective 02/04/1991."
            ],
            confidence=0.6,
            limitations="",
            option_evidence=[],
        )
        effective_response = LegalQueryResponse(
            short_answer="Unknown",
            reasoning="No date found.",
            citations=["§ 7-1"],
            supporting_passages=[
                "This chapter was adopted by ordinance on 07/19/1989. Takes effect 02/04/1991."
            ],
            confidence=0.6,
            limitations="",
            option_evidence=[],
        )
        sections = [
            SectionResult(
                section_id="s1",
                heading_text="History",
                body_text=(
                    "This chapter was adopted by ordinance on 07/19/1989. "
                    "Takes effect 02/04/1991."
                ),
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        enacted_validated = query_module._apply_date_surface_validators(
            enacted_response,
            sections,
            {
                "query_id": "dp_enacted",
                "response_options": "Responses: <enactment date> OR Unknown",
            },
        )
        effective_validated = query_module._apply_date_surface_validators(
            effective_response,
            sections,
            {
                "query_id": "dp_effective_dt",
                "response_options": "Responses: <effective date> OR Unknown",
            },
        )

        assert enacted_validated.short_answer == "07/19/1989"
        assert effective_validated.short_answer == "02/04/1991"

    def test_ssp_permit_validator_promotes_no_to_yes_for_explicit_permit_regime(self):
        response = LegalQueryResponse(
            short_answer="No",
            reasoning="Initial answer missed the permit regime.",
            citations=["§ 9-15-4"],
            supporting_passages=[
                "No person shall operate a syringe exchange facility at any location in the City without having a valid permit for such syringe exchange facility in accordance with this ordinance."
            ],
            confidence=0.7,
            limitations="",
            option_evidence=[],
        )
        sections = [
            SectionResult(
                section_id="s2",
                heading_text="# Permit",
                body_text=(
                    "No person shall operate a syringe exchange facility at any location in the City without having a valid permit for such syringe exchange facility in accordance with this ordinance."
                ),
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        validated = query_module._apply_ssp_permit_validator(
            response,
            sections,
            {
                "query_id": "ssp_permit",
                "response_options": (
                    "Responses: No OR Yes OR Yes, only if a local public health emergency or disease outbreak has been declared"
                ),
            },
        )

        assert validated.short_answer == "Yes"

    def test_answer_review_decision_flags_reasoning_short_answer_conflict(self):
        response = LegalQueryResponse(
            short_answer="No",
            reasoning=(
                "The ordinance explicitly authorizes syringe exchange operation once a valid local permit is obtained."
            ),
            citations=["§ 9-15-4"],
            supporting_passages=[
                "No person shall operate a syringe exchange facility without having a valid permit."
            ],
            confidence=0.61,
            limitations="",
            option_evidence=[],
        )
        sections = [
            SectionResult(
                section_id="s-review-1",
                heading_text="# Permit",
                body_text=(
                    "No person shall operate a syringe exchange facility without having a valid permit."
                ),
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=1.0,
                segment_count=1,
            )
        ]

        decision = query_module._build_answer_review_decision(
            response=response,
            sections=sections,
            query_metadata={
                "variable_name": "ssp_permit",
                "guidance_topic": "ssp_authorization",
                "response_options": (
                    "No OR Yes OR Yes, only if a local public health emergency or disease outbreak has been declared"
                ),
            },
            settings=QuerySettings(
                llm=LLMConfig(client=Mock()),
                enable_answer_review=True,
                answer_review_topics=("ssp_authorization",),
            ),
        )

        assert decision.should_rerun is True
        assert any(
            signal.issue == "reasoning_conflicts_with_short_answer"
            for signal in decision.reasons
        )

    def test_resolve_completion_sections_relaxes_threshold_when_filter_empties(
        self, monkeypatch
    ):
        sections = [
            SectionResult(
                section_id="s-fallback-1",
                heading_text="# Permit",
                body_text="A valid local permit is required to operate the syringe exchange facility.",
                heading_level=1,
                parent_id=None,
                matching_segments=[],
                relevance_score=0.2,
                segment_count=1,
            )
        ]
        retrieval_results = SectionCollection(
            sections=sections,
            query_info=QueryInfo(original_query="query"),
        )
        calls: list[float] = []

        def _fake_filter_sections(**kwargs):
            calls.append(kwargs["relevance_threshold"])
            if len(calls) == 1:
                return SectionCollection(
                    sections=[],
                    query_info=retrieval_results.query_info,
                    filtering_metadata=FilteringMetadata(
                        original_count=1,
                        filtered_count=0,
                        threshold=kwargs["relevance_threshold"],
                        assessments=[
                            {
                                "index": 0,
                                "section_id": "s-fallback-1",
                                "relevance_score": 0.55,
                                "reasoning": "Borderline on first pass.",
                                "kept": False,
                                "keep_reason": "below_threshold",
                            }
                        ],
                    ),
                )
            return SectionCollection(
                sections=sections,
                query_info=retrieval_results.query_info,
                filtering_metadata=FilteringMetadata(
                    original_count=1,
                    filtered_count=1,
                    threshold=kwargs["relevance_threshold"],
                    assessments=[
                        {
                            "index": 0,
                            "section_id": "s-fallback-1",
                            "relevance_score": 0.55,
                            "reasoning": "Recovered at relaxed threshold.",
                            "kept": True,
                            "keep_reason": "threshold_relaxation",
                        }
                    ],
                ),
            )

        monkeypatch.setattr(query_module, "filter_sections", _fake_filter_sections)
        monkeypatch.setattr(
            query_module,
            "resolve_relevance_filter_client_factory",
            lambda _llm: None,
        )

        debug_capture = {"relevance": {}, "query": {}}
        resolved = query_module._resolve_completion_sections(
            retrieval_results,
            "Does the ordinance explicitly authorize SSPs?",
            QuerySettings(
                llm=LLMConfig(client=Mock()),
                filter_relevance=True,
                relevance_threshold=0.7,
            ),
            debug_capture=debug_capture,
        )

        assert len(resolved) == 1
        assert calls == [0.7, 0.5]
        assert debug_capture["relevance"]["empty_filter_fallback_attempted"] is True
        assert debug_capture["relevance"]["empty_filter_fallback_recovered"] is True

    def test_dependency_decision_treats_abstaining_parent_as_non_blocking(self):
        hierarchy = QueryHierarchy(query_id="child", boolean_parent_ids=["parent"])
        parent_state = query_module.QueryExecutionState(
            query_id="parent",
            question="Parent question",
            prompt_question="Parent question",
            status="completed",
            short_answer="I cannot answer your question as no relevant legal provisions were found after filtering.",
        )

        decision = query_module._evaluate_dependency_decision(
            hierarchy=hierarchy,
            state_by_query_id={"parent": parent_state},
        )

        assert decision.should_skip is False
        assert any(
            rule.get("status") == "parent_abstained_non_blocking"
            for rule in decision.dependency_rules_evaluated
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

    def test_load_queries_truncates_ragged_rows_for_exported_csvs(self):
        """Extra trailing fields in exported benchmark rows should not abort query loading."""
        csv_content = (
            "question,variable_name,category\n"
            "Question 1,var1,general,unexpected_extra_value\n"
            "Question 2,var2,specific\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            queries = load_queries(temp_path, adjust_for_dataset=False)
            assert len(queries) == 2
            assert queries[0].question == "Question 1"
            assert queries[0].variable_name == "var1"
            assert queries[0].metadata == {"category": "general"}
            assert queries[1].metadata == {"category": "specific"}
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

        assert [section.section_id for section in augmented] == ["s_penalty", "s0"]
        assert [section.section_id for section in deduped] == ["s_penalty", "s0"]

    def test_augment_sections_with_same_text_cross_references_handles_provided_in_reference(
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
                    "A violation is punishable as provided in Section 10.99.",
                    "A violation is punishable by a fine not to exceed $500 or imprisonment for up to 60 days.",
                ],
                "heading_level": [1, 3],
                "parent_id": [None, None],
                "context_path": [None, "Chapter 10 > Section 10.99"],
            }
        ).write_parquet(sections_path)

        source_section = SectionResult(
            section_id="s0",
            heading_text="# Drug Paraphernalia",
            body_text="A violation is punishable as provided in Section 10.99.",
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

        assert [section.section_id for section in augmented] == ["s_penalty", "s0"]

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
            section.section_id for section in execution_capture["completion_sections"]
        ] == ["s_penalty", "s0"]
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

    def test_query_legal_documents_reruns_review_for_year_only_imputed_date(self):
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
        first_response = LegalQueryResponse(
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
        second_response = first_response.model_copy()
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
                "On which date was the ordinance enacted?",
                settings,
                query_metadata={
                    "response_options": "Responses: <enactment date> OR Unknown",
                },
                debug_capture=debug_capture,
            )

        assert len(prompts) == 2
        assert response.short_answer == "07/15/2022"
        assert debug_capture["query"]["review_rerun_triggered"] is True
        assert (
            "date_answer_uses_year_only_imputation"
            in debug_capture["query"]["review_rerun_reasons"]
        )

    def test_query_legal_documents_reruns_when_current_through_answer_lacks_explicit_date_support(
        self,
    ):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Weeds and litter",
                    body_text="(Ord. 2025-009)",
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
        first_response = LegalQueryResponse(
            short_answer="02/21/2025",
            reasoning="Fallback date.",
            citations=["§ 11-1-1-1"],
            supporting_passages=["(Ord. 2025-009)"],
            confidence=0.8,
            limitations="None",
            option_evidence=[],
        )
        second_response = first_response.model_copy()
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
                "What is the current-through date of the ordinance?",
                settings,
                query_metadata={
                    "query_id": "dp_collected",
                    "guidance_topic": "date_current_through",
                    "response_options": "Responses: <current-through date>",
                },
                debug_capture=debug_capture,
            )

        assert len(prompts) == 2
        assert response.short_answer == "02/21/2025"
        assert debug_capture["query"]["review_rerun_triggered"] is True
        assert (
            "current_through_answer_lacks_explicit_date_support"
            in debug_capture["query"]["review_rerun_reasons"]
        )

    def test_query_legal_documents_reruns_when_date_answer_has_invalid_calendar_value(
        self,
    ):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Current through",
                    body_text="Current through July 2, 2025.",
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
        first_response = LegalQueryResponse(
            short_answer="13/02/2025",
            reasoning="Using the displayed date.",
            citations=["§ 1-1"],
            supporting_passages=["Current through July 2, 2025."],
            confidence=0.62,
            limitations="None",
            option_evidence=[],
        )
        second_response = LegalQueryResponse(
            short_answer="07/02/2025",
            reasoning="The explicit current-through date is July 2, 2025.",
            citations=["§ 1-1"],
            supporting_passages=["Current through July 2, 2025."],
            confidence=0.85,
            limitations="None",
            option_evidence=[],
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
                "What is the current-through date of the ordinance?",
                settings,
                query_metadata={
                    "query_id": "dp_collected",
                    "guidance_topic": "date_current_through",
                    "response_options": "Responses: <current-through date>",
                },
                debug_capture=debug_capture,
            )

        assert len(prompts) == 2
        assert response.short_answer == "07/02/2025"
        assert debug_capture["query"]["review_rerun_triggered"] is True
        assert (
            "current_through_answer_lacks_explicit_date_support"
            in debug_capture["query"]["review_rerun_reasons"]
        )

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

    def test_query_legal_documents_reruns_when_multi_select_still_includes_other(self):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# Restrictions",
                    body_text="The operator shall obtain a permit before operating the SSP.",
                    heading_level=1,
                    parent_id=None,
                    matching_segments=[],
                    relevance_score=0.1,
                    segment_count=1,
                )
            ],
            query_info=QueryInfo(
                original_query="restrictions",
                total_segments_found=1,
                unique_sections=1,
            ),
        )
        first_response = LegalQueryResponse(
            short_answer="Permit or license required for operation AND/OR Other restrictions",
            reasoning="The permit requirement appears explicit.",
            citations=["§ 6-4-2"],
            supporting_passages=[
                "The operator shall obtain a permit before operating the SSP."
            ],
            confidence=0.7,
            limitations="None",
            option_evidence=[
                ResponseOptionEvidence(
                    option="Permit or license required for operation",
                    selected=True,
                    citations=["§ 6-4-2"],
                    supporting_passages=[
                        "The operator shall obtain a permit before operating the SSP."
                    ],
                ),
                ResponseOptionEvidence(
                    option="Other restrictions",
                    selected=True,
                    citations=["§ 6-4-2"],
                    supporting_passages=[
                        "The operator shall obtain a permit before operating the SSP."
                    ],
                ),
            ],
        )
        second_response = first_response.model_copy(
            update={"short_answer": "Permit or license required for operation"}
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
                "What restrictions apply to SSPs?",
                settings,
                query_metadata={
                    "response_options": (
                        "Responses: Permit or license required for operation AND/OR Other restrictions"
                    ),
                },
                debug_capture=debug_capture,
            )

        assert len(prompts) == 2
        assert response.short_answer == "Permit or license required for operation"
        assert debug_capture["query"]["review_rerun_triggered"] is True
        assert (
            "multi_select_includes_other"
            in debug_capture["query"]["review_rerun_reasons"]
        )

    def test_query_legal_documents_reruns_when_scalar_citation_conflicts_with_parent_family(
        self,
    ):
        mock_client = Mock(spec=Instructor)
        retrieval_results = SectionCollection(
            sections=[
                SectionResult(
                    section_id="s0",
                    heading_text="# State law reference",
                    body_text="This section also mentions Section 26-2C-1 of the Harm Reduction Act.",
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
        first_response = LegalQueryResponse(
            short_answer="§ 26-2C-1",
            reasoning="The local text mentions the Harm Reduction Act.",
            citations=["§ 12-4-10"],
            supporting_passages=[
                "This section also mentions Section 26-2C-1 of the Harm Reduction Act."
            ],
            confidence=0.8,
            limitations="None",
            option_evidence=[],
        )
        second_response = first_response.model_copy(
            update={"short_answer": "§ 30-31-1"}
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
                "If yes, what is the citation of the relevant law?",
                settings,
                query_metadata={
                    "response_options": "Responses: <citation> OR Unknown",
                    "parent_contexts": [
                        {
                            "query_id": "Q1",
                            "question": "Does local law require outside-law review?",
                            "short_answer": "Yes",
                            "option_evidence": [
                                {
                                    "option": "Yes",
                                    "selected": True,
                                    "citations": ["Sections 30-31-1 et seq. NMSA 1978"],
                                    "supporting_passages": [
                                        "This section incorporates the State Controlled Substances Act, Sections 30-31-1 et seq. NMSA 1978."
                                    ],
                                }
                            ],
                        }
                    ],
                },
                debug_capture=debug_capture,
            )

        assert len(prompts) == 2
        assert response.short_answer == "§ 30-31-1"
        assert debug_capture["query"]["review_rerun_triggered"] is True
        assert (
            "citation_family_conflicts_with_parent_dependency_rationale"
            in debug_capture["query"]["review_rerun_reasons"]
        )

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

    def test_build_no_sections_response_uses_structured_ssp_fallback_when_no_anchor_text(
        self,
    ):
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

    def test_build_no_sections_response_keeps_abstention_when_anchor_text_was_seen(
        self,
    ):
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

    def test_run_queries_skips_child_when_dp_exemption_option_evidence_omits_blocker_label(
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
            short_answer="I cannot answer your question as no relevant legal provisions were found after filtering.",
            reasoning="Parent abstained after gating.",
            citations=[],
            supporting_passages=[],
            confidence=0.9,
            limitations="None",
            option_evidence=[
                ResponseOptionEvidence(option="None", selected=True),
                ResponseOptionEvidence(option="Syringes, generally", selected=False),
            ],
        )
        child_response = LegalQueryResponse(
            short_answer="Possession",
            reasoning="Child answer.",
            citations=[],
            supporting_passages=[],
            confidence=0.8,
            limitations="None",
        )

        parent_hierarchy = QueryHierarchy(query_id="Q1")
        child_hierarchy = QueryHierarchy(
            query_id="Q1.1",
            parent_ids=("dp_exemption",),
            label_blockers=(
                LabelBlockerRule(
                    parent_query_id="dp_exemption",
                    blocker_labels=("Syringes, generally",),
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
                )

                results_df = run_queries(
                    collection=Mock(),
                    sections_parquet_path=str(sections_path),
                    queries=[
                        QueryInput(
                            question="Parent",
                            variable_name="dp_exemption",
                            metadata={
                                "response_options": "None AND/OR Syringes, generally",
                                "hierarchy": hierarchy_to_metadata(parent_hierarchy),
                            },
                            query_id="Q1",
                        ),
                        QueryInput(
                            question="Child",
                            variable_name="dp_exempt_sygen_activity",
                            metadata={
                                "hierarchy": hierarchy_to_metadata(child_hierarchy),
                            },
                            query_id="Q1.1",
                        ),
                    ],
                    jurisdiction_id="IL-WindyTown",
                    settings=settings,
                )

        child_row = results_df.filter(pl.col("query_id") == "Q1.1")
        assert mock_query.call_count == 1
        assert child_row[0, "query_status"] == "skipped"
        assert child_row[0, "skip_reason"] == "label_blocker_not_satisfied"

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
