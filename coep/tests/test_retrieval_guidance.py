"""Tests for COEP-specific retrieval guidance hooks."""

from coep.src.retrieval_guidance import get_drug_paraphernalia_retrieval_guidance
from legiscope.query_hierarchy import (
    LabelBlockerRule,
    QueryHierarchy,
    hierarchy_to_metadata,
)
from legiscope.retrieval_guidance import ParentQueryContext, RetrievalGuidanceRequest


class TestCoepRetrievalGuidance:
    """Validate fine-grained variable-to-family mapping."""

    def test_returns_guidance_for_enactment_date_variable(self):
        request = RetrievalGuidanceRequest(
            query="When was this enacted?",
            variable_name="dp_enacted",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "date_enactment"
        assert guidance.shared_context is not None
        assert (
            "drug paraphernalia used with controlled substances"
            in guidance.shared_context
        )
        assert "enacted" in guidance.anchor_terms
        assert "became law" in guidance.anchor_terms
        assert guidance.retrieval_instructions is not None
        assert (
            "Retrieve ordinance metadata and amendment history"
            in guidance.retrieval_instructions
        )
        assert guidance.relevance_instructions is not None
        assert (
            "Prefer ordinance metadata and text that explicitly states when the law was enacted"
            in guidance.relevance_instructions
        )
        assert guidance.retrieval_query is not None
        assert "Legal context:" in guidance.retrieval_query
        assert "Question: When was this enacted?" in guidance.retrieval_query
        assert "High-value legal terms:" in guidance.retrieval_query
        assert guidance.completion_instructions is not None
        assert "Query context:" in guidance.completion_instructions
        assert "Variable family: date enactment." in guidance.completion_instructions
        assert "Prefer ordinance metadata" not in guidance.completion_instructions

    def test_returns_guidance_for_penalty_variable(self):
        request = RetrievalGuidanceRequest(
            query="What penalties apply?",
            variable_name="dp_penalties",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "penalty"
        assert "forfeiture" in guidance.anchor_terms
        assert guidance.retrieval_instructions is not None
        assert "Retrieve penalty sections" in guidance.retrieval_instructions
        assert "Exact legal labels matter" in guidance.relevance_instructions
        assert guidance.retrieval_query is not None
        assert "misdemeanor" in guidance.retrieval_query
        assert guidance.completion_instructions is not None
        assert "license revocation" in guidance.completion_instructions
        assert "SHOULD NOT be coded as Other" in guidance.completion_instructions

    def test_returns_guidance_for_exemption_variable_with_strict_other_rules(self):
        request = RetrievalGuidanceRequest(
            query="Which exemptions exist?",
            variable_name="dp_exemption",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "exemption_presence"
        assert guidance.completion_instructions is not None
        assert "tobacco-only exceptions" in guidance.completion_instructions
        assert "SHOULD NOT be coded as Other" in guidance.completion_instructions
        assert (
            "favor does not apply, does not include, exception"
            in guidance.completion_instructions
        )

    def test_returns_guidance_for_exemption_activity_variable(self):
        request = RetrievalGuidanceRequest(
            query="If cannabis paraphernalia is exempted, which activities are exempted?",
            variable_name="dp_exempt_can_activity",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "exemption_activity_scope"
        assert "cannabis" in guidance.anchor_terms
        assert "marijuana accessories" in guidance.anchor_terms
        assert guidance.retrieval_instructions is not None
        assert (
            "Prioritize cannabis, marijuana, marihuana"
            in guidance.retrieval_instructions
        )
        assert "which activities remain allowed" in guidance.relevance_instructions
        assert guidance.retrieval_query is not None
        assert "Retrieval target: exemption activity scope" in guidance.retrieval_query
        assert guidance.completion_instructions is not None
        assert (
            "Completion-relevant legal anchors and terms"
            in guidance.completion_instructions
        )

    def test_exemption_activity_guidance_uses_parent_contexts(self):
        hierarchy = QueryHierarchy(
            query_id="Q2.1",
            parent_ids=("Q1",),
            label_blockers=(
                LabelBlockerRule(
                    parent_query_id="Q1",
                    blocker_labels=("Custom cannabis label",),
                ),
            ),
        )
        request = RetrievalGuidanceRequest(
            query="If cannabis paraphernalia is exempted, which activities are exempted?",
            variable_name="dp_exempt_can_activity",
            metadata={
                "prepend_text": (
                    "This query refers to legal municipal ordinance that prohibits "
                    "drug paraphernalia used with controlled substances."
                ),
                "hierarchy": hierarchy_to_metadata(hierarchy),
            },
            parent_contexts=[
                ParentQueryContext(
                    query_id="Q1",
                    variable_name="dp_exemption",
                    question="Which paraphernalia exemptions exist?",
                    short_answer="Paraphernalia for consumption of cannabis, generally or medical use",
                ),
                ParentQueryContext(
                    query_id="Q2",
                    variable_name="dp_activity",
                    question="Which activities are prohibited?",
                    short_answer="Sales AND/OR Use",
                ),
            ],
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.shared_context is not None
        assert "Previously coded exemption answer" in guidance.shared_context
        assert (
            "Previously coded prohibited activities: Sales AND/OR Use."
            in guidance.shared_context
        )
        assert (
            "only in scope if the earlier exemption answer included"
            in guidance.shared_context
        )
        assert (
            "This subquestion is only in scope if the earlier exemption answer included: Custom cannabis label."
            in guidance.shared_context
        )

    def test_returns_guidance_for_existence_variable(self):
        request = RetrievalGuidanceRequest(
            query="Does a law exist?",
            variable_name="dp_law",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "existence_scope"
        assert "drug paraphernalia" in guidance.anchor_terms
        assert guidance.retrieval_query is not None
        assert "Question: Does a law exist?" in guidance.retrieval_query

    def test_retrieval_query_prefers_query_text_over_full_composed_prompt(self):
        request = RetrievalGuidanceRequest(
            query=(
                "Context: This query refers to legal municipal ordinance that prohibits drug "
                "paraphernalia-related activities.\n\n"
                "Question: On which date was the ordinance enacted?\n\n"
                "Coding instructions: Use the enacted date if explicitly stated."
            ),
            variable_name="dp_enacted",
            metadata={
                "query_text": "On which date was the ordinance enacted?",
                "prepend_text": (
                    "This query refers to legal municipal ordinance that prohibits "
                    "drug paraphernalia used with controlled substances."
                ),
                "coding_instructions": "Use the enacted date if explicitly stated.",
            },
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.retrieval_query is not None
        assert (
            "Legal context: This query refers to legal municipal ordinance that prohibits drug paraphernalia used with controlled substances."
            in guidance.retrieval_query
        )
        assert (
            "Question: On which date was the ordinance enacted?"
            in guidance.retrieval_query
        )
        assert "Context:" not in guidance.retrieval_query
        assert "Coding instructions:" not in guidance.retrieval_query

    def test_completion_instructions_preserve_context_for_vague_questions(self):
        request = RetrievalGuidanceRequest(
            query="On which date did the ordinance go into effect if known?",
            variable_name="dp_effective_dt",
            metadata={
                "query_text": "On which date did the ordinance go into effect if known?",
                "prepend_text": (
                    "This query refers to legal municipal ordinance that prohibits "
                    "drug paraphernalia used with controlled substances."
                ),
            },
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.retrieval_query is not None
        assert (
            "Legal context: This query refers to legal municipal ordinance that prohibits drug paraphernalia used with controlled substances."
            in guidance.retrieval_query
        )
        assert guidance.completion_instructions is not None
        assert (
            "Query context: This query refers to legal municipal ordinance that prohibits drug paraphernalia used with controlled substances."
            in guidance.completion_instructions
        )

    def test_reference_necessity_guidance_rejects_mere_citations(self):
        request = RetrievalGuidanceRequest(
            query=(
                "1. Does local law reference state or federal law, review of which is necessary "
                "to answer survey questions? 2. If yes, what is the citation of the relevant law?"
            ),
            variable_name="dp_state_fed_combined",
            metadata={
                "query_text": (
                    "1. Does local law reference state or federal law, review of which is necessary "
                    "to answer survey questions? 2. If yes, what is the citation of the relevant law?"
                ),
                "prepend_text": (
                    "This query refers to legal municipal ordinance that prohibits "
                    "drug paraphernalia used with controlled substances."
                ),
            },
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "reference_necessity"
        assert guidance.retrieval_instructions is not None
        assert "self-contained definition text" in guidance.retrieval_instructions
        assert (
            "Mere citations and authority references are low value"
            in guidance.retrieval_instructions
        )
        assert guidance.relevance_instructions is not None
        assert "A mere citation is not enough." in guidance.relevance_instructions
        assert (
            "the correct answer is no even if state or federal law is cited"
            in guidance.relevance_instructions
        )
        assert guidance.completion_instructions is not None
        assert (
            "Decision rule: Answer Yes only when the local ordinance expressly incorporates"
            in guidance.completion_instructions
        )
        assert (
            "Answer No when the local ordinance is self-contained"
            in guidance.completion_instructions
        )

    def test_existence_guidance_adds_controlled_substance_anchors(self):
        request = RetrievalGuidanceRequest(
            query="Does a local drug paraphernalia law exist?",
            variable_name="dp_law",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert "hypodermic" in guidance.anchor_terms
        assert "roach clip" in guidance.anchor_terms
        assert "injection device" in guidance.anchor_terms
        assert guidance.completion_instructions is not None
        assert "used with controlled substances" in guidance.completion_instructions

    def test_completion_instructions_do_not_reuse_relevance_filter_language(self):
        request = RetrievalGuidanceRequest(
            query="Which types of drug paraphernalia are covered?",
            variable_name="dp_type",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.relevance_instructions is not None
        assert "Prefer definitions" in guidance.relevance_instructions
        assert guidance.completion_instructions is not None
        assert "Prefer definitions" not in guidance.completion_instructions
        assert "used with controlled substances" in guidance.completion_instructions
        assert (
            "ground the answer in the legal definition"
            in guidance.completion_instructions
        )

    def test_returns_guidance_for_split_reference_variable(self):
        request = RetrievalGuidanceRequest(
            query="Does local law require review of outside law?",
            variable_name="dp_state_fed_reference",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "reference_necessity"
        assert "state law" in guidance.anchor_terms
        assert "A mere citation is not enough." in guidance.relevance_instructions

    def test_returns_guidance_for_split_reference_citation_variable(self):
        request = RetrievalGuidanceRequest(
            query="What outside-law citation is actually relevant?",
            variable_name="dp_state_fed_citation",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "reference_necessity"
        assert guidance.retrieval_instructions is not None
        assert "self-contained definition text" in guidance.retrieval_instructions

    def test_returns_guidance_for_split_current_through_variable(self):
        request = RetrievalGuidanceRequest(
            query="As of what date is the code current through?",
            variable_name="dp_valid_imp",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "date_current_through"
        assert guidance.retrieval_instructions is not None
        assert "current-through notices" in guidance.retrieval_instructions

    def test_returns_none_for_unmapped_variable(self):
        request = RetrievalGuidanceRequest(
            query="Unknown variable",
            variable_name="not_a_real_variable",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is None
