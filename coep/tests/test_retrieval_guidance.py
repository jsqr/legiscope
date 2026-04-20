"""Tests for COEP-specific retrieval guidance hooks."""

from coep.src.retrieval_guidance import get_drug_paraphernalia_retrieval_guidance
from legiscope.retrieval_guidance import RetrievalGuidanceRequest


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
        assert "drug paraphernalia-related activities" in guidance.shared_context
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
        assert "Relevant legal anchors and terms" in guidance.completion_instructions

    def test_exemption_activity_guidance_uses_prior_answer_context(self):
        request = RetrievalGuidanceRequest(
            query="If cannabis paraphernalia is exempted, which activities are exempted?",
            variable_name="dp_exempt_can_activity",
            metadata={
                "prepend_text": (
                    "This query refers to legal municipal ordinance that prohibits "
                    "drug paraphernalia-related activities."
                ),
                "prior_answers": {
                    "dp_exemption": {
                        "short_answer": "Paraphernalia for consumption of cannabis, generally or medical use"
                    },
                    "dp_activity": {"short_answer": "Sales AND/OR Use"},
                },
            },
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
                    "drug paraphernalia-related activities."
                ),
                "coding_instructions": "Use the enacted date if explicitly stated.",
            },
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.retrieval_query is not None
        assert (
            "Legal context: This query refers to legal municipal ordinance that prohibits drug paraphernalia-related activities."
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
                    "drug paraphernalia-related activities."
                ),
            },
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.retrieval_query is not None
        assert (
            "Legal context: This query refers to legal municipal ordinance that prohibits drug paraphernalia-related activities."
            in guidance.retrieval_query
        )
        assert guidance.completion_instructions is not None
        assert (
            "Query context: This query refers to legal municipal ordinance that prohibits drug paraphernalia-related activities."
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
                    "drug paraphernalia-related activities."
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
        assert (
            "do not elevate incidental citations as the relevant law"
            in guidance.completion_instructions
        )

    def test_returns_none_for_unmapped_variable(self):
        request = RetrievalGuidanceRequest(
            query="Unknown variable",
            variable_name="not_a_real_variable",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is None
