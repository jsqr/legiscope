"""Tests for COEP-specific retrieval guidance hooks."""

from coep.src.retrieval_guidance import get_drug_paraphernalia_retrieval_guidance
from legiscope.query_hierarchy import (
    LabelBlockerRule,
    QueryHierarchy,
    hierarchy_to_metadata,
)
from legiscope.retrieval_guidance import (
    ParentOptionEvidence,
    ParentQueryContext,
    RetrievalGuidanceRequest,
)


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
        assert "Penalty, see §" in guidance.retrieval_instructions
        assert "Exact legal labels matter" in guidance.relevance_instructions
        assert guidance.retrieval_query is not None
        assert "misdemeanor" in guidance.retrieval_query
        assert guidance.completion_instructions is not None
        assert "assign the exact labels" in guidance.completion_instructions
        assert "DO NOT answer Unlawful only" in guidance.completion_instructions
        assert "license revocation" in guidance.completion_instructions
        assert "SHOULD NOT be coded as Other" in guidance.completion_instructions

    def test_returns_guidance_for_activity_variable_with_operative_only_rules(self):
        request = RetrievalGuidanceRequest(
            query="Which activities are prohibited?",
            variable_name="dp_activity",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "prohibited_activity"
        assert guidance.relevance_instructions is not None
        assert (
            "Code only activities directly prohibited by the legal text"
            in guidance.relevance_instructions
        )
        assert "minors-only access restrictions" in guidance.relevance_instructions
        assert guidance.completion_instructions is not None
        assert (
            "only code items found directly in the legal text"
            in guidance.completion_instructions
        )
        assert (
            "Advertising or display language by itself does not prove Sales"
            in guidance.completion_instructions
        )

    def test_returns_guidance_for_exemption_variable_with_strict_other_rules(self):
        request = RetrievalGuidanceRequest(
            query="Which exemptions exist?",
            variable_name="dp_exemption",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "exemption_presence"
        assert guidance.completion_instructions is not None
        assert (
            "only code labels found directly in operative exemption text"
            in guidance.completion_instructions
        )
        assert "tobacco-only exceptions" in guidance.completion_instructions
        assert "SHOULD NOT be coded as Other" in guidance.completion_instructions
        assert (
            "syringe-exchange-facility text expressly authorizes"
            in guidance.completion_instructions
        )
        assert "SSP and DCE labels" in guidance.completion_instructions
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
        assert "same operative sentence/subsection" in guidance.completion_instructions
        assert "cannabis use or commerce" in guidance.completion_instructions

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

    def test_exemption_activity_guidance_uses_parent_option_evidence_and_inherited_anchors(
        self,
    ):
        hierarchy = QueryHierarchy(
            query_id="Q2.1",
            parent_ids=("Q1",),
            label_blockers=(
                LabelBlockerRule(
                    parent_query_id="Q1",
                    blocker_labels=(
                        "Paraphernalia for consumption of cannabis, generally or medical use",
                    ),
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
                    option_evidence=[
                        ParentOptionEvidence(
                            option="Paraphernalia for consumption of cannabis, generally or medical use",
                            selected=True,
                            confidence=0.88,
                            citations=["§ 12-4-10(C)(3)"],
                            supporting_passages=[
                                "Nothing in this section shall be construed to establish a criminal penalty for possession of paraphernalia for the exclusive purpose of cannabis use."
                            ],
                        )
                    ],
                )
            ],
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.shared_context is not None
        assert "Selected exemption option with evidence" in guidance.shared_context
        assert "Citation: § 12-4-10(C)(3)." in guidance.shared_context
        assert guidance.anchor_terms is not None
        assert "cannabis" in guidance.anchor_terms
        assert "marijuana" in guidance.anchor_terms

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
        assert guidance.completion_instructions is not None
        assert "sales-to-minors rules" in guidance.completion_instructions
        assert "SSP administration" in guidance.completion_instructions

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
        assert (
            "smallest specific state or federal citation"
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
        assert guidance.shared_context is not None
        assert (
            "Outside-law review is only relevant if it is necessary to answer one of these exact non-date benchmark questions"
            in guidance.shared_context
        )
        assert (
            "What types of drug paraphernalia are included in the law?"
            in guidance.shared_context
        )
        assert (
            "Which specific drug paraphernalia-related activities are prohibited?"
            in guidance.shared_context
        )
        assert guidance.completion_instructions is not None
        assert (
            "DO NOT answer Yes for bare citations" in guidance.completion_instructions
        )
        assert "controlled-substances definition" in guidance.completion_instructions
        assert "exemption or carve-out depend on whether conduct complies" in guidance.completion_instructions
        assert (
            "Treat the benchmark-question list in the query context as exhaustive"
            in guidance.completion_instructions
        )
        assert "in accordance with" in guidance.anchor_terms

    def test_exemption_guidance_rejects_unrelated_cannabis_business_noise(self):
        request = RetrievalGuidanceRequest(
            query="Are there any exemptions, such as for syringes, drug test strips, or other paraphernalia?",
            variable_name="dp_exemption",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.retrieval_instructions is not None
        assert "commercial-cannabis" in guidance.retrieval_instructions
        assert guidance.completion_instructions is not None
        assert (
            "Do NOT treat unrelated commercial-cannabis or marijuana-business provisions as cannabis exemptions"
            in guidance.completion_instructions
        )
        assert (
            "bona fide religious ritual or ceremony carve-outs to Other"
            in guidance.completion_instructions
        )

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
        assert guidance.shared_context is not None
        assert (
            "Does the ordinance specify any of the following types of violations or penalties for violating drug paraphernalia laws?"
            in guidance.shared_context
        )
        assert guidance.completion_instructions is not None
        assert "smallest specific statutory unit" in guidance.completion_instructions
        assert (
            "same sentence, subsection, or immediately adjacent chunk"
            in guidance.completion_instructions
        )

    def test_returns_guidance_for_ssp_reference_variable_with_ssp_question_scope(self):
        request = RetrievalGuidanceRequest(
            query="Does local law require review of outside SSP law?",
            variable_name="ssp_state_fed_reference",
            metadata={
                "prepend_text": (
                    "This query refers to legal municipal ordinance governing syringe "
                    "service programs (SSPs)."
                )
            },
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "ssp_reference_necessity"
        assert guidance.shared_context is not None
        assert (
            "Does the ordinance specifically prohibit all SSPs?"
            in guidance.shared_context
        )
        assert (
            "Does the ordinance explicitly authorize SSPs?"
            in guidance.shared_context
        )
        assert (
            "Does the ordinance require any of the following restrictions on SSPs?"
            in guidance.shared_context
        )
        assert guidance.completion_instructions is not None
        assert (
            "Treat the benchmark-question list in the query context as exhaustive"
            in guidance.completion_instructions
        )
        assert "state registration mentions" in guidance.completion_instructions

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

    def test_all_ssp_variables_are_mapped(self):
        variable_names = [
            "ssp_law",
            "ssp_enacted",
            "ssp_effective_dt",
            "ssp_collected",
            "ssp_current_imp",
            "ssp_state_fed_reference",
            "ssp_state_fed_citation",
            "ssp_prohibit",
            "ssp_permit",
            "ssp_restrict",
        ]

        for variable_name in variable_names:
            guidance = get_drug_paraphernalia_retrieval_guidance(
                RetrievalGuidanceRequest(
                    query=f"Guidance for {variable_name}",
                    variable_name=variable_name,
                    metadata={
                        "prepend_text": (
                            "This query refers to legal municipal ordinance governing "
                            "syringe service programs (SSPs)."
                        )
                    },
                )
            )

            assert guidance is not None, variable_name

    def test_returns_guidance_for_ssp_existence_variable(self):
        request = RetrievalGuidanceRequest(
            query="Does the jurisdiction have an SSP law?",
            variable_name="ssp_law",
            metadata={
                "prepend_text": (
                    "This query refers to legal municipal ordinance governing syringe "
                    "service programs (SSPs)."
                )
            },
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "ssp_scope"
        assert "syringe exchange" in guidance.anchor_terms
        assert "syringe exchange program" in guidance.anchor_terms
        assert "needle exchange program" in guidance.anchor_terms
        assert "clean needle" in guidance.anchor_terms
        assert "local public health emergency" in guidance.anchor_terms
        assert "sterile needle" in guidance.anchor_terms
        assert "hypodermic" not in guidance.anchor_terms
        assert guidance.no_context_fallback_short_answer == "No"
        assert guidance.enable_relevance_backfill is False
        assert guidance.retrieval_instructions is not None
        assert (
            "Distinguish true SSP programs from syringe buyback"
            in guidance.retrieval_instructions
        )
        assert (
            "authorization of clean needle or needle-and-syringe exchange projects"
            in guidance.retrieval_instructions
        )
        assert guidance.completion_instructions is not None
        assert "Do not count syringe buyback" in guidance.completion_instructions
        assert (
            "authorize clean needle or needle-and-syringe exchange projects"
            in guidance.completion_instructions
        )

    def test_returns_guidance_for_ssp_authorization_variable(self):
        request = RetrievalGuidanceRequest(
            query="Does the ordinance explicitly authorize SSPs?",
            variable_name="ssp_permit",
            metadata={
                "prepend_text": (
                    "This query refers to legal municipal ordinance governing syringe "
                    "service programs (SSPs)."
                )
            },
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "ssp_authorization"
        assert "local public health emergency" in guidance.anchor_terms
        assert "clean needle and syringe exchange project" in guidance.anchor_terms
        assert guidance.relevance_instructions is not None
        assert (
            "Treat authorization of clean needle or needle-and-syringe exchange projects"
            in guidance.relevance_instructions
        )
        assert guidance.completion_instructions is not None
        assert (
            "Treat authorization of clean needle or needle-and-syringe exchange projects as SSP authorization"
            in guidance.completion_instructions
        )
        assert "site approval, state registration" in guidance.completion_instructions

    def test_returns_guidance_for_ssp_current_imp_variable(self):
        request = RetrievalGuidanceRequest(
            query="Is the current-through date known or imputed?",
            variable_name="ssp_current_imp",
            metadata={
                "prepend_text": (
                    "This query refers to legal municipal ordinance governing syringe "
                    "service programs (SSPs)."
                )
            },
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "ssp_current_through_status"
        assert guidance.relevance_instructions is not None
        assert "official current-through notices" in guidance.relevance_instructions
        assert guidance.completion_instructions is not None
        assert (
            "ratified ordinance used as the fallback for ssp_collected"
            in guidance.completion_instructions
        )
        assert "date of data collection" in guidance.completion_instructions

    def test_returns_guidance_for_ssp_restriction_variable(self):
        request = RetrievalGuidanceRequest(
            query="Does the ordinance require any restrictions on SSPs?",
            variable_name="ssp_restrict",
            metadata={
                "prepend_text": (
                    "This query refers to legal municipal ordinance governing syringe "
                    "service programs (SSPs)."
                )
            },
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "ssp_restriction"
        assert guidance.retrieval_instructions is not None
        assert (
            "distance buffers from schools or parks" in guidance.retrieval_instructions
        )
        assert "syringe exchange facility" in guidance.anchor_terms
        assert "exchange only basis" in guidance.anchor_terms
        assert guidance.completion_instructions is not None
        assert (
            "Do not count outright bans as restrictions"
            in guidance.completion_instructions
        )
        assert (
            "notice, registration, or approval of a site or mobile unit"
            in guidance.completion_instructions
        )
        assert "No restrictions listed" in guidance.completion_instructions

    def test_returns_guidance_that_disables_backfill_for_exemption_presence(self):
        request = RetrievalGuidanceRequest(
            query="Are there any exemptions?",
            variable_name="dp_exemption",
            metadata={
                "prepend_text": (
                    "This query concerns a local municipal ordinance regulating "
                    "drug paraphernalia used with controlled substances."
                )
            },
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "exemption_presence"
        assert guidance.enable_relevance_backfill is False

    def test_returns_guidance_for_ssp_reference_citation_variable(self):
        request = RetrievalGuidanceRequest(
            query="If yes, what is the citation of the relevant law?",
            variable_name="ssp_state_fed_citation",
            metadata={
                "prepend_text": (
                    "This query refers to legal municipal ordinance governing syringe "
                    "service programs (SSPs)."
                )
            },
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is not None
        assert guidance.guidance_topic == "ssp_reference_necessity"
        assert guidance.retrieval_instructions is not None
        assert (
            "state or federal law must actually be read to determine the local SSP rule"
            in guidance.retrieval_instructions
        )
        assert guidance.completion_instructions is not None
        assert (
            "smallest specific statutory, regulatory, or administrative unit"
            in guidance.completion_instructions
        )
        assert "immediately adjacent chunk" in guidance.completion_instructions
        assert "harm reduction act" in guidance.anchor_terms

    def test_returns_none_for_unmapped_variable(self):
        request = RetrievalGuidanceRequest(
            query="Unknown variable",
            variable_name="not_a_real_variable",
        )

        guidance = get_drug_paraphernalia_retrieval_guidance(request)

        assert guidance is None
