"""COEP-specific retrieval guidance provider hooks."""

from legiscope.retrieval_guidance import RetrievalGuidance, RetrievalGuidanceRequest


_DEFAULT_QUERY_CONTEXT = (
    "This query concerns a local municipal ordinance regulating "
    "drug paraphernalia used with controlled substances."
)


_RETRIEVAL_INSTRUCTIONS_BY_FAMILY = {
    "existence_scope": (
        "Retrieve operative ordinance text that establishes whether the jurisdiction bans or regulates "
        "drug paraphernalia used with controlled substances beyond narrow business-only rules."
    ),
    "date_enactment": (
        "Retrieve ordinance metadata and amendment history that identify enactment, adoption, approval, "
        "or became-law dates for the target paraphernalia ordinance."
    ),
    "date_effective": (
        "Retrieve effective-date clauses and ordinance metadata stating when the target paraphernalia law "
        "takes effect or becomes effective."
    ),
    "date_current_through": (
        "Retrieve code-level recency statements, current-through notices, edition headers, or update "
        "metadata for the municipal code source."
    ),
    "reference_necessity": (
        "Retrieve definition, incorporation, adoption, and scope clauses showing whether state or federal "
        "law must actually be read to interpret the local paraphernalia ordinance. Also retrieve local "
        "self-contained definition text that shows the ordinance can be answered from local law alone. "
        "Mere citations and authority references are low value. "
    ),
    "definition_type": (
        "Retrieve definition sections and closely linked operative text that describe covered paraphernalia "
        "types, functions, or item lists used with controlled substances."
    ),
    "prohibited_activity": (
        "Retrieve operative prohibition text enumerating what acts are barred, especially sale, delivery, "
        "distribution, possession-with-intent, use, display, advertising, or manufacture of drug paraphernalia "
        "used with controlled substances."
    ),
    "penalty": (
        "Retrieve penalty sections, general penalty cross-references, and sanction language tied to the "
        "paraphernalia ordinance."
    ),
    "exemption_presence": (
        "Retrieve exception, exclusion, does-not-apply, authorized-use, and incorporated state-definition "
        "language that could create paraphernalia exemptions."
    ),
    "exemption_activity_scope": (
        "Retrieve exemption text and nearby operative activity language showing which acts remain allowed "
        "under the exemption."
    ),
}


_RETRIEVAL_OVERRIDE_BY_VARIABLE = {
    "dp_exempt_sygen_activity": (
        "Prioritize exemption language mentioning syringes, needles, hypodermic equipment, or injection equipment."
    ),
    "dp_exempt_sy_ssp_activity": (
        "Prioritize syringe-service, harm-reduction, supervised-use, or needle-exchange exemption language for syringes."
    ),
    "dp_exempt_can_activity": (
        "Prioritize cannabis, marijuana, marihuana, marijuana-accessory, or medical-marijuana exemption language."
    ),
    "dp_exempt_DCEgen_activity": (
        "Prioritize exemption language for test strips, testing equipment, checking equipment, or drug-checking tools."
    ),
    "dp_exempt_fentDCE_activity": (
        "Prioritize exemption language for fentanyl or fentanyl-analogue test strips and testing equipment."
    ),
    "dp_exempt_xyDCE_activity": (
        "Prioritize exemption language for xylazine test strips and testing equipment."
    ),
    "dp_exempt_DCE_ssp_activity": (
        "Prioritize SSP or harm-reduction exemption language for drug-checking or testing equipment."
    ),
    "dp_exempt_fentDCE_ssp_activity": (
        "Prioritize SSP or harm-reduction exemption language for fentanyl testing equipment."
    ),
    "dp_exempt_xyDCE_ssp_activity": (
        "Prioritize SSP or harm-reduction exemption language for xylazine testing equipment."
    ),
    "dp_exempt_SEgen_activity": (
        "Prioritize exemption language for pipes, smoking equipment, inhalation equipment, or marijuana accessories."
    ),
    "dp_exempt_SE_ssp_activity": (
        "Prioritize SSP or harm-reduction exemption language for pipes and smoking equipment."
    ),
    "dp_exempt_unspec_ssp_activity": (
        "Prioritize broad SSP or harm-reduction exemptions that do not name a specific paraphernalia subtype."
    ),
}


_LEGACY_EXEMPTION_DEPENDENCY_LABELS_BY_VARIABLE = {
    "dp_exempt_sygen_activity": ["Syringes, generally"],
    "dp_exempt_sy_ssp_activity": [
        "Syringes from syringe services, harm reduction programs, or supervised use sites"
    ],
    "dp_exempt_can_activity": [
        "Paraphernalia for consumption of cannabis, generally",
        "Paraphernalia for consumption of cannabis, generally or medical use",
    ],
    "dp_exempt_DCEgen_activity": ["Drug checking/testing equipment, generally"],
    "dp_exempt_fentDCE_activity": [
        "Drug checking/testing equipment for fentanyl or fentanyl analogues"
    ],
    "dp_exempt_xyDCE_activity": ["Drug checking/testing equipment for xylazine"],
    "dp_exempt_DCE_ssp_activity": [
        "Drug checking equipment, in the context of syringe services, harm reduction programs, or supervised use sites"
    ],
    "dp_exempt_fentDCE_ssp_activity": [
        "Fentanyl checking/testing equipment specifically, in the context of syringe services, harm reduction programs, or supervised use sites"
    ],
    "dp_exempt_xyDCE_ssp_activity": [
        "Xylazine checking/testing equipment specifically, in the context syringe services, harm reduction programs, or supervised use sites"
    ],
    "dp_exempt_SEgen_activity": ["Pipes/smoking equipment, generally"],
    "dp_exempt_SE_ssp_activity": [
        "Pipes/smoking equipment, in the context syringe services, harm reduction programs, or supervised use sites"
    ],
    "dp_exempt_unspec_ssp_activity": [
        "Unspecified or other paraphernalia, in the context of syringe services, harm reduction programs, or supervised use sites"
    ],
}


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        stripped = value.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        deduped.append(stripped)
    return deduped


def _expected_exemption_dependency_labels(
    request: RetrievalGuidanceRequest,
) -> list[str]:
    """Prefer CSV-configured blocker labels, falling back to legacy aliases."""
    hierarchy = request.metadata.get("hierarchy") or {}
    configured_labels: list[str] = []

    if isinstance(hierarchy, dict):
        for rule in hierarchy.get("label_blockers", []):
            if not isinstance(rule, dict):
                continue
            blocker_labels = rule.get("blocker_labels") or []
            if not isinstance(blocker_labels, list):
                continue
            for label in blocker_labels:
                if isinstance(label, str):
                    configured_labels.append(label)

    deduped_labels = _dedupe_preserving_order(configured_labels)
    if deduped_labels:
        return deduped_labels

    return list(
        _LEGACY_EXEMPTION_DEPENDENCY_LABELS_BY_VARIABLE.get(
            request.variable_name or "",
            [],
        )
    )


_FAMILY_BY_VARIABLE = {
    "dp_law": "existence_scope",
    "dp_enacted": "date_enactment",
    "dp_effective_dt": "date_effective",
    "dp_collected": "date_current_through",
    "dp_valid_imp": "date_current_through",
    "dp_collected_combined": "date_current_through",
    "dp_state_fed_reference": "reference_necessity",
    "dp_state_fed_citation": "reference_necessity",
    "dp_state_fed_combined": "reference_necessity",
    "dp_type": "definition_type",
    "dp_activity": "prohibited_activity",
    "dp_penalties": "penalty",
    "dp_exemption": "exemption_presence",
    "dp_exempt_sygen_activity": "exemption_activity_scope",
    "dp_exempt_sy_ssp_activity": "exemption_activity_scope",
    "dp_exempt_can_activity": "exemption_activity_scope",
    "dp_exempt_DCEgen_activity": "exemption_activity_scope",
    "dp_exempt_fentDCE_activity": "exemption_activity_scope",
    "dp_exempt_xyDCE_activity": "exemption_activity_scope",
    "dp_exempt_DCE_ssp_activity": "exemption_activity_scope",
    "dp_exempt_fentDCE_ssp_activity": "exemption_activity_scope",
    "dp_exempt_xyDCE_ssp_activity": "exemption_activity_scope",
    "dp_exempt_SEgen_activity": "exemption_activity_scope",
    "dp_exempt_SE_ssp_activity": "exemption_activity_scope",
    "dp_exempt_unspec_ssp_activity": "exemption_activity_scope",
}


_GUIDANCE_BY_FAMILY = {
    "existence_scope": RetrievalGuidance(
        guidance_topic="existence_scope",
        relevance_instructions=(
            "Prefer operative local ordinance text that actually prohibits or regulates drug "
            "paraphernalia used with controlled substances. Treat broad prohibitions that apply beyond a narrow "
            "business context as high value. Reject unrelated mentions, tobacco-only language, "
            "non-controlled-substance paraphernalia, and business-only display or head-shop rules "
            "unless the text clearly applies more broadly or creates a generally applicable ban."
        ),
        anchor_terms=[
            "drug paraphernalia",
            "paraphernalia",
            "bong",
            "pipe",
            "syringe",
            "needle",
            "hypodermic",
            "roach clip",
            "injection device",
            "ingestion device",
            "inhalation device",
        ],
    ),
    "date_enactment": RetrievalGuidance(
        guidance_topic="date_enactment",
        relevance_instructions=(
            "Prefer ordinance metadata and text that explicitly states when the law was enacted, "
            "passed, adopted, approved, or became law. Amendment dates are also high value when "
            "the coding logic asks for the most recent amendment in the target window. Reject "
            "substantive paraphernalia provisions that do not anchor the ordinance in time, and do "
            "not confuse effective dates or current-through dates with enactment dates unless the "
            "date label is genuinely unknown and the ordinance-end date appears to function as the "
            "only enactment marker."
        ),
        anchor_terms=[
            "enacted",
            "passed",
            "adopted",
            "approved",
            "became law",
            "ordinance no",
            "bill no",
            "amended",
        ],
    ),
    "date_effective": RetrievalGuidance(
        guidance_topic="date_effective",
        relevance_instructions=(
            "Prefer text explicitly labeled as effective date or language such as shall take effect, "
            "takes effect, become effective, or eff. Reject enactment dates, approval dates, and "
            "unlabeled dates. If the text does not clearly tie a date to effectiveness, it is low value."
        ),
        anchor_terms=[
            "effective",
            "eff",
            "take effect",
            "takes effect",
            "become effective",
            "effective date",
        ],
    ),
    "date_current_through": RetrievalGuidance(
        guidance_topic="date_current_through",
        relevance_instructions=(
            "Prefer source- or code-level recency statements describing when the municipal code was "
            "last updated, such as current through language, edition notices, update banners, or "
            "publisher metadata. Reject ordinary section amendment dates unless the coding logic says "
            "to fall back to the most recently ratified ordinance because no official current-through "
            "date is present."
        ),
        anchor_terms=[
            "current through",
            "current to",
            "current as of",
            "edition",
            "supplement",
            "updated",
            "ordinances passed through",
        ],
    ),
    "reference_necessity": RetrievalGuidance(
        guidance_topic="reference_necessity",
        relevance_instructions=(
            "Prefer text showing whether a cited state or federal law must actually be consulted to "
            "answer the survey question. A mere citation is not enough. High-value text either says "
            "the local ordinance adopts or incorporates an external definition, or makes clear that the "
            "local text is self-contained. Count this as necessary only when the outside law must be "
            "reviewed to determine the local ordinance's meaning, scope, or elements. If the local text "
            "can be answered from the ordinance itself, the correct answer is no even if state or federal "
            "law is cited, mirrored, used as background authority, or referenced in a penalty or "
            "enforcement cross-reference. "
        ),
        anchor_terms=[
            "incorporate",
            "adopt",
            "defined in",
            "pursuant to",
            "as provided in",
            "as defined in",
            "state law",
            "federal law",
            "35 p.s.",
            "u.s.c.",
        ],
    ),
    "definition_type": RetrievalGuidance(
        guidance_topic="definition_type",
        relevance_instructions=(
            "Prefer definitions and closely linked operative text that describe what kinds of "
            "paraphernalia used with controlled substances are covered. High-value text includes verbs like inject, inhale, test, "
            "analyze, ingest, prepare, conceal, or pack, and item lists such as pipes, syringes, "
            "kits, roach clips, hypodermic devices, or containers. Reject date, penalty, and exemption text unless it directly adds to "
            "the operative definition."
        ),
        anchor_terms=[
            "inject",
            "inhale",
            "test",
            "analyze",
            "ingest",
            "pipe",
            "syringe",
            "hypodermic",
            "roach clip",
            "injection device",
            "kit",
        ],
    ),
    "prohibited_activity": RetrievalGuidance(
        guidance_topic="prohibited_activity",
        relevance_instructions=(
            "Prefer text that explicitly enumerates prohibited acts. High-value passages state verbs "
            "such as sell, offer for sale, deliver, distribute, transfer, possess with intent, use, "
            "display, advertise, or manufacture. Apply the coding logic carefully: simple possession "
            "is different from possession with intent to sell or deliver; use should only count if the "
            "text explicitly prohibits use; and business-context activity may still be narrower than a "
            "general prohibition. Focus on acts involving drug paraphernalia used with controlled substances. Reject nearby definition, date, or penalty text unless it directly "
            "states the operative activity."
        ),
        anchor_terms=[
            "sell",
            "offer for sale",
            "deliver",
            "distribute",
            "transfer",
            "possess with intent",
            "use",
            "manufacture",
            "display",
            "advertise",
        ],
    ),
    "penalty": RetrievalGuidance(
        guidance_topic="penalty",
        relevance_instructions=(
            "Prefer text that states penalties, sanctions, violation classes, fines, "
            "imprisonment, general penalty provisions, or explicit cross-references to penalty "
            "sections. Exact legal labels matter here: misdemeanor, infraction, felony, forfeiture, "
            "seizure, civil fine, criminal fine, and unlawful only should be grounded in the actual "
            "text. Reject definitions and operative restrictions that do not specify consequences."
        ),
        anchor_terms=[
            "penalty",
            "fine",
            "misdemeanor",
            "infraction",
            "felony",
            "civil penalty",
            "criminal fine",
            "unlawful",
            "imprisonment",
            "jail",
            "forfeiture",
            "seizure",
        ],
    ),
    "exemption_presence": RetrievalGuidance(
        guidance_topic="exemption_presence",
        relevance_instructions=(
            "Prefer text that contains exceptions, carve-outs, exclusions, does-not-apply language, "
            "or references to authorized uses. Search both the local ordinance and any specifically "
            "referenced state-law definition sections when the local text incorporates them. High-value "
            "passages may mention cannabis, marijuana, syringe services, harm reduction programs, "
            "medical use, professionals acting in the course of business, or public officials. Reject "
            "ordinary prohibitions, zoning restrictions, and business permissions unless they actually "
            "function as a paraphernalia exemption under the coding logic."
        ),
        anchor_terms=[
            "except",
            "does not apply",
            "does not include",
            "authorized",
            "exemption",
            "exception",
            "hypodermic",
            "injection device",
            "roach clip",
            "medical marijuana",
            "cannabis",
            "marihuana",
            "syringe services",
            "harm reduction",
        ],
    ),
    "exemption_activity_scope": RetrievalGuidance(
        guidance_topic="exemption_activity_scope",
        relevance_instructions=(
            "Prefer exemption text that clarifies which activities remain allowed. High-value passages "
            "either explicitly enumerate exempted activities such as possession, use, distribution, or "
            "sales, or tie the exemption to a definition that implicitly spans all prohibited activities. "
            "Read the exemption together with the operative activity language when needed. Reject text "
            "that only proves an exemption exists but does not help determine its activity scope."
        ),
        anchor_terms=[
            "this section shall not apply",
            "nothing in this section shall prohibit",
            "distribution",
            "possession",
            "sell",
            "use",
            "give away",
            "manufacture",
        ],
    ),
}


_VARIABLE_OVERRIDES = {
    "dp_penalties": RetrievalGuidance(
        relevance_instructions=(
            "Focus on operative sanction text. Do not elevate business-license revocation, permitting consequences, or other collateral remedies into Other when the coding rules exclude them."
        ),
        completion_instructions=(
            "For dp_penalties, include Other only when the ordinance imposes a genuine residual penalty that does not fit another named option. If the text supports only Unlawful and Civil Fine, do not add Other as a hedge."
        ),
        anchor_terms=["civil penalty", "license revocation", "sanction"],
    ),
    "dp_exemption": RetrievalGuidance(
        relevance_instructions=(
            "Focus on true paraphernalia carve-outs. Reject tobacco-only exceptions, zoning permissions, retail-business permissions, and other non-paraphernalia exceptions unless they clearly function as coded exemptions under the survey rules."
        ),
        completion_instructions=(
            "For dp_exemption, include Other only for a real paraphernalia exemption that does not fit any listed label. Do not use Other to capture tobacco exceptions, business permissions, or other carve-outs that the coding instructions exclude."
        ),
        anchor_terms=["exception", "does not apply", "authorized", "medical marijuana"],
    ),
    "dp_state_fed_combined": RetrievalGuidance(
        relevance_instructions=(
            "For the combined survey question, answer yes only if the ordinance expressly incorporates, "
            "adopts, or depends on the external law such that reviewing that outside law is required. "
            "Answer no when the local ordinance remains self-contained and the outside law is only cited "
            "or mentioned."
        ),
        anchor_terms=[
            "incorporated by reference",
            "adopts",
            "incorporates",
            "defined by reference",
        ],
    ),
    "dp_exempt_sygen_activity": RetrievalGuidance(
        relevance_instructions=(
            "Focus on exemption language involving syringes or injection equipment generally. High-value "
            "text may say syringe, needle, hypodermic, or injection equipment and should be evaluated for "
            "which activities are expressly exempted."
        ),
        anchor_terms=["syringe", "needle", "hypodermic", "injection equipment"],
    ),
    "dp_exempt_sy_ssp_activity": RetrievalGuidance(
        relevance_instructions=(
            "Focus on syringe exemptions tied to syringe services programs, harm reduction programs, or "
            "supervised use sites. High-value text often names governmental or authorized program actors."
        ),
        anchor_terms=[
            "syringe services",
            "harm reduction",
            "supervised use",
            "needle exchange",
            "governmental agency",
        ],
    ),
    "dp_exempt_can_activity": RetrievalGuidance(
        relevance_instructions=(
            "Focus on cannabis- or marijuana-related exemptions. High-value text may use cannabis, "
            "marijuana, marihuana, marijuana accessories, or medical marijuana language and may place the "
            "exemption either in the definition or directly alongside prohibited activities."
        ),
        anchor_terms=[
            "cannabis",
            "marijuana",
            "marihuana",
            "marijuana accessories",
            "medical marijuana",
        ],
    ),
    "dp_exempt_DCEgen_activity": RetrievalGuidance(
        relevance_instructions=(
            "Focus on exemptions for drug checking or testing equipment generally. High-value text may use "
            "testing equipment, test strip, checking equipment, or drug checking language."
        ),
        anchor_terms=[
            "test strip",
            "testing equipment",
            "drug checking",
            "drug testing",
        ],
    ),
    "dp_exempt_fentDCE_activity": RetrievalGuidance(
        relevance_instructions=(
            "Focus on exemptions for fentanyl, fentanyl analogue, or synthetic opioid testing equipment."
        ),
        anchor_terms=[
            "fentanyl",
            "fentanyl analogue",
            "synthetic opioid",
            "test strip",
        ],
    ),
    "dp_exempt_xyDCE_activity": RetrievalGuidance(
        relevance_instructions=(
            "Focus on exemptions for xylazine testing equipment. High-value text should connect xylazine "
            "to testing, strips, or checking equipment."
        ),
        anchor_terms=["xylazine", "test strip", "testing equipment"],
    ),
    "dp_exempt_DCE_ssp_activity": RetrievalGuidance(
        relevance_instructions=(
            "Focus on drug checking or testing equipment exemptions within syringe services, harm reduction, "
            "or supervised use contexts."
        ),
        anchor_terms=[
            "drug checking",
            "drug testing",
            "syringe services",
            "harm reduction",
            "supervised use",
        ],
    ),
    "dp_exempt_fentDCE_ssp_activity": RetrievalGuidance(
        relevance_instructions=(
            "Focus on fentanyl testing equipment exemptions specifically linked to syringe services, harm "
            "reduction programs, or supervised use sites."
        ),
        anchor_terms=[
            "fentanyl",
            "test strip",
            "syringe services",
            "harm reduction",
            "supervised use",
        ],
    ),
    "dp_exempt_xyDCE_ssp_activity": RetrievalGuidance(
        relevance_instructions=(
            "Focus on xylazine testing equipment exemptions specifically linked to syringe services, harm "
            "reduction programs, or supervised use sites."
        ),
        anchor_terms=[
            "xylazine",
            "test strip",
            "syringe services",
            "harm reduction",
            "supervised use",
        ],
    ),
    "dp_exempt_SEgen_activity": RetrievalGuidance(
        relevance_instructions=(
            "Focus on smoking equipment exemptions generally. High-value text may use pipe, smoking "
            "equipment, inhalation equipment, or marijuana accessories language."
        ),
        anchor_terms=[
            "pipe",
            "smoking equipment",
            "inhalation equipment",
            "marijuana accessories",
        ],
    ),
    "dp_exempt_SE_ssp_activity": RetrievalGuidance(
        relevance_instructions=(
            "Focus on smoking-equipment exemptions linked to syringe services, harm reduction programs, or "
            "supervised use sites."
        ),
        anchor_terms=[
            "pipe",
            "smoking equipment",
            "syringe services",
            "harm reduction",
            "supervised use",
        ],
    ),
    "dp_exempt_unspec_ssp_activity": RetrievalGuidance(
        relevance_instructions=(
            "Focus on broad or unspecified paraphernalia exemptions linked to syringe services, harm "
            "reduction programs, or supervised use sites, especially when the exemption does not name a "
            "specific paraphernalia type."
        ),
        anchor_terms=[
            "paraphernalia",
            "syringe services",
            "harm reduction",
            "supervised use",
            "government approved",
        ],
    ),
}


_COMPLETION_RULES_BY_FAMILY = {
    "existence_scope": (
        "Interpret the question as asking whether the ordinance prohibits or regulates drug paraphernalia "
        "used with controlled substances, not tobacco-only or other non-controlled-substance paraphernalia."
    ),
    "definition_type": (
        "Describe only paraphernalia types tied to controlled-substance use and ground the answer in the legal "
        "definition or closely linked operative text."
    ),
    "prohibited_activity": (
        "List only activities that the ordinance expressly prohibits for drug paraphernalia used with controlled substances."
    ),
    "penalty": (
        "Code only penalties or sanction labels that the ordinance actually imposes. Use Other only for a genuine residual penalty that is clearly imposed and does not fit any named option. Do not use Other as a hedge, and do not treat excluded collateral remedies as Other unless the coding rules expressly require them."
    ),
    "exemption_presence": (
        "Identify only exemption language that actually creates a paraphernalia carve-out under the coding rules. Do not use Other for tobacco-only exceptions, zoning permissions, business permissions, or other carve-outs that the coding rules exclude from paraphernalia exemptions."
    ),
    "exemption_activity_scope": (
        "Explain which activities remain allowed under the exemption, using the exemption text together with the operative activity language when necessary."
    ),
    "reference_necessity": (
        "Decision rule: Answer Yes only when the local ordinance expressly incorporates, adopts, or "
        "depends on a state or federal statute or definition such that reviewing that outside law is "
        "required to determine the local ordinance's meaning, scope, or elements. Answer No when the "
        "local ordinance is self-contained, or when outside law is only cited as background authority, "
        "parallel wording, a penalty cross-reference, an enforcement reference, or some other incidental "
        "mention. Do not treat a bare citation as making outside-law review necessary. If the answer is "
        "Yes, identify the specific state or federal citation that must be reviewed. If the answer is No, "
        "say that no outside-law review is necessary and do not elevate incidental citations as the "
        "relevant law."
    ),
}


def _merge_guidance(
    base: RetrievalGuidance,
    override: RetrievalGuidance | None,
) -> RetrievalGuidance:
    """Merge family-level guidance with optional variable-specific overrides."""
    if override is None:
        return base

    retrieval_parts = []
    if base.retrieval_instructions:
        retrieval_parts.append(base.retrieval_instructions.strip())
    if override and override.retrieval_instructions:
        retrieval_parts.append(override.retrieval_instructions.strip())

    prompt_parts = []
    if base.relevance_instructions:
        prompt_parts.append(base.relevance_instructions.strip())
    if override and override.relevance_instructions:
        prompt_parts.append(override.relevance_instructions.strip())

    completion_parts = []
    if base.completion_instructions:
        completion_parts.append(base.completion_instructions.strip())
    if override and override.completion_instructions:
        completion_parts.append(override.completion_instructions.strip())

    override_terms = override.anchor_terms if override else []
    merged_keywords = list(dict.fromkeys(base.anchor_terms + override_terms))

    return RetrievalGuidance(
        guidance_topic=(override.guidance_topic if override else None)
        or base.guidance_topic,
        retrieval_instructions=" ".join(retrieval_parts) if retrieval_parts else None,
        relevance_instructions=" ".join(prompt_parts) if prompt_parts else None,
        anchor_terms=merged_keywords,
        completion_instructions=(
            " ".join(completion_parts) if completion_parts else None
        ),
    )


def _build_query_context(request: RetrievalGuidanceRequest) -> str:
    """Build concise legal-scope context for ambiguous subquestions."""
    prepend_text = (request.metadata.get("prepend_text") or "").strip()
    context_parts = []
    if prepend_text:
        context_parts.append(prepend_text.rstrip(". ") + ".")
    else:
        context_parts.append(_DEFAULT_QUERY_CONTEXT)

    prior_answers = request.metadata.get("prior_answers") or {}
    if (
        _FAMILY_BY_VARIABLE.get(request.variable_name or "")
        == "exemption_activity_scope"
    ):
        context_by_variable = {
            context.variable_name: context
            for context in request.parent_contexts
            if context.variable_name
        }
        exemption_context = context_by_variable.get("dp_exemption")
        activity_context = context_by_variable.get("dp_activity")
        exemption_short_answer = (
            exemption_context.short_answer if exemption_context is not None else None
        )
        activity_short_answer = (
            activity_context.short_answer if activity_context is not None else None
        )

        if exemption_short_answer is None and isinstance(prior_answers, dict):
            exemption_answer = prior_answers.get("dp_exemption", {})
            if isinstance(exemption_answer, dict):
                exemption_short_answer = exemption_answer.get("short_answer")

        if activity_short_answer is None and isinstance(prior_answers, dict):
            activity_answer = prior_answers.get("dp_activity", {})
            if isinstance(activity_answer, dict):
                activity_short_answer = activity_answer.get("short_answer")

        if exemption_short_answer:
            context_parts.append(
                f"Previously coded exemption answer: {exemption_short_answer}."
            )
        if activity_short_answer:
            context_parts.append(
                f"Previously coded prohibited activities: {activity_short_answer}."
            )

        expected_labels = _expected_exemption_dependency_labels(request)
        if expected_labels:
            context_parts.append(
                "This subquestion is only in scope if the earlier exemption answer included: "
                + " OR ".join(expected_labels)
                + "."
            )

    return " ".join(context_parts)


def _build_retrieval_instructions(
    variable_name: str | None,
    family: str,
) -> str:
    """Build retrieval-stage instructions that are concise and search-oriented."""
    parts = [_RETRIEVAL_INSTRUCTIONS_BY_FAMILY[family]]

    override = _RETRIEVAL_OVERRIDE_BY_VARIABLE.get(variable_name or "")
    if override:
        parts.append(override)

    return " ".join(parts)


def _build_retrieval_query(
    request: RetrievalGuidanceRequest,
    guidance: RetrievalGuidance,
) -> str:
    """Build a retrieval-optimized query string for semantic search."""
    query_text = (request.metadata.get("query_text") or "").strip()
    if not query_text:
        query_text = request.query.strip()

    parts = []

    if guidance.shared_context:
        parts.append(f"Legal context: {guidance.shared_context}")

    parts.append(f"Question: {query_text}")

    if guidance.guidance_topic:
        parts.append(
            "Retrieval target: "
            + guidance.guidance_topic.replace("_", " ")
            + " in operative legal text"
        )

    if guidance.retrieval_instructions:
        parts.append(f"Retrieval focus: {guidance.retrieval_instructions.strip()}")

    if guidance.anchor_terms:
        parts.append("High-value legal terms: " + ", ".join(guidance.anchor_terms))

    return "\n\n".join(parts)


def _build_completion_instructions(guidance: RetrievalGuidance) -> str | None:
    """Build completion-time variable-specific guidance."""
    parts = []

    if guidance.shared_context:
        parts.append(f"Query context: {guidance.shared_context}")

    if guidance.guidance_topic:
        parts.append(
            "Variable family: " + guidance.guidance_topic.replace("_", " ") + "."
        )

    family_rule = _COMPLETION_RULES_BY_FAMILY.get(guidance.guidance_topic or "")
    if family_rule:
        parts.append(family_rule)

    if guidance.anchor_terms:
        parts.append(
            "Completion-relevant legal anchors and terms: "
            + ", ".join(guidance.anchor_terms)
            + "."
        )

    if guidance.completion_instructions:
        parts.append(guidance.completion_instructions.strip())

    return " ".join(parts) if parts else None


def get_drug_paraphernalia_retrieval_guidance(
    request: RetrievalGuidanceRequest,
) -> RetrievalGuidance | None:
    """Return query-family guidance for COEP drug paraphernalia variables."""
    if not request.variable_name:
        return None

    family = _FAMILY_BY_VARIABLE.get(request.variable_name)
    if family is None:
        return None

    guidance = _merge_guidance(
        _GUIDANCE_BY_FAMILY[family],
        _VARIABLE_OVERRIDES.get(request.variable_name),
    )
    query_context = _build_query_context(request)
    guidance = RetrievalGuidance(
        guidance_topic=guidance.guidance_topic,
        shared_context=query_context,
        retrieval_instructions=_build_retrieval_instructions(
            request.variable_name,
            family,
        ),
        relevance_instructions=guidance.relevance_instructions,
        anchor_terms=guidance.anchor_terms,
    )

    return RetrievalGuidance(
        guidance_topic=guidance.guidance_topic,
        shared_context=guidance.shared_context,
        retrieval_query=_build_retrieval_query(request, guidance),
        retrieval_instructions=guidance.retrieval_instructions,
        relevance_instructions=guidance.relevance_instructions,
        anchor_terms=guidance.anchor_terms,
        completion_instructions=_build_completion_instructions(guidance),
    )
