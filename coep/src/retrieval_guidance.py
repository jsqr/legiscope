"""COEP-specific retrieval guidance provider hooks."""

import re

from legiscope.retrieval_guidance import RetrievalGuidance, RetrievalGuidanceRequest


_DEFAULT_QUERY_CONTEXT = (
    "This query concerns a local municipal ordinance regulating "
    "drug paraphernalia used with controlled substances."
)


_REFERENCE_SCOPE_QUESTIONS_BY_FAMILY = {
    "reference_necessity": [
        "Does the jurisdiction have an ordinance that prohibits drug paraphernalia-related activities?",
        "What types of drug paraphernalia are included in the law?",
        "Which specific drug paraphernalia-related activities are prohibited?",
        "Does the ordinance specify any of the following types of violations or penalties for violating drug paraphernalia laws?",
        "Are there any exemptions, such as for syringes, drug test strips, or other paraphernalia?",
        "If an exemption exists, for which prohibited activities does it apply?",
    ],
    "ssp_reference_necessity": [
        "Does the jurisdiction have a law that authorizes, prohibits, or limits syringe service programs (SSPs)?",
        "Does the ordinance specifically prohibit all SSPs?",
        "Does the ordinance explicitly authorize SSPs?",
        "Does the ordinance require any of the following restrictions on SSPs?",
    ],
}


_RETRIEVAL_INSTRUCTIONS_BY_FAMILY = {
    "existence_scope": (
        "Retrieve operative ordinance text that establishes whether the jurisdiction bans or regulates "
        "drug paraphernalia used with controlled substances beyond narrow business-only rules. We are "
        "looking for a law that governs drug paraphernalia legality itself, not a small business-only "
        "mention. Do not treat sales-to-minors rules, head-shop zoning, retail-display restrictions, "
        "business-license conditions, location-only business restrictions, or SSP-only program text as a "
        "general drug-paraphernalia ordinance unless the same text also creates an ordinance-wide "
        "paraphernalia prohibition or regulation that applies beyond businesses."
    ),
    "ssp_scope": (
        "Retrieve operative municipal ordinance text establishing whether the jurisdiction authorizes, "
        "prohibits, or limits syringe service programs, syringe exchanges, needle exchanges, harm-reduction "
        "syringe distribution, or closely synonymous SSP operations. Treat emergency-conditioned authorization "
        "of clean needle or needle-and-syringe exchange projects as in-scope SSP text when the ordinance empowers "
        "a local official to declare an outbreak or public health emergency that activates the program. Actively "
        "look for clauses like `Declaration of Local Public Health Emergency` and sentences stating that the mayor "
        "or another local official is empowered to authorize clean needle and syringe exchange projects. Distinguish "
        "true SSP programs from syringe buyback or disposal-only programs."
    ),
    "date_enactment": (
        "Retrieve ordinance metadata and amendment history that identify enactment, adoption, approval, "
        "or became-law dates for the target paraphernalia ordinance."
    ),
    "ssp_date_enactment": (
        "Retrieve ordinance metadata and amendment history that identify enactment, adoption, approval, "
        "or passed dates for the target SSP ordinance."
    ),
    "date_effective": (
        "Retrieve effective-date clauses and ordinance metadata stating when the target paraphernalia law "
        "takes effect or becomes effective."
    ),
    "ssp_date_effective": (
        "Retrieve effective-date clauses and ordinance metadata stating when the target SSP ordinance takes "
        "effect or becomes effective."
    ),
    "date_current_through": (
        "Retrieve code-level recency statements, current-through notices, edition headers, or update "
        "metadata for the municipal code source."
    ),
    "ssp_date_current_through": (
        "Retrieve code-level recency statements, current-through notices, edition headers, update metadata, "
        "or ordinance-ratification dates relevant to when the SSP ordinance source was current."
    ),
    "ssp_current_through_status": (
        "Retrieve official current-through notices, partial current-through statements, edition metadata, "
        "or if none exist the most recent ratified ordinance date used as the fallback source for the SSP "
        "current-through answer."
    ),
    "reference_necessity": (
        "Retrieve definition, incorporation, adoption, and scope clauses showing whether state or federal "
        "law must actually be read to interpret the local paraphernalia ordinance. Also retrieve local "
        "self-contained definition text that shows the ordinance can be answered from local law alone. "
        "Mere citations and authority references are low value. "
    ),
    "ssp_reference_necessity": (
        "Retrieve incorporation, adoption, authorization, prohibition, and restriction clauses showing whether "
        "state or federal law must actually be read to determine the local SSP rule. Also retrieve local "
        "self-contained SSP text showing the answer can be resolved from local law alone. Mere citations and "
        "authority references are low value."
    ),
    "ssp_prohibition": (
        "Retrieve operative ordinance text stating whether SSPs are prohibited, banned, unlawful, not "
        "permitted, or barred, including any carve-out for programs authorized by a state health department "
        "or other state entity."
    ),
    "ssp_authorization": (
        "Retrieve operative ordinance text stating whether SSPs are authorized, permitted, approved, allowed, "
        "or established, including any condition limiting authorization to a declared public health emergency "
        "or disease outbreak. Treat provisions authorizing clean needle or needle-and-syringe exchange projects "
        "during a declared emergency as affirmative SSP authorization."
    ),
    "ssp_restriction": (
        "Retrieve operative SSP restriction text covering caps on program sites, distance buffers from schools "
        "or parks, visit frequency limits, syringe-quantity limits, mobile-site limits, permit or license "
        "requirements, or similar operational restrictions."
    ),
    "definition_type": (
        "Retrieve definition sections and closely linked operative text that describe covered paraphernalia "
        "types, functions, or item lists used with controlled substances."
    ),
    "prohibited_activity": (
        "Retrieve operative prohibition text enumerating what acts are barred, especially sale, delivery, "
        "distribution, possession-with-intent, use, display, advertising, or manufacture of drug paraphernalia "
        "used with controlled substances. Prefer directly operative prohibitions, not zoning, land-use, licensing, "
        "head-shop, retail-display, or minors-only access restrictions unless that same text expressly creates the "
        "general paraphernalia prohibition being coded."
    ),
    "penalty": (
        "Retrieve penalty sections, general penalty cross-references, and sanction language tied to the "
        "paraphernalia ordinance. Follow explicit cross-references such as `Penalty, see § ...` and treat the cited "
        "penalty section as controlling sanction text. Retrieve the actual cited penalty section when possible. "
        "Do not translate an offense class or generic criminality into a benchmark label unless the text itself states "
        "that exact sanction or the cross-referenced section states it."
    ),
    "exemption_presence": (
        "Retrieve exception, exclusion, does-not-apply, authorized-use, and incorporated state-definition "
        "language that could create paraphernalia exemptions. Also retrieve syringe-service, syringe-exchange, harm-"
        "reduction, supervised-use, and drug-checking/test-strip provisions when they can function as direct "
        "paraphernalia carve-outs. Do not treat unrelated marijuana-business, commercial-cannabis, zoning, or "
        "employment provisions as exemption evidence merely because they mention cannabis or syringes. Prefer the "
        "smallest operative exemption sentence or subsection over broad surrounding business-regulation context."
    ),
    "exemption_activity_scope": (
        "Retrieve exemption text and nearby operative activity language showing which acts remain allowed "
        "under the exemption. Prioritize the same sentence, clause, or subsection as the exemption itself so the "
        "model does not over-expand one exemption into every activity label."
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


_RELEVANCE_FILTER_ENABLED_VARIABLES = {
    "dp_law",
    "ssp_law",
    "dp_activity",
    "dp_penalties",
    "dp_exemption",
    "dp_state_fed_reference",
    "ssp_state_fed_reference",
    "ssp_restrict",
}


def _coep_relevance_filter_enabled(variable_name: str | None) -> bool:
    """Return whether COEP guidance should allow relevance filtering for this variable."""
    if not variable_name:
        return False
    if variable_name in _RELEVANCE_FILTER_ENABLED_VARIABLES:
        return True
    return variable_name.startswith("dp_exempt_")


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


def _format_reference_scope_questions(family: str | None) -> str | None:
    questions = _REFERENCE_SCOPE_QUESTIONS_BY_FAMILY.get(family or "", [])
    if not questions:
        return None

    formatted_questions = " | ".join(f'"{question}"' for question in questions)
    return (
        "Outside-law review is only relevant if it is necessary to answer one of these exact non-date benchmark questions: "
        + formatted_questions
        + "."
    )


def _normalize_label_text(value: str) -> str:
    """Normalize benchmark option labels for conservative matching."""
    normalized = value.strip().lower()
    normalized = normalized.replace("and/or", " and or ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[\[\](){}]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


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


def _matching_selected_option_evidence(
    request: RetrievalGuidanceRequest,
) -> list[tuple[str, str | None, str | None, list[str]]]:
    """Return selected exemption-option evidence relevant to the current child query."""
    context_by_variable = {
        context.variable_name: context
        for context in request.parent_contexts
        if context.variable_name
    }
    exemption_context = context_by_variable.get("dp_exemption")
    if exemption_context is None or not exemption_context.option_evidence:
        return []

    expected_labels = _expected_exemption_dependency_labels(request)
    expected_keys = {_normalize_label_text(label) for label in expected_labels}
    matched: list[tuple[str, str | None, str | None, list[str]]] = []
    fallback: list[tuple[str, str | None, str | None, list[str]]] = []
    for item in exemption_context.option_evidence:
        if not item.selected:
            continue
        citation = item.citations[0] if item.citations else None
        passage = item.supporting_passages[0] if item.supporting_passages else None
        anchor_terms = item.anchor_terms if item.anchor_terms else []
        record = (item.option, citation, passage, anchor_terms)
        fallback.append(record)
        if expected_keys and _normalize_label_text(item.option) in expected_keys:
            matched.append(record)

    if matched:
        return matched
    return fallback


def _truncate_context_passage(text: str | None, limit: int = 220) -> str | None:
    """Trim carried-forward evidence passages so child prompts stay compact."""
    if text is None:
        return None
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3].rstrip() + "..."


_FAMILY_BY_VARIABLE = {
    "dp_law": "existence_scope",
    "ssp_law": "ssp_scope",
    "dp_enacted": "date_enactment",
    "ssp_enacted": "ssp_date_enactment",
    "dp_effective_dt": "date_effective",
    "ssp_effective_dt": "ssp_date_effective",
    "dp_collected": "date_current_through",
    "dp_valid_imp": "date_current_through",
    "dp_collected_combined": "date_current_through",
    "ssp_collected": "ssp_date_current_through",
    "ssp_current_imp": "ssp_current_through_status",
    "dp_state_fed_reference": "reference_necessity",
    "dp_state_fed_citation": "reference_necessity",
    "dp_state_fed_combined": "reference_necessity",
    "ssp_state_fed_reference": "ssp_reference_necessity",
    "ssp_state_fed_citation": "ssp_reference_necessity",
    "dp_type": "definition_type",
    "dp_activity": "prohibited_activity",
    "dp_penalties": "penalty",
    "dp_exemption": "exemption_presence",
    "ssp_prohibit": "ssp_prohibition",
    "ssp_permit": "ssp_authorization",
    "ssp_restrict": "ssp_restriction",
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
            "non-controlled-substance paraphernalia, SSP administration that does not itself create a generally applicable "
            "drug-paraphernalia prohibition, and business-only display or head-shop rules "
            "unless the text clearly applies more broadly or creates a generally applicable ban. Business-only restrictions do not "
            "count unless the same law governs drug paraphernalia legality beyond the business setting."
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
    "ssp_scope": RetrievalGuidance(
        guidance_topic="ssp_scope",
        relevance_instructions=(
            "Prefer operative local ordinance text that actually authorizes, prohibits, or limits syringe service "
            "programs. High-value text expressly uses SSP, syringe exchange, needle exchange, harm reduction, or "
            "similar program language, including emergency-conditioned authorization of clean needle or needle-and-"
            "syringe exchange projects. Reject syringe buyback, disposal-only, or generic public-health background "
            "text, and reject generic hypodermic, syringe, or HIV/AIDS references unless they directly create the local SSP "
            "rule being coded. Reject general local-emergency, nuisance-abatement, building-code, or emergency-response text "
            "that mentions emergencies without also authorizing or regulating clean needle, needle-and-syringe exchange, or SSP operation. A public-health-emergency or HIV/AIDS "
            "finding section counts only when it empowers local officials to authorize or operate exchange projects. Treat text like "
            "`Declaration of Local Public Health Emergency` or `The Mayor is hereby empowered` to authorize clean needle and syringe "
            "exchange projects as especially high value only when that text itself creates operative authorization or another operative SSP rule."
        ),
        anchor_terms=[
            "syringe service program",
            "syringe services",
            "syringe exchange facility",
            "syringe exchange program",
            "needle exchange",
            "needle exchange program",
            "needle and syringe exchange",
            "needle and syringe exchange project",
            "clean needle",
            "clean needle and syringe exchange",
            "clean needle and syringe exchange project",
            "syringe exchange",
            "sterile needle",
            "sterile syringe",
            "harm reduction",
            "needle dispensary",
            "local public health emergency",
            "public health emergency",
            "disease outbreak",
        ],
        negative_anchor_terms=[
            "harm reduction background",
            "public health findings",
            "hiv/aids findings",
            "overdose prevention background",
            "syringe disposal",
            "sharps disposal",
            "syringe buyback",
            "state registration",
            "annual reporting",
            "site approval",
            "complaint procedures",
        ],
        no_context_fallback_short_answer="No",
        enable_relevance_backfill=False,
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
    "ssp_date_enactment": RetrievalGuidance(
        guidance_topic="ssp_date_enactment",
        relevance_instructions=(
            "Prefer ordinance metadata and text that explicitly states when the SSP law was enacted, passed, "
            "adopted, or approved. Amendment dates are also high value when the coding logic asks for the most "
            "recent amendment in the target window. Reject effective dates and current-through dates unless the "
            "date label is genuinely unknown and the ordinance-end date functions as the only enactment marker."
        ),
        anchor_terms=[
            "enacted",
            "passed",
            "adopted",
            "approved",
            "ordinance no",
            "amended",
            "needle exchange",
            "syringe service",
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
    "ssp_date_effective": RetrievalGuidance(
        guidance_topic="ssp_date_effective",
        relevance_instructions=(
            "Prefer text explicitly labeled as the effective date for the SSP ordinance or language such as shall "
            "take effect, takes effect, become effective, or eff. Reject enactment dates, approval dates, and "
            "unlabeled dates. If the text does not clearly tie a date to effectiveness, it is low value."
        ),
        anchor_terms=[
            "effective",
            "eff",
            "take effect",
            "takes effect",
            "become effective",
            "effective date",
            "syringe service",
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
    "ssp_date_current_through": RetrievalGuidance(
        guidance_topic="ssp_date_current_through",
        relevance_instructions=(
            "Prefer source- or code-level recency statements describing when the municipal code or SSP ordinance "
            "source was last updated, such as current through language, edition notices, update banners, or publisher "
            "metadata. Reject ordinary section amendment dates unless the coding logic says to fall back to the most "
            "recently ratified ordinance because no official current-through date is present."
        ),
        anchor_terms=[
            "current through",
            "current as of",
            "supplement",
            "edition",
            "updated",
            "ordinances passed through",
            "syringe service",
        ],
    ),
    "ssp_current_through_status": RetrievalGuidance(
        guidance_topic="ssp_current_through_status",
        relevance_instructions=(
            "Prefer official current-through notices, partial published current-through dates, edition metadata, or "
            "the enactment date of the most recently ratified ordinance used as the fallback source for the SSP "
            "current-through answer. A pure data-collection date is low value unless no ordinance-derived date exists."
        ),
        anchor_terms=[
            "current through",
            "supplement",
            "updated",
            "ratified ordinance",
            "passed",
            "adopted",
        ],
    ),
    "reference_necessity": RetrievalGuidance(
        guidance_topic="reference_necessity",
        relevance_instructions=(
            "Prefer text showing whether a cited state or federal law must actually be consulted to "
            "answer the survey question. A mere citation is not enough. High-value text either says "
            "the local ordinance adopts or incorporates an external definition, or makes clear that the "
            "local text is self-contained. Count this as necessary when the outside law must be "
            "reviewed to determine the local ordinance's meaning, scope, or elements. If the local text "
            "can be answered from the ordinance itself, the correct answer is no even if state or federal "
            "law is cited, mirrored, used as background authority, or referenced in a penalty or "
            "enforcement cross-reference. Exemptions or carve-outs that apply only when conduct is "
            "authorized, lawful, or in accordance with a cited outside statute do count as true outside-law "
            "dependency because the external law must be reviewed to know the exemption's scope. Treat citations used only to "
            "define controlled substances, controlled-substance schedules, drug schedules, or other background legal categories as low-value "
            "noise unless the local ordinance makes that outside material dispositive for a benchmarked paraphernalia question. Prefer the smallest "
            "sentence or clause that actually makes outside law necessary, and reject nearby definitional or background citations that do not change the answer. "
        ),
        anchor_terms=[
            "incorporate",
            "adopt",
            "defined in",
            "pursuant to",
            "in accordance with",
            "in compliance with",
            "as provided in",
            "as defined in",
            "state law",
            "federal law",
            "35 p.s.",
            "u.s.c.",
        ],
    ),
    "ssp_reference_necessity": RetrievalGuidance(
        guidance_topic="ssp_reference_necessity",
        relevance_instructions=(
            "Prefer text showing whether a cited state or federal law must actually be consulted to answer the SSP "
            "survey question. A mere citation is not enough. High-value text either says the local ordinance adopts, "
            "incorporates, or depends on an outside authorization, prohibition, or restriction standard, or makes clear "
            "that the local SSP rule is self-contained. Do not treat a local approval, registration, reporting, or site-review "
            "requirement as an outside-law dependency unless the ordinance makes the outside legal standard dispositive."
        ),
        anchor_terms=[
            "incorporate",
            "adopt",
            "as defined in",
            "state health department",
            "state law",
            "federal law",
            "public health law",
            "harm reduction act",
            "administrative code",
        ],
        negative_anchor_terms=[
            "state registration",
            "annual reporting",
            "site approval",
            "implementation background",
            "public health findings",
        ],
        enable_relevance_backfill=False,
    ),
    "ssp_prohibition": RetrievalGuidance(
        guidance_topic="ssp_prohibition",
        relevance_instructions=(
            "Prefer operative text that expressly bans, prohibits, or makes SSP operation unlawful. High-value passages "
            "state that syringe exchange, needle exchange, or syringe service programs may not operate or are prohibited. "
            "Reject mere permit requirements or narrower restrictions unless the same text creates a full ban."
        ),
        anchor_terms=[
            "prohibited",
            "unlawful",
            "shall not operate",
            "may not operate",
            "needle exchange",
            "syringe exchange facility",
            "syringe exchange program",
            "syringe service",
            "state health department",
        ],
    ),
    "ssp_authorization": RetrievalGuidance(
        guidance_topic="ssp_authorization",
        relevance_instructions=(
            "Prefer operative text that expressly authorizes, permits, or establishes SSPs. High-value passages say SSPs "
            "may operate, are authorized, are permitted, or are allowed, including conditions tied to a declared public "
            "health emergency or disease outbreak. Treat authorization of clean needle or needle-and-syringe exchange "
            "projects during a declared emergency as affirmative SSP authorization. Reject restrictions, state-registration "
            "references, site-approval language, or the mere absence of a ban unless the text itself creates affirmative "
            "authorization for SSP operation. Reject general local-emergency findings, outbreak declarations, nuisance-abatement text, and generic public-health "
            "powers unless the same text expressly says a clean needle, needle-and-syringe exchange, syringe exchange, or SSP program may operate."
        ),
        anchor_terms=[
            "authorized",
            "permitted",
            "may operate",
            "allowed",
            "public health emergency",
            "local public health emergency",
            "disease outbreak",
            "clean needle",
            "clean needle and syringe exchange",
            "clean needle and syringe exchange project",
            "needle exchange",
            "needle and syringe exchange project",
            "syringe exchange facility",
            "syringe exchange program",
            "syringe service",
        ],
        negative_anchor_terms=[
            "state registration",
            "annual reporting",
            "site approval",
            "complaint procedures",
            "coordination duties",
            "implementation background",
            "public health findings",
        ],
    ),
    "ssp_restriction": RetrievalGuidance(
        guidance_topic="ssp_restriction",
        relevance_instructions=(
            "Prefer operative SSP restriction text that expressly limits how programs may operate. High-value passages "
            "state caps on sites, distance restrictions from schools or parks, visit-frequency limits, syringe-quantity "
            "limits, mobile-site limits, permit or license requirements, or similar operational conditions. Reject outright "
            "prohibitions when the question asks about restrictions rather than total bans. Treat permit or license required "
            "for operation as high value only when the ordinance makes formal local operating authorization mandatory, not "
            "when it merely requires notice, registration, or approval of a particular site. Reject emergency-trigger text, coordination duties, reporting, "
            "state-registration references, and site-review language unless they impose a listed operational limit such as a permit requirement, buffer, cap, "
            "mobile-site rule, visit limit, or syringe-quantity limit. Prefer option-specific clauses over umbrella operational text, and avoid assigning multiple labels "
            "from a single broad sentence unless each selected label has its own distinct direct support."
        ),
        anchor_terms=[
            "syringe service program",
            "syringe exchange facility",
            "needle exchange program",
            "distance",
            "schools",
            "childcare",
            "parks",
            "mobile",
            "permit",
            "license",
            "exchange only basis",
            "quantity of syringes",
            "frequency of visits",
        ],
        negative_anchor_terms=[
            "harm reduction background",
            "public health findings",
            "state registration",
            "annual reporting",
            "site approval",
            "coordination duties",
            "complaint procedures",
            "implementation plan",
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
            "Prefer text that explicitly enumerates operative prohibited acts. High-value passages state verbs "
            "such as sell, offer for sale, deliver, distribute, transfer, possess with intent, use, "
            "display, advertise, or manufacture. Code only activities directly prohibited by the legal text. "
            "Simple possession is different from possession with intent to sell or deliver; use should only count "
            "if the text explicitly prohibits use; and business-context activity may be narrower than a general "
            "prohibition. Do not count zoning, land-use, business-license, head-shop, retail-display, or minors-only "
            "access restrictions as general prohibited activities unless the same text expressly creates the operative "
            "paraphernalia prohibition. Treat advertising/display as distinct from sale, and do not infer sale or "
            "delivery unless the text independently prohibits sale, offer for sale, delivery, distribution, transfer, "
            "furnishing, exchange, or possession with intent. Focus on acts involving drug paraphernalia used with "
            "controlled substances. Reject nearby definition, date, or penalty text unless it directly states the "
            "operative activity."
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
            "sections. Follow explicit penalty cross-references such as `Penalty, see § ...` and treat the "
            "cited section as governing sanction text. Exact legal labels matter here: misdemeanor, infraction, "
            "felony, forfeiture, seizure, civil fine, criminal fine, and unlawful only should be grounded in the "
            "actual text. Do not treat bare unlawfulness language as the full answer when a nearby or directly "
            "cross-referenced section states a fine, imprisonment, offense class, forfeiture, or seizure. Reject "
            "definitions and operative restrictions that do not specify consequences. Do not infer a benchmark label from generic "
            "criminality or offense classification unless the text itself states that exact sanction or the governing cross-reference states it. Treat generic default-penalty "
            "sections and offense classes as low value unless the paraphernalia ordinance itself, or its direct penalty cross-reference, ties them to a benchmarked sanction concept."
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
            "medical use, professionals acting in the course of business, public officials, needle exchange, "
            "syringe exchange facilities, syringe service program participants, test strips, fentanyl/xylazine "
            "checking language, or drug checking. Code only exemptions directly supported "
            "by the legal text. Reject ordinary prohibitions, zoning restrictions, business permissions, and "
            "other nonoperative context unless they actually function as a paraphernalia exemption under the "
            "coding logic. Prefer the exact exemption clause over broad surrounding context so the answer stays label-by-label rather "
            "than overinclusive. Treat cannabis decriminalization, medical-marijuana zoning, marijuana-business permissions, general business-use permissions, and tobacco carve-outs as low-value "
            "noise unless the same text expressly narrows the operative paraphernalia prohibition or definition."
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
            "needle exchange",
            "syringe exchange",
            "syringe service program participant",
            "clean needle and syringe exchange",
            "harm reduction",
            "test strip",
            "fentanyl test strip",
            "xylazine test strip",
            "drug testing equipment",
            "does not include",
            "drug checking",
        ],
        enable_relevance_backfill=False,
    ),
    "exemption_activity_scope": RetrievalGuidance(
        guidance_topic="exemption_activity_scope",
        relevance_instructions=(
            "Prefer exemption text that clarifies which activities remain allowed. High-value passages "
            "either explicitly enumerate exempted activities such as possession, use, distribution, or "
            "sales, or tie the exemption to a definition that implicitly spans all prohibited activities. "
            "Read the exemption together with the operative activity language when needed. Reject text "
            "that only proves an exemption exists but does not help determine its activity scope. Do not expand a broad phrase into "
            "every activity label unless the exemption truly operates at the full definition or full-ordinance level."
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
    "dp_activity": RetrievalGuidance(
        relevance_instructions=(
            "For dp_activity, code only parent-query activity labels that appear directly in the operative prohibition text. "
            "Do not count paraphernalia-shop zoning rules, retail-display or business-access restrictions, or minors-only "
            "sales/display provisions as general prohibited activities unless that same text expressly creates the benchmarked "
            "ordinance-wide prohibition. Treat illegal-smoking-product text as noise unless the same sentence or subsection also expressly targets illegal smoking paraphernalia."
        ),
        completion_instructions=(
            "For dp_activity, only code items found directly in the legal text. Count only operative prohibitions. DO NOT code "
            "zoning, land-use, head-shop, retail-display, business-access, or minors-only restrictions as general activity labels. "
            "DO NOT code Sales unless the text itself prohibits sale, offer for sale, possession with intent to sell, or a clearly "
            "equivalent sales act. Advertising or display language by itself does not prove Sales. Do not treat illegal-smoking-product sales clauses as paraphernalia activity support unless the same operative text also prohibits conduct involving illegal smoking paraphernalia."
            " Do not map illegal-smoking-product delivery or transfer clauses to paraphernalia delivery labels unless the same clause explicitly targets paraphernalia conduct."
        ),
        anchor_terms=["unlawful", "prohibited", "offer for sale"],
        enable_relevance_backfill=False,
    ),
    "dp_penalties": RetrievalGuidance(
        relevance_instructions=(
            "Focus on operative sanction text. Follow direct cross-references such as `Penalty, see § ...`. Do not elevate "
            "business-license revocation, permitting consequences, or other collateral remedies into Other when the coding rules "
            "exclude them. Retrieve and use the actual cross-referenced penalty section before answering. Do not translate offense "
            "classes into benchmark labels unless the text itself states the exact benchmark label or the directly relevant sanction. Treat generic default-penalty language, offense grades, and "
            "boilerplate classification clauses as low value unless the ordinance or direct penalty cross-reference states the benchmark sanction concept itself. Ignore nuisance, licensing, zoning, or business-remedy text unless it is the direct operative penalty for the paraphernalia violation."
        ),
        completion_instructions=(
            "For dp_penalties, assign the exact labels supported by the exact penalty terms found in the legal text. Treat `Penalty, "
            "see § ...` or similar cross-references as part of the governing penalty text. If the text states a fine, imprisonment, "
            "or both, DO NOT answer Unlawful only. Use Criminal Fine only when the text expressly ties the fine to a criminal "
            "offense class or conviction; otherwise use Unspecified Fine unless another named fine label is directly supported. "
            "Do NOT translate a misdemeanor degree, conviction language, or other offense classification into Infraction, Civil Fine, "
            "Criminal Fine, Incarceration, Forfeiture/Seizure, or Other unless the text itself states those exact sanctions or an "
            "explicitly cross-referenced section states them. Generic default penalties, offense classes, and boilerplate criminal classifications do not count by themselves unless the text states the benchmark concept directly. "
            "If penalties are provided in another section, retrieve and code that section before choosing Unlawful only. "
            "If any concrete sanction appears (fine amount/class, incarceration term, forfeiture, seizure), suppress Unlawful only and code the concrete labels instead. "
            "Use Criminal Fine when fine language is tied to misdemeanor/felony class or explicit penalty class references. "
            "Keep Civil Fine separate from Forfeiture/Seizure; do not infer one from the other. "
            "Include Other only when the ordinance imposes a genuine residual penalty that does not fit another named option. If "
            "the text supports only Unlawful and Civil Fine, do not add Other as a hedge. IGNORE business-side licensing or "
            "permitting consequences such as license revocation, suspension, or denial unless the coding rules explicitly require "
            "them; those consequences SHOULD NOT be coded as Other."
        ),
        anchor_terms=[
            "civil penalty",
            "license revocation",
            "sanction",
            "penalty, see",
            "punishable as provided in",
            "general penalty section",
            "imprisonment",
            "jail",
            "fine",
        ],
    ),
    "dp_exemption": RetrievalGuidance(
        relevance_instructions=(
            "Focus on true paraphernalia carve-outs. Reject tobacco-only exceptions, zoning permissions, retail-business "
            "permissions, and other non-paraphernalia exceptions unless they clearly function as coded exemptions under the "
            "survey rules. If syringe-service, syringe-exchange, harm-reduction, supervised-use, or test-strip/drug-checking "
            "text is retrieved, treat it as high-value only when it directly creates an operative exemption. Reject unrelated "
            "commercial-cannabis, marijuana-business, employment, or administrative provisions that mention cannabis or syringes "
            "without narrowing the paraphernalia prohibition or definition. Treat cannabis decriminalization, medical-marijuana zoning, marijuana-business permissions, "
            "general business-use permissions, and tobacco carve-outs as low-value noise unless the same text expressly narrows the paraphernalia prohibition or definition. Prefer the exact exemption clause over nearby business, "
            "zoning, or facility-regulation text so the answer does not become overinclusive. For SSP-related exemptions, prefer text that explicitly exempts syringes, needles, test strips, fentanyl/xylazine checking equipment, or similar paraphernalia."
        ),
        completion_instructions=(
            "For dp_exemption, only code labels found directly in operative exemption text, incorporated definition text, or other "
            "true carve-out language. Include Other only for a real paraphernalia exemption that does not fit any listed label. "
            "IGNORE tobacco-only exceptions, tobacco packaging carve-outs, zoning permissions, retail-business permissions, and "
            "other non-paraphernalia business carve-outs; they SHOULD NOT be coded as Other. When the ordinance incorporates a "
            "state definition, do not infer a broad exemption from narrow sales-to-minors or prescription-device exceptions unless "
            "the same text actually narrows the operative paraphernalia prohibition or definition. When the ordinance incorporates a "
            "state definition, favor does not apply, does not include, exception, or incorporated definitional carve-out language "
            "over business-context permissions. If syringe-service, syringe-exchange, harm-reduction, supervised-use, or syringe-"
            "exchange-facility text expressly authorizes syringes, needles, test strips, drug-checking equipment, or similar "
            "paraphernalia, code the corresponding SSP and DCE labels instead of defaulting to None or cannabis-only. Do NOT "
            "treat cannabis decriminalization, medical-marijuana zoning, marijuana-business permissions, general business-use permissions, or unrelated commercial-cannabis provisions as paraphernalia exemptions. Map prescription or "
            "licensed-physician-or-dentist carve-outs to the approved-medical-use labels, public-official or governmental-duty "
            "carve-outs to the public-official label, and bona fide religious ritual or ceremony carve-outs to Other. Evaluate each "
            "exemption label independently. Do not add an SSP, DCE, cannabis, medical-use, or professional label just because nearby "
            "retrieved context mentions it; add a label only when the exemption text itself supports that specific label. If no specific "
            "listed label is directly supported, do not guess and do not use Other as a fallback. "
            "Map lawful use of hypodermic syringes to approved medical-use labels only when medical scope is explicit (e.g., diabetes, insulin, practitioner/prescription context). "
            "For drug-checking equipment labels, use SSP/harm-reduction context-specific DCE only when SSP/harm-reduction context is explicit; otherwise keep the general DCE label. "
            "If any explicit exemption label is supported, do not return None."
        ),
        anchor_terms=[
            "exception",
            "does not apply",
            "authorized",
            "medical marijuana",
            "needle exchange",
            "syringe exchange",
            "syringe service program participant",
            "clean needle and syringe exchange",
            "test strip",
            "fentanyl test strip",
            "xylazine test strip",
            "drug testing equipment",
            "drug checking",
        ],
    ),
    "dp_state_fed_reference": RetrievalGuidance(
        relevance_instructions=(
            "For dp_state_fed_reference, answer Yes only when the local text expressly incorporates, adopts, or depends on an "
            "outside statute or definition for meaning, scope, or elements. Mere citations, authority statements, mirrored text, "
            "penalty references, or background references are not enough. A citation used only to identify controlled substances, "
            "drug schedules, or other background legal categories is not enough unless the outside law must actually be reviewed to "
            "resolve a benchmarked paraphernalia question. Treat the benchmark-question list in the query context as exhaustive. "
            "Definitional citations to controlled-substance schedules, controlled-substances acts, or background state-law categories do not count unless the ordinance makes them dispositive. Treat an exemption or carve-out that turns on conduct "
            "being authorized by, lawful under, or in accordance with an outside statute as a real dependency."
        ),
        completion_instructions=(
            "For dp_state_fed_reference, answer Yes only when the local ordinance expressly makes outside law necessary to interpret "
            "the benchmarked paraphernalia rule. DO NOT answer Yes for bare citations, general authority references, parallel wording, "
            "or penalty/enforcement cross-references. DO NOT answer Yes just because the ordinance cites a state drug schedule, a "
            "controlled-substances definition, or similar background law unless that outside definition must actually be consulted to "
            "answer the benchmark questions. Definitional citations to controlled-substance schedules, controlled-substances acts, or background state-law categories do not count by themselves. Answer Yes when the ordinance makes an exemption or carve-out depend on whether conduct "
            "complies with a cited outside statute, because reviewing that outside law is necessary to know the exemption's scope. If retrieved context mixes broader cannabis, public-health, or business chapters with the operative dependency clause, trust only the clause that actually makes outside law necessary."
        ),
        anchor_terms=[
            "incorporated by reference",
            "as defined in",
            "pursuant to",
            "in accordance with",
        ],
        enable_relevance_backfill=False,
    ),
    "dp_state_fed_citation": RetrievalGuidance(
        completion_instructions=(
            "For dp_state_fed_citation, return only the smallest specific statutory unit that the ordinance actually incorporates or "
            "depends on. Cite only provisions appearing in the same sentence, subsection, or immediately adjacent chunk as the "
            "dependency-triggering language. DO NOT dump every citation appearing anywhere in the retrieved chapter set. If the "
            "ordinance stays self-contained on the benchmark question, do not elevate incidental cannabis, public-health, or "
            "controlled-substances citations as the relevant outside law. Keep the returned citation aligned with the same outside-law "
            "family that justified the parent dependency answer unless the local text for this query clearly points to a different "
            "controlling family. When the local ordinance contains both a narrow exemption carve-out citation and a broader operative "
            "paraphernalia or controlled-substances citation, prefer the citation that governs the benchmarked paraphernalia rule over "
            "the narrower carve-out citation."
        ),
        anchor_terms=["incorporated by reference", "as defined in", "et seq."],
    ),
    "ssp_state_fed_reference": RetrievalGuidance(
        relevance_instructions=(
            "For ssp_state_fed_reference, answer Yes only when the local SSP ordinance expressly makes outside law necessary to "
            "decide whether SSPs are authorized, prohibited, or restricted. Mere citations, state-registration references, local "
            "site-approval language, or implementation background are not enough. References to the State Harm Reduction Act only as "
            "background authorization or definitional context are not enough by themselves."
        ),
        completion_instructions=(
            "For ssp_state_fed_reference, answer Yes only when the ordinance expressly depends on outside law to determine the local "
            "SSP rule. DO NOT answer Yes for bare citations, public-health background references, state registration mentions, or "
            "local approval prerequisites when the operative local rule is otherwise self-contained. Treat statements that an SSP is "
            "authorized by the State Harm Reduction Act, or generic clauses making the program subject to state or federal law, as "
            "insufficient unless the outside law is actually dispositive of the local benchmarked rule."
        ),
        anchor_terms=["as defined in", "state health department", "harm reduction act"],
        enable_relevance_backfill=False,
    ),
    "ssp_current_imp": RetrievalGuidance(
        completion_instructions=(
            "For ssp_current_imp, code a Known option whenever the answer can be grounded in an official current-through notice, "
            "a partial published current-through notice with month or day imputed, or the enactment date of the most recently "
            "ratified ordinance used as the fallback for ssp_collected. Use Unknown, reflects date of data collection only when "
            "the date truly comes from the data-collection date rather than any ordinance-derived date."
        ),
        anchor_terms=[
            "current through",
            "supplement",
            "ratified ordinance",
            "data collection",
        ],
    ),
    "ssp_state_fed_citation": RetrievalGuidance(
        completion_instructions=(
            "For ssp_state_fed_citation, return only the smallest specific statutory, regulatory, or administrative unit that the "
            "local SSP ordinance actually depends on. Cite only provisions appearing in the same sentence, subsection, or "
            "immediately adjacent chunk as the dependency-triggering language. Do not dump every state or federal citation "
            "appearing anywhere in the retrieved materials."
        ),
        anchor_terms=[
            "state health department",
            "public health law",
            "administrative code",
        ],
    ),
    "dp_state_fed_combined": RetrievalGuidance(
        relevance_instructions=(
            "For the combined survey question, answer yes only if the ordinance expressly incorporates, "
            "adopts, or depends on the external law such that reviewing that outside law is required. "
            "Answer no when the local ordinance remains self-contained and the outside law is only cited "
            "or mentioned."
        ),
        completion_instructions=(
            "For dp_state_fed_combined, answer Yes only when outside law is actually necessary to interpret the local rule, and if Yes "
            "return only the smallest specific statutory unit that must be reviewed. Do not dump every citation in the surrounding "
            "chapter set."
        ),
        anchor_terms=[
            "incorporated by reference",
            "adopts",
            "incorporates",
            "defined by reference",
        ],
    ),
    "ssp_law": RetrievalGuidance(
        relevance_instructions=(
            "For ssp_law, answer Yes whenever the ordinance expressly authorizes, prohibits, or limits SSPs. Do not require an explicit "
            "authorization clause. A total ban, a conditional authorization, or an operative restriction all count as evidence that an SSP law exists. "
            "Answer No only when the ordinance is silent on SSP authorization, prohibition, and operational limits. Treat emergency "
            "authorization language for clean needle or needle-and-syringe exchange projects as especially high value even if the text "
            "does not use the modern `SSP` label, but only when the ordinance itself grants or triggers that authorization."
        ),
        completion_instructions=(
            "For ssp_law, answer Yes whenever any operative local ordinance text authorizes, prohibits, or limits SSPs. Do not require an explicit authorization clause. "
            "If the ordinance empowers a mayor or other local official to authorize clean needle, needle exchange, or "
            "needle-and-syringe exchange projects during a declared local public health emergency or disease outbreak, that "
            "counts as local SSP authorization and ssp_law should be coded as Yes. Reserve the conditional answer wording "
            "for ssp_permit or other authorization-detail questions, not for the broader ssp_law existence question. Treat text such as "
            "`Declaration of Local Public Health Emergency` and `The Mayor is hereby empowered to declare the existence of a Local "
            "Public Health Emergency when the authorization of clean needle and syringe exchange projects would abate the spread of HIV "
            "and AIDS` as qualifying evidence only when that same text or an immediately linked operative provision actually creates local SSP authorization. Do not "
            "generalize from public-health background or emergency text that does not itself create the operative SSP rule."
        ),
    ),
    "ssp_permit": RetrievalGuidance(
        relevance_instructions=(
            "For ssp_permit, focus on text that expressly permits, authorizes, allows, or establishes SSP operation. Do not treat a mere "
            "restriction, reporting rule, site-approval prerequisite, or the existence of an SSP law as authorization unless the text itself says the program may operate. "
            "A rule that no person may operate a syringe exchange facility without a valid local permit counts as an authorization regime, not just a restriction. Do not treat state registration, annual reporting, complaint-procedure requirements, or mayoral/site approval alone as authorization."
        ),
        completion_instructions=(
            "For ssp_permit, distinguish explicit authorization from broader SSP-law existence. A jurisdiction can have an SSP law because it bans or restricts programs, "
            "yet ssp_permit is still No unless the ordinance expressly authorizes operation. Emergency-conditioned clean needle or needle-and-syringe exchange authority counts here only when the ordinance says the program may operate once the emergency condition is met."
            " Treat authorization of clean needle or needle-and-syringe exchange projects as SSP authorization only when the ordinance itself authorizes operation, and do not infer authorization from site approval, state registration, or the mere existence of an SSP law. A permit-required operating regime for a syringe exchange facility is an explicit authorization regime and should be coded Yes rather than No."
            " If text only states operation at approved sites or state registration prerequisites, do not treat that as explicit authorization unless a separate clause affirmatively authorizes SSP operation."
        ),
    ),
    "ssp_prohibit": RetrievalGuidance(
        relevance_instructions=(
            "For ssp_prohibit, focus on text that bans all SSP operation. Reject operational restrictions, permit requirements, conditional authorizations, or narrower site rules unless the same text makes all SSP operation unlawful."
        ),
        completion_instructions=(
            "For ssp_prohibit, answer Yes only for a true SSP ban. Do not code Yes when the ordinance merely restricts SSPs, limits locations, requires approval, or authorizes SSPs only in emergencies."
        ),
    ),
    "ssp_restrict": RetrievalGuidance(
        relevance_instructions=(
            "For ssp_restrict, focus on operational limits after SSPs are otherwise recognized by the ordinance. Do not treat a total ban or a bare authorization clause as a restriction. Count only listed operating conditions such as permits, caps, buffers, mobile-site limits, visit limits, or syringe-quantity limits. Require explicit lexical triggers for each label (e.g., quantity caps/max/per participant/per visit; mobile/vehicle/van/roving terms; permit/license/registration required for operation)."
        ),
        completion_instructions=(
            "For ssp_restrict, distinguish operational restrictions from both total bans and bare authorization. If the ordinance only says SSPs are authorized, prohibited, or tied to a declared emergency without imposing a listed operating condition, use No restrictions listed. Do not infer `Other restrictions` from general operating requirements, exchange-only language, or coordination duties unless the ordinance states a concrete residual restriction that fits no named label. Do not infer `Restrictions on quantity of syringes that may be provided or exchanged` unless the ordinance expressly limits number, amount, or quantity. Treat site approval, notice, or registration as `Permit or license required for operation` only when the ordinance requires a formal local operating permit or license. Treat mobile-site restrictions only when the ordinance expressly limits mobile or non-fixed-location operation."
            " Apply a one-vs-many guard: if the evidence contains only one explicit restriction sentence, do not spread that sentence across multiple labels unless each selected label has its own direct lexical trigger in that sentence."
            " If your selected labels conflict with your own reasoning, align the final labels to the cited operative clauses before returning the answer."
        ),
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
            "exemption either in the definition or directly alongside prohibited activities. Prefer the exact exemption clause and the "
            "same sentence or subsection for activity mapping."
        ),
        completion_instructions=(
            "For dp_exempt_can_activity, code only the activities actually enumerated in the exemption text or in the same operative "
            "sentence/subsection as the exemption. If the exemption does not enumerate activities but clearly functions as a definitional "
            "carve-out, fall back to the prohibited activities selected in dp_activity. Do not expand broad phrases such as cannabis use "
            "or commerce into every response option unless the exemption truly operates at the definition-wide level. When a phrase like "
            "`possession ... or activities associated with cannabis use or commerce` appears, do not automatically code Use, Distribution, "
            "Sales, and Other all together unless the legal text clearly makes those activities exempt. Treat umbrella language as non-specific unless concrete activity verbs are explicitly listed. "
            "Only map Distribution/Sales/Use when those verbs (or direct equivalents) are explicitly present in the exemption text. Do not infer them from broad possession/authorization clauses. "
            "Require at least one direct quote per selected activity label in option evidence. Be conservative and label-by-label."
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
        "used with controlled substances, not tobacco-only or other non-controlled-substance paraphernalia. "
        "Answer No when the retrieved text is limited to sales-to-minors rules, head-shop zoning, retail-display "
        "or business-license limits, location-only restrictions, or SSP administration that does not itself create "
        "a generally applicable drug-paraphernalia prohibition or regulation. Business-only restrictions do not count. "
        "We are interested in full statutes governing drug paraphernalia legality, not small business-only mentions."
    ),
    "ssp_scope": (
        "Interpret the question as asking whether a local ordinance expressly authorizes, prohibits, or limits syringe service "
        "programs, syringe exchange programs, needle exchange programs, or closely synonymous harm-reduction syringe programs. "
        "Do not count syringe buyback or disposal-only programs as SSP authorization. Emergency-conditioned provisions count "
        "when they empower a local official to authorize clean needle or needle-and-syringe exchange projects during a declared "
        "public health emergency or disease outbreak. Do not count generic syringe, hypodermic, HIV/AIDS, or facility-location text "
        "unless it directly creates an SSP authorization, prohibition, or operational limit. If the ordinance says the mayor or another "
        "local official may declare a public health emergency and thereby authorize clean needle and syringe exchange projects, that is "
        "an SSP law only when that same text or an immediately linked operative provision actually creates local authorization."
    ),
    "ssp_date_enactment": (
        "Return only the enacted, passed, adopted, or approved date for the SSP ordinance. Do not substitute effective dates or "
        "current-through dates unless the coding logic explicitly treats an unlabeled ordinance-end date as the only enactment marker."
    ),
    "ssp_date_effective": (
        "Return only a date clearly tied to effectiveness for the SSP ordinance. If the date is unlabeled or tied only to passage or "
        "approval, it is not enough."
    ),
    "ssp_date_current_through": (
        "Prefer the official current-through source date for the SSP ordinance. If no official current-through date exists, the coding "
        "logic may fall back to the most recently ratified ordinance date."
    ),
    "ssp_current_through_status": (
        "For status coding, Known includes both published current-through notices and ordinance-derived fallback dates. Unknown is "
        "reserved for dates that truly reflect only the date of data collection."
    ),
    "definition_type": (
        "Describe only paraphernalia types tied to controlled-substance use and ground the answer in the legal "
        "definition or closely linked operative text."
    ),
    "prohibited_activity": (
        "List only activity labels that the ordinance expressly and operatively prohibits for drug paraphernalia used "
        "with controlled substances. Code an activity only when the legal text directly bars that activity or a clearly "
        "synonymous act. Do not infer sale from advertising or display language, and do not convert zoning, land-use, "
        "business-license, retail-display, or minors-only access restrictions into general prohibited activities."
        " Business-only sale or display restrictions do not create the general Sales label unless the ordinance expressly prohibits sale beyond the business-only context."
    ),
    "penalty": (
        "Assign penalty labels from the exact terms found in the legal text or directly cited penalty section. Do not "
        "infer misdemeanor, felony, infraction, civil fine, or criminal fine unless the text actually supports that label. "
        "If the ordinance states a fine, imprisonment, forfeiture, seizure, or offense class, do not answer Unlawful only. "
        "Use Unlawful only only when the operative and directly cross-referenced penalty text contains no specific sanction "
        "beyond unlawfulness. Use Other only for a genuine residual penalty that is clearly imposed and does not fit any "
        "named option. Do not use Other as a hedge, and do not treat excluded collateral remedies as Other unless the coding "
        "rules expressly require them. Do not translate offense classes, generic default penalties, or boilerplate criminal classifications "
        "into benchmark labels unless the text states the benchmark concept directly."
    ),
    "exemption_presence": (
        "Identify only labels directly supported by exemption text, incorporated definition text, or other operative carve-out "
        "language. Code a label only when the legal text itself expressly creates that exemption or clearly includes it within "
        "the carve-out. Do not use Other for tobacco-only exceptions, zoning permissions, business permissions, or other "
        "carve-outs that the coding rules exclude from paraphernalia exemptions. If retrieved text includes syringe services, "
        "harm reduction, supervised use, needle exchange, or syringe exchange facility provisions, evaluate whether they "
        "expressly allow syringes, needles, test strips, drug-checking equipment, or other paraphernalia before answering "
        "None or cannabis-only. Do not treat unrelated marijuana-business, commercial-cannabis, employment, or other non-"
        "paraphernalia cannabis provisions as a cannabis exemption. Cannabis decriminalization, medical-marijuana zoning, marijuana-business permissions, general business-use permissions, and tobacco carve-outs do not count as paraphernalia exemptions unless the same text expressly narrows the operative paraphernalia prohibition or definition. If no listed exemption label is directly supported, leave it out rather than guessing or defaulting to Other."
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
        "mention. Do not treat a bare citation as making outside-law review necessary. A citation that merely defines controlled "
        "substances, drug schedules, or background statutory authority is still No unless the local ordinance makes that outside "
        "definition dispositive for a benchmarked paraphernalia question. Definitional citations to controlled-substance schedules, controlled-substances acts, or background state-law categories do not count by themselves. However, answer Yes when a local exemption or "
        "carve-out applies only if conduct is authorized by, lawful under, or in accordance with the cited outside law, because the "
        "external law must then be reviewed to know the exemption's scope. If the answer is "
        "Yes, identify only the smallest specific state or federal citation that must be reviewed. If the answer is No, "
        "say that no outside-law review is necessary and do not elevate incidental citations as the "
        "relevant law. Treat the benchmark-question list in the query context as exhaustive. Do not dump every citation found in the retrieved chapter set."
    ),
    "ssp_reference_necessity": (
        "Decision rule: Answer Yes only when the local SSP ordinance expressly incorporates, adopts, or depends on a state or "
        "federal statute, regulation, or agency authorization such that reviewing that outside law is required to determine whether "
        "SSPs are authorized, prohibited, or restricted. Answer No when the local ordinance is self-contained or the outside law is "
        "only cited as background authority, implementation context, an incidental reference, or a state-registration backdrop. Treat the benchmark-question list in the "
        "query context as exhaustive. If the answer is Yes, identify only the smallest specific outside-law citation that must be reviewed."
    ),
    "ssp_prohibition": (
        "Answer Yes only when the ordinance expressly prohibits all SSPs. Restrictions, permit requirements, or regulation short of an "
        "outright ban are not enough. Use the state-authorized exception label only when the ordinance bans SSPs but expressly carves "
        "out programs authorized by a state health department or other state entity."
    ),
    "ssp_authorization": (
        "Answer Yes only when the ordinance expressly authorizes or permits SSP operation. Restrictions or the mere absence of a ban "
        "are not enough. Treat authorization of clean needle or needle-and-syringe exchange projects as SSP authorization when the "
        "ordinance expressly empowers a local official to activate them during a declared emergency. Use the emergency-conditional label "
        "only when authorization is expressly limited to a declared local public health emergency or disease outbreak. Do not treat "
        "site approval, state registration, or compliance prerequisites as authorization unless the ordinance expressly says the SSP may operate. General local-emergency or public-health findings do not count unless the same text actually authorizes exchange or SSP operation."
    ),
    "ssp_restriction": (
        "Select only restrictions explicitly supported by the operative legal text. Do not count outright bans as restrictions. Use No "
        "restrictions listed only when the ordinance addresses SSPs but does not impose any listed operational restriction. Treat Permit "
        "or license required for operation as present only when formal local operating authorization is mandatory, not when the ordinance "
        "merely requires notice, registration, or approval of a site or mobile unit. Do not infer a restriction from emergency-trigger language, reporting duties, coordination requirements, or generic site review unless the text states a listed operational limit."
    ),
}


def _direct_topic_scope_sentence(family: str | None) -> str | None:
    """Return a generic prompt guardrail that narrows attention to the right legal surface."""
    paraphernalia_families = {
        "existence_scope",
        "date_enactment",
        "date_effective",
        "date_current_through",
        "reference_necessity",
        "definition_type",
        "prohibited_activity",
        "penalty",
        "exemption_presence",
        "exemption_activity_scope",
    }
    ssp_families = {
        "ssp_scope",
        "ssp_date_enactment",
        "ssp_date_effective",
        "ssp_date_current_through",
        "ssp_current_through_status",
        "ssp_reference_necessity",
        "ssp_prohibition",
        "ssp_authorization",
        "ssp_restriction",
    }

    if family in paraphernalia_families:
        return (
            "Only rely on sections that directly address drug paraphernalia used with controlled substances "
            "or official source metadata for that ordinance. Ignore unrelated cannabis-business, tobacco, zoning, "
            "licensing, SSP, or generic public-health text unless it directly creates the operative rule being coded."
        )

    if family in ssp_families:
        return (
            "Only rely on sections that directly authorize, prohibit, or limit syringe service programs, syringe exchange "
            "programs, needle exchange programs, or official source metadata for that SSP ordinance. Ignore unrelated public-"
            "health background, generic syringe or HIV/AIDS references, disposal-only programs, school or park provisions, "
            "or generic permitting text unless it directly creates the operative SSP rule being coded."
        )

    return None


_EXEMPTION_OPTION_ANCHOR_TERMS_BY_LABEL = {
    _normalize_label_text(label): list(
        (
            _VARIABLE_OVERRIDES.get(variable_name).anchor_terms
            if _VARIABLE_OVERRIDES.get(variable_name)
            else []
        )
    )
    for variable_name, labels in _LEGACY_EXEMPTION_DEPENDENCY_LABELS_BY_VARIABLE.items()
    for label in labels
}


def _inherited_exemption_option_anchor_terms(
    request: RetrievalGuidanceRequest,
) -> list[str]:
    """Return option-specific anchor terms inherited from selected parent evidence."""
    inherited: list[str] = []
    for option, _citation, _passage, anchor_terms in _matching_selected_option_evidence(
        request
    ):
        if anchor_terms:
            inherited.extend(anchor_terms)
            continue
        inherited.extend(
            _EXEMPTION_OPTION_ANCHOR_TERMS_BY_LABEL.get(
                _normalize_label_text(option),
                [],
            )
        )
    return _dedupe_preserving_order(inherited)


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
    override_negative_terms = override.negative_anchor_terms if override else []
    merged_negative_keywords = list(
        dict.fromkeys(base.negative_anchor_terms + override_negative_terms)
    )

    return RetrievalGuidance(
        guidance_topic=(override.guidance_topic if override else None)
        or base.guidance_topic,
        retrieval_instructions=" ".join(retrieval_parts) if retrieval_parts else None,
        relevance_instructions=" ".join(prompt_parts) if prompt_parts else None,
        anchor_terms=merged_keywords,
        negative_anchor_terms=merged_negative_keywords,
        completion_instructions=(
            " ".join(completion_parts) if completion_parts else None
        ),
        no_context_fallback_short_answer=(
            override.no_context_fallback_short_answer
            if override and override.no_context_fallback_short_answer is not None
            else base.no_context_fallback_short_answer
        ),
        enable_relevance_filter=(
            override.enable_relevance_filter
            if override and override.enable_relevance_filter is not None
            else base.enable_relevance_filter
        ),
        enable_relevance_backfill=(
            override.enable_relevance_backfill
            if override and override.enable_relevance_backfill is not None
            else base.enable_relevance_backfill
        ),
    )


def _build_query_context(request: RetrievalGuidanceRequest) -> str:
    """Build concise legal-scope context for ambiguous subquestions."""
    prepend_text = (request.metadata.get("prepend_text") or "").strip()
    family = _FAMILY_BY_VARIABLE.get(request.variable_name or "")
    context_parts = []
    if prepend_text:
        context_parts.append(prepend_text.rstrip(". ") + ".")
    else:
        context_parts.append(_DEFAULT_QUERY_CONTEXT)

    direct_topic_scope = _direct_topic_scope_sentence(family)
    if direct_topic_scope:
        context_parts.append(direct_topic_scope)

    reference_scope = _format_reference_scope_questions(family)
    if reference_scope:
        context_parts.append(reference_scope)

    prior_answers = request.metadata.get("prior_answers") or {}
    if family == "exemption_activity_scope":
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
        selected_evidence = _matching_selected_option_evidence(request)
        for option, citation, passage, _anchor_terms in selected_evidence:
            evidence_line = f"Selected exemption option with evidence: {option}."
            if citation:
                evidence_line += f" Citation: {citation}."
            if passage:
                evidence_line += " Passage: " + _truncate_context_passage(passage)
            context_parts.append(evidence_line)
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
    if guidance.negative_anchor_terms:
        parts.append(
            "Low-value/noise terms unless paired with operative authorization, prohibition, or restriction text: "
            + ", ".join(guidance.negative_anchor_terms)
        )

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
    if guidance.negative_anchor_terms:
        parts.append(
            "Completion-stage noise cues to discount unless the same clause creates the operative rule: "
            + ", ".join(guidance.negative_anchor_terms)
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
    inherited_anchor_terms = (
        _inherited_exemption_option_anchor_terms(request)
        if family == "exemption_activity_scope"
        else []
    )
    guidance = RetrievalGuidance(
        guidance_topic=guidance.guidance_topic,
        shared_context=query_context,
        retrieval_instructions=_build_retrieval_instructions(
            request.variable_name,
            family,
        ),
        relevance_instructions=guidance.relevance_instructions,
        anchor_terms=_dedupe_preserving_order(
            list(guidance.anchor_terms) + inherited_anchor_terms
        ),
        negative_anchor_terms=list(guidance.negative_anchor_terms),
        completion_instructions=guidance.completion_instructions,
        no_context_fallback_short_answer=guidance.no_context_fallback_short_answer,
        enable_relevance_filter=_coep_relevance_filter_enabled(request.variable_name),
        enable_relevance_backfill=guidance.enable_relevance_backfill,
    )

    return RetrievalGuidance(
        guidance_topic=guidance.guidance_topic,
        shared_context=guidance.shared_context,
        retrieval_query=_build_retrieval_query(request, guidance),
        retrieval_instructions=guidance.retrieval_instructions,
        relevance_instructions=guidance.relevance_instructions,
        anchor_terms=guidance.anchor_terms,
        negative_anchor_terms=guidance.negative_anchor_terms,
        completion_instructions=_build_completion_instructions(guidance),
        no_context_fallback_short_answer=guidance.no_context_fallback_short_answer,
        enable_relevance_filter=guidance.enable_relevance_filter,
        enable_relevance_backfill=guidance.enable_relevance_backfill,
    )
