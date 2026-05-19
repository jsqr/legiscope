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
        "drug paraphernalia used with controlled substances beyond narrow business-only rules."
    ),
    "ssp_scope": (
        "Retrieve operative municipal ordinance text establishing whether the jurisdiction authorizes, "
        "prohibits, or limits syringe service programs, syringe exchanges, needle exchanges, harm-reduction "
        "syringe distribution, or closely synonymous SSP operations. Distinguish true SSP programs from "
        "syringe buyback or disposal-only programs."
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
        "or disease outbreak."
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
        "penalty section as controlling sanction text."
    ),
    "exemption_presence": (
        "Retrieve exception, exclusion, does-not-apply, authorized-use, and incorporated state-definition "
        "language that could create paraphernalia exemptions. Also retrieve syringe-service, syringe-exchange, harm-"
        "reduction, supervised-use, and drug-checking/test-strip provisions when they can function as direct "
        "paraphernalia carve-outs."
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
    "ssp_scope": RetrievalGuidance(
        guidance_topic="ssp_scope",
        relevance_instructions=(
            "Prefer operative local ordinance text that actually authorizes, prohibits, or limits syringe service "
            "programs. High-value text expressly uses SSP, syringe exchange, needle exchange, harm reduction, or "
            "similar program language. Reject syringe buyback, disposal-only, or generic public-health background "
            "text unless it directly creates the local SSP rule being coded."
        ),
        anchor_terms=[
            "syringe service program",
            "syringe services",
            "syringe exchange facility",
            "syringe exchange program",
            "needle exchange",
            "needle exchange program",
            "needle and syringe exchange",
            "syringe exchange",
            "sterile needle",
            "sterile syringe",
            "harm reduction",
            "needle dispensary",
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
    "ssp_reference_necessity": RetrievalGuidance(
        guidance_topic="ssp_reference_necessity",
        relevance_instructions=(
            "Prefer text showing whether a cited state or federal law must actually be consulted to answer the SSP "
            "survey question. A mere citation is not enough. High-value text either says the local ordinance adopts, "
            "incorporates, or depends on an outside authorization, prohibition, or restriction standard, or makes clear "
            "that the local SSP rule is self-contained."
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
            "health emergency or disease outbreak. Reject restrictions or the mere absence of a ban unless the text itself "
            "creates affirmative authorization."
        ),
        anchor_terms=[
            "authorized",
            "permitted",
            "may operate",
            "allowed",
            "public health emergency",
            "disease outbreak",
            "needle exchange",
            "syringe exchange facility",
            "syringe exchange program",
            "syringe service",
        ],
    ),
    "ssp_restriction": RetrievalGuidance(
        guidance_topic="ssp_restriction",
        relevance_instructions=(
            "Prefer operative SSP restriction text that expressly limits how programs may operate. High-value passages "
            "state caps on sites, distance restrictions from schools or parks, visit-frequency limits, syringe-quantity "
            "limits, mobile-site limits, permit or license requirements, or similar operational conditions. Reject outright "
            "prohibitions when the question asks about restrictions rather than total bans."
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
            "definitions and operative restrictions that do not specify consequences."
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
            "syringe exchange facilities, test strips, or drug checking. Code only exemptions directly supported "
            "by the legal text. Reject ordinary prohibitions, zoning restrictions, business permissions, and "
            "other nonoperative context unless they actually function as a paraphernalia exemption under the "
            "coding logic."
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
            "harm reduction",
            "test strip",
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
    "dp_activity": RetrievalGuidance(
        relevance_instructions=(
            "For dp_activity, code only parent-query activity labels that appear directly in the operative prohibition text. "
            "Do not count paraphernalia-shop zoning rules, retail-display or business-access restrictions, or minors-only "
            "sales/display provisions as general prohibited activities unless that same text expressly creates the benchmarked "
            "ordinance-wide prohibition."
        ),
        completion_instructions=(
            "For dp_activity, only code items found directly in the legal text. Count only operative prohibitions. DO NOT code "
            "zoning, land-use, head-shop, retail-display, business-access, or minors-only restrictions as general activity labels. "
            "DO NOT code Sales unless the text itself prohibits sale, offer for sale, possession with intent to sell, or a clearly "
            "equivalent sales act. Advertising or display language by itself does not prove Sales."
        ),
        anchor_terms=["unlawful", "prohibited", "offer for sale"],
    ),
    "dp_penalties": RetrievalGuidance(
        relevance_instructions=(
            "Focus on operative sanction text. Follow direct cross-references such as `Penalty, see § ...`. Do not elevate "
            "business-license revocation, permitting consequences, or other collateral remedies into Other when the coding rules "
            "exclude them."
        ),
        completion_instructions=(
            "For dp_penalties, assign the exact labels supported by the exact penalty terms found in the legal text. Treat `Penalty, "
            "see § ...` or similar cross-references as part of the governing penalty text. If the text states a fine, imprisonment, "
            "or both, DO NOT answer Unlawful only. Use Criminal Fine only when the text expressly ties the fine to a criminal "
            "offense class or conviction; otherwise use Unspecified Fine unless another named fine label is directly supported. "
            "Include Other only when the ordinance imposes a genuine residual penalty that does not fit another named option. If "
            "the text supports only Unlawful and Civil Fine, do not add Other as a hedge. IGNORE business-side licensing or "
            "permitting consequences such as license revocation, suspension, or denial unless the coding rules explicitly require "
            "them; those consequences SHOULD NOT be coded as Other."
        ),
        anchor_terms=["civil penalty", "license revocation", "sanction"],
    ),
    "dp_exemption": RetrievalGuidance(
        relevance_instructions=(
            "Focus on true paraphernalia carve-outs. Reject tobacco-only exceptions, zoning permissions, retail-business "
            "permissions, and other non-paraphernalia exceptions unless they clearly function as coded exemptions under the "
            "survey rules. If syringe-service, syringe-exchange, harm-reduction, supervised-use, or test-strip/drug-checking "
            "text is retrieved, treat it as high-value only when it directly creates an operative exemption."
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
            "paraphernalia, code the corresponding SSP and DCE labels instead of defaulting to None or cannabis-only."
        ),
        anchor_terms=[
            "exception",
            "does not apply",
            "authorized",
            "medical marijuana",
            "needle exchange",
            "syringe exchange",
            "test strip",
            "drug checking",
        ],
    ),
    "dp_state_fed_reference": RetrievalGuidance(
        relevance_instructions=(
            "For dp_state_fed_reference, answer Yes only when the local text expressly incorporates, adopts, or depends on an "
            "outside statute or definition for meaning, scope, or elements. Mere citations, authority statements, mirrored text, "
            "penalty references, or background references are not enough."
        ),
        completion_instructions=(
            "For dp_state_fed_reference, answer Yes only when the local ordinance expressly makes outside law necessary to interpret "
            "the benchmarked paraphernalia rule. DO NOT answer Yes for bare citations, general authority references, parallel wording, "
            "or penalty/enforcement cross-references."
        ),
        anchor_terms=["incorporated by reference", "as defined in", "pursuant to"],
    ),
    "dp_state_fed_citation": RetrievalGuidance(
        completion_instructions=(
            "For dp_state_fed_citation, return only the smallest specific statutory unit that the ordinance actually incorporates or "
            "depends on. Cite only provisions appearing in the same sentence, subsection, or immediately adjacent chunk as the "
            "dependency-triggering language. DO NOT dump every citation appearing anywhere in the retrieved chapter set."
        ),
        anchor_terms=["incorporated by reference", "as defined in", "et seq."],
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
    "ssp_scope": (
        "Interpret the question as asking whether a local ordinance expressly authorizes, prohibits, or limits syringe service "
        "programs, syringe exchange programs, needle exchange programs, or closely synonymous harm-reduction syringe programs. "
        "Do not count syringe buyback or disposal-only programs as SSP authorization."
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
    ),
    "penalty": (
        "Assign penalty labels from the exact terms found in the legal text or directly cited penalty section. Do not "
        "infer misdemeanor, felony, infraction, civil fine, or criminal fine unless the text actually supports that label. "
        "If the ordinance states a fine, imprisonment, forfeiture, seizure, or offense class, do not answer Unlawful only. "
        "Use Unlawful only only when the operative and directly cross-referenced penalty text contains no specific sanction "
        "beyond unlawfulness. Use Other only for a genuine residual penalty that is clearly imposed and does not fit any "
        "named option. Do not use Other as a hedge, and do not treat excluded collateral remedies as Other unless the coding "
        "rules expressly require them."
    ),
    "exemption_presence": (
        "Identify only labels directly supported by exemption text, incorporated definition text, or other operative carve-out "
        "language. Code a label only when the legal text itself expressly creates that exemption or clearly includes it within "
        "the carve-out. Do not use Other for tobacco-only exceptions, zoning permissions, business permissions, or other "
        "carve-outs that the coding rules exclude from paraphernalia exemptions. If retrieved text includes syringe services, "
        "harm reduction, supervised use, needle exchange, or syringe exchange facility provisions, evaluate whether they "
        "expressly allow syringes, needles, test strips, drug-checking equipment, or other paraphernalia before answering "
        "None or cannabis-only."
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
        "Yes, identify only the smallest specific state or federal citation that must be reviewed. If the answer is No, "
        "say that no outside-law review is necessary and do not elevate incidental citations as the "
        "relevant law. Treat the benchmark-question list in the query context as exhaustive. Do not dump every citation found in the retrieved chapter set."
    ),
    "ssp_reference_necessity": (
        "Decision rule: Answer Yes only when the local SSP ordinance expressly incorporates, adopts, or depends on a state or "
        "federal statute, regulation, or agency authorization such that reviewing that outside law is required to determine whether "
        "SSPs are authorized, prohibited, or restricted. Answer No when the local ordinance is self-contained or the outside law is "
        "only cited as background authority, implementation context, or an incidental reference. Treat the benchmark-question list in the "
        "query context as exhaustive. If the answer is Yes, identify only the smallest specific outside-law citation that must be reviewed."
    ),
    "ssp_prohibition": (
        "Answer Yes only when the ordinance expressly prohibits all SSPs. Restrictions, permit requirements, or regulation short of an "
        "outright ban are not enough. Use the state-authorized exception label only when the ordinance bans SSPs but expressly carves "
        "out programs authorized by a state health department or other state entity."
    ),
    "ssp_authorization": (
        "Answer Yes only when the ordinance expressly authorizes or permits SSP operation. Restrictions or the mere absence of a ban "
        "are not enough. Use the emergency-conditional label only when authorization is expressly limited to a declared local public "
        "health emergency or disease outbreak."
    ),
    "ssp_restriction": (
        "Select only restrictions explicitly supported by the operative legal text. Do not count outright bans as restrictions. Use No "
        "restrictions listed only when the ordinance addresses SSPs but does not impose any listed operational restriction."
    ),
}


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

    return RetrievalGuidance(
        guidance_topic=(override.guidance_topic if override else None)
        or base.guidance_topic,
        retrieval_instructions=" ".join(retrieval_parts) if retrieval_parts else None,
        relevance_instructions=" ".join(prompt_parts) if prompt_parts else None,
        anchor_terms=merged_keywords,
        completion_instructions=(
            " ".join(completion_parts) if completion_parts else None
        ),
        no_context_fallback_short_answer=(
            override.no_context_fallback_short_answer
            if override and override.no_context_fallback_short_answer is not None
            else base.no_context_fallback_short_answer
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
        completion_instructions=guidance.completion_instructions,
        no_context_fallback_short_answer=guidance.no_context_fallback_short_answer,
        enable_relevance_backfill=guidance.enable_relevance_backfill,
    )

    return RetrievalGuidance(
        guidance_topic=guidance.guidance_topic,
        shared_context=guidance.shared_context,
        retrieval_query=_build_retrieval_query(request, guidance),
        retrieval_instructions=guidance.retrieval_instructions,
        relevance_instructions=guidance.relevance_instructions,
        anchor_terms=guidance.anchor_terms,
        completion_instructions=_build_completion_instructions(guidance),
        no_context_fallback_short_answer=guidance.no_context_fallback_short_answer,
        enable_relevance_backfill=guidance.enable_relevance_backfill,
    )
