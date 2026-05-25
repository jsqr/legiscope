"""Raw-element LLM scanning, verification & scoring."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import polars as pl
from instructor import Instructor
from instructor.core.exceptions import InstructorRetryException

from legiscope.llm_config import Config
from legiscope.params import load_params
from legiscope.parse.elements import split_elements
from legiscope.parse.find_code_start import find_code_start
from legiscope.parse.headings import HeadingLevel, HeadingStructure
from legiscope.utils import create_structured_completion


# ── Constants ──────────────────────────────────────────────────────────

_params = load_params()
DEFAULT_SCAN_MAX_LINES = _params.get("convert", {}).get("scan_max_lines", 200)
DEFAULT_TEMPERATURE = _params.get("llm", {}).get(
    "temperature", 0.0
)  # Low temperature for consistent legal text analysis
DEFAULT_MAX_RETRIES = _params.get("llm", {}).get("max_retries", 3)
DEFAULT_LLM_TIMEOUT_SECONDS = float(_params.get("llm", {}).get("timeout", 300))
SCAN_CREATE_MAX_RETRIES = DEFAULT_MAX_RETRIES
DEFAULT_SCAN_INITIAL_SAMPLE_COUNT = 200
DEFAULT_SCAN_SCORE_THRESHOLD = 0.7
DEFAULT_SCAN_MAX_ITERATIONS = 5
DEFAULT_SCAN_MAX_TOKENS = 1600


def _get_scan_params() -> dict[str, object]:
    """Return parse-scan-specific params with sane fallbacks."""
    try:
        params = load_params()
    except FileNotFoundError:
        return {}
    return params.get("parse", {}).get("scan", {}) or {}


def _get_scan_initial_sample_count() -> int:
    """Return the initial representative element sample size for scan_headings."""
    value = _get_scan_params().get(
        "initial_sample_count", DEFAULT_SCAN_INITIAL_SAMPLE_COUNT
    )
    if not isinstance(value, int) or value <= 0:
        return DEFAULT_SCAN_INITIAL_SAMPLE_COUNT
    return value


def _get_scan_max_iterations() -> int:
    """Return the maximum number of scan refinement iterations."""
    value = _get_scan_params().get("max_iterations", DEFAULT_SCAN_MAX_ITERATIONS)
    if not isinstance(value, int) or value <= 0:
        return DEFAULT_SCAN_MAX_ITERATIONS
    return value


def _get_scan_score_threshold() -> float:
    """Return the early-stop score threshold for scan refinement."""
    value = _get_scan_params().get("score_threshold", DEFAULT_SCAN_SCORE_THRESHOLD)
    if not isinstance(value, (int, float)) or value <= 0:
        return DEFAULT_SCAN_SCORE_THRESHOLD
    return float(value)


def _get_scan_create_max_retries() -> int:
    """Return scan-stage LLM retry count, falling back to the global setting."""
    value = _get_scan_params().get("max_retries", DEFAULT_MAX_RETRIES)
    if not isinstance(value, int) or value < 0:
        return DEFAULT_MAX_RETRIES
    return value


def _get_scan_timeout_seconds() -> float:
    """Return scan-stage LLM timeout in seconds, falling back to the global setting."""
    value = _get_scan_params().get("timeout", DEFAULT_LLM_TIMEOUT_SECONDS)
    if not isinstance(value, (int, float)) or value <= 0:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    return float(value)


def _get_scan_max_tokens() -> int:
    """Return scan-stage completion cap for HeadingStructure generation."""
    value = _get_scan_params().get("max_tokens", DEFAULT_SCAN_MAX_TOKENS)
    if not isinstance(value, int) or value <= 0:
        return DEFAULT_SCAN_MAX_TOKENS
    return value


# ── Heading-like line heuristics ───────────────────────────────────────

_KEYWORD_PAT = re.compile(
    r"^(?:TITLE|CHAPTER|Ch\.|ARTICLE|Art\.|SECTION|Sec\.|SEC\."
    r"|PART|DIVISION|SUBDIVISION|SUBCHAPTER|APPENDIX"
    r"|RULE|REGULATION|CLAUSE|SCHEDULE|ANNEX|EXHIBIT"
    r"|SUBPART|PREAMBLE|AMENDMENT|ORDINANCE|RESOLUTION|SUBARTICLE)\b",
    re.IGNORECASE,
)
_COMPOUND_NUM_PAT = re.compile(r"^\d+[.\-]\d+")
_SECTION_SYMBOL_PAT = re.compile(r"^§")
_PAREN_LABEL_PAT = re.compile(r"^\([a-zA-Z0-9]{1,4}\)")
_ROMAN_HEADING_PAT = re.compile(r"^[IVXLCDM]+\.?\s+[A-Z]")
_NUMBERED_HEADING_PAT = re.compile(r"^\d{1,4}\.\s+\S")
_LETTERED_HEADING_PAT = re.compile(r"^[A-Z]\.\s+[A-Z]")
_DASH_SECTION_PAT = re.compile(r"^[-\u2013\u2014=_]{3,}\s*$")
_TITLE_HEADING_PAT = re.compile(r"^TITLE\b", re.IGNORECASE)
_ARTICLE_HEADING_PAT = re.compile(r"^ARTICLE\b", re.IGNORECASE)
_CHAPTER_HEADING_PAT = re.compile(r"^CHAPTER\b", re.IGNORECASE)
_APPENDIX_HEADING_PAT = re.compile(r"^APPENDIX\b", re.IGNORECASE)
_PREAMBLE_HEADING_PAT = re.compile(r"^PREAMBLE\b", re.IGNORECASE)
_ANNOTATION_HEADING_PAT = re.compile(r"^ANNOTATION\b", re.IGNORECASE)
_NOTES_HEADING_PAT = re.compile(r"^NOTES$", re.IGNORECASE)
_COMPOUND_IDENTIFIER_HEADING_PAT = re.compile(
    r"^(?:§\s*)?[A-Z0-9]+(?:[-.][A-Z0-9]+)+\b",
    re.IGNORECASE,
)


def _is_excluded_from_heading_like_recall(line: str) -> bool:
    """Return True for heading-like lines that are usually non-structural."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(
        _PAREN_LABEL_PAT.match(stripped)
        or _NUMBERED_HEADING_PAT.match(stripped)
        or _DASH_SECTION_PAT.match(stripped)
        or _ANNOTATION_HEADING_PAT.match(stripped)
        or _NOTES_HEADING_PAT.match(stripped)
    )


def _counts_toward_heading_like_recall(line: str) -> bool:
    """Return True for heading-like lines that should count toward recall."""
    stripped = line.strip()
    return bool(
        stripped
        and is_heading_like(stripped)
        and not _is_excluded_from_heading_like_recall(stripped)
    )


def is_heading_like(line: str) -> bool:
    """Return True if a line looks like it could be a heading."""
    stripped = line.strip()
    if not stripped:
        return False
    if _KEYWORD_PAT.match(stripped):
        return True
    if _COMPOUND_NUM_PAT.match(stripped):
        return True
    if _SECTION_SYMBOL_PAT.match(stripped):
        return True
    if _PAREN_LABEL_PAT.match(stripped):
        return True
    if len(stripped) <= 120:
        alpha_chars = [c for c in stripped if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio >= 0.60:
                return True
    if _ROMAN_HEADING_PAT.match(stripped):
        return True
    if _NUMBERED_HEADING_PAT.match(stripped):
        return True
    if _LETTERED_HEADING_PAT.match(stripped):
        return True
    if _DASH_SECTION_PAT.match(stripped):
        return True
    return False


# ── Raw element formatting ─────────────────────────────────────────────


def _format_raw_elements(elements_df: pl.DataFrame) -> str:
    """Format elements as numbered text for LLM consumption."""
    parts = []
    for row in elements_df.to_dicts():
        eid = row["element_id"]
        first_line = row["text"].split("\n")[0].strip()
        n = row["n_lines"]
        if n > 1:
            parts.append(f"E{eid}: {first_line}  [{n} lines]")
        else:
            parts.append(f"E{eid}: {first_line}")
    return "\n".join(parts)


def _variant_signature_for_line(line: str) -> str | None:
    """Return a coarse signature for a structural heading variant."""
    stripped = line.strip()
    if not stripped:
        return None

    if _TITLE_HEADING_PAT.match(stripped):
        token = _extract_identifier_token(stripped, "TITLE")
        if token and "." in token:
            return "title_decimal"
        return "title_plain"

    if _ARTICLE_HEADING_PAT.match(stripped):
        token = _extract_identifier_token(stripped, "ARTICLE")
        if token is None:
            return "article"
        if re.fullmatch(r"[IVXLCDM]+", token, re.IGNORECASE):
            return "article_roman"
        if re.search(r"[-.]", token):
            return "article_compound"
        return "article_plain"

    if _CHAPTER_HEADING_PAT.match(stripped):
        token = _extract_identifier_token(stripped, "CHAPTER")
        if token and re.search(r"[-.]", token):
            return "chapter_compound"
        return "chapter_plain"

    if _APPENDIX_HEADING_PAT.match(stripped):
        token = _extract_identifier_token(stripped, "APPENDIX")
        return "appendix_token" if token else "appendix_plain"

    if _PREAMBLE_HEADING_PAT.match(stripped):
        return "preamble"

    if _SECTION_SYMBOL_PAT.match(stripped):
        token = _extract_identifier_token(stripped)
        if token and re.search(r"[-.]", token):
            return "section_symbol_compound"
        return "section_symbol"

    if re.match(r"^SECTION\b", stripped, re.IGNORECASE):
        token = _extract_identifier_token(stripped, "SECTION")
        if token and re.search(r"[-.]", token):
            return "section_keyword_compound"
        return "section_keyword"

    if _COMPOUND_IDENTIFIER_HEADING_PAT.match(stripped):
        token = _extract_identifier_token(stripped)
        if token and token[:1].isalpha():
            return "bare_compound_alpha"
        return "bare_compound_numeric"

    return None


def _build_scan_variant_guidance(elements_df: pl.DataFrame) -> str:
    """Summarize structural format variants visible in the sample for the LLM."""
    grouped_examples: dict[str, dict[str, str]] = {}
    display_order = ["title", "article", "chapter", "section", "appendix"]

    for row in elements_df.to_dicts():
        first_line = row["text"].split("\n")[0].strip()
        label = _classify_scan_candidate(first_line)
        if label not in {
            "title",
            "article",
            "chapter",
            "section",
            "appendix",
            "compound_id",
        }:
            continue
        group_label = "section" if label == "compound_id" else label
        signature = _variant_signature_for_line(first_line)
        if signature is None:
            continue
        examples_for_group = grouped_examples.setdefault(group_label, {})
        examples_for_group.setdefault(signature, first_line)

    lines: list[str] = []
    for group_label in display_order:
        variants = grouped_examples.get(group_label, {})
        if len(variants) < 2:
            continue
        variant_examples = "; ".join(
            f"`{example[:90]}`" for example in list(variants.values())[:4]
        )
        lines.append(f"- {group_label}: {variant_examples}")

    if not lines:
        return ""

    return (
        "FORMAT VARIANTS SEEN IN SAMPLE:\n"
        "If multiple examples below belong to the same logical heading level, keep one level and put the alternate regexes in `regex_patterns` instead of creating separate levels.\n"
        + "\n".join(lines)
        + "\n"
    )


def _classify_scan_candidate(line: str) -> str:
    """Classify a first-line heading candidate for sampling diagnostics."""
    stripped = line.strip()
    if not stripped:
        return "other"
    if _TITLE_HEADING_PAT.match(stripped):
        return "title"
    if _ARTICLE_HEADING_PAT.match(stripped):
        return "article"
    if _CHAPTER_HEADING_PAT.match(stripped):
        return "chapter"
    if _APPENDIX_HEADING_PAT.match(stripped):
        return "appendix"
    if _PREAMBLE_HEADING_PAT.match(stripped):
        return "preamble"
    if _SECTION_SYMBOL_PAT.match(stripped):
        return "section"
    if _ANNOTATION_HEADING_PAT.match(stripped):
        return "annotation"
    if _NOTES_HEADING_PAT.match(stripped):
        return "notes"
    if _COMPOUND_IDENTIFIER_HEADING_PAT.match(stripped):
        return "compound_id"
    if _counts_toward_heading_like_recall(stripped):
        return "heading_like"
    return "other"


def _select_scan_sample(code_elements: pl.DataFrame, sample_count: int) -> pl.DataFrame:
    """Build a representative scan sample instead of taking only the first N elements.

    The scan prompt still needs the early TOC/body region, but relying exclusively on
    the first N elements can miss later structural headings entirely. This sampler keeps
    an early contiguous block, then adds later title/article/chapter exemplars and a
    spaced set of heading-like elements from across the document.
    """
    target_count = min(sample_count, code_elements.height)
    if code_elements.height <= target_count:
        return code_elements

    rows = code_elements.to_dicts()
    selected_ids: set[int] = set()

    def add_element_id(element_id: int) -> None:
        if len(selected_ids) < target_count:
            selected_ids.add(element_id)

    def add_spaced_rows(candidate_rows: list[dict[str, object]], quota: int) -> None:
        if quota <= 0 or not candidate_rows or len(selected_ids) >= target_count:
            return

        available = [
            row for row in candidate_rows if row["element_id"] not in selected_ids
        ]
        if not available:
            return

        take_count = min(quota, len(available), target_count - len(selected_ids))
        if take_count <= 0:
            return

        if len(available) <= take_count:
            for row in available:
                add_element_id(row["element_id"])
            return

        max_index = len(available) - 1
        for offset in range(take_count):
            index = round(offset * max_index / max(1, take_count - 1))
            add_element_id(available[index]["element_id"])

    front_quota = min(target_count, max(50, target_count // 2))
    for row in rows[:front_quota]:
        add_element_id(row["element_id"])

    class_quotas = {
        "title": max(4, target_count // 25),
        "article": max(4, target_count // 25),
        "chapter": max(6, target_count // 20),
        "section": max(4, target_count // 30),
        "compound_id": max(4, target_count // 25),
        "appendix": min(3, max(1, target_count // 60)),
        "preamble": 1,
    }

    for class_name, quota in class_quotas.items():
        class_rows = [
            row
            for row in rows[front_quota:]
            if _classify_scan_candidate(row["text"].split("\n")[0].strip())
            == class_name
        ]
        add_spaced_rows(class_rows, quota)

    heading_like_rows = [
        row
        for row in rows
        if _classify_scan_candidate(row["text"].split("\n")[0].strip())
        == "heading_like"
    ]
    remaining = target_count - len(selected_ids)
    if remaining > 0 and heading_like_rows:
        add_spaced_rows(heading_like_rows, remaining)

    if len(selected_ids) < target_count:
        max_index = len(rows) - 1
        remaining = target_count - len(selected_ids)
        for offset in range(remaining * 2):
            if len(selected_ids) >= target_count:
                break
            index = round(offset * max_index / max(1, remaining * 2 - 1))
            add_element_id(rows[index]["element_id"])

    selected_element_ids = sorted(selected_ids)
    return code_elements.filter(pl.col("element_id").is_in(selected_element_ids))


def _sample_diagnostics(elements_df: pl.DataFrame) -> dict[str, int]:
    """Return compact diagnostic counts for a scan sample."""
    diagnostics = {
        "title": 0,
        "article": 0,
        "chapter": 0,
        "section": 0,
        "appendix": 0,
        "preamble": 0,
        "compound_id": 0,
        "annotation": 0,
        "notes": 0,
        "heading_like": 0,
        "chars": 0,
    }
    for row in elements_df.to_dicts():
        first_line = row["text"].split("\n")[0].strip()
        diagnostics["chars"] += len(first_line)
        label = _classify_scan_candidate(first_line)
        if label in diagnostics and label != "heading_like":
            diagnostics[label] += 1
        if label == "heading_like" or _counts_toward_heading_like_recall(first_line):
            diagnostics["heading_like"] += 1
    return diagnostics


def _reduce_sample_count_after_generation_failure(
    sample_count: int,
    exc: Exception,
) -> int:
    """Shrink the sample after generation failures so retries are materially different."""
    if _is_output_length_error(exc):
        if sample_count <= 40:
            return max(20, sample_count - 8)
        return max(20, int(sample_count * 0.5))

    if _is_context_length_error(exc):
        return max(60, int(sample_count * 0.6))

    lowered = str(exc).lower()
    if "timed out" in lowered or "timeout" in lowered:
        return max(80, int(sample_count * 0.75))

    return max(100, sample_count - 40)


def _reduce_sample_count_after_scoring_retry(sample_count: int) -> int:
    """Shrink the sample modestly after a scored retry so the next attempt differs."""
    if sample_count <= 60:
        return sample_count
    return max(60, int(sample_count * 0.85))


def _is_context_length_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a model context-length failure."""
    if _is_output_length_error(exc):
        return False

    err = str(exc).lower()
    return (
        "maximum context length" in err
        or "context length" in err
        or "input_tokens" in err
        or "max model len" in err
    )


def _completion_finish_reasons(completion: object | None) -> list[str]:
    """Extract finish reasons from provider completion objects when available."""
    reasons: list[str] = []
    choices = getattr(completion, "choices", None)
    if choices is None:
        return reasons

    try:
        for choice in list(choices)[:5]:
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is not None:
                reasons.append(str(finish_reason).lower())
    except Exception:
        return reasons

    return reasons


def _is_output_length_error(exc: Exception) -> bool:
    """Return True if *exc* indicates generation stopped at the output cap."""
    lowered = str(exc).lower()
    if (
        "max_tokens length limit" in lowered
        or "output is incomplete due to a max_tokens length limit" in lowered
        or "finish_reason=length" in lowered
        or "finish reason=length" in lowered
        or "finish_reason 'length'" in lowered
    ):
        return True

    if not isinstance(exc, InstructorRetryException):
        return False

    for reason in _completion_finish_reasons(exc.last_completion):
        if reason == "length":
            return True

    for failed_attempt in exc.failed_attempts or []:
        for reason in _completion_finish_reasons(
            getattr(failed_attempt, "completion", None)
        ):
            if reason == "length":
                return True

    return False


def _summarize_generation_error(exc: Exception) -> str:
    """Condense verbose Instructor/provider errors into short prompt feedback."""
    err = str(exc)
    lowered = err.lower()

    if _is_output_length_error(exc):
        return (
            "Previous attempt hit the output length limit. Return the smallest "
            "valid HeadingStructure JSON object and omit optional fields."
        )

    if "timed out" in lowered or "timeout" in lowered:
        return (
            "Previous attempt timed out before a valid response was returned. "
            "Respond with a compact HeadingStructure JSON object only."
        )

    if _is_context_length_error(exc):
        return (
            "Previous attempt exceeded the model context window. Return only one "
            "compact JSON object with the required top-level keys and no schema metadata."
        )

    if "validation errors for headingstructure" in lowered:
        if "$defs" in err or "properties" in err:
            return (
                "Previous attempt returned a JSON schema wrapper instead of a "
                "HeadingStructure instance. Put `heading_levels`, `total_levels`, "
                "and `file_sample_size` at the top level."
            )
        return (
            "Previous attempt was not a valid HeadingStructure object. Return one "
            "JSON object with all required top-level fields."
        )

    compact = " ".join(err.split())
    return compact[:240]


def _generation_feedback_from_exception(exc: Exception) -> list[str]:
    """Extract concise, de-duplicated feedback from Instructor retry errors."""
    feedback: list[str] = []

    if isinstance(exc, InstructorRetryException):
        failed_attempts = exc.failed_attempts or []
        for failed_attempt in failed_attempts[-3:]:
            message = _summarize_generation_error(failed_attempt.exception)
            if message and message not in feedback:
                feedback.append(message)

    top_level_message = _summarize_generation_error(exc)
    if top_level_message and top_level_message not in feedback:
        feedback.append(top_level_message)

    if not feedback:
        feedback.append(
            "Previous attempt failed to produce valid JSON output. Return one "
            "HeadingStructure JSON object only."
        )

    return feedback[:5]


def _is_timeout_error(exc: Exception) -> bool:
    """Return True if an exception message indicates a timeout."""
    lowered = str(exc).lower()
    return "timed out" in lowered or "timeout" in lowered


def _serialize_completion_debug(completion: object | None) -> dict[str, object] | None:
    """Extract lightweight provider completion metadata when available."""
    if completion is None:
        return None

    snapshot: dict[str, object] = {"type": type(completion).__name__}

    try:
        choices = getattr(completion, "choices", None)
        if choices is not None:
            finish_reasons: list[object] = []
            for choice in list(choices)[:5]:
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason is not None:
                    finish_reasons.append(finish_reason)
            if finish_reasons:
                snapshot["finish_reasons"] = finish_reasons
    except Exception:
        pass

    try:
        usage = getattr(completion, "usage", None)
        if usage is not None:
            if hasattr(usage, "model_dump"):
                snapshot["usage"] = usage.model_dump(mode="json")
            elif isinstance(usage, dict):
                snapshot["usage"] = usage
            else:
                usage_dict = {
                    key: getattr(usage, key)
                    for key in dir(usage)
                    if not key.startswith("_")
                    and isinstance(getattr(usage, key), (int, float, str, type(None)))
                }
                if usage_dict:
                    snapshot["usage"] = usage_dict
    except Exception:
        pass

    return snapshot


def _exception_debug_snapshot(exc: Exception) -> dict[str, object]:
    """Capture compact provider/debug metadata from scan failures."""
    snapshot: dict[str, object] = {
        "type": type(exc).__name__,
        "message": " ".join(str(exc).split())[:500],
        "is_timeout": _is_timeout_error(exc),
        "is_output_length": _is_output_length_error(exc),
        "is_context_length": _is_context_length_error(exc),
    }

    for attr_name in ("status_code", "code", "param"):
        attr_value = getattr(exc, attr_name, None)
        if attr_value is not None:
            snapshot[attr_name] = attr_value

    if isinstance(exc, InstructorRetryException):
        snapshot["n_attempts"] = exc.n_attempts
        if exc.create_kwargs:
            allowed_keys = {
                "model",
                "max_retries",
                "timeout",
                "max_tokens",
                "temperature",
            }
            snapshot["create_kwargs"] = {
                key: value
                for key, value in exc.create_kwargs.items()
                if key in allowed_keys
            }
        if exc.last_completion is not None:
            snapshot["last_completion"] = _serialize_completion_debug(
                exc.last_completion
            )

        failed_attempts: list[dict[str, object]] = []
        for failed_attempt in (exc.failed_attempts or [])[-3:]:
            attempt_snapshot = {
                "attempt_number": failed_attempt.attempt_number,
                "exception_type": type(failed_attempt.exception).__name__,
                "message": " ".join(str(failed_attempt.exception).split())[:500],
                "is_timeout": _is_timeout_error(failed_attempt.exception),
                "is_output_length": _is_output_length_error(failed_attempt.exception),
                "is_context_length": _is_context_length_error(failed_attempt.exception),
            }
            completion_snapshot = _serialize_completion_debug(
                getattr(failed_attempt, "completion", None)
            )
            if completion_snapshot is not None:
                attempt_snapshot["completion"] = completion_snapshot
            failed_attempts.append(attempt_snapshot)
        if failed_attempts:
            snapshot["failed_attempts"] = failed_attempts

    cause = exc.__cause__
    if cause is not None:
        snapshot["cause_type"] = type(cause).__name__
        snapshot["cause_message"] = " ".join(str(cause).split())[:500]

    return snapshot


def _format_exception_debug_summary(exception_debug: dict[str, object]) -> str:
    """Format a one-line summary for stderr/log output."""

    def _append_finish_reasons(
        reasons: list[str], completion_snapshot: object | None
    ) -> None:
        if not isinstance(completion_snapshot, dict):
            return
        finish_reasons = completion_snapshot.get("finish_reasons")
        if not isinstance(finish_reasons, list):
            return
        for finish_reason in finish_reasons:
            if finish_reason is not None:
                reasons.append(str(finish_reason))

    finish_reasons: list[str] = []
    _append_finish_reasons(finish_reasons, exception_debug.get("last_completion"))
    failed_attempts = exception_debug.get("failed_attempts")
    if isinstance(failed_attempts, list):
        for failed_attempt in failed_attempts:
            if isinstance(failed_attempt, dict):
                _append_finish_reasons(
                    finish_reasons,
                    failed_attempt.get("completion"),
                )

    deduped_finish_reasons = list(dict.fromkeys(finish_reasons))
    parts = [
        f"type={exception_debug.get('type', 'UnknownError')}",
        f"timeout={bool(exception_debug.get('is_timeout'))}",
        f"context_length={bool(exception_debug.get('is_context_length'))}",
        "finish_reason="
        + (",".join(deduped_finish_reasons) if deduped_finish_reasons else "unknown"),
    ]
    for key in ("status_code", "code", "cause_type"):
        value = exception_debug.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    message = exception_debug.get("message")
    if isinstance(message, str) and message:
        parts.append(f"message={message[:200]}")
    return " | ".join(parts)


def _serialize_heading_structure(structure: HeadingStructure) -> dict[str, object]:
    """Serialize a heading structure for debug artifacts."""
    return structure.model_dump(mode="json", by_alias=True)


def _write_scan_debug_artifact(
    *,
    debug_output_path: str | Path,
    file_path: str,
    code_elements_height: int,
    code_start_element_id: int | None,
    code_start_line: int | None,
    best_iteration: int,
    best_score: float,
    iteration_records: list[dict[str, object]],
) -> None:
    """Persist per-iteration heading scan diagnostics for later audit/debug use."""
    payload = {
        "file_path": file_path,
        "code_elements": code_elements_height,
        "code_start": {
            "element_id": code_start_element_id,
            "source_line": code_start_line,
        },
        "best_iteration": best_iteration,
        "best_score": best_score,
        "iterations": iteration_records,
    }

    output_path = Path(debug_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(payload, indent=2)
    output_path.write_text(payload_json)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_path = output_path.with_name(
        f"{output_path.stem}_{timestamp}{output_path.suffix}"
    )
    if timestamped_output_path != output_path:
        timestamped_output_path.write_text(payload_json)


# ── System prompt ──────────────────────────────────────────────────────

SCAN_SYSTEM_PROMPT = """\
You are a legal text analyst. You receive raw ELEMENTS from a legal document
and must identify the heading hierarchy.

ELEMENT FORMAT:
- `E{id}: text` — an element (first line shown)
- `E{id}: text  [N lines]` — multi-line element (N total lines)

Most elements are body text (paragraphs, clauses, definitions). Your job is to
identify which elements are HEADINGS and group them by hierarchical level.

TASK: Identify heading elements, group by hierarchical level, and define regex patterns.

RULES:

1. HIERARCHY: level 1 = most general (title/part), increasing = more specific.
   Each level number used exactly once. Up to 8 levels maximum.

2. TOC ENTRIES: Legal documents often have a Table of Contents near the start.
   TOC entries duplicate body headings — use them to confirm patterns, not as
   separate levels. Format variants (Ch./CHAPTER, Sec./SECTION) belong in one
   level's `regex_patterns` list.

3. INFERRED LEVELS: if compound identifiers (e.g. 7-4-010) imply a parent that
   never appears as a heading, mark it `inferred: true` with empty `regex_patterns`.
   Inferred parents get LOWER level numbers than the children they were deduced from.

3a. VARIANT CONSOLIDATION: if the same logical heading level appears in multiple
    text forms, keep it as one level and place the alternate regexes in
    `regex_patterns`. Do not split levels solely because one example uses Roman
    numerals, another uses alphanumeric identifiers, or one form has punctuation
    like `§`, `.`, or `-`.

4. REGEX PATTERNS:
   - Anchor with `^`, single-line only (no `\\n`)
   - No capturing groups — use `(?:...)` for grouping
   - Patterns must be unique across levels
   - Handle case/format variants in one level's list
   - End with `.*$` or `(?:\\s+.*)?$` as appropriate

5. MARKDOWN PREFIX: optional. If omitted, the pipeline assigns `#`, `##`, `###`,
    or `####` based on normalized level order. Levels 5-8 all use `####`.

6. EXAMPLE_HEADING: complete verbatim text from the elements (not abbreviated).

7. TYPE_LABEL: short lowercase label ("title", "chapter", "section", etc.).

8. NUMBER_REGEX: regex for just the identifier portion, no anchors. null if none.

9. MULTILINE: true if heading keyword is on one line and title on the next.

10. BODY TEXT: Most elements are NOT headings. Do not assign body paragraphs,
    enumerated clauses like `(a)`, `(1)`, or `(i)`, or definitions to heading levels.
    Only structural division markers (titles, chapters, articles, sections, parts, etc.)
    are headings.

OUTPUT REQUIREMENTS:
- Return exactly one JSON object representing a HeadingStructure instance.
- `heading_levels` is required.
- `total_levels`, `file_sample_size`, `toc_line_ranges`, `outline_warnings`,
    `quality_score`, and `iterations` are optional and may be omitted for brevity.
- Each item in `heading_levels` should include `level`, `example_heading`,
    `type_label`, and either `regex_patterns` or `regex_pattern`.
- `markdown_prefix`, `number_regex`, `multiline`, and `inferred` are optional
    when unknown or unnecessary.
- Do not return JSON Schema or metadata. Never include keys like `$defs`, `properties`,
    `required`, `title`, `type`, or `description`.
- Do not wrap the answer inside `properties` or any other container.
- No commentary, no Markdown fences, no prose.

OUTPUT TEMPLATE:
{
    "heading_levels": [...]
}"""


# ── Verification ───────────────────────────────────────────────────────


def _verify_compile_patterns(
    structure: HeadingStructure,
) -> tuple[list[tuple[int, "re.Pattern[str]", str]], list[str]]:
    compiled: list[tuple[int, re.Pattern[str], str]] = []
    warnings: list[str] = []
    for level in structure.levels:
        if level.inferred:
            continue
        for pat_str in level.regex_patterns:
            try:
                c = re.compile(pat_str, re.IGNORECASE | re.MULTILINE)
                compiled.append((level.level, c, pat_str))
            except re.error as e:
                warnings.append(f"Level {level.level}: invalid regex '{pat_str}': {e}")
    return compiled, warnings


def _pattern_matches_element(pattern: "re.Pattern[str]", element_text: str) -> bool:
    """Return True when a regex matches either the first line or joined element text."""
    first_line = element_text.split("\n")[0].strip()
    if first_line and pattern.match(first_line):
        return True

    joined = " ".join(element_text.split())
    return bool(joined and pattern.match(joined))


_BODY_CUE_PAT = re.compile(
    r"\b(?:shall|must|may|means|include(?:s|d)?|require(?:s|d)?|"
    r"prohibit(?:s|ed)?|provide(?:s|d)?|govern(?:s|ed)?|apply|applies|"
    r"applicable|subject\s+to|pursuant\s+to)\b",
    re.IGNORECASE,
)


def _keyword_prefix_for_level(level: HeadingLevel) -> str | None:
    """Return the canonical heading keyword prefix for a structural level."""
    keyword_map = {
        "title": "TITLE",
        "article": "ARTICLE",
        "chapter": "CHAPTER",
        "section": "SECTION",
        "appendix": "APPENDIX",
        "preamble": "PREAMBLE",
    }
    return keyword_map.get(level.type_label.lower().strip())


def _split_heading_identifier_and_remainder(
    line: str,
    level: HeadingLevel,
) -> tuple[str | None, str]:
    """Split a heading line into identifier token and trailing title portion."""
    stripped = line.strip()
    if not stripped:
        return None, ""

    working = stripped
    keyword_prefix = _keyword_prefix_for_level(level)
    if keyword_prefix and re.match(rf"^{keyword_prefix}\b", working, re.IGNORECASE):
        keyword_match = re.match(rf"^{keyword_prefix}\s+", working, re.IGNORECASE)
        if keyword_match:
            working = working[keyword_match.end() :]
    elif working.startswith("§"):
        symbol_match = re.match(r"^§\s*", working)
        if symbol_match:
            working = working[symbol_match.end() :]

    token_match = re.match(r"[A-Z0-9]+(?:[-.][A-Z0-9]+)*", working, re.IGNORECASE)
    if not token_match:
        return None, working

    return token_match.group(0), working[token_match.end() :]


def _delimiter_family(remainder: str) -> str:
    """Classify the delimiter that follows a heading identifier."""
    if not remainder:
        return "none"
    if remainder.startswith("."):
        return "dot"
    if remainder.startswith(":"):
        return "colon"
    if re.match(r"^\s{2,}", remainder):
        return "multi_space"
    if re.match(r"^\s*[\-\u2013\u2014]\s*", remainder):
        return "dash"
    if re.match(r"^\s+", remainder):
        return "single_space"
    return "attached"


def _normalized_heading_tail(remainder: str) -> str:
    """Return the title-like tail of a heading after delimiter cleanup."""
    return remainder.lstrip(" .:-\t\u2013\u2014")


def _starts_with_upper_alpha(text: str) -> bool:
    """Return True when the first alphabetic character in *text* is uppercase."""
    for char in text:
        if char.isalpha():
            return char.isupper()
    return False


def _lowercase_initial_ratio(text: str) -> float:
    """Return the fraction of words in *text* that begin with lowercase letters."""
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", text)
    if not words:
        return 0.0
    lowercase_initial = sum(1 for word in words if word[0].islower())
    return lowercase_initial / len(words)


def _looks_body_like_heading_tail(
    tail_text: str,
    *,
    expect_initial_upper: bool,
) -> bool:
    """Return True when a matched heading tail looks more like body text than a title."""
    if not tail_text:
        return False

    if expect_initial_upper and not _starts_with_upper_alpha(tail_text):
        return True

    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", tail_text)
    if len(words) < 6:
        return False

    lowercase_ratio = _lowercase_initial_ratio(tail_text)
    if lowercase_ratio >= 0.6 and _BODY_CUE_PAT.search(tail_text):
        return True

    return False


def _structural_precision_score(
    elements_df: pl.DataFrame,
    structure: HeadingStructure,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
) -> tuple[list[str], float]:
    """Score whether matched headings look structurally consistent with their level."""
    warnings: list[str] = []
    level_scores: list[float] = []
    element_rows = elements_df.to_dicts()

    for level in structure.levels:
        if level.inferred:
            continue

        matched_lines: list[str] = []
        for row in element_rows:
            element_text = row["text"]
            if level.level in _matched_levels_for_element(element_text, compiled):
                first_line = element_text.split("\n")[0].strip()
                if first_line:
                    matched_lines.append(first_line)

        if not matched_lines:
            continue

        _example_identifier, example_remainder = (
            _split_heading_identifier_and_remainder(
                level.example_heading,
                level,
            )
        )
        expected_delimiter = _delimiter_family(example_remainder)
        example_tail = _normalized_heading_tail(example_remainder)
        expect_initial_upper = _starts_with_upper_alpha(example_tail)

        delimiter_mismatches = 0
        body_like_matches = 0
        match_scores: list[float] = []
        for line in matched_lines:
            _identifier, remainder = _split_heading_identifier_and_remainder(
                line, level
            )
            actual_delimiter = _delimiter_family(remainder)
            tail_text = _normalized_heading_tail(remainder)
            line_score = 1.0

            if expected_delimiter in {"multi_space", "colon", "dash"} and (
                actual_delimiter != expected_delimiter
            ):
                delimiter_mismatches += 1
                line_score -= 0.45

            if _looks_body_like_heading_tail(
                tail_text,
                expect_initial_upper=expect_initial_upper,
            ):
                body_like_matches += 1
                line_score -= 0.55

            match_scores.append(max(0.0, line_score))

        level_score = sum(match_scores) / len(match_scores)
        level_scores.append(level_score)

        if level_score < 0.85 and (delimiter_mismatches > 0 or body_like_matches > 0):
            warnings.append(
                f"Low structural precision at level {level.level}: {delimiter_mismatches} delimiter mismatches and "
                f"{body_like_matches} body-like matches across {len(matched_lines)} matched elements "
                f"(score {level_score:.0%})"
            )

    if not level_scores:
        return warnings, 1.0

    return warnings, sum(level_scores) / len(level_scores)


def _classify_generation_failure(exc: Exception) -> str:
    """Classify generation failures into a small set of retry-relevant buckets."""
    if _is_output_length_error(exc):
        return "output_length"
    if _is_context_length_error(exc):
        return "context_length"
    if _is_timeout_error(exc):
        return "timeout"

    lowered = str(exc).lower()
    if "validation errors for headingstructure" in lowered:
        return "schema_validation"
    return "other_generation"


def _classify_scored_structure_errors(errors: list[str]) -> list[dict[str, object]]:
    """Group scored-iteration errors into compact retry buckets."""
    category_order = [
        "low_recall",
        "structural_precision",
        "ambiguity",
        "pattern_validity",
        "parent_child",
        "sibling_ordering",
    ]
    counts: dict[str, int] = {}
    samples: dict[str, str] = {}

    for error in errors:
        lowered = error.lower()
        category: str | None = None
        if lowered.startswith("low recall"):
            category = "low_recall"
        elif "structural precision" in lowered:
            category = "structural_precision"
        elif "ambiguous" in lowered:
            category = "ambiguity"
        elif (
            "pattern has 0 matches" in lowered
            or "invalid regex" in lowered
            or "no elements matched" in lowered
        ):
            category = "pattern_validity"
        elif "parent-child mismatch" in lowered:
            category = "parent_child"
        elif "out-of-order siblings" in lowered:
            category = "sibling_ordering"

        if category is None:
            continue

        counts[category] = counts.get(category, 0) + 1
        samples.setdefault(category, error)

    classified: list[dict[str, object]] = []
    for category in category_order:
        count = counts.get(category)
        if not count:
            continue
        message = " ".join(samples[category].split())
        if count > 1 and category in {
            "ambiguity",
            "pattern_validity",
            "parent_child",
            "sibling_ordering",
        }:
            message = f"{message} (+{count - 1} more)"
        classified.append(
            {
                "category": category,
                "count": count,
                "message": message[:180],
            }
        )
    return classified


def _build_scored_retry_feedback(classified_errors: list[dict[str, object]]) -> str:
    """Build a compact scored-retry footer from classified structure errors."""
    if not classified_errors:
        return ""

    lines = ["RETRY_FEEDBACK:"]
    for item in classified_errors[:3]:
        lines.append(f"- {item['category']}: {item['message']}")
    return "\n".join(lines)


def _matched_levels_for_element(
    element_text: str,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
) -> list[int]:
    """Return all heading levels whose regexes match an element."""
    return sorted(
        set(
            level
            for level, pattern, _ in compiled
            if _pattern_matches_element(pattern, element_text)
        )
    )


def _identifier_sort_key(
    identifier: str,
) -> tuple[tuple[int, int | str], ...] | None:
    """Build a natural sort key for identifiers like 1-100, 2-3, or A-10."""
    parts = re.findall(r"\d+|[A-Za-z]+", identifier)
    if not parts:
        return None

    key: list[tuple[int, int | str]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def _dedupe_patterns(patterns: list[str]) -> list[str]:
    """Preserve pattern order while removing duplicates and blanks."""
    seen: set[str] = set()
    ordered: list[str] = []
    for pattern in patterns:
        stripped = pattern.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        ordered.append(stripped)
    return ordered


def _extract_identifier_token(example: str, keyword: str | None = None) -> str | None:
    """Extract the first identifier token from a heading example."""
    if keyword is not None:
        match = re.match(
            rf"^{re.escape(keyword)}\s+([A-Z0-9]+(?:[-.][A-Z0-9]+)*)\.?(?=\s|$)",
            example,
            re.IGNORECASE,
        )
    else:
        match = re.match(
            r"^(?:§\s*)?([A-Z0-9]+(?:[-.][A-Z0-9]+)*)\.?(?=\s|$)",
            example,
            re.IGNORECASE,
        )
    if not match:
        return None
    return match.group(1).rstrip(".:;")


def _identifier_pattern_from_token(token: str) -> str:
    """Generalize a heading identifier token into a regex fragment."""
    cleaned = token.strip().upper().rstrip(".:;")
    if not cleaned:
        return r"[A-Z0-9]+(?:[-.][A-Z0-9]+)*"
    if re.fullmatch(r"[IVXLCDM]+", cleaned, re.IGNORECASE):
        return r"[IVXLCDM]+"
    if re.fullmatch(r"\d+(?:\.\d+)*", cleaned):
        return r"\d+(?:\.\d+)*"
    if re.fullmatch(r"[A-Z]", cleaned, re.IGNORECASE):
        return r"[A-Z]"
    if re.fullmatch(r"[A-Z]+", cleaned, re.IGNORECASE):
        return r"[A-Z]+"
    if re.fullmatch(r"[A-Z0-9]+(?:[-.][A-Z0-9]+)+", cleaned, re.IGNORECASE):
        return r"[A-Z0-9]+(?:[-.][A-Z0-9]+)+"
    if re.fullmatch(r"[A-Z0-9]+", cleaned, re.IGNORECASE):
        return r"[A-Z0-9]+"
    return r"[A-Z0-9]+(?:[-.][A-Z0-9]+)*"


def _update_level_patterns(
    level: HeadingLevel,
    refined_patterns: list[str],
    *,
    number_regex: str | None = None,
) -> None:
    """Merge refined patterns ahead of existing variants for a level."""
    existing_patterns = list(level.regex_patterns or [])
    if level.regex_pattern:
        existing_patterns.insert(0, level.regex_pattern)

    merged = _dedupe_patterns(refined_patterns + existing_patterns)
    if merged:
        level.regex_patterns = merged
        level.regex_pattern = merged[0]
    if number_regex is not None:
        level.number_regex = number_regex


def _apply_example_based_pattern_refinement(level: HeadingLevel) -> None:
    """Tighten obvious heading regexes using the example heading without losing variants."""
    example = level.example_heading.strip()
    label = level.type_label.lower().strip()

    if label == "title":
        token = _extract_identifier_token(example, "TITLE")
        if token:
            number_pattern = _identifier_pattern_from_token(token)
            _update_level_patterns(
                level,
                [rf"^TITLE\s+{number_pattern}\.?(?:\s+.*)?$"],
                number_regex=number_pattern,
            )
        return

    if label == "article":
        token = _extract_identifier_token(example, "ARTICLE")
        if token:
            number_pattern = _identifier_pattern_from_token(token)
            _update_level_patterns(
                level,
                [rf"^ARTICLE\s+{number_pattern}\.?(?:\s+.*)?$"],
                number_regex=number_pattern,
            )
        return

    if label == "chapter":
        token = _extract_identifier_token(example, "CHAPTER")
        if token:
            number_pattern = _identifier_pattern_from_token(token)
            _update_level_patterns(
                level,
                [rf"^CHAPTER\s+{number_pattern}\.?(?:\s+.*)?$"],
                number_regex=number_pattern,
            )
        return

    if label in {"section", "code_section"}:
        if re.match(r"^SECTION\b", example, re.IGNORECASE):
            token = _extract_identifier_token(example, "SECTION")
            if token:
                number_pattern = _identifier_pattern_from_token(token)
                _update_level_patterns(
                    level,
                    [rf"^SECTION\s+{number_pattern}\.?(?:\s+.*)?$"],
                    number_regex=number_pattern,
                )
            return

        token = _extract_identifier_token(example)
        if token:
            number_pattern = _identifier_pattern_from_token(token)
            _update_level_patterns(
                level,
                [rf"^(?:§\s*)?{number_pattern}(?:\.\s*.*|\:\s*.*|\s+.*)$"],
                number_regex=number_pattern,
            )
        return

    if label == "appendix" or _APPENDIX_HEADING_PAT.match(example):
        token = _extract_identifier_token(example, "APPENDIX")
        if token:
            number_pattern = _identifier_pattern_from_token(token)
            refined = rf"^APPENDIX\s+{number_pattern}\.?(?:\s+.*)?$"
        else:
            refined = r"^APPENDIX(?:\s+.*)?$"
            number_pattern = None
        _update_level_patterns(level, [refined], number_regex=number_pattern)
        return

    if label == "preamble" or _PREAMBLE_HEADING_PAT.match(example):
        _update_level_patterns(level, [r"^PREAMBLE(?:\s+.*)?$"])


def _normalize_scanned_structure(structure: HeadingStructure) -> HeadingStructure:
    """Apply conservative post-processing to LLM output before scoring."""
    structure.levels = sorted(structure.levels, key=lambda level: level.level)

    for new_level, level in enumerate(structure.levels, start=1):
        level.level = new_level
        _apply_example_based_pattern_refinement(level)
        level.markdown_prefix = "#" * min(new_level, 4)

    structure.total_levels = len(structure.levels)
    return structure


def _check_completeness(
    elements_df: pl.DataFrame,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
) -> list[str]:
    """Check elements for ambiguous pattern matches."""
    warnings: list[str] = []
    ambiguous = 0
    for row in elements_df.to_dicts():
        eid = row["element_id"]
        element_text = row["text"]
        first_line = element_text.split("\n")[0].strip()
        if not first_line:
            continue
        matching_levels = _matched_levels_for_element(element_text, compiled)
        if len(matching_levels) > 1:
            if ambiguous < 10:
                warnings.append(
                    f"Ambiguous match E{eid}: levels {matching_levels}: {first_line[:60]}"
                )
            ambiguous += 1

    if ambiguous > 10:
        warnings.append(f"... and {ambiguous - 10} more ambiguous elements")
    return warnings


def _check_parent_child(
    structure: HeadingStructure,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
    elements_df: pl.DataFrame,
) -> list[str]:
    """Check parent-child ID relationships in element texts."""
    warnings: list[str] = []
    sep_pat = re.compile(r"\b(\d+)([.\-])(\d+)")
    separator = None
    for level in structure.levels:
        if level.inferred:
            continue
        m = sep_pat.search(level.example_heading)
        if m:
            separator = m.group(2)
            break

    if not separator:
        return warnings

    # Collect IDs per level from element texts
    element_rows = elements_df.to_dicts()

    level_ids: dict[int, list[str]] = {}
    for level in structure.levels:
        if level.inferred or not level.number_regex:
            continue
        try:
            num_pat = re.compile(level.number_regex)
        except re.error:
            continue
        ids = []
        for _lvl, pat, _ in compiled:
            if _lvl != level.level:
                continue
            for row in element_rows:
                element_text = row["text"]
                first_line = element_text.split("\n")[0].strip()
                if _pattern_matches_element(pat, element_text):
                    nm = num_pat.search(first_line)
                    if nm:
                        ids.append(nm.group(0))
        level_ids[level.level] = ids

    sorted_levels = sorted(level_ids.keys())
    for idx in range(1, len(sorted_levels)):
        child_lvl = sorted_levels[idx]
        parent_lvl = sorted_levels[idx - 1]
        child_ids = level_ids.get(child_lvl, [])
        parent_ids = set(level_ids.get(parent_lvl, []))
        if not parent_ids or not child_ids:
            continue
        for cid in child_ids[:20]:
            parts = cid.rsplit(separator, 1)
            if len(parts) == 2:
                prefix = parts[0]
                if prefix and prefix not in parent_ids:
                    warnings.append(
                        f"Parent-child mismatch: child {cid} (level {child_lvl}) "
                        f"prefix '{prefix}' not found in level {parent_lvl} ids"
                    )
                    break
    return warnings


def _check_sibling_ordering(
    structure: HeadingStructure,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
    elements_df: pl.DataFrame,
) -> list[str]:
    """Check sibling ordering across element texts."""
    warnings: list[str] = []
    element_rows = elements_df.to_dicts()

    for level in structure.levels:
        if level.inferred or not level.number_regex:
            continue
        try:
            num_pat = re.compile(level.number_regex)
        except re.error:
            continue
        prev_id: str | None = None
        prev_key: tuple[tuple[int, int | str], ...] | None = None
        for row in element_rows:
            element_text = row["text"]
            first_line = element_text.split("\n")[0].strip()
            if not first_line:
                continue

            matching_levels = _matched_levels_for_element(element_text, compiled)
            if any(matched_level < level.level for matched_level in matching_levels):
                prev_id = None
                prev_key = None

            matched_this_level = level.level in matching_levels
            if not matched_this_level:
                continue
            nm = num_pat.search(first_line)
            if not nm:
                continue
            current_id = nm.group(0)
            current_key = _identifier_sort_key(current_id)
            if prev_id is not None and prev_key is not None and current_key is not None:
                if current_key < prev_key:
                    warnings.append(
                        f"Out-of-order siblings at level {level.level}: "
                        f"'{current_id}' after '{prev_id}'"
                    )
            prev_id = current_id
            prev_key = current_key
    return warnings


def verify_structure(
    structure: HeadingStructure,
    elements_df: pl.DataFrame,
) -> list[str]:
    """Verify the LLM's heading structure against elements."""
    compiled, warnings = _verify_compile_patterns(structure)

    warnings.extend(_check_completeness(elements_df, compiled))
    warnings.extend(_check_parent_child(structure, compiled, elements_df))
    warnings.extend(_check_sibling_ordering(structure, compiled, elements_df))

    for _lvl, pat, pat_str in compiled:
        if not any(
            _pattern_matches_element(pat, row["text"]) for row in elements_df.to_dicts()
        ):
            warnings.append(f"Pattern has 0 matches in full text: {pat_str[:70]}")

    return warnings


# ── Quality scoring ────────────────────────────────────────────────────


def score_structure(
    elements_df: pl.DataFrame,
    structure: HeadingStructure,
    *,
    outline_elements_df: pl.DataFrame | None = None,
) -> tuple[float, list[str]]:
    """Compute a 0.0-1.0 quality score and return error messages."""
    del outline_elements_df

    compiled, compile_warnings = _verify_compile_patterns(structure)
    errors = list(compile_warnings)

    # If all patterns failed to compile, score is 0
    if compile_warnings and not compiled:
        return 0.0, errors

    # Count elements matched by patterns and heading-like elements for recall
    matched_count = 0
    ambiguous_count = 0
    heading_like_count = 0
    heading_like_matched = 0
    for row in elements_df.to_dicts():
        element_text = row["text"]
        first_line = element_text.split("\n")[0].strip()
        if not first_line:
            continue
        is_hl = _counts_toward_heading_like_recall(first_line)
        if is_hl:
            heading_like_count += 1
        matching = _matched_levels_for_element(element_text, compiled)
        if len(matching) >= 1:
            matched_count += 1
            if is_hl:
                heading_like_matched += 1
        if len(matching) > 1:
            ambiguous_count += 1

    # If patterns match nothing at all, score is 0 — patterns are wrong
    if matched_count == 0:
        errors.append("No elements matched any pattern")
        return 0.0, errors

    # Precision (0.15) — matched exactly once / total matched
    exactly_one = matched_count - ambiguous_count
    precision = exactly_one / matched_count if matched_count > 0 else 1.0

    # Recall (0.25) — fraction of heading-like elements captured by patterns
    if heading_like_count > 0:
        recall = heading_like_matched / heading_like_count
    else:
        recall = 1.0 if matched_count > 0 else 0.0
    if recall < 1.0:
        errors.append(
            f"Low recall: patterns matched {heading_like_matched} of "
            f"{heading_like_count} heading-like elements ({recall:.0%})"
        )

    # Pattern validity (0.15) — fraction of non-inferred patterns matching >= 1 element
    valid_patterns = 0
    total_patterns = 0
    for _lvl, pat, pat_str in compiled:
        total_patterns += 1
        if any(
            _pattern_matches_element(pat, row["text"]) for row in elements_df.to_dicts()
        ):
            valid_patterns += 1
        else:
            errors.append(f"Pattern has 0 matches: {pat_str[:70]}")
    pattern_validity = valid_patterns / total_patterns if total_patterns > 0 else 1.0

    # Sibling ordering (0.1)
    sibling_warnings = _check_sibling_ordering(structure, compiled, elements_df)
    out_of_order = len(sibling_warnings)
    errors.extend(sibling_warnings)
    total_sibling_pairs = max(1, matched_count - len(structure.levels))
    sibling_score = max(0.0, 1.0 - out_of_order / total_sibling_pairs)

    # No ambiguity (0.1)
    ambiguity_score = (
        1.0 - ambiguous_count / matched_count if matched_count > 0 else 1.0
    )

    # Parent-child (0.1)
    pc_warnings = _check_parent_child(structure, compiled, elements_df)
    errors.extend(pc_warnings)
    pc_score = 0.0 if pc_warnings else 1.0

    # Structural precision (0.15)
    structural_precision_warnings, structural_precision = _structural_precision_score(
        elements_df,
        structure,
        compiled,
    )
    errors.extend(structural_precision_warnings)

    # Completeness warnings for error feedback
    completeness_warnings = _check_completeness(elements_df, compiled)
    errors.extend(completeness_warnings)

    weighted_score = (
        0.15 * precision
        + 0.25 * recall
        + 0.15 * pattern_validity
        + 0.1 * sibling_score
        + 0.1 * ambiguity_score
        + 0.1 * pc_score
        + 0.15 * structural_precision
    )
    score = weighted_score

    # Quality gates: cap score when critical metrics are poor
    if recall < 0.5:
        score = min(score, recall + 0.3)
    if structural_precision < 0.6:
        score = min(score, structural_precision + 0.25)

    return score, errors


# ── Iterative scan loop ───────────────────────────────────────────────


def scan_headings(
    file_path: str,
    client: Instructor | None = None,
    max_iterations: int | None = None,
    score_threshold: float | None = None,
    debug_output_path: str | Path | None = None,
) -> tuple[HeadingStructure, float, int]:
    """Iteratively scan legal text with a self-correcting feedback loop.

    Returns the best normalized heading structure found, along with the score
    and iteration count. The returned structure also includes the detected
    ``code_start_element_id`` and ``code_start_line`` used by parse output.
    """
    from loguru import logger

    if client is None:
        client = Config.get_powerful_client()

    if max_iterations is None:
        max_iterations = _get_scan_max_iterations()
    if score_threshold is None:
        score_threshold = _get_scan_score_threshold()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Split file into elements
    elements_df = split_elements(file_path)
    if elements_df.height == 0:
        raise ValueError(f"File is empty: {file_path}")

    # Find code start using element-based scanner
    code_start = find_code_start(client, elements_df)
    logger.info(f"Code starts at element {code_start.element_id}")
    code_elements = elements_df.filter(pl.col("element_id") >= code_start.element_id)

    sample_count = _get_scan_initial_sample_count()
    llm_timeout_seconds = _get_scan_timeout_seconds()
    scan_create_max_retries = _get_scan_create_max_retries()
    scan_max_tokens = _get_scan_max_tokens()
    best_structure: HeadingStructure | None = None
    best_score = 0.0
    best_iteration = 0
    last_generation_error: list[str] = []
    scored_retry_feedback = ""
    iteration_records: list[dict[str, object]] = []

    for iteration in range(1, max_iterations + 1):
        logger.info(
            f"Iteration {iteration}/{max_iterations}, sample_count={sample_count}"
        )

        # Phase 1: Format raw elements for LLM
        scan_count = min(sample_count, code_elements.height)
        sample_elements = _select_scan_sample(code_elements, scan_count)
        raw_text = _format_raw_elements(sample_elements)
        variant_guidance = _build_scan_variant_guidance(sample_elements)
        sample_stats = _sample_diagnostics(sample_elements)
        sample_element_ids = sample_elements["element_id"].to_list()
        logger.info(
            "Iteration {} sample spans E{}-E{} with {} elements (chars={}, titles={}, articles={}, chapters={}, sections={}, notes={}, heading_like={})",
            iteration,
            sample_element_ids[0],
            sample_element_ids[-1],
            sample_elements.height,
            sample_stats["chars"],
            sample_stats["title"],
            sample_stats["article"],
            sample_stats["chapter"],
            sample_stats["section"],
            sample_stats["notes"],
            sample_stats["heading_like"],
        )
        iteration_record: dict[str, object] = {
            "iteration": iteration,
            "sample_count": sample_count,
            "sampled_elements": sample_elements.height,
            "sample_element_start_id": sample_element_ids[0],
            "sample_element_end_id": sample_element_ids[-1],
            "sample_diagnostics": dict(sample_stats),
            "prompt_mode": "standard",
            "raw_text_chars": len(raw_text),
            "variant_guidance_chars": len(variant_guidance),
            "llm_timeout_seconds": llm_timeout_seconds,
            "scan_max_tokens": scan_max_tokens,
            "scan_create_max_retries": scan_create_max_retries,
        }

        # Phase 2: LLM call
        user_prompt = (
            f"Analyze the heading structure in these legal text elements:\n\n"
            f"{raw_text}\n\n"
            f"These are {sample_elements.height} representative elements from the document "
            f"({code_elements.height} total).\n"
            f"Identify which elements are headings, group by level, and create regex "
            f"patterns.\n"
            f"When the same logical level appears in multiple observed formats, return "
            f"multiple entries in `regex_patterns` for that level instead of splitting it "
            f"into separate levels.\n"
            f"Keep the JSON compact and omit optional fields when possible.\n"
            f"Return a single JSON object only. Use `heading_levels` as the top-level "
            f"array key. Do not return schema keys like `$defs` or `properties`.\n"
        )
        if variant_guidance:
            user_prompt += f"\n{variant_guidance}"
        if scored_retry_feedback:
            user_prompt += f"\n\n{scored_retry_feedback}\n"
        iteration_record["user_prompt_chars"] = len(user_prompt)
        iteration_record["system_prompt_chars"] = len(SCAN_SYSTEM_PROMPT)
        if scored_retry_feedback:
            iteration_record["retry_feedback"] = scored_retry_feedback

        try:
            structure = create_structured_completion(
                client=client,
                messages=[
                    {
                        "role": "system",
                        "content": SCAN_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_model=HeadingStructure,
                retry_label="parse scan headings",
                **Config.get_llm_params(
                    max_retries=scan_create_max_retries,
                    timeout=llm_timeout_seconds,
                    max_tokens=scan_max_tokens,
                ),
            )
            structure = _normalize_scanned_structure(structure)
        except Exception as exc:
            generation_feedback = _generation_feedback_from_exception(exc)
            last_generation_error = generation_feedback
            exception_debug = _exception_debug_snapshot(exc)
            logger.warning(
                "Iteration {} failed before scoring: {}",
                iteration,
                generation_feedback[0],
            )
            logger.warning(
                "Iteration {} exception_debug: {}",
                iteration,
                _format_exception_debug_summary(exception_debug),
            )
            iteration_record.update(
                {
                    "status": "generation_error",
                    "generation_feedback": generation_feedback,
                    "exception_debug": exception_debug,
                    "retry_classification": {
                        "kind": "generation",
                        "category": _classify_generation_failure(exc),
                    },
                }
            )
            iteration_records.append(iteration_record)
            scored_retry_feedback = ""
            reduced_sample_count = _reduce_sample_count_after_generation_failure(
                sample_count,
                exc,
            )
            if reduced_sample_count < sample_count:
                logger.warning(
                    "Reducing sample_count from {} to {} after generation failure",
                    sample_count,
                    reduced_sample_count,
                )
                sample_count = reduced_sample_count
            continue

        # Phase 3: Evaluate on full code elements
        score, errors = score_structure(
            code_elements,
            structure,
        )
        logger.info(f"Iteration {iteration}: score={score:.3f}, errors={len(errors)}")
        classified_scored_errors: list[dict[str, object]] = []
        next_retry_feedback = ""
        if score < score_threshold:
            classified_scored_errors = _classify_scored_structure_errors(errors)
            next_retry_feedback = _build_scored_retry_feedback(classified_scored_errors)
        iteration_record.update(
            {
                "status": "scored",
                "score": score,
                "error_count": len(errors),
                "errors": errors,
                "generated_structure": _serialize_heading_structure(structure),
            }
        )
        if classified_scored_errors:
            iteration_record["retry_classification"] = {
                "kind": "scored",
                "categories": [
                    str(item["category"]) for item in classified_scored_errors
                ],
            }
        iteration_records.append(iteration_record)

        if score > best_score or best_structure is None:
            best_score = score
            best_structure = structure
            best_structure.toc_line_ranges = []
            best_iteration = iteration

        if score >= score_threshold:
            break

        scored_retry_feedback = next_retry_feedback

        reduced_sample_count = _reduce_sample_count_after_scoring_retry(sample_count)
        if reduced_sample_count < sample_count:
            logger.info(
                "Reducing sample_count from {} to {} for next retry after score {:.3f}",
                sample_count,
                reduced_sample_count,
                score,
            )
            sample_count = reduced_sample_count
    sample_count = min(code_elements.height, sample_count + 50)

    if best_structure is None:
        if debug_output_path is not None:
            _write_scan_debug_artifact(
                debug_output_path=debug_output_path,
                file_path=file_path,
                code_elements_height=code_elements.height,
                code_start_element_id=code_start.element_id,
                code_start_line=code_start.start_line,
                best_iteration=best_iteration,
                best_score=best_score,
                iteration_records=iteration_records,
            )
        detail = " ".join(last_generation_error).strip()
        if detail:
            raise RuntimeError(
                "Failed to generate heading structure after "
                f"{max_iterations} attempts. {detail}"
            )
        raise RuntimeError(
            f"Failed to generate heading structure after {max_iterations} attempts."
        )

    # Finalize
    verification_warnings = verify_structure(best_structure, code_elements)
    best_structure.outline_warnings = verification_warnings
    best_structure.quality_score = best_score
    best_structure.iterations = best_iteration
    best_structure.code_start_element_id = code_start.element_id
    best_structure.code_start_line = code_start.start_line
    if best_structure.total_levels != len(best_structure.levels):
        best_structure.total_levels = len(best_structure.levels)
    best_structure.file_sample_size = code_elements.height

    if debug_output_path is not None:
        _write_scan_debug_artifact(
            debug_output_path=debug_output_path,
            file_path=file_path,
            code_elements_height=code_elements.height,
            code_start_element_id=code_start.element_id,
            code_start_line=code_start.start_line,
            best_iteration=best_iteration,
            best_score=best_score,
            iteration_records=iteration_records,
        )

    return best_structure, best_score, best_iteration


def scan_legal_text(
    client: Instructor,
    file_path: str,
    max_lines: int = DEFAULT_SCAN_MAX_LINES,
    model: str | None = None,
    debug_output_path: str | Path | None = None,
) -> HeadingStructure:
    """Analyze legal text to identify heading structure and patterns.

    Delegates to ``scan_headings()`` for self-correcting multi-pass analysis
    and returns a structure enriched with the detected start of code proper.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    structure, score, iterations = scan_headings(
        file_path=file_path,
        client=client,
        max_iterations=None,
        score_threshold=None,
        debug_output_path=debug_output_path,
    )
    return structure
