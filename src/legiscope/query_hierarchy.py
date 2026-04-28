"""Helpers for normalized parent/child query hierarchy metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRES_YES_COLUMN = 'Requires "yes" from upstream question:'
REQUIRES_DATA_COLUMN = "Requires data from upstream question:"
REQUIRES_LABELS_COLUMN = "Requires label(s) from upstream question:"


def _dedupe_preserve_order(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _split_pipe_delimited_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, str):
        value = str(value)
    return _dedupe_preserve_order(value.split("||"))


@dataclass(frozen=True)
class LabelBlockerRule:
    """A child query only runs when the parent answer matches one of these labels."""

    parent_query_id: str
    blocker_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryHierarchy:
    """Normalized dependency metadata attached to a single query row."""

    query_id: str
    parent_ids: tuple[str, ...] = ()
    boolean_parent_ids: tuple[str, ...] = ()
    context_parent_ids: tuple[str, ...] = ()
    label_blockers: tuple[LabelBlockerRule, ...] = ()
    pass_parent_question: bool = True
    pass_parent_short_answer: bool = True
    inherit_parent_retrieval: bool = False

    def has_dependencies(self) -> bool:
        return bool(
            self.parent_ids
            or self.boolean_parent_ids
            or self.context_parent_ids
            or self.label_blockers
        )


def _parse_label_blockers(value: Any) -> tuple[LabelBlockerRule, ...]:
    if value is None:
        return ()
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    if not text:
        return ()

    rules: list[LabelBlockerRule] = []
    for raw_rule in text.split(";;"):
        rule_text = raw_rule.strip()
        if not rule_text or "=>" not in rule_text:
            continue
        parent_query_id, label_text = rule_text.split("=>", 1)
        parent_query_id = parent_query_id.strip()
        blocker_labels = _dedupe_preserve_order(label_text.split("||"))
        if not parent_query_id or not blocker_labels:
            continue
        rules.append(
            LabelBlockerRule(
                parent_query_id=parent_query_id,
                blocker_labels=blocker_labels,
            )
        )
    return tuple(rules)


def _row_value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        normalized_key = key.replace('"', '""')
        if normalized_key != key:
            value = row.get(normalized_key)
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def build_query_hierarchy(
    row: dict[str, Any],
    *,
    fallback_query_id: str,
) -> QueryHierarchy:
    """Build normalized hierarchy metadata from a CSV row."""

    query_id = str(
        _row_value(row, "query_id")
        or _row_value(row, "question_number")
        or _row_value(row, "variable_name")
        or fallback_query_id
    ).strip()
    boolean_parent_ids = _split_pipe_delimited_ids(_row_value(row, REQUIRES_YES_COLUMN))
    context_parent_ids = _split_pipe_delimited_ids(
        _row_value(row, REQUIRES_DATA_COLUMN)
    )
    label_blockers = _parse_label_blockers(_row_value(row, REQUIRES_LABELS_COLUMN))
    parent_ids = _dedupe_preserve_order(
        [
            *boolean_parent_ids,
            *context_parent_ids,
            *(rule.parent_query_id for rule in label_blockers),
        ]
    )

    return QueryHierarchy(
        query_id=query_id,
        parent_ids=parent_ids,
        boolean_parent_ids=boolean_parent_ids,
        context_parent_ids=context_parent_ids,
        label_blockers=label_blockers,
        pass_parent_question=bool(context_parent_ids),
        pass_parent_short_answer=bool(context_parent_ids),
        inherit_parent_retrieval=bool(context_parent_ids),
    )


def hierarchy_to_metadata(hierarchy: QueryHierarchy) -> dict[str, Any]:
    """Serialize hierarchy metadata into result/query metadata structures."""

    return {
        "query_id": hierarchy.query_id,
        "parent_ids": list(hierarchy.parent_ids),
        "boolean_parent_ids": list(hierarchy.boolean_parent_ids),
        "context_parent_ids": list(hierarchy.context_parent_ids),
        "label_blockers": [
            {
                "parent_query_id": rule.parent_query_id,
                "blocker_labels": list(rule.blocker_labels),
            }
            for rule in hierarchy.label_blockers
        ],
        "pass_parent_question": hierarchy.pass_parent_question,
        "pass_parent_short_answer": hierarchy.pass_parent_short_answer,
        "inherit_parent_retrieval": hierarchy.inherit_parent_retrieval,
    }


def hierarchy_from_metadata(payload: Any) -> QueryHierarchy | None:
    """Deserialize hierarchy metadata from query/result metadata."""

    if not isinstance(payload, dict):
        return None

    query_id = str(payload.get("query_id") or "").strip()
    if not query_id:
        return None

    label_blockers: list[LabelBlockerRule] = []
    for item in payload.get("label_blockers", []):
        if not isinstance(item, dict):
            continue
        parent_query_id = str(item.get("parent_query_id") or "").strip()
        blocker_labels = _dedupe_preserve_order(item.get("blocker_labels", []))
        if not parent_query_id or not blocker_labels:
            continue
        label_blockers.append(
            LabelBlockerRule(
                parent_query_id=parent_query_id,
                blocker_labels=blocker_labels,
            )
        )

    return QueryHierarchy(
        query_id=query_id,
        parent_ids=_dedupe_preserve_order(list(payload.get("parent_ids", []))),
        boolean_parent_ids=_dedupe_preserve_order(
            list(payload.get("boolean_parent_ids", []))
        ),
        context_parent_ids=_dedupe_preserve_order(
            list(payload.get("context_parent_ids", []))
        ),
        label_blockers=tuple(label_blockers),
        pass_parent_question=bool(payload.get("pass_parent_question", True)),
        pass_parent_short_answer=bool(payload.get("pass_parent_short_answer", True)),
        inherit_parent_retrieval=bool(payload.get("inherit_parent_retrieval", False)),
    )
