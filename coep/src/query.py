"""COEP-specific query preprocessing helpers."""

import polars as pl


def _first_nonempty_text(row: dict, *keys: str) -> str:
    """Return the first non-empty string value from the provided row keys."""
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _build_query_text(row: dict) -> str:
    """Compose a single LLM-readable query from the structured CSV columns."""
    parts: list[str] = []

    query = _first_nonempty_text(row, "query_text", "question")
    if query:
        parts.append(f"Question: {query}")

    coding = _first_nonempty_text(row, "coding_instructions")
    if coding:
        parts.append(f"Coding instructions: {coding}")

    options = _first_nonempty_text(row, "response_options")
    if options:
        parts.append(f"Response options: {options}")

    return "\n\n".join(parts)


def adjust_drug_paraphernalia_queries(df: pl.DataFrame) -> pl.DataFrame:
    """Apply COEP-specific query adjustments for drug paraphernalia datasets.

    Expects the new CSV schema with columns: question_number, variable_name,
    prepend_text, query_text, response_options, coding_instructions.
    Composes completion-oriented text into a single ``question`` column while
    preserving split-query dependency columns in metadata for the generic
    hierarchy engine. ``prepend_text`` intentionally stays metadata-only so the
    retrieval-guidance hook can reuse it without duplicating context in the
    composed prompt.
    """
    if df.is_empty():
        return df

    # Compose question from structured columns
    if "query_text" in df.columns:
        questions = [_build_query_text(row) for row in df.to_dicts()]
        df = df.with_columns(pl.Series("question", questions))

    # Exclude non-queryable variables
    if "variable_name" in df.columns:
        rows_to_exclude = ["dp_database", "dp_url", "dp_note"]
        df = df.filter(~pl.col("variable_name").is_in(rows_to_exclude))

    return df
