"""COEP-specific query preprocessing helpers."""

import polars as pl


def _build_query_text(row: dict) -> str:
    """Compose a single LLM-readable query from the structured CSV columns."""
    parts: list[str] = []

    prepend = (row.get("prepend_text") or "").strip()
    if prepend:
        parts.append(f"Context: {prepend}")

    query = (row.get("query_text") or "").strip()
    if query:
        parts.append(f"Question: {query}")

    coding = (row.get("coding_instructions") or "").strip()
    if coding:
        parts.append(f"Coding instructions: {coding}")

    options = (row.get("response_options") or "").strip()
    if options:
        parts.append(f"Response options: {options}")

    return "\n\n".join(parts)


def adjust_drug_paraphernalia_queries(df: pl.DataFrame) -> pl.DataFrame:
    """Apply COEP-specific query adjustments for drug paraphernalia datasets.

    Expects the new CSV schema with columns: question_number, variable_name,
    prepend_text, query_text, response_options, coding_instructions.
    Composes them into a single ``question`` column.
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
