"""COEP-specific query preprocessing helpers."""

import polars as pl


def adjust_drug_paraphernalia_queries(df: pl.DataFrame) -> pl.DataFrame:
    """Apply COEP-specific query adjustments for drug paraphernalia datasets."""
    if df.is_empty():
        return df

    first_query = str(df["question"][0]).lower()
    if "drug paraphernalia" not in first_query:
        return df

    context = (
        "This query is about ordinance that prohibits "
        "drug paraphernalia-related activities."
    )
    adjusted = df.with_columns(
        (pl.lit(context) + " " + pl.col("question").cast(pl.String)).alias("question")
    )

    if "variable_name" in adjusted.columns:
        rows_to_exclude = ["dp_database", "dp_url", "dp_note"]
        adjusted = adjusted.filter(~pl.col("variable_name").is_in(rows_to_exclude))

    return adjusted
