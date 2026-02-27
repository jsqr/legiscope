"""Display and formatting utilities for heading structure analysis."""

from __future__ import annotations

import json
import sys

import polars as pl

from legiscope.parse.headings import HeadingStructure
from legiscope.parse.scan import (
    ScoreBreakdown,
    _per_level_quality,
    _verify_compile_patterns,
    score_structure_detailed,
)


# ── Private helpers ───────────────────────────────────────────────────


def _trunc(s: str, width: int) -> str:
    """Truncate string to *width* chars, adding '...' if needed."""
    if len(s) <= width:
        return s
    return s[: width - 3] + "..."


def _pad_table(rows: list[list[str]], headers: list[str]) -> str:
    """Build a column-aligned ASCII table from rows and headers.

    All values are left-aligned except numeric-looking columns which are
    right-aligned.
    """
    all_rows = [headers] + rows
    n_cols = len(headers)

    # Compute column widths
    widths = [0] * n_cols
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Detect right-align columns (header excluded, check if most data cells are numeric)
    right_align = [False] * n_cols
    for col in range(n_cols):
        numeric_count = sum(
            1
            for row in rows
            if row[col].strip().replace(".", "").replace("-", "").replace("%", "").isdigit()
        )
        if rows and numeric_count > len(rows) // 2:
            right_align[col] = True

    def fmt_row(row: list[str]) -> str:
        parts = []
        for i, cell in enumerate(row):
            if right_align[i]:
                parts.append(cell.rjust(widths[i]))
            else:
                parts.append(cell.ljust(widths[i]))
        return "  ".join(parts)

    lines = [fmt_row(all_rows[0])]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────


def format_structure(structure: HeadingStructure) -> str:
    """Render the heading hierarchy as a readable indented tree.

    Returns a multi-line string showing each level's pattern, example,
    and flags like ``[multiline]`` or ``[inferred]``.
    """
    non_inferred_count = sum(1 for hl in structure.levels if not hl.inferred)
    total_elements = structure.file_sample_size
    header = (
        f"Heading Structure "
        f"({non_inferred_count} levels, "
        f"score={structure.quality_score:.2f}, "
        f"{structure.iterations} iterations, "
        f"{total_elements} elements)"
    )

    lines = [header, ""]
    sorted_levels = sorted(structure.levels, key=lambda hl: hl.level)

    for hl in sorted_levels:
        flags = []
        if hl.multiline:
            flags.append("[multiline]")
        if hl.inferred:
            flags.append("[inferred]")
        flag_str = "  " + " ".join(flags) if flags else ""

        if hl.inferred:
            pattern_display = "(no patterns)"
            md_display = "\u2500\u2500 \u2500"
        else:
            pattern_display = _trunc(hl.regex_pattern, 35)
            md_display = hl.markdown_prefix.rstrip()

        label = hl.type_label if hl.type_label else "(unnamed)"
        example_display = _trunc(hl.example_heading, 20)

        lines.append(
            f"  L{hl.level}  {label:<12s} {md_display:<8s} "
            f"{pattern_display:<37s} "
            f'"{example_display}"{flag_str}'
        )

    if structure.outline_warnings:
        lines.append("")
        lines.append(f"Warnings ({len(structure.outline_warnings)}):")
        for w in structure.outline_warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


_WEIGHTS = {
    "coverage": 0.35,
    "pattern_validity": 0.20,
    "sibling_ordering": 0.15,
    "ambiguity": 0.10,
    "parent_child": 0.10,
    "density": 0.10,
}

_LABELS = {
    "coverage": "Coverage",
    "pattern_validity": "Pattern validity",
    "sibling_ordering": "Sibling ordering",
    "ambiguity": "Ambiguity",
    "parent_child": "Parent-child",
    "density": "Density",
}


def format_score_breakdown(
    elements_df: pl.DataFrame,
    structure: HeadingStructure,
) -> str:
    """Render the six score components and per-level quality.

    Calls ``score_structure_detailed()`` and ``_per_level_quality()``
    internally.
    """
    bd: ScoreBreakdown = score_structure_detailed(elements_df, structure)
    compiled, _ = _verify_compile_patterns(structure)
    plq = _per_level_quality(elements_df, structure, compiled)

    lines = [f"Score Breakdown ({bd['total']:.3f})", ""]

    # Component table
    comp_headers = ["Component", "Weight", "Score", "Weighted"]
    comp_rows = []
    for key in _WEIGHTS:
        w = _WEIGHTS[key]
        s = bd[key]  # type: ignore[literal-required]
        comp_rows.append([
            _LABELS[key],
            f"{w:.2f}",
            f"{s:.2f}",
            f"{w * s:.3f}",
        ])
    comp_rows.append(["", "", "Total:", f"{bd['total']:.3f}"])
    lines.append(_pad_table(comp_rows, comp_headers))

    # Match summary
    matched = bd["matched_count"]
    total = bd["total_elements"]
    ambig = bd["ambiguous_count"]
    density_pct = matched / total * 100 if total > 0 else 0.0
    lines.append("")
    lines.append(
        f"  Matched: {matched}/{total} elements ({density_pct:.1f}%), "
        f"{ambig} ambiguous"
    )

    # Per-level quality
    if plq:
        lines.append("")
        lines.append("Per-Level Quality:")
        for lvl_num in sorted(plq.keys()):
            info = plq[lvl_num]
            label = info["type_label"] or "(unnamed)"
            mc = info["match_count"]
            ap = info["ambiguous_pct"] * 100
            status = "good" if info["good"] else "WARN"
            if info["marker_only"]:
                status = "WARN: marker-only"
            elif not info["good"] and ap >= 10:
                status = "WARN: ambiguous"
            elif not info["good"] and info["over_class_pct"] >= 0.20:
                status = "WARN: over-classified"
            lines.append(
                f"  L{lvl_num} {label:<14s} {mc:>3d} matches  "
                f"{ap:>3.0f}% ambig   {status}"
            )

    # Top errors
    errs = bd["errors"]
    if errs:
        shown = errs[:5]
        lines.append("")
        lines.append(f"Top Errors ({len(errs)}):")
        for i, e in enumerate(shown, 1):
            lines.append(f"  {i}. {e}")
        if len(errs) > 5:
            lines.append(f"  ... and {len(errs) - 5} more")

    return "\n".join(lines)


def make_batch_entry(
    jurisdiction: str,
    structure: HeadingStructure,
    elements_df: pl.DataFrame,
    threshold: float = 0.7,
) -> dict:
    """Build a standardized result dict for one jurisdiction.

    Returns a dict with keys: jurisdiction, score, iterations, levels,
    headings, total_elements, density_pct, errors, top_issues, status.
    """
    bd = score_structure_detailed(elements_df, structure)
    non_inferred = sum(1 for hl in structure.levels if not hl.inferred)
    matched = bd["matched_count"]
    total = bd["total_elements"]
    density = matched / total * 100 if total > 0 else 0.0
    errs = bd["errors"]

    return {
        "jurisdiction": jurisdiction,
        "score": round(bd["total"], 3),
        "iterations": structure.iterations,
        "levels": non_inferred,
        "headings": matched,
        "total_elements": total,
        "density_pct": round(density, 1),
        "errors": len(errs),
        "top_issues": errs[:3],
        "status": "pass" if bd["total"] >= threshold else "FAIL",
    }


def format_batch_summary(
    results: list[dict],
    threshold: float = 0.7,
) -> str:
    """Render a comparison table across jurisdictions.

    *results* is a list of dicts as returned by ``make_batch_entry()``.
    """
    n = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    scores = [r["score"] for r in results]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    lines = [
        f"Batch Parse Results ({n} jurisdictions, threshold={threshold:.2f})",
        "",
    ]

    headers = [
        "Jurisdiction",
        "Score",
        "Iters",
        "Lvls",
        "Headings",
        "Density",
        "Errors",
        "Status",
    ]
    rows = []
    for r in results:
        rows.append([
            r["jurisdiction"],
            f"{r['score']:.3f}",
            str(r["iterations"]),
            str(r["levels"]),
            str(r["headings"]),
            f"{r['density_pct']:.1f}%",
            str(r["errors"]),
            r["status"],
        ])

    lines.append(_pad_table(rows, headers))
    lines.append("")
    lines.append(f"Summary: {passed}/{n} passed, mean={mean_score:.2f}")

    # Failed details
    failed = [r for r in results if r["status"] == "FAIL"]
    if failed:
        lines.append("")
        lines.append("Failed:")
        for r in failed:
            lines.append(f"  {r['jurisdiction']} ({r['score']:.2f}):")
            for issue in r["top_issues"]:
                lines.append(f"    - {issue}")

    return "\n".join(lines)


# ── CLI entry point ───────────────────────────────────────────────────


if __name__ == "__main__":
    usage = (
        "Usage:\n"
        "  python -m legiscope.parse.display structure <structure.json>\n"
        "  python -m legiscope.parse.display breakdown <structure.json> <elements.parquet>"
    )

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    command = sys.argv[1]
    structure_path = sys.argv[2]

    with open(structure_path) as f:
        data = json.load(f)
    struct = HeadingStructure(**data)

    if command == "structure":
        print(format_structure(struct))
    elif command == "breakdown":
        if len(sys.argv) < 4:
            print("breakdown requires an elements parquet path")
            print(usage)
            sys.exit(1)
        elements_path = sys.argv[3]
        df = pl.read_parquet(elements_path)
        print(format_score_breakdown(df, struct))
    else:
        print(f"Unknown command: {command}")
        print(usage)
        sys.exit(1)
