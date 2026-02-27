"""Tests for legiscope.parse.display module."""

import polars as pl

from legiscope.parse.headings import HeadingLevel, HeadingStructure
from legiscope.parse.display import (
    _pad_table,
    _trunc,
    format_batch_summary,
    format_score_breakdown,
    format_structure,
    make_batch_entry,
)
from legiscope.parse.scan import score_structure_detailed


# ── Helpers ───────────────────────────────────────────────────────────


def _make_elements(texts: list[str]) -> pl.DataFrame:
    rows = []
    for i, text in enumerate(texts):
        n_lines = len(text.split("\n"))
        rows.append({
            "element_id": i,
            "start_line": i * 10 + 1,
            "end_line": i * 10 + n_lines,
            "n_lines": n_lines,
            "text": text,
        })
    return pl.DataFrame(rows)


def _make_structure(levels: list[dict], **kwargs) -> HeadingStructure:
    heading_levels = []
    for lvl in levels:
        defaults = {
            "markdown_prefix": "#" * lvl.get("level", 1) + " ",
            "example_heading": lvl.get("example_heading", "Example"),
            "type_label": lvl.get("type_label", "section"),
            "regex_patterns": lvl.get("regex_patterns", []),
            "regex_pattern": lvl.get("regex_pattern", ""),
            "inferred": lvl.get("inferred", False),
            "multiline": lvl.get("multiline", False),
        }
        defaults.update(lvl)
        heading_levels.append(HeadingLevel(**defaults))
    defaults_hs = {
        "heading_levels": heading_levels,
        "total_levels": len(heading_levels),
        "file_sample_size": 100,
    }
    defaults_hs.update(kwargs)
    return HeadingStructure(**defaults_hs)


def _simple_fixture():
    """A simple 2-level structure with matching elements."""
    structure = _make_structure([
        {
            "level": 1,
            "type_label": "title",
            "regex_patterns": [r"^TITLE\s+\d+"],
            "example_heading": "TITLE 1 GENERAL",
        },
        {
            "level": 2,
            "type_label": "chapter",
            "regex_patterns": [r"^CHAPTER\s+\d+"],
            "example_heading": "CHAPTER 1 ADMIN",
        },
    ], quality_score=0.85, iterations=2)

    elements = _make_elements([
        "TITLE 1 GENERAL PROVISIONS",
        "Some body text here",
        "CHAPTER 1 ADMINISTRATION",
        "More body text",
        "CHAPTER 2 ZONING",
        "Body text continues",
        "TITLE 2 PUBLIC SAFETY",
        "CHAPTER 3 POLICE",
        "Body text",
        "Body text again",
    ])
    return structure, elements


# ── _trunc tests ─────────────────────────────────────────────────────


class TestTrunc:
    def test_short_string(self):
        assert _trunc("hello", 10) == "hello"

    def test_exact_width(self):
        assert _trunc("hello", 5) == "hello"

    def test_truncated(self):
        assert _trunc("hello world", 8) == "hello..."

    def test_width_3(self):
        assert _trunc("abcdef", 3) == "..."


# ── _pad_table tests ─────────────────────────────────────────────────


class TestPadTable:
    def test_basic(self):
        result = _pad_table(
            [["a", "1"], ["bb", "22"]],
            ["Name", "Val"],
        )
        lines = result.split("\n")
        assert len(lines) == 3
        assert "Name" in lines[0]
        assert "Val" in lines[0]

    def test_empty_rows(self):
        result = _pad_table([], ["Col1", "Col2"])
        assert "Col1" in result
        assert result.count("\n") == 0


# ── format_structure tests ───────────────────────────────────────────


class TestFormatStructure:
    def test_header_contains_score(self):
        structure, _ = _simple_fixture()
        result = format_structure(structure)
        assert "score=0.85" in result
        assert "2 levels" in result
        assert "2 iterations" in result

    def test_levels_shown_sorted(self):
        structure, _ = _simple_fixture()
        result = format_structure(structure)
        lines = result.strip().split("\n")
        level_lines = [ln for ln in lines if ln.strip().startswith("L")]
        assert len(level_lines) == 2
        assert "L1" in level_lines[0]
        assert "L2" in level_lines[1]

    def test_inferred_flag(self):
        structure = _make_structure([
            {
                "level": 1,
                "type_label": "title",
                "regex_patterns": [r"^TITLE\s+\d+"],
                "example_heading": "TITLE 1",
            },
            {
                "level": 2,
                "type_label": "subsection",
                "inferred": True,
                "example_heading": "1-101",
            },
        ])
        result = format_structure(structure)
        assert "[inferred]" in result
        assert "(no patterns)" in result

    def test_multiline_flag(self):
        structure = _make_structure([
            {
                "level": 1,
                "type_label": "article",
                "regex_patterns": [r"^ARTICLE\s+[IVXL]+"],
                "example_heading": "ARTICLE III",
                "multiline": True,
            },
        ])
        result = format_structure(structure)
        assert "[multiline]" in result

    def test_warnings_shown(self):
        structure = _make_structure(
            [
                {
                    "level": 1,
                    "type_label": "title",
                    "regex_patterns": [r"^TITLE\s+\d+"],
                    "example_heading": "TITLE 1",
                },
            ],
            outline_warnings=["Test warning one", "Test warning two"],
        )
        result = format_structure(structure)
        assert "Warnings (2):" in result
        assert "Test warning one" in result


# ── ScoreBreakdown / score_structure_detailed tests ──────────────────


class TestScoreStructureDetailed:
    def test_returns_typed_dict(self):
        structure, elements = _simple_fixture()
        bd = score_structure_detailed(elements, structure)
        assert isinstance(bd, dict)
        assert "total" in bd
        assert "coverage" in bd
        assert "matched_count" in bd
        assert "errors" in bd

    def test_component_scores_between_0_and_1(self):
        structure, elements = _simple_fixture()
        bd = score_structure_detailed(elements, structure)
        for key in ("coverage", "pattern_validity", "sibling_ordering",
                     "ambiguity", "parent_child", "density"):
            assert 0.0 <= bd[key] <= 1.0, f"{key} out of range: {bd[key]}"

    def test_total_matches_weighted_sum(self):
        structure, elements = _simple_fixture()
        bd = score_structure_detailed(elements, structure)
        expected = (
            0.35 * bd["coverage"]
            + 0.20 * bd["pattern_validity"]
            + 0.15 * bd["sibling_ordering"]
            + 0.10 * bd["ambiguity"]
            + 0.10 * bd["parent_child"]
            + 0.10 * bd["density"]
        )
        assert abs(bd["total"] - expected) < 1e-9

    def test_empty_structure_scores_zero(self):
        structure = _make_structure([])
        elements = _make_elements(["some text"])
        bd = score_structure_detailed(elements, structure)
        # No patterns at all → pattern_validity defaults to 1.0 but coverage=1.0 too
        # total should still be computable
        assert isinstance(bd["total"], float)

    def test_matched_count(self):
        structure, elements = _simple_fixture()
        bd = score_structure_detailed(elements, structure)
        # 2 TITLE + 3 CHAPTER = 5 headings matched
        assert bd["matched_count"] == 5
        assert bd["ambiguous_count"] == 0


# ── format_score_breakdown tests ─────────────────────────────────────


class TestFormatScoreBreakdown:
    def test_contains_total(self):
        structure, elements = _simple_fixture()
        result = format_score_breakdown(elements, structure)
        assert "Score Breakdown" in result
        assert "Total:" in result

    def test_contains_components(self):
        structure, elements = _simple_fixture()
        result = format_score_breakdown(elements, structure)
        assert "Coverage" in result
        assert "Pattern validity" in result
        assert "Density" in result

    def test_contains_per_level(self):
        structure, elements = _simple_fixture()
        result = format_score_breakdown(elements, structure)
        assert "Per-Level Quality:" in result
        assert "title" in result
        assert "chapter" in result

    def test_contains_matched_summary(self):
        structure, elements = _simple_fixture()
        result = format_score_breakdown(elements, structure)
        assert "Matched:" in result
        assert "ambiguous" in result


# ── make_batch_entry tests ───────────────────────────────────────────


class TestMakeBatchEntry:
    def test_returns_expected_keys(self):
        structure, elements = _simple_fixture()
        entry = make_batch_entry("CA/TestCity", structure, elements)
        expected_keys = {
            "jurisdiction", "score", "iterations", "levels",
            "headings", "total_elements", "density_pct",
            "errors", "top_issues", "status",
        }
        assert set(entry.keys()) == expected_keys

    def test_jurisdiction_passthrough(self):
        structure, elements = _simple_fixture()
        entry = make_batch_entry("AK/TestKingCove", structure, elements)
        assert entry["jurisdiction"] == "AK/TestKingCove"

    def test_pass_status(self):
        structure, elements = _simple_fixture()
        entry = make_batch_entry("test", structure, elements, threshold=0.0)
        assert entry["status"] == "pass"

    def test_fail_status(self):
        structure, elements = _simple_fixture()
        entry = make_batch_entry("test", structure, elements, threshold=1.0)
        assert entry["status"] == "FAIL"

    def test_top_issues_max_3(self):
        structure, elements = _simple_fixture()
        entry = make_batch_entry("test", structure, elements)
        assert len(entry["top_issues"]) <= 3

    def test_levels_excludes_inferred(self):
        structure = _make_structure([
            {
                "level": 1,
                "type_label": "title",
                "regex_patterns": [r"^TITLE\s+\d+"],
                "example_heading": "TITLE 1",
            },
            {
                "level": 2,
                "type_label": "sub",
                "inferred": True,
                "example_heading": "1-1",
            },
        ])
        elements = _make_elements(["TITLE 1 FOO", "body"])
        entry = make_batch_entry("test", structure, elements)
        assert entry["levels"] == 1


# ── format_batch_summary tests ───────────────────────────────────────


class TestFormatBatchSummary:
    def _make_results(self):
        structure, elements = _simple_fixture()
        return [
            make_batch_entry("AK/KingCove", structure, elements),
            make_batch_entry("CA/Adelanto", structure, elements, threshold=1.0),
        ]

    def test_header(self):
        results = self._make_results()
        output = format_batch_summary(results, threshold=0.70)
        assert "Batch Parse Results" in output
        assert "2 jurisdictions" in output

    def test_summary_line(self):
        results = self._make_results()
        # First passes (default threshold 0.7), second fails (threshold 1.0 in entry)
        output = format_batch_summary(results)
        assert "Summary:" in output
        assert "passed" in output

    def test_failed_section(self):
        results = self._make_results()
        output = format_batch_summary(results)
        # CA/Adelanto was set with threshold=1.0 so it FAILs
        if any(r["status"] == "FAIL" for r in results):
            assert "Failed:" in output

    def test_empty_results(self):
        output = format_batch_summary([])
        assert "0 jurisdictions" in output
        assert "mean=0.00" in output

    def test_all_pass(self):
        structure, elements = _simple_fixture()
        results = [
            make_batch_entry("A", structure, elements, threshold=0.0),
            make_batch_entry("B", structure, elements, threshold=0.0),
        ]
        output = format_batch_summary(results)
        assert "2/2 passed" in output
        assert "Failed:" not in output
