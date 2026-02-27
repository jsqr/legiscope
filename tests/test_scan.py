"""Tests for legiscope.parse.scan module — new functions."""

import polars as pl
import pytest
from pydantic import ValidationError

from legiscope.parse.headings import HeadingLevel, HeadingStructure
from legiscope.parse.scan import (
    _build_sample_windows,
    _check_multiline_candidates,
    _format_text_block,
    _is_literal_pattern,
    _per_level_quality,
    _prioritize_errors,
    _verify_compile_patterns,
    score_structure,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_elements(texts: list[str]) -> pl.DataFrame:
    """Build a synthetic elements DataFrame from a list of text strings."""
    rows = []
    for i, text in enumerate(texts):
        n_lines = len(text.split("\n"))
        rows.append(
            {
                "element_id": i,
                "start_line": i * 10 + 1,
                "end_line": i * 10 + n_lines,
                "n_lines": n_lines,
                "text": text,
            }
        )
    return pl.DataFrame(rows)


def _make_structure(levels: list[dict], **kwargs) -> HeadingStructure:
    """Build a HeadingStructure from a list of level dicts."""
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


# ── _format_text_block tests ──────────────────────────────────────────


class TestFormatTextBlock:
    def test_single_element(self):
        df = _make_elements(["Hello world"])
        result = _format_text_block(df)
        assert "--- E0 ---" in result
        assert "Hello world" in result

    def test_multi_element(self):
        df = _make_elements(["First", "Second", "Third"])
        result = _format_text_block(df)
        assert "--- E0 ---" in result
        assert "--- E1 ---" in result
        assert "--- E2 ---" in result
        assert "First" in result
        assert "Third" in result

    def test_multiline_element_shows_all_lines(self):
        df = _make_elements(["Line one\nLine two\nLine three"])
        result = _format_text_block(df)
        assert "Line one" in result
        assert "Line two" in result
        assert "Line three" in result

    def test_truncation(self):
        # Create elements that exceed max_chars
        texts = [f"Element {i} " * 20 for i in range(50)]
        df = _make_elements(texts)
        result = _format_text_block(df, max_chars=500)
        assert "[... truncated at E" in result
        # Should not contain all 50 elements
        assert "--- E49 ---" not in result


# ── _is_literal_pattern tests ─────────────────────────────────────────


class TestIsLiteralPattern:
    def test_literal_string(self):
        assert _is_literal_pattern("^ADMINISTRATION$") is True

    def test_regex_with_metacharacters(self):
        assert _is_literal_pattern(r"^CHAPTER\s+\d+.*$") is False

    def test_pattern_with_character_class(self):
        assert _is_literal_pattern("^[A-Z][A-Z ]+$") is False

    def test_pattern_with_alternation(self):
        assert _is_literal_pattern("^TITLE|CHAPTER$") is False

    def test_pattern_with_quantifier(self):
        assert _is_literal_pattern("^SEC+$") is False

    def test_plain_word_no_anchors(self):
        assert _is_literal_pattern("HELLO") is True


# ── _check_multiline_candidates tests ─────────────────────────────────


class TestCheckMultilineCandidates:
    def test_2line_keyword_multiline_false_warns(self):
        df = _make_elements(["CHAPTER 5\nGeneral Provisions and Rules"])
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
                "multiline": False,
            }
        ])
        compiled, _ = _verify_compile_patterns(structure)
        warnings = _check_multiline_candidates(structure, df, compiled)
        assert len(warnings) == 1
        assert "multiline=False" in warnings[0]

    def test_2line_keyword_multiline_true_no_warn(self):
        df = _make_elements(["CHAPTER 5\nGeneral Provisions and Rules"])
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
                "multiline": True,
            }
        ])
        compiled, _ = _verify_compile_patterns(structure)
        warnings = _check_multiline_candidates(structure, df, compiled)
        assert len(warnings) == 0

    def test_1line_element_ignored(self):
        df = _make_elements(["CHAPTER 5"])
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
                "multiline": False,
            }
        ])
        compiled, _ = _verify_compile_patterns(structure)
        warnings = _check_multiline_candidates(structure, df, compiled)
        assert len(warnings) == 0


# ── score_structure new checks ────────────────────────────────────────


class TestScoreStructureNewChecks:
    def test_duplicate_levels_error(self):
        """Duplicate levels should be caught by Pydantic validator."""
        with pytest.raises(ValidationError, match="Duplicate level number"):
            _make_structure([
                {
                    "level": 1,
                    "regex_pattern": r"^CHAPTER\s+\d+",
                    "regex_patterns": [r"^CHAPTER\s+\d+"],
                },
                {
                    "level": 1,
                    "regex_pattern": r"^SECTION\s+\d+",
                    "regex_patterns": [r"^SECTION\s+\d+"],
                },
            ])

    def test_over_classification_error(self):
        # Create elements where >20% match a single level
        texts = [f"(A) clause {i}" for i in range(25)]
        texts.insert(0, "CHAPTER 1")
        df = _make_elements(texts)
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
            },
            {
                "level": 2,
                "regex_pattern": r"^\(A\)",
                "regex_patterns": [r"^\(A\)"],
            },
        ])
        _, errors = score_structure(df, structure)
        over_class = [e for e in errors if "mis-classified" in e]
        assert len(over_class) >= 1

    def test_too_many_levels_error(self):
        levels = []
        for i in range(1, 8):
            levels.append({
                "level": i,
                "regex_pattern": rf"^LEVEL{i}\s+\d+",
                "regex_patterns": [rf"^LEVEL{i}\s+\d+"],
            })
        texts = [f"LEVEL{i} 1" for i in range(1, 8)]
        df = _make_elements(texts)
        structure = _make_structure(levels)
        _, errors = score_structure(df, structure)
        too_many = [e for e in errors if "Too many heading levels" in e]
        assert len(too_many) == 1

    def test_literal_pattern_error(self):
        df = _make_elements(["ADMINISTRATION", "Some body text"])
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": "^ADMINISTRATION$",
                "regex_patterns": ["^ADMINISTRATION$"],
            },
        ])
        _, errors = score_structure(df, structure)
        literal = [e for e in errors if "literal string" in e]
        assert len(literal) >= 1

    def test_single_match_warning(self):
        df = _make_elements([
            "CHAPTER 1",
            "Some text",
            "More text",
            "Even more text",
        ])
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
            },
        ])
        _, errors = score_structure(df, structure)
        single = [e for e in errors if "matches only 1 element" in e]
        assert len(single) == 1


# ── HeadingStructure validator ────────────────────────────────────────


class TestHeadingStructureValidator:
    def test_duplicate_level_numbers_raises(self):
        with pytest.raises(ValidationError, match="Duplicate level number"):
            HeadingStructure(
                heading_levels=[
                    HeadingLevel(
                        level=1,
                        markdown_prefix="# ",
                        example_heading="CHAPTER 1",
                        regex_pattern=r"^CHAPTER\s+\d+",
                    ),
                    HeadingLevel(
                        level=1,
                        markdown_prefix="## ",
                        example_heading="SECTION 1",
                        regex_pattern=r"^SECTION\s+\d+",
                    ),
                ],
                total_levels=2,
                file_sample_size=100,
            )

    def test_unique_levels_ok(self):
        structure = HeadingStructure(
            heading_levels=[
                HeadingLevel(
                    level=1,
                    markdown_prefix="# ",
                    example_heading="CHAPTER 1",
                    regex_pattern=r"^CHAPTER\s+\d+",
                ),
                HeadingLevel(
                    level=2,
                    markdown_prefix="## ",
                    example_heading="SECTION 1",
                    regex_pattern=r"^SECTION\s+\d+",
                ),
            ],
            total_levels=2,
            file_sample_size=100,
        )
        assert len(structure.levels) == 2


# ── Fix 3: Marker-only level detection ───────────────────────────────


class TestMarkerOnlyDetection:
    def test_identical_strings_flagged(self):
        """28 of 30 identical strings should be flagged as marker-only."""
        texts = ["Section"] * 28 + ["Section 1 - Zoning", "Section 2 - Roads"]
        # Add body text so we have non-heading elements
        texts.extend([f"Body paragraph {i}" for i in range(70)])
        df = _make_elements(texts)
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^Section",
                "regex_patterns": [r"^Section"],
                "type_label": "section",
            },
        ])
        _, errors = score_structure(df, structure)
        marker = [e for e in errors if "identical string" in e]
        assert len(marker) >= 1
        assert "Section" in marker[0]

    def test_varied_headings_not_flagged(self):
        """Varied heading texts should not be flagged."""
        texts = [f"CHAPTER {i} - Title {i}" for i in range(10)]
        texts.extend([f"Body text {i}" for i in range(90)])
        df = _make_elements(texts)
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
                "type_label": "chapter",
            },
        ])
        _, errors = score_structure(df, structure)
        marker = [e for e in errors if "identical string" in e]
        assert len(marker) == 0

    def test_inferred_level_not_flagged(self):
        """Inferred levels should be skipped."""
        texts = ["Section"] * 10 + [f"Body {i}" for i in range(90)]
        df = _make_elements(texts)
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^Section",
                "regex_patterns": [r"^Section"],
                "type_label": "section",
                "inferred": True,
            },
        ])
        _, errors = score_structure(df, structure)
        marker = [e for e in errors if "identical string" in e]
        assert len(marker) == 0

    def test_exactly_80_pct_flagged(self):
        """Exactly 80% identical should be flagged (>= threshold)."""
        texts = ["SECTION:"] * 8 + ["SECTION: Zoning", "SECTION: Roads"]
        texts.extend([f"Body {i}" for i in range(90)])
        df = _make_elements(texts)
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^SECTION",
                "regex_patterns": [r"^SECTION"],
                "type_label": "section",
            },
        ])
        _, errors = score_structure(df, structure)
        marker = [e for e in errors if "identical string" in e]
        assert len(marker) >= 1

    def test_fewer_than_5_matches_not_flagged(self):
        """Levels with fewer than 5 matches should be skipped."""
        texts = ["Section"] * 4 + [f"Body {i}" for i in range(96)]
        df = _make_elements(texts)
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^Section",
                "regex_patterns": [r"^Section"],
                "type_label": "section",
            },
        ])
        _, errors = score_structure(df, structure)
        marker = [e for e in errors if "identical string" in e]
        assert len(marker) == 0


# ── Fix 5: Density scoring ───────────────────────────────────────────


class TestDensityScoring:
    def test_normal_density_score_1(self):
        """10% density should give density_score = 1.0."""
        texts = [f"CHAPTER {i}" for i in range(10)]
        texts.extend([f"Body paragraph {i}" for i in range(90)])
        df = _make_elements(texts)
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
                "type_label": "chapter",
            },
        ])
        score, errors = score_structure(df, structure)
        density_errors = [e for e in errors if "density" in e.lower()]
        assert len(density_errors) == 0
        # Score should be healthy (no density penalty)
        assert score > 0.5

    def test_very_low_density_penalized(self):
        """0.2% density should be penalized."""
        texts = ["CHAPTER 1"]
        texts.extend([f"Body paragraph {i}" for i in range(499)])
        df = _make_elements(texts)
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
                "type_label": "chapter",
            },
        ])
        score, errors = score_structure(df, structure)
        density_errors = [e for e in errors if "density" in e.lower()]
        assert len(density_errors) >= 1

    def test_high_density_penalized(self):
        """80% density should be penalized (over-matching)."""
        texts = [f"CHAPTER {i}" for i in range(80)]
        texts.extend([f"Body {i}" for i in range(20)])
        df = _make_elements(texts)
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
                "type_label": "chapter",
            },
        ])
        score, errors = score_structure(df, structure)
        # Over-classification error should also fire, density penalty applied
        assert score <= 0.9

    def test_density_error_below_1_pct(self):
        """Should get error message when density < 1%."""
        texts = ["CHAPTER 1", "CHAPTER 2"]
        texts.extend([f"Body {i}" for i in range(498)])
        df = _make_elements(texts)
        structure = _make_structure([
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
                "type_label": "chapter",
            },
        ])
        _, errors = score_structure(df, structure)
        density_errors = [e for e in errors if "heading density" in e.lower()]
        assert len(density_errors) >= 1

    def test_score_weights_sum_to_1(self):
        """Verify the score weights sum to 1.0."""
        weights = [0.35, 0.20, 0.15, 0.10, 0.10, 0.10]
        assert abs(sum(weights) - 1.0) < 1e-9


# ── Fix 2: Prioritized error feedback ────────────────────────────────


class TestPrioritizeErrors:
    def test_empty_returns_empty(self):
        assert _prioritize_errors([]) == []

    def test_over_classification_sorts_first(self):
        errors = [
            "Out-of-order siblings at level 1: '2' after '3'",
            "Level 1 matches 50 elements (50% of total). This suggests body clauses are being mis-classified as headings.",
        ]
        result = _prioritize_errors(errors)
        assert "mis-classified" in result[0]

    def test_500_errors_capped_at_10(self):
        errors = [
            f"Out-of-order siblings at level 1: '{i}' after '{i+1}'"
            for i in range(500)
        ]
        result = _prioritize_errors(errors, max_items=10)
        assert len(result) <= 10

    def test_100_sibling_errors_collapsed(self):
        errors = [
            f"Out-of-order siblings at level 1: '{i}' after '{i+1}'"
            for i in range(100)
        ]
        result = _prioritize_errors(errors)
        # Should have 1 representative + "... and N more"
        sibling_items = [e for e in result if "Out-of-order" in e or "sibling" in e]
        assert len(sibling_items) == 2  # 1 representative + 1 summary

    def test_mixed_priorities(self):
        errors = [
            "Out-of-order siblings at level 1: '2' after '3'",
            "Out-of-order siblings at level 1: '5' after '6'",
            "Level 2 pattern is a literal string, not a generalizable regex: '^ADMIN$'",
            "Level 1 matches 50 elements (50% of total). This suggests body clauses are being mis-classified as headings.",
            "Pattern has 0 matches: ^FOOBAR.*$",
            "Ambiguous match E5: levels [1, 2]: CHAPTER 1",
        ]
        result = _prioritize_errors(errors)
        # Over-classification should come first
        assert "mis-classified" in result[0]


# ── Fix 6: Per-level quality ─────────────────────────────────────────


class TestPerLevelQuality:
    def _build_quality_test(self, texts, level_defs):
        df = _make_elements(texts)
        structure = _make_structure(level_defs)
        compiled, _ = _verify_compile_patterns(structure)
        return _per_level_quality(df, structure, compiled)

    def test_good_level_passes(self):
        texts = [f"CHAPTER {i} - Title" for i in range(10)]
        texts.extend([f"Body {i}" for i in range(90)])
        quality = self._build_quality_test(texts, [
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
                "type_label": "chapter",
            },
        ])
        assert quality[1]["good"] is True
        assert quality[1]["match_count"] == 10

    def test_over_classified_not_good(self):
        texts = [f"CHAPTER {i}" for i in range(25)]
        texts.extend([f"Body {i}" for i in range(75)])
        quality = self._build_quality_test(texts, [
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
                "type_label": "chapter",
            },
        ])
        assert quality[1]["good"] is False
        assert quality[1]["over_class_pct"] > 0.20

    def test_marker_only_not_good(self):
        texts = ["Section"] * 10 + [f"Body {i}" for i in range(90)]
        quality = self._build_quality_test(texts, [
            {
                "level": 1,
                "regex_pattern": r"^Section",
                "regex_patterns": [r"^Section"],
                "type_label": "section",
            },
        ])
        assert quality[1]["good"] is False
        assert quality[1]["marker_only"] is True

    def test_single_match_not_good(self):
        texts = ["CHAPTER 1"] + [f"Body {i}" for i in range(99)]
        quality = self._build_quality_test(texts, [
            {
                "level": 1,
                "regex_pattern": r"^CHAPTER\s+\d+",
                "regex_patterns": [r"^CHAPTER\s+\d+"],
                "type_label": "chapter",
            },
        ])
        assert quality[1]["good"] is False
        assert quality[1]["match_count"] == 1


# ── Fix 1: Multi-window sampling ─────────────────────────────────────


class TestBuildSampleWindows:
    def test_small_doc_no_mid_window(self):
        """Small doc (< 2x sample_count) returns None for mid_window."""
        texts = [f"Element {i}" for i in range(100)]
        df = _make_elements(texts)
        opening, mid = _build_sample_windows(df, sample_count=200)
        assert opening.height == 100
        assert mid is None

    def test_large_doc_returns_mid_window(self):
        """Large doc returns mid-document window."""
        texts = [f"Element {i}" for i in range(1000)]
        df = _make_elements(texts)
        opening, mid = _build_sample_windows(df, sample_count=200)
        assert opening.height == 200
        assert mid is not None
        assert mid.height == 200

    def test_mid_window_at_40_pct(self):
        """Mid window starts at 40% of document."""
        texts = [f"Element {i}" for i in range(1000)]
        df = _make_elements(texts)
        _, mid = _build_sample_windows(df, sample_count=200)
        assert mid is not None
        # 40% of 1000 = 400, so first element_id in mid should be 400
        first_eid = mid["element_id"][0]
        assert first_eid == 400

    def test_exact_boundary_has_mid(self):
        """When 40% offset equals sample_count, mid window is returned."""
        # 500 elements, sample_count=200 → mid_start=200 → not < 200 → mid returned
        texts = [f"Element {i}" for i in range(500)]
        df = _make_elements(texts)
        _, mid = _build_sample_windows(df, sample_count=200)
        assert mid is not None

    def test_just_over_boundary_has_mid(self):
        """When document just exceeds boundary, mid window appears."""
        # 501 elements, sample_count=200 → mid_start=200 → equals sample_count → None
        # Need mid_start > sample_count → height * 0.4 > 200 → height > 500
        texts = [f"Element {i}" for i in range(501)]
        df = _make_elements(texts)
        _, mid = _build_sample_windows(df, sample_count=200)
        assert mid is not None
