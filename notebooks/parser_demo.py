import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")

with app.setup:
    import json
    import os
    from pathlib import Path

    import polars as pl


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    _project_root = Path(__file__).resolve().parent.parent
    os.chdir(_project_root)
    project_root = _project_root
    return (project_root,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parser Diagnostics Viewer

    Interactive dashboard for parse diagnostics.  No LLM calls or API keys required.

    Run `./scripts/parse_samples.sh` (or `./scripts/dvc_repro.sh --stage parse`) to generate data.
    """)
    return


@app.cell
def _(mo, project_root):
    _laws_dir = project_root / "data" / "laws"
    diagnostics = {}
    for diag_path in sorted(_laws_dir.rglob("parse_diagnostics.json")):
        rel = diag_path.relative_to(_laws_dir)
        parts = rel.parts
        if len(parts) >= 4:
            label = f"{parts[0]}/{parts[1]}/{parts[2]}"
            with open(diag_path) as f:
                data = json.load(f)
            # Look for sibling parquet
            parquet_path = diag_path.parent / "classified_elements.parquet"
            diagnostics[label] = {
                "data": data,
                "parquet_path": str(parquet_path) if parquet_path.exists() else None,
            }

    if not diagnostics:
        mo.output.replace(
            mo.callout(
                mo.md(
                    "No `parse_diagnostics.json` files found under `data/laws/`.\n\n"
                    "Run `./scripts/parse_samples.sh` to generate data."
                ),
                kind="warn",
            )
        )

    # Build overview dataframe from batch_entry dicts
    _batch_rows = []
    for label, info in diagnostics.items():
        entry = info["data"].get("batch_entry", {})
        _batch_rows.append({
            "jurisdiction": label,
            "score": entry.get("score", 0.0),
            "iterations": entry.get("iterations", 0),
            "levels": entry.get("levels", 0),
            "headings": entry.get("headings", 0),
            "total_elements": entry.get("total_elements", 0),
            "density_pct": entry.get("density_pct", 0.0),
            "errors": entry.get("errors", 0),
            "status": entry.get("status", ""),
        })
    overview_df = pl.DataFrame(_batch_rows) if _batch_rows else pl.DataFrame()
    return diagnostics, overview_df


@app.cell
def _(mo, overview_df):
    if overview_df.is_empty():
        mo.output.replace(mo.md(""))
    else:
        _n = overview_df.height
        _pass = overview_df.filter(pl.col("status") == "pass").height
        _fail = _n - _pass
        _mean = overview_df["score"].mean()

        _stats = mo.hstack([
            mo.stat(value=_n, label="Jurisdictions"),
            mo.stat(value=_pass, label="Pass"),
            mo.stat(value=_fail, label="Fail"),
            mo.stat(value=f"{_mean:.3f}", label="Mean Score"),
        ])

        _table = mo.ui.table(
            overview_df.to_dicts(),
            label="Batch results",
        )

        mo.output.replace(
            mo.vstack([
                mo.md("## Overview"),
                _stats,
                _table,
            ])
        )
    return


@app.cell
def _(diagnostics, mo):
    _labels = list(diagnostics.keys())
    jurisdiction_picker = mo.ui.dropdown(
        options=_labels,
        value=_labels[0] if _labels else None,
        label="Jurisdiction",
    ) if _labels else None
    jurisdiction_picker
    return (jurisdiction_picker,)


@app.cell
def _(diagnostics, jurisdiction_picker, mo):
    if not jurisdiction_picker or not jurisdiction_picker.value:
        mo.output.replace(mo.md(""))
    else:
        _label = jurisdiction_picker.value
        _info = diagnostics[_label]
        _data = _info["data"]
        _bd = _data.get("score_breakdown", {})
        _struct = _data.get("structure", {})
        _plq = _data.get("per_level_quality", {})
        _levels = _struct.get("levels", [])

        # ── Tab 1: Score Components ──────────────────────────────
        _weights = {
            "coverage": 0.35, "pattern_validity": 0.20,
            "sibling_ordering": 0.15, "ambiguity": 0.10,
            "parent_child": 0.10, "density": 0.10,
        }
        _comp_rows = []
        for k, w in _weights.items():
            s = _bd.get(k, 0.0)
            _comp_rows.append({
                "component": k.replace("_", " ").title(),
                "weight": w, "raw_score": round(s, 3),
                "weighted": round(w * s, 3),
            })
        _comp_table = mo.ui.table(_comp_rows, label="Score components")
        _score_stats = mo.hstack([
            mo.stat(value=f"{_bd.get('total', 0):.3f}", label="Total Score"),
            mo.stat(value=_struct.get("iterations", "?"), label="Iterations"),
            mo.stat(value=_struct.get("file_sample_size", "?"), label="Sample Size"),
        ])
        _tab1 = mo.vstack([_score_stats, _comp_table])

        # ── Tab 2: Heading Levels ────────────────────────────────
        _level_rows = []
        for lv in sorted(_levels, key=lambda x: x.get("level", 0)):
            _level_rows.append({
                "level": lv.get("level"),
                "type_label": lv.get("type_label", ""),
                "markdown_prefix": lv.get("markdown_prefix", ""),
                "regex_pattern": (lv.get("regex_pattern", "") or "")[:60],
                "example_heading": lv.get("example_heading", ""),
                "multiline": lv.get("multiline", False),
                "inferred": lv.get("inferred", False),
            })
        _levels_table = mo.ui.table(_level_rows, label="Heading levels") if _level_rows else mo.md("No levels")

        # Per-level quality table
        _plq_rows = []
        for lvl_key in sorted(_plq.keys(), key=lambda x: int(x)):
            q = _plq[lvl_key]
            _status = "good" if q.get("good") else "WARN"
            if q.get("marker_only"):
                _status = "WARN: marker-only"
            _plq_rows.append({
                "level": int(lvl_key),
                "type_label": q.get("type_label", ""),
                "match_count": q.get("match_count", 0),
                "ambiguous_pct": f"{q.get('ambiguous_pct', 0) * 100:.0f}%",
                "over_class_pct": f"{q.get('over_class_pct', 0) * 100:.0f}%",
                "marker_only": q.get("marker_only", False),
                "status": _status,
            })
        _plq_table = mo.ui.table(_plq_rows, label="Per-level quality") if _plq_rows else mo.md("No per-level data")

        # Heading hierarchy tree
        _tree_data = {}
        for lv in sorted(_levels, key=lambda x: x.get("level", 0)):
            _tl = lv.get("type_label", f"Level {lv.get('level')}")
            _ex = lv.get("example_heading", "")
            _tree_data[f"L{lv.get('level')} — {_tl}"] = [_ex] if _ex else []
        _tree = mo.tree(_tree_data) if _tree_data else mo.md("")

        _tab2 = mo.vstack([_levels_table, _plq_table, mo.md("### Hierarchy"), _tree])

        # ── Tab 3: Elements ──────────────────────────────────────
        _parquet_path = _info.get("parquet_path")
        if _parquet_path:
            _el_df = pl.read_parquet(_parquet_path)
            _heading_count = _el_df.filter(pl.col("classification") != "body").height
            _body_count = _el_df.filter(pl.col("classification") == "body").height
            _ambig_count = _el_df.filter(pl.col("is_ambiguous")).height
            _el_stats = mo.hstack([
                mo.stat(value=_el_df.height, label="Total Elements"),
                mo.stat(value=_heading_count, label="Headings"),
                mo.stat(value=_body_count, label="Body"),
                mo.stat(value=_ambig_count, label="Ambiguous"),
            ])
            # Truncate text for display
            _display_df = _el_df.with_columns(
                pl.col("text").str.slice(0, 80).alias("text"),
            )
            _el_table = mo.ui.table(
                _display_df.to_dicts(),
                label="Classified elements",
            )
            _tab3 = mo.vstack([_el_stats, _el_table])
        else:
            _tab3 = mo.callout(
                mo.md("No `classified_elements.parquet` found. Re-run the parse stage to generate it."),
                kind="warn",
            )

        # ── Tab 4: Errors & Warnings ─────────────────────────────
        _errors = _bd.get("errors", [])
        _warnings = _struct.get("outline_warnings", [])
        _issues = []
        if _errors:
            _issues.append(mo.md("### Errors"))
            for e in _errors:
                _issues.append(mo.md(f"- {e}"))
        if _warnings:
            _issues.append(mo.md("### Outline Warnings"))
            for w in _warnings:
                _issues.append(mo.md(f"- {w}"))
        if not _errors and not _warnings:
            _issues.append(mo.callout(mo.md("No issues found"), kind="success"))
        _tab4 = mo.vstack(_issues)

        # ── Assemble tabs ────────────────────────────────────────
        _tabs = mo.ui.tabs({
            "Score Components": _tab1,
            "Heading Levels": _tab2,
            "Elements": _tab3,
            "Errors & Warnings": _tab4,
        })

        mo.output.replace(mo.vstack([
            mo.md(f"## {_label}"),
            _tabs,
        ]))
    return


if __name__ == "__main__":
    app.run()
