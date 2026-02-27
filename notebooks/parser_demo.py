import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")

with app.setup:
    import json
    import os
    from pathlib import Path

    from legiscope.parse.display import format_batch_summary


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

    Displays pre-computed parse diagnostics from `parse_diagnostics.json` files
    written by the parse stage.  No LLM calls or API keys required.

    Run `./scripts/dvc_repro.sh --stage parse` to generate diagnostics.
    """)
    return


@app.cell
def _(mo, project_root):
    _laws_dir = project_root / "data" / "laws"
    diagnostics = []
    for diag_path in sorted(_laws_dir.rglob("parse_diagnostics.json")):
        rel = diag_path.relative_to(_laws_dir)
        parts = rel.parts
        if len(parts) >= 4:
            label = f"{parts[0]}/{parts[1]}/{parts[2]}"
            with open(diag_path) as f:
                data = json.load(f)
            diagnostics.append({"label": label, "path": str(diag_path), "data": data})

    _rows = [{"jurisdiction": d["label"], "score": d["data"]["batch_entry"]["score"]} for d in diagnostics]
    jurisdiction_table = mo.ui.table(
        _rows,
        label=f"Discovered jurisdictions ({len(diagnostics)})",
    )
    show_button = mo.ui.run_button(label="Show results")
    mo.vstack([jurisdiction_table, show_button])
    return diagnostics, show_button


@app.cell
def _(diagnostics, mo, show_button):
    if not show_button.value or not diagnostics:
        mo.output.replace(mo.md("*Press **Show results** to display diagnostics.*"))
    else:
        _batch_entries = [d["data"]["batch_entry"] for d in diagnostics]
        _summary = format_batch_summary(_batch_entries)
        _parts = [
            mo.md("## Batch Summary"),
            mo.md(f"```\n{_summary}\n```"),
            mo.ui.table(_batch_entries, label="Raw batch results"),
        ]
        mo.output.replace(mo.vstack(_parts))
    return


@app.cell
def _(diagnostics, mo, show_button):
    if not show_button.value or not diagnostics:
        mo.output.replace(mo.md(""))
    else:
        _parts = [mo.md("## Per-Jurisdiction Details")]
        for _d in diagnostics:
            _struct_text = _d["data"]["format_structure"]
            _breakdown_text = _d["data"]["format_score_breakdown"]
            _parts.append(mo.md(f"### {_d['label']}"))
            _parts.append(mo.md(f"```\n{_struct_text}\n```"))
            _parts.append(mo.md(f"```\n{_breakdown_text}\n```"))
        mo.output.replace(mo.vstack(_parts))
    return


if __name__ == "__main__":
    app.run()
