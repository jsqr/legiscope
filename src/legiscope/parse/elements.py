"""Break raw text into blank-line-separated elements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def split_elements(file_path: str | Path) -> pl.DataFrame:
    """Split a text file into elements (blank-line-separated blocks).

    Each element is a contiguous block of non-empty lines. Blank lines
    (empty or whitespace-only) act as separators.

    Returns a polars DataFrame with columns:
        element_id  (Int64)  — sequential 0-based index
        start_line  (Int64)  — 1-based line number of first line
        end_line    (Int64)  — 1-based line number of last line
        n_lines     (Int64)  — number of lines in the element
        text        (String) — full text of the element (lines joined with newline)
    """
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()

    elements: list[dict[str, Any]] = []
    current_lines: list[str] = []
    start_line: int | None = None

    for i, line in enumerate(lines, start=1):
        if line.strip():
            if start_line is None:
                start_line = i
            current_lines.append(line)
        else:
            if current_lines:
                elements.append(
                    {
                        "element_id": len(elements),
                        "start_line": start_line,
                        "end_line": i - 1,
                        "n_lines": len(current_lines),
                        "text": "\n".join(current_lines),
                    }
                )
                current_lines = []
                start_line = None

    # Flush any remaining element
    if current_lines:
        elements.append(
            {
                "element_id": len(elements),
                "start_line": start_line,
                "end_line": len(lines),
                "n_lines": len(current_lines),
                "text": "\n".join(current_lines),
            }
        )

    return pl.DataFrame(
        elements,
        schema={
            "element_id": pl.Int64,
            "start_line": pl.Int64,
            "end_line": pl.Int64,
            "n_lines": pl.Int64,
            "text": pl.String,
        },
    )
