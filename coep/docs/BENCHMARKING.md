# Benchmarking Workflow

This document describes how to evaluate the RAG pipeline against a human-verified dataset.

> **Note:** Benchmarking is now a DVC stage (`benchmark`). It runs automatically
> as part of `dvc repro` / `./scripts/dvc_repro.sh`. You can also run it
> individually with `dvc repro benchmark` or standalone via the script directly.

**Important limitations in current state**:
- The system is currently only configured to accept Drug Paraphernalia MonQcle records.
- The benchmarking system is currently hardcoded to function on the CA-LosAngeles jurisdiction 
(Los Angeles, Los Angeles County, California, United States)


## Setup

1.  **Prepare your Dataset**: Input file should be a CSV with each row corresponding to a MonQcle 
database record with columns corresponding to MonQcle variables (questions). The format should be as provided
via a "Standard Report" download from MonQcle.

2. **Prepare your Queries**: Queries should be a CSV file with each row corresponding to a question
in a MonQcle record, for example the Drug Paraphernalia record. Follow-up questions should stay split as
their own rows; parent-child behavior is driven by explicit dependency columns rather than manual prompt
merging.
Columns should include `question_number`, `variable_name`, `query_text`, and any optional dependency
columns such as `Requires "yes" from upstream question:`, `Requires data from upstream question:`, and
`Requires label(s) from upstream question:`. The COEP query adjuster composes the benchmark prompt from
`query_text`, `coding_instructions`, and `response_options`, while preserving hierarchy metadata for the
generic query engine.

3.  **Environment Variables**: Ensure your `.env` file has the necessary API keys 
(`OPENAI_API_KEY` or `OPENROUTER_API_KEY`) and the `LEGISCOPE_COLLECTION_NAME` is set correctly.

## Running the Benchmark

The benchmark is a DVC stage and runs as part of the full pipeline:

```bash
# Run as part of full DVC pipeline
dvc repro
# Or: ./scripts/dvc_repro.sh

# Run only the benchmark stage (requires embed stage outputs)
dvc repro benchmark
# Or: ./scripts/dvc_repro.sh --stage benchmark
```

You can also run the script directly (standalone, outside DVC).
This script is **config-driven**:

- Jurisdiction is read from `params.yaml` (`jurisdiction.*`)
- Query/retrieval behavior is read from `params.yaml` (via `BatchQuerySettings`)
- Paths are read from `config.yaml` (`paths.default_queries_file`, `paths.monqcle_report`, `paths.output_dir`)
- Benchmark series title is read from `params.yaml` at `benchmark.series_title`

### Quick Start (Standalone)
```bash
# Run with defaults from params.yaml + config.yaml
uv run python coep/scripts/benchmark_pipeline.py

# Run with explicit jurisdiction (used by DVC stage)
uv run python coep/scripts/benchmark_pipeline.py --state CA --locality LosAngeles --code-slug municipal-code

# Run with limited query count for quick testing
uv run python coep/scripts/benchmark_pipeline.py --test-limit 5
```

### Full Configuration
```bash
# Example config-first workflow
# 1) Set jurisdiction + benchmark options in params.yaml 
#    (to get transparent debug files, set retrieval.debug to true)
# 2) Set query/report/output paths in config.yaml
# 3) Run:
uv run python coep/scripts/benchmark_pipeline.py
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--state` | Two-letter state code | From `params.yaml` |
| `--locality` | PascalCase locality name | From `params.yaml` |
| `--code-slug` | Code slug identifier | From `params.yaml` |
| `--test-limit` | Limit number of queries (for testing) | None |

> **Note on Debugging:** The `--debug` flag has been removed. Debug mode is now entirely controlled by setting `debug: true` under the `retrieval` section in `params.yaml`. This toggle produces debug files for retrieved sections, LLM relevance assessments, contextual prompts, imported CSVs, and joined pipeline states in `data/output/<jurisdiction_id>/debug`.

### Key Resolved Inputs (from config/params)

- Queries CSV: `config.default_queries_path()`
- MonQcle report CSV: `config.monqcle_report_path()` (repo-root-relative by default, not under `data_dir`)
- Output file: `config.output_dir() / {jurisdiction_id} / benchmark_results.csv` (DVC-tracked)
- Timestamped copy: `config.output_dir() / {jurisdiction_id} / benchmark_results_{timestamp}.csv`
- Metrics: `config.output_dir() / {jurisdiction_id} / benchmark_metrics.json` (DVC metrics)
- Timestamped metrics copy: `config.output_dir() / {jurisdiction_id} / benchmark_metrics_{timestamp}.json`
- Series title: `params["benchmark"]["series_title"]` (fallback: `DPL_2025_Consolidated`)

Jurisdiction, retrieval settings (including HYDE/relevance filtering and debug outputs), and query validation settings are read from `params.yaml`.

## How it Works

1.  **Generation**: The script executes the RAG pipeline for the configured queries CSV (optionally truncated by `--test-limit`).
    - It uses `run_queries` with `n-results` retrieved segments.
    - Optionally applies **HYDE** (Hypothetical Document Embeddings) to improve retrieval.
    - Optionally applies **Relevance Filtering** to remove irrelevant segments before generation.
    - It reads in the MonQcle Standard Report, isolates the correct record, and melts it to a question-answer pair format.
    - It applies COEP-specific query preprocessing via `adjust_drug_paraphernalia_queries()`.
        - Split MonQcle variables are the primary benchmark surface. Legacy combined variables such as
            `dp_collected_combined` and `dp_state_fed_combined` are expanded only as compatibility aliases when
            an older query file still requests them.
        - Child queries can inherit parent retrieval context. The merge keeps all distinct parent and child
            retrieval units, then coalesces exact duplicate chunks or sections.
2.  **Evaluation**: It uses an "LLM-as-a-judge" ("powerful" model) to compare the generated answer against the ground truth.
    - **Note**: The evaluator is provided with the full **Comprehensive Answer** which includes:
        - The Short Answer
        - The detailed Reasoning
        - The Supporting Passages/Citations
    - This ensures the judge evaluates the entire context of the generated response, not just the final conclusion.
        - Rows skipped because an explicit dependency rule was not satisfied are scored deterministically instead:
            blank ground truth counts as correct, non-blank ground truth counts as incorrect, and no judge-model call is made.
3.  **Scoring**: The judge assigns a score (0-10) based on accuracy and provides a reasoning.
    - `benchmark_metrics.json` now exposes `primary_score` / `weighted_query_score` as the headline metric.
    - A timestamped metrics copy is also written for historical auditing alongside the canonical DVC metrics file.
    - Each scorable original benchmark query is worth an equal share of 100 total points.
    - If a query expands into multiple `AND/OR` response-option rows, that query's share is split evenly across those rows so the query can earn partial credit without inflating its weight.
    - Queries with missing or excluded ground truth are omitted from the weighted-score denominator rather than reducing the score ceiling; their counts remain available in the metrics as `weighted_query_unscored` and related fields.
    - The legacy row-level `accuracy_rate` and strict collapsed `collapsed_query_accuracy_rate` remain in the metrics for comparison.
4.  **Output**: A new CSV is saved containing original questions, generated answers (comprehensive), human answers, scores, and reasonings.


## Code Structure

-   `coep/src/eval.py`: Contains the `Evaluator` class and the `EvaluationResult` schema (using Instructor).
-   `coep/scripts/benchmark_pipeline.py`: The executable workflow script.
