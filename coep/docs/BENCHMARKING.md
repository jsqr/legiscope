# Benchmarking Workflow

This document describes how to evaluate the RAG pipeline against a human-verified dataset.

**Important limitations in current state**:
- The system is currently only configured to accept Drug Paraphernalia MonQcle records.
- The benchmarking system is currently hardcoded to function on the CA-LosAngeles jurisdiction 
(Los Angeles, Los Angeles County, California, United States)


## Setup

1.  **Prepare your Dataset**: Input file should be a CSV with each row corresponding to a MonQcle 
database record with columns corresponding to MonQcle variables (questions). The format should be as provided
via a "Standard Report" download from MonQcle.

2. **Prepare your Queries**: Queries should be a CSV file with each row corresponding to a question
in a MonQcle record, for example the Drug Paraphernalia record. Follow-up questions should be merged
with the parent question to provide adequate context during query to the LLM.
Columns should include `question` and the `variable_name` from MonQcle.

3.  **Environment Variables**: Ensure your `.env` file has the necessary API keys 
(`OPENAI_API_KEY` or `MISTRAL_API_KEY`) and the `LEGISCOPE_COLLECTION_NAME` is set correctly.

## Running the Benchmark

Run the pipeline script from the project root.

This script is now mostly **config-driven**:

- Jurisdiction is read from `params.yaml` (`jurisdiction.*`)
- Query/retrieval behavior is read from `params.yaml` (via `BatchQuerySettings`)
- Paths are read from `config.yaml` (`paths.default_queries_file`, `paths.monqcle_report`, `paths.output_dir`)
- Benchmark series title is read from `params.yaml` at `benchmark.series_title`

### Quick Start
```bash
# Run with defaults from params.yaml + config.yaml
uv run python coep/scripts/benchmark_pipeline.py

# Debug run with limited query count
uv run python coep/scripts/benchmark_pipeline.py --test-limit 5 --debug
```

### Full Configuration
```bash
# Example config-first workflow
# 1) Set jurisdiction + benchmark options in params.yaml
# 2) Set query/report/output paths in config.yaml
# 3) Run:
uv run python coep/scripts/benchmark_pipeline.py --debug
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--test-limit` | Limit number of queries (for testing) | None |
| `--debug` | Enable debug mode (saves intermediate CSVs) | `False` |

### Key Resolved Inputs (from config/params)

- Queries CSV: `config.default_queries_path()`
- MonQcle report CSV: `config.monqcle_report_path()`
- Output file: `config.output_dir() / {jurisdiction_id} / benchmark_results.csv`
- Series title: `params["benchmark"]["series_title"]` (fallback: `DPL_2025_Consolidated`)

Jurisdiction, retrieval settings (including HYDE/relevance filtering), and query
validation settings are read from `params.yaml`.

## How it Works

1.  **Generation**: The script executes the RAG pipeline for the configured queries CSV (optionally truncated by `--test-limit`).
    - It uses `run_queries` with `n-results` retrieved segments.
    - Optionally applies **HYDE** (Hypothetical Document Embeddings) to improve retrieval.
    - Optionally applies **Relevance Filtering** to remove irrelevant segments before generation.
    - It reads in the MonQcle Standard Report, isolates the correct record, and melts it to a question-answer pair format.
    - It applies COEP-specific query preprocessing via `adjust_drug_paraphernalia_queries()`.
2.  **Evaluation**: It uses an "LLM-as-a-judge" ("powerful" model) to compare the generated answer against the ground truth.
    - **Note**: The evaluator is provided with the full **Comprehensive Answer** which includes:
        - The Short Answer
        - The detailed Reasoning
        - The Supporting Passages/Citations
    - This ensures the judge evaluates the entire context of the generated response, not just the final conclusion.
3.  **Scoring**: The judge assigns a score (0-10) based on accuracy and provides a reasoning.
4.  **Output**: A new CSV is saved containing original questions, generated answers (comprehensive), human answers, scores, and reasonings.


## Code Structure

-   `coep/src/eval.py`: Contains the `Evaluator` class and the `EvaluationResult` schema (using Instructor).
-   `coep/scripts/benchmark_pipeline.py`: The executable workflow script.
