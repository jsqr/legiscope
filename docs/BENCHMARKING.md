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

Run the pipeline script from the project root. You can customize the behavior using various command-line arguments.

### Quick Start
```bash
# Test run (limit to 5 queries)
uv run python scripts/benchmark_pipeline.py \
--queries-path data/queries/drug_paraphernalia_queries_clean.csv \
--monqcle-path data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv \
--series-title DPL_2025_Consolidated \
--jurisdiction-id CA-LosAngeles \
--output data/output/CA-LosAngeles/benchmark_results.csv \
--n-results 10 \
--test-limit 5
```

### Full Configuration
```bash
# Full run with all options explicitly set
uv run python scripts/benchmark_pipeline.py \
--queries-path data/queries/drug_paraphernalia_queries_clean.csv \
--monqcle-path data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv \
--series-title DPL_2025_Consolidated \
--jurisdiction-id CA-LosAngeles \
--output data/output/CA-LosAngeles/benchmark_results.csv \
--n-results 10 \
--use-hyde False \
--filter-relevance True \
--relevance-threshold 0.5 \
--validate-supporting-passages True \
--debug
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--queries-path` | Path to queries CSV (required) | - |
| `--jurisdiction-id` | Jurisdiction ID (e.g., CA-LosAngeles) (required) | - |
| `--monqcle-path` | Path to MonQcle CSV | `data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv` |
| `--output` | Output file path | `data/output/benchmark_results.csv` |
| `--n-results` | Number of segments to retrieve | `10` |
| `--test-limit` | Limit number of queries (for testing) | None |
| `--series-title` | MonQcle series title | `DPL_2025_Consolidated` |
| `--use-hyde` | Enable HYDE query rewriting | `False` |
| `--filter-relevance` | Enable LLM relevance filtering of results | `False` |
| `--relevance-threshold` | Threshold for filtering (0.0-1.0) | `0.5` |
| `--validate-supporting-passages` | Validate LLM citations against text | `True` |
| `--debug` | Enable debug mode (saves intermediate CSVs) | `False` |

## How it Works

1.  **Generation**: The script executes the RAG pipeline for `test-limit` queries from the `queries-path` CSV file.
    - It uses `run_queries` with `n-results` retrieved segments.
    - Optionally applies **HYDE** (Hypothetical Document Embeddings) to improve retrieval.
    - Optionally applies **Relevance Filtering** to remove irrelevant segments before generation.
    - It reads in the MonQcle Standard Report, isolates the correct record, and melts it to a question-answer pair format.
2.  **Evaluation**: It uses an "LLM-as-a-judge" ("powerful" model) to compare the generated answer against the ground truth.
    - **Note**: The evaluator is provided with the full **Comprehensive Answer** which includes:
        - The Short Answer
        - The detailed Reasoning
        - The Supporting Passages/Citations
    - This ensures the judge evaluates the entire context of the generated response, not just the final conclusion.
3.  **Scoring**: The judge assigns a score (0-10) based on accuracy and provides a reasoning.
4.  **Output**: A new CSV is saved containing original questions, generated answers (comprehensive), human answers, scores, and reasonings.


## Code Structure

-   `src/legiscope/eval.py`: Contains the `Evaluator` class and the `EvaluationResult` schema (using Instructor).
-   `scripts/benchmark_pipeline.py`: The executable workflow script.
