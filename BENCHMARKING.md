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

Run the pipeline script from the project root:

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

# Full run
uv run python scripts/benchmark_pipeline.py \
--queries-path data/queries/drug_paraphernalia_queries_clean.csv \
--monqcle-path data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv \
--series-title DPL_2025_Consolidated \
--jurisdiction-id CA-LosAngeles \
--output data/output/CA-LosAngeles/benchmark_results.csv \
--n-results 10
```

## How it Works

1.  **Generation**: The script executes the RAG pipeline for `test-limit` queries from the `queries-path` CSV file (one row per query), using the `run_queries` function with `n-results` retrieved segments from the jurisdiction corresponding to `jurisdiction-id`.
It also reads in the MonQcle Standard Report, isolates the correct record (row), and 
pivots (melts) the row to a format of one question-answer pair per row. The query results and MonQcle records are joined by `variable_name`.
2.  **Evaluation**: It uses an "LLM-as-a-judge" ("powerful" model) to compare the generated answer against the ground truth.
3.  **Scoring**: The judge assigns a score (0-10) for each query and provides a reasoning. This score
is averaged to provide a global score.
4.  **Output**: A new CSV is saved containing the original questions, generated answers, human answers, scores, and reasonings.


## Code Structure

-   `src/legiscope/eval.py`: Contains the `Evaluator` class and the `EvaluationResult` schema (using Instructor).
-   `scripts/benchmark_pipeline.py`: The executable workflow script.
