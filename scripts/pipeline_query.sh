#!/bin/bash

# DEPRECATED: Use scripts/run_queries.py directly instead.
# Query execution is not a DVC stage.
#
# pipeline_query.sh - Run queries against processed codes
# Executes batch queries and saves results

set -e  # Exit on error

# Configuration
STATE="$1"
LOCALITY="$2"
CODE_SLUG="$3"
QUERIES_FILE="$4"

# Check arguments
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <STATE> <LOCALITY|-> <CODE_SLUG> <QUERIES_FILE>"
    echo "  Use '-' for LOCALITY for state-level codes."
    echo ""
    echo "Examples:"
    echo "  $0 CA LosAngeles municipal-code data/queries/example.csv"
    echo "  $0 CA - penal-code data/queries/example.csv"
    exit 1
fi

# Build common args
COMMON_ARGS="--state $STATE --code-slug $CODE_SLUG"
if [[ "$LOCALITY" != "-" ]]; then
    COMMON_ARGS="$COMMON_ARGS --locality $LOCALITY"
fi

echo "Running queries: state=$STATE locality=$LOCALITY code=$CODE_SLUG..."

# Check if queries file exists
if [[ ! -f "$QUERIES_FILE" ]]; then
    echo "Error: Queries file not found: $QUERIES_FILE"
    exit 1
fi

# Check if ChromaDB exists (rough check)
if [[ ! -d "data/chroma_db" ]]; then
    echo "Error: ChromaDB index not found at data/chroma_db"
    echo "Run pipeline_process.sh first to build the index"
    exit 1
fi

# Run queries
OUTPUT_PATH="data/output/query_results.csv"
echo "Running queries from $QUERIES_FILE..."

source .venv/bin/activate && python scripts/run_queries.py \
    $COMMON_ARGS \
    --queries-path "$QUERIES_FILE" \
    --output "$OUTPUT_PATH"

echo "Queries completed successfully!"
echo "Results saved to: $OUTPUT_PATH"
