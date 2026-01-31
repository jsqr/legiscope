#!/bin/bash

# pipeline.sh - Jurisdiction processing pipeline
# Basic workflow from DOCX files to searchable embeddings

set -e  # Exit on error

# Configuration
STATE="$1"
MUNICIPALITY="$2"
CODE_SLUG="$3"
QUERIES_FILE="$4"  # Optional queries file

# Check basic arguments
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <STATE> <MUNICIPALITY|-> <CODE_SLUG> [QUERIES_FILE]"
    echo "  Use '-' for MUNICIPALITY for state-level codes."
    echo ""
    echo "Examples:"
    echo "  $0 CA LosAngeles municipal-code"
    echo "  $0 CA - penal-code"
    echo "  $0 CA LosAngeles municipal-code data/queries/example.csv"
    exit 1
fi

# Build common args
COMMON_ARGS="--state $STATE --code-slug $CODE_SLUG"
if [[ "$MUNICIPALITY" != "-" ]]; then
    COMMON_ARGS="$COMMON_ARGS --municipality $MUNICIPALITY"
fi

echo "Starting pipeline for state=$STATE municipality=$MUNICIPALITY code=$CODE_SLUG..."

# Step 1: Create directory structure and register jurisdiction/code
echo "Step 1: Creating directory structure..."
source .venv/bin/activate && python scripts/create_jurisdiction.py $COMMON_ARGS

# Step 2: Convert DOCX to text (if DOCX files exist)
# Determine the data directory
if [[ "$MUNICIPALITY" != "-" ]]; then
    CODE_DIR="data/laws/$STATE/$MUNICIPALITY/$CODE_SLUG"
else
    CODE_DIR="data/laws/$STATE/State/$CODE_SLUG"
fi
RAW_DIR="$CODE_DIR/raw"

if [[ -d "$RAW_DIR" ]] && [[ -n "$(ls -A "$RAW_DIR"/*.docx 2>/dev/null)" ]]; then
    echo "Step 2: Converting DOCX to text..."
    ./scripts/convert_docx.sh "$RAW_DIR"
else
    echo "Step 2: Skipping DOCX conversion (no DOCX files found)"
fi

# Step 3: Convert text to Markdown
echo "Step 3: Converting text to structured Markdown..."
source .venv/bin/activate && python scripts/convert_to_markdown.py $COMMON_ARGS

# Step 4: Segment legal code
echo "Step 4: Segmenting Markdown into sections..."
source .venv/bin/activate && python scripts/segment_legal_code.py $COMMON_ARGS

# Step 5: Create embeddings
echo "Step 5: Generating embeddings..."
source .venv/bin/activate && python scripts/create_embeddings.py $COMMON_ARGS

# Step 6: Build ChromaDB index
echo "Step 6: Building ChromaDB index..."
source .venv/bin/activate && python scripts/build_chroma_index.py

# Step 7: Run queries (if queries file provided)
if [[ -n "$QUERIES_FILE" ]] && [[ -f "$QUERIES_FILE" ]]; then
    echo "Step 7: Running queries from $QUERIES_FILE..."
    OUTPUT_PATH="data/output/query_results.csv"

    source .venv/bin/activate && python scripts/run_queries.py \
        $COMMON_ARGS \
        --queries-path "$QUERIES_FILE" \
        --output "$OUTPUT_PATH"

    echo "Query results saved to: $OUTPUT_PATH"
elif [[ -n "$QUERIES_FILE" ]]; then
    echo "Step 7: Skipping queries (file not found: $QUERIES_FILE)"
else
    echo "Step 7: Skipping queries (no queries file provided)"
fi

echo "Pipeline completed successfully!"
echo "Files created in: $CODE_DIR"
