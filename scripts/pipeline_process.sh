#!/bin/bash

# pipeline_process.sh - Create embeddings and build search index
# Segments code, creates embeddings, and builds ChromaDB index

set -e  # Exit on error

# Configuration
STATE="$1"
MUNICIPALITY="$2"
CODE_SLUG="$3"

# Check arguments
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <STATE> <MUNICIPALITY|-> <CODE_SLUG>"
    echo "  Use '-' for MUNICIPALITY for state-level codes."
    echo ""
    echo "Examples:"
    echo "  $0 CA LosAngeles municipal-code"
    echo "  $0 CA - penal-code"
    exit 1
fi

# Build common args
COMMON_ARGS="--state $STATE --code-slug $CODE_SLUG"
if [[ "$MUNICIPALITY" != "-" ]]; then
    COMMON_ARGS="$COMMON_ARGS --municipality $MUNICIPALITY"
fi

echo "Processing legal code: state=$STATE municipality=$MUNICIPALITY code=$CODE_SLUG..."

# Determine the data directory
if [[ "$MUNICIPALITY" != "-" ]]; then
    CODE_DIR="data/laws/$STATE/$MUNICIPALITY/$CODE_SLUG"
else
    CODE_DIR="data/laws/$STATE/State/$CODE_SLUG"
fi

# Check if code.md exists
if [[ ! -f "$CODE_DIR/code.md" ]]; then
    echo "Error: code.md not found at $CODE_DIR/code.md"
    echo "Run pipeline_parse.sh first to create Markdown file"
    exit 1
fi

# Step 1: Segment legal code
echo "Step 1: Segmenting Markdown into sections..."
source .venv/bin/activate && python scripts/segment_legal_code.py $COMMON_ARGS

# Step 2: Create embeddings
echo "Step 2: Generating embeddings..."
source .venv/bin/activate && python scripts/create_embeddings.py $COMMON_ARGS

# Step 3: Build ChromaDB index
echo "Step 3: Building ChromaDB index..."
source .venv/bin/activate && python scripts/build_chroma_index.py

echo "Processing completed successfully!"
echo "Files created:"
echo "  - $CODE_DIR/sections.parquet"
echo "  - $CODE_DIR/segments.parquet"
echo "  - $CODE_DIR/embeddings.parquet"
echo "  - ChromaDB index updated"
echo ""
echo "Next steps:"
echo "  - Run queries: make query STATE=$STATE MUNICIPALITY=${MUNICIPALITY} CODE_SLUG=$CODE_SLUG QUERIES=<path>"
