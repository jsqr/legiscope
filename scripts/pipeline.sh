#!/bin/bash

# pipeline_simple.sh - Simplified jurisdiction processing pipeline
# Basic workflow from DOCX files to searchable embeddings

set -e  # Exit on error

# Basic configuration
STATE="$1"
MUNICIPALITY="$2"
JURISDICTION_NAME="${STATE}-${MUNICIPALITY}"

# Check basic arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <STATE> <MUNICIPALITY>"
    echo "Example: $0 NY \"New York\""
    exit 1
fi

echo "Starting pipeline for $JURISDICTION_NAME..."

# Step 1: Create directory structure
echo "Step 1: Creating directory structure..."
source .venv/bin/activate && python scripts/create_jurisdiction.py "$STATE" "$MUNICIPALITY"

# Step 2: Convert DOCX to text (if DOCX files exist)
RAW_DIR="data/laws/$JURISDICTION_NAME/raw"
if [[ -d "$RAW_DIR" ]] && [[ -n "$(ls -A "$RAW_DIR"/*.docx 2>/dev/null)" ]]; then
    echo "Step 2: Converting DOCX to text..."
    ./scripts/convert_docx.sh "$JURISDICTION_NAME"
else
    echo "Step 2: Skipping DOCX conversion (no DOCX files found)"
fi

# Step 3: Convert text to Markdown
echo "Step 3: Converting text to structured Markdown..."
source .venv/bin/activate && python scripts/convert_to_markdown.py "data/laws/$JURISDICTION_NAME"

# Step 4: Segment legal code
echo "Step 4: Segmenting Markdown into sections..."
source .venv/bin/activate && python scripts/segment_legal_code.py "data/laws/$JURISDICTION_NAME"

# Step 5: Create embeddings
echo "Step 5: Generating embeddings..."
source .venv/bin/activate && python scripts/create_embeddings.py "data/laws/$JURISDICTION_NAME"

echo "Pipeline completed successfully for $JURISDICTION_NAME!"
echo "Files created in: data/laws/$JURISDICTION_NAME"
