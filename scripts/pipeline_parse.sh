#!/bin/bash

# DEPRECATED: Use the DVC pipeline instead:
#   ./scripts/dvc_repro.sh --state STATE --locality LOCALITY --code-slug SLUG --stage parse
#
# pipeline_parse.sh - Convert raw files to structured Markdown
# Converts DOCX to text (if present) and text to Markdown

set -e  # Exit on error

# Configuration
STATE="$1"
LOCALITY="$2"
CODE_SLUG="$3"

# Check arguments
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <STATE> <LOCALITY|-> <CODE_SLUG>"
    echo "  Use '-' for LOCALITY for state-level codes."
    echo ""
    echo "Examples:"
    echo "  $0 CA LosAngeles municipal-code"
    echo "  $0 CA - penal-code"
    exit 1
fi

# Build common args
COMMON_ARGS="--state $STATE --code-slug $CODE_SLUG"
if [[ "$LOCALITY" != "-" ]]; then
    COMMON_ARGS="$COMMON_ARGS --locality $LOCALITY"
fi

echo "Parsing legal code: state=$STATE locality=$LOCALITY code=$CODE_SLUG..."

# Determine the data directory
if [[ "$LOCALITY" != "-" ]]; then
    CODE_DIR="data/laws/$STATE/$LOCALITY/$CODE_SLUG"
else
    CODE_DIR="data/laws/$STATE/State/$CODE_SLUG"
fi
RAW_DIR="$CODE_DIR/raw"

# Check if directory exists
if [[ ! -d "$CODE_DIR" ]]; then
    echo "Error: Directory does not exist: $CODE_DIR"
    echo "Run pipeline_init.sh first to create directory structure"
    exit 1
fi

# Step 1: Convert DOCX to text (if DOCX files exist)
if [[ -d "$RAW_DIR" ]] && [[ -n "$(ls -A "$RAW_DIR"/*.docx 2>/dev/null)" ]]; then
    echo "Step 1: Converting DOCX to text..."
    ./scripts/convert_docx.sh "$RAW_DIR"
else
    echo "Step 1: Skipping DOCX conversion (no DOCX files found)"
fi

# Step 2: Convert text to Markdown
echo "Step 2: Converting text to structured Markdown..."
source .venv/bin/activate && python scripts/convert_to_markdown.py $COMMON_ARGS

echo "Parsing completed successfully!"
echo "Output: $CODE_DIR/code.md"
echo ""
echo "Next steps:"
echo "  1. Review the Markdown file: $CODE_DIR/code.md"
echo "  2. Run: make process STATE=$STATE LOCALITY=${LOCALITY} CODE_SLUG=$CODE_SLUG"
