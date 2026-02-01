#!/bin/bash

# pipeline_init.sh - Initialize jurisdiction directory structure
# Creates directory structure and registers jurisdiction/code in Parquet files

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

echo "Initializing jurisdiction: state=$STATE municipality=$MUNICIPALITY code=$CODE_SLUG..."

# Step 1: Create directory structure and register jurisdiction/code
echo "Creating directory structure..."
source .venv/bin/activate && python scripts/create_jurisdiction.py $COMMON_ARGS \
    --name "$(echo $CODE_SLUG | sed 's/-/ /g' | awk '{for(i=1;i<=NF;i++)sub(/./,toupper(substr($i,1,1)),$i)}1')" \
    --code-type municipal

# Determine the data directory
if [[ "$MUNICIPALITY" != "-" ]]; then
    CODE_DIR="data/laws/$STATE/$MUNICIPALITY/$CODE_SLUG"
else
    CODE_DIR="data/laws/$STATE/State/$CODE_SLUG"
fi

echo "Initialization complete!"
echo "Directory created: $CODE_DIR"
echo ""
echo "Next steps:"
echo "  1. Place your raw files (DOCX, TXT, etc.) in: $CODE_DIR/raw/"
echo "  2. Run: make parse STATE=$STATE MUNICIPALITY=${MUNICIPALITY} CODE_SLUG=$CODE_SLUG"
