#!/bin/bash

# pipeline.sh - Complete jurisdiction processing pipeline
# Automates the entire workflow from DOCX files to searchable embeddings

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURATION & DEFAULTS
# =============================================================================

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Default configuration
DEFAULT_MODEL="gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL="embeddinggemma"
DEFAULT_MAX_LINES=150
DEFAULT_TOKEN_LIMIT=1024
DEFAULT_WORDS_PER_TOKEN=0.78
DEFAULT_CHROMA_DB_PATH="data/chroma_db"
DEFAULT_COLLECTION_NAME="legal_code_all"
DEFAULT_INPUT_FILE="code.txt"
DEFAULT_MARKDOWN_FILE="code.md"
DEFAULT_SEGMENTS_FILE="segments.parquet"
DEFAULT_EMBEDDINGS_FILE="embeddings.parquet"

# Global variables
VERBOSE=false
DRY_RUN=false
SKIP_DOCX=false
SKIP_MARKDOWN=false
SKIP_SEGMENT=false
SKIP_EMBEDDINGS=false
RESUME_FROM=""
LOG_FILE=""

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_stage() {
    local stage_num=$1
    local total_stages=$2
    local description=$3
    echo -e "${PURPLE}[$stage_num/$total_stages]${NC} $description..."
}

# Log function
log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" >> "$LOG_FILE"
    if [[ "$VERBOSE" == true ]]; then
        echo "$message"
    fi
}

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 <STATE> "<MUNICIPALITY>" [OPTIONS]

Positional arguments:
  STATE           Two-letter state abbreviation (e.g., NY, CA, IL)
  MUNICIPALITY    Municipality name (e.g., "New York", LosAngeles)

Pipeline control:
  --skip-docx          Skip DOCX to text conversion
  --skip-markdown      Skip text to Markdown conversion
  --skip-segment       Skip Markdown segmentation
  --skip-embeddings    Skip embedding generation
  --resume-from STAGE  Resume from specific stage (docx|markdown|segment|embeddings)

File configuration:
  --input-file FILE        Input text filename (default: $DEFAULT_INPUT_FILE)
  --markdown-file FILE     Markdown filename (default: $DEFAULT_MARKDOWN_FILE)
  --segments-file FILE     Segments parquet filename (default: $DEFAULT_SEGMENTS_FILE)
  --embeddings-file FILE   Embeddings parquet filename (default: $DEFAULT_EMBEDDINGS_FILE)

Model configuration:
  --model MODEL            OpenAI model for text analysis (default: $DEFAULT_MODEL)
  --embedding-model MODEL  Embedding model (default: $DEFAULT_EMBEDDING_MODEL)

Processing configuration:
  --max-lines NUM          Max lines for heading analysis (default: $DEFAULT_MAX_LINES)
  --token-limit NUM        Max tokens per segment (default: $DEFAULT_TOKEN_LIMIT)
  --words-per-token NUM    Words per token ratio (default: $DEFAULT_WORDS_PER_TOKEN)

Database configuration:
  --chroma-db-path PATH    ChromaDB directory path (default: $DEFAULT_CHROMA_DB_PATH)
  --collection-name NAME   ChromaDB collection name (default: $DEFAULT_COLLECTION_NAME)
  --no-shared-chroma       Use parquet-only approach instead of shared ChromaDB

General options:
  --verbose                Enable detailed output
  --dry-run                Show commands without executing
  --log-file FILE          Log file path (default: pipeline_YYYY-MM-DD_HH-MM-SS.log)
  --help                   Show this help message

Examples:
  $0 NY "New York"
  $0 CA LosAngeles --verbose
  $0 TX Houston --model gpt-4 --embedding-model nomic-embed-text
  $0 IL Chicago --skip-docx --input-file municipal_code.txt
  $0 FL Miami --resume-from embeddings --verbose

EOF
}

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

validate_environment() {
    log "Validating environment..."
    
    # Check if we're in the project root
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "scripts" ]] || [[ ! -d "src" ]]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Check Python virtual environment
    if [[ ! -d ".venv" ]]; then
        print_error "Python virtual environment not found. Run 'make env' first"
        exit 1
    fi
    
    # Check required tools
    if ! command -v pandoc &> /dev/null; then
        print_error "pandoc is not installed. Please install it before running this script."
        exit 1
    fi
    
    # Check OpenAI API key
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        print_error "OPENAI_API_KEY environment variable is not set"
        exit 1
    fi
    
    # Check ollama service (only if not skipping embeddings)
    if [[ "$SKIP_EMBEDDINGS" == false ]]; then
        if ! command -v ollama &> /dev/null; then
            print_error "ollama is not installed. Install with: pip install ollama"
            exit 1
        fi
        
        # Test ollama service
        if ! ollama list &> /dev/null; then
            print_error "ollama service is not running. Start it with: ollama serve"
            exit 1
        fi
    fi
    
    # Check if required models are available
    if [[ "$SKIP_EMBEDDINGS" == false ]]; then
        if ! ollama list | grep -q "$EMBEDDING_MODEL"; then
            print_warning "Embedding model '$EMBEDDING_MODEL' not found in ollama"
            print_info "Pull it with: ollama pull $EMBEDDING_MODEL"
        fi
    fi
    
    print_success "Environment validation passed"
}

# =============================================================================
# PIPELINE STAGES
# =============================================================================

stage_1_create_directory() {
    print_stage 1 5 "Creating directory structure"
    
    local cmd="source .venv/bin/activate && python scripts/create_jurisdiction.py $STATE \"$MUNICIPALITY\""
    if [[ "$VERBOSE" == true ]]; then
        cmd="$cmd --verbose"
    fi
    
    log "Creating directory structure for $STATE-$MUNICIPALITY"
    execute_command "$cmd" "Directory structure creation"
    
    # Verify directory was created (skip in dry-run mode)
    if [[ "$DRY_RUN" == false ]]; then
        local jurisdiction_dir="data/laws/$JURISDICTION_NAME"
        if [[ ! -d "$jurisdiction_dir" ]]; then
            print_error "Failed to create directory structure"
            exit 1
        fi
        print_success "Directory structure created: $jurisdiction_dir"
    else
        print_success "Directory structure would be created: data/laws/$JURISDICTION_NAME"
    fi
}

stage_2_convert_docx() {
    print_stage 2 5 "Converting DOCX to plain text"
    
    # Check if raw directory exists and has DOCX files (skip in dry-run mode)
    if [[ "$DRY_RUN" == false ]]; then
        local raw_dir="data/laws/$JURISDICTION_NAME/raw"
        if [[ ! -d "$raw_dir" ]]; then
            print_error "Raw directory not found: $raw_dir"
            print_info "Please place DOCX files in: $raw_dir"
            exit 1
        fi
        
        local docx_files=("$raw_dir"/*.docx)
        if [[ ! -f "${docx_files[0]}" ]]; then
            print_warning "No DOCX files found in $raw_dir"
            print_info "Skipping DOCX conversion stage"
            return
        fi
    fi
    
    local cmd="./convert_docx.sh $JURISDICTION_NAME"
    log "Converting DOCX files for $JURISDICTION_NAME"
    execute_command "$cmd" "DOCX conversion"
    
    # Verify output file was created (skip in dry-run mode)
    if [[ "$DRY_RUN" == false ]]; then
        local output_file="data/laws/$JURISDICTION_NAME/$INPUT_FILE"
        if [[ ! -f "$output_file" ]]; then
            print_error "Failed to create output file: $output_file"
            exit 1
        fi
        
        local file_size=$(stat -f%z "$output_file" 2>/dev/null || stat -c%s "$output_file" 2>/dev/null || echo "unknown")
        print_success "DOCX conversion completed: $output_file (${file_size} bytes)"
    else
        print_success "DOCX conversion would create: data/laws/$JURISDICTION_NAME/$INPUT_FILE"
    fi
}

stage_3_convert_to_markdown() {
    print_stage 3 5 "Converting text to structured Markdown"
    
    # Check if input file exists (skip in dry-run mode)
    if [[ "$DRY_RUN" == false ]]; then
        local input_path="data/laws/$JURISDICTION_NAME/$INPUT_FILE"
        if [[ ! -f "$input_path" ]]; then
            print_error "Input file not found: $input_path"
            exit 1
        fi
    fi
    
    local cmd="source .venv/bin/activate && python scripts/convert_to_markdown.py data/laws/$JURISDICTION_NAME"
    cmd="$cmd --input-file $INPUT_FILE"
    cmd="$cmd --output-file $MARKDOWN_FILE"
    cmd="$cmd --max-lines $MAX_LINES"
    cmd="$cmd --model $MODEL"
    
    if [[ "$VERBOSE" == true ]]; then
        cmd="$cmd --verbose"
    fi
    
    log "Converting text to Markdown for $JURISDICTION_NAME"
    execute_command "$cmd" "Markdown conversion"
    
    # Verify output file was created (skip in dry-run mode)
    if [[ "$DRY_RUN" == false ]]; then
        local output_path="data/laws/$JURISDICTION_NAME/$MARKDOWN_FILE"
        if [[ ! -f "$output_path" ]]; then
            print_error "Failed to create Markdown file: $output_path"
            exit 1
        fi
        print_success "Markdown conversion completed: $output_path"
    else
        print_success "Markdown conversion would create: data/laws/$JURISDICTION_NAME/$MARKDOWN_FILE"
    fi
}

stage_4_segment_legal_code() {
    print_stage 4 5 "Segmenting Markdown into sections and segments"
    
    # Check if markdown file exists (skip in dry-run mode)
    if [[ "$DRY_RUN" == false ]]; then
        local markdown_path="data/laws/$JURISDICTION_NAME/$MARKDOWN_FILE"
        if [[ ! -f "$markdown_path" ]]; then
            print_error "Markdown file not found: $markdown_path"
            exit 1
        fi
    fi
    
    local cmd="source .venv/bin/activate && python scripts/segment_legal_code.py data/laws/$JURISDICTION_NAME"
    cmd="$cmd --markdown-file $MARKDOWN_FILE"
    cmd="$cmd --token-limit $TOKEN_LIMIT"
    cmd="$cmd --words-per-token $WORDS_PER_TOKEN"
    
    if [[ "$VERBOSE" == true ]]; then
        cmd="$cmd --verbose"
    fi
    
    log "Segmenting legal code for $JURISDICTION_NAME"
    execute_command "$cmd" "Legal code segmentation"
    
    # Verify output files were created (skip in dry-run mode)
    if [[ "$DRY_RUN" == false ]]; then
        local tables_dir="data/laws/$JURISDICTION_NAME/tables"
        local sections_file="$tables_dir/sections.parquet"
        local segments_file="$tables_dir/$DEFAULT_SEGMENTS_FILE"
        
        if [[ ! -f "$sections_file" ]] || [[ ! -f "$segments_file" ]]; then
            print_error "Failed to create segment files"
            exit 1
        fi
        print_success "Segmentation completed: sections.parquet and segments.parquet"
    else
        print_success "Segmentation would create: sections.parquet and segments.parquet"
    fi
}

stage_5_create_embeddings() {
    print_stage 5 5 "Generating embeddings and populating ChromaDB"
    
    # Check if segments file exists (skip in dry-run mode)
    if [[ "$DRY_RUN" == false ]]; then
        local segments_path="data/laws/$JURISDICTION_NAME/tables/$DEFAULT_SEGMENTS_FILE"
        if [[ ! -f "$segments_path" ]]; then
            print_error "Segments file not found: $segments_path"
            exit 1
        fi
    fi
    
    local cmd="source .venv/bin/activate && python scripts/create_embeddings.py data/laws/$JURISDICTION_NAME"
    cmd="$cmd --model $EMBEDDING_MODEL"
    cmd="$cmd --segments-file $DEFAULT_SEGMENTS_FILE"
    cmd="$cmd --embeddings-file $DEFAULT_EMBEDDINGS_FILE"
    cmd="$cmd --chroma-db-path $CHROMA_DB_PATH"
    cmd="$cmd --collection-name $COLLECTION_NAME"
    
    if [[ "$NO_SHARED_CHROMA" == true ]]; then
        cmd="$cmd --no-shared-chroma"
    fi
    
    if [[ "$VERBOSE" == true ]]; then
        cmd="$cmd --verbose"
    fi
    
    log "Creating embeddings for $JURISDICTION_NAME"
    execute_command "$cmd" "Embedding generation"
    
    # Verify output files were created (skip in dry-run mode)
    if [[ "$DRY_RUN" == false ]]; then
        local embeddings_path="data/laws/$JURISDICTION_NAME/tables/$DEFAULT_EMBEDDINGS_FILE"
        if [[ ! -f "$embeddings_path" ]]; then
            print_error "Failed to create embeddings file: $embeddings_path"
            exit 1
        fi
        print_success "Embeddings completed: $DEFAULT_EMBEDDINGS_FILE"
    else
        print_success "Embeddings would create: $DEFAULT_EMBEDDINGS_FILE"
    fi
}

# =============================================================================
# COMMAND EXECUTION
# =============================================================================

execute_command() {
    local cmd="$1"
    local description="$2"
    
    if [[ "$DRY_RUN" == true ]]; then
        print_info "[DRY RUN] Would execute: $cmd"
        return 0
    fi
    
    log "Executing: $cmd"
    
    if [[ "$VERBOSE" == true ]]; then
        print_info "Running: $description"
    fi
    
    # Execute command and capture output
    local output
    local exit_code
    
    if output=$(eval "$cmd" 2>&1); then
        exit_code=0
    else
        exit_code=$?
    fi
    
    # Log output
    if [[ -n "$output" ]]; then
        echo "$output" >> "$LOG_FILE"
        if [[ "$VERBOSE" == true ]]; then
            echo "$output"
        fi
    fi
    
    if [[ $exit_code -ne 0 ]]; then
        print_error "$description failed with exit code $exit_code"
        print_error "Check log file for details: $LOG_FILE"
        exit $exit_code
    fi
    
    return $exit_code
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Initialize log file
    if [[ -z "$LOG_FILE" ]]; then
        LOG_FILE="pipeline_$(date '+%Y-%m-%d_%H-%M-%S').log"
    fi
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Validate inputs
    validate_arguments
    
    # Print configuration
    print_configuration
    
    # Validate environment
    validate_environment
    
    # Execute pipeline stages
    execute_pipeline
    
    # Print summary
    print_summary
}

parse_arguments() {
    # Check for help first
    if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        show_usage
        exit 0
    fi
    
    # Extract positional arguments first (STATE and MUNICIPALITY)
    # STATE is always the first argument
    if [[ $# -lt 2 ]]; then
        print_error "Both STATE and MUNICIPALITY are required"
        show_usage
        exit 1
    fi
    
    STATE="$1"
    shift
    
    # MUNICIPALITY is everything up to the first option (starts with --)
    MUNICIPALITY=""
    while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        if [[ -n "$MUNICIPALITY" ]]; then
            MUNICIPALITY="$MUNICIPALITY $1"
        else
            MUNICIPALITY="$1"
        fi
        shift
    done
    
    # Now parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_usage
                exit 0
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-docx)
                SKIP_DOCX=true
                shift
                ;;
            --skip-markdown)
                SKIP_MARKDOWN=true
                shift
                ;;
            --skip-segment)
                SKIP_SEGMENT=true
                shift
                ;;
            --skip-embeddings)
                SKIP_EMBEDDINGS=true
                shift
                ;;
            --resume-from)
                RESUME_FROM="$2"
                shift 2
                ;;
            --input-file)
                INPUT_FILE="$2"
                shift 2
                ;;
            --markdown-file)
                MARKDOWN_FILE="$2"
                shift 2
                ;;
            --segments-file)
                DEFAULT_SEGMENTS_FILE="$2"
                shift 2
                ;;
            --embeddings-file)
                DEFAULT_EMBEDDINGS_FILE="$2"
                shift 2
                ;;
            --model)
                MODEL="$2"
                shift 2
                ;;
            --embedding-model)
                EMBEDDING_MODEL="$2"
                shift 2
                ;;
            --max-lines)
                MAX_LINES="$2"
                shift 2
                ;;
            --token-limit)
                TOKEN_LIMIT="$2"
                shift 2
                ;;
            --words-per-token)
                WORDS_PER_TOKEN="$2"
                shift 2
                ;;
            --chroma-db-path)
                CHROMA_DB_PATH="$2"
                shift 2
                ;;
            --collection-name)
                COLLECTION_NAME="$2"
                shift 2
                ;;
            --no-shared-chroma)
                NO_SHARED_CHROMA=true
                shift
                ;;
            --log-file)
                LOG_FILE="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Set defaults for variables that weren't set
    MODEL="${MODEL:-$DEFAULT_MODEL}"
    EMBEDDING_MODEL="${EMBEDDING_MODEL:-$DEFAULT_EMBEDDING_MODEL}"
    MAX_LINES="${MAX_LINES:-$DEFAULT_MAX_LINES}"
    TOKEN_LIMIT="${TOKEN_LIMIT:-$DEFAULT_TOKEN_LIMIT}"
    WORDS_PER_TOKEN="${WORDS_PER_TOKEN:-$DEFAULT_WORDS_PER_TOKEN}"
    CHROMA_DB_PATH="${CHROMA_DB_PATH:-$DEFAULT_CHROMA_DB_PATH}"
    COLLECTION_NAME="${COLLECTION_NAME:-$DEFAULT_COLLECTION_NAME}"
    INPUT_FILE="${INPUT_FILE:-$DEFAULT_INPUT_FILE}"
    MARKDOWN_FILE="${MARKDOWN_FILE:-$DEFAULT_MARKDOWN_FILE}"
    DEFAULT_SEGMENTS_FILE="${DEFAULT_SEGMENTS_FILE:-segments.parquet}"
    DEFAULT_EMBEDDINGS_FILE="${DEFAULT_EMBEDDINGS_FILE:-embeddings.parquet}"
    NO_SHARED_CHROMA="${NO_SHARED_CHROMA:-false}"
}

validate_arguments() {
    # Check required arguments
    if [[ -z "${STATE:-}" ]] || [[ -z "${MUNICIPALITY:-}" ]]; then
        print_error "Both STATE and MUNICIPALITY are required"
        show_usage
        exit 1
    fi
    
    # Validate state format
    if [[ ! "$STATE" =~ ^[A-Za-z]{2}$ ]]; then
        print_error "State must be a 2-letter abbreviation"
        exit 1
    fi
    
    # Validate resume stage
    if [[ -n "$RESUME_FROM" ]]; then
        case "$RESUME_FROM" in
            docx|markdown|segment|embeddings)
                ;;
            *)
                print_error "Invalid resume stage: $RESUME_FROM"
                print_error "Valid stages: docx, markdown, segment, embeddings"
                exit 1
                ;;
        esac
    fi
    
    # Set jurisdiction name
    JURISDICTION_NAME="${STATE}-${MUNICIPALITY}"
    
    # Validate numeric arguments
    if [[ ! "$MAX_LINES" =~ ^[0-9]+$ ]] || [[ "$MAX_LINES" -lt 1 ]]; then
        print_error "max-lines must be a positive integer"
        exit 1
    fi
    
    if [[ ! "$TOKEN_LIMIT" =~ ^[0-9]+$ ]] || [[ "$TOKEN_LIMIT" -lt 1 ]]; then
        print_error "token-limit must be a positive integer"
        exit 1
    fi
    
    if [[ ! "$WORDS_PER_TOKEN" =~ ^[0-9]*\.?[0-9]+$ ]] || [[ "$(echo "$WORDS_PER_TOKEN <= 0" | bc -l)" -eq 1 ]]; then
        print_error "words-per-token must be a positive number"
        exit 1
    fi
}

print_configuration() {
    echo
    print_info "Pipeline Configuration:"
    echo "  State: $STATE"
    echo "  Municipality: $MUNICIPALITY"
    echo "  Jurisdiction: $JURISDICTION_NAME"
    echo "  Model: $MODEL"
    echo "  Embedding Model: $EMBEDDING_MODEL"
    echo "  Max Lines: $MAX_LINES"
    echo "  Token Limit: $TOKEN_LIMIT"
    echo "  Words per Token: $WORDS_PER_TOKEN"
    echo "  ChromaDB Path: $CHROMA_DB_PATH"
    echo "  Collection Name: $COLLECTION_NAME"
    echo "  Shared ChromaDB: $([ "$NO_SHARED_CHROMA" == true ] && echo "No" || echo "Yes")"
    echo "  Verbose: $VERBOSE"
    echo "  Dry Run: $DRY_RUN"
    echo "  Log File: $LOG_FILE"
    echo
    
    if [[ "$DRY_RUN" == true ]]; then
        print_warning "DRY RUN MODE - No commands will be executed"
        echo
    fi
}

execute_pipeline() {
    local start_time=$(date +%s)
    
    print_info "Starting pipeline for $JURISDICTION_NAME"
    log "Pipeline started for $JURISDICTION_NAME"
    
    # Determine which stages to run
    local stages_to_run=()
    
    if [[ -n "$RESUME_FROM" ]]; then
        case "$RESUME_FROM" in
            docx)
                stages_to_run=("docx" "markdown" "segment" "embeddings")
                ;;
            markdown)
                stages_to_run=("markdown" "segment" "embeddings")
                ;;
            segment)
                stages_to_run=("segment" "embeddings")
                ;;
            embeddings)
                stages_to_run=("embeddings")
                ;;
        esac
    else
        # Normal execution - check skip flags
        if [[ "$SKIP_DOCX" == false ]]; then
            stages_to_run+=("docx")
        fi
        if [[ "$SKIP_MARKDOWN" == false ]]; then
            stages_to_run+=("markdown")
        fi
        if [[ "$SKIP_SEGMENT" == false ]]; then
            stages_to_run+=("segment")
        fi
        if [[ "$SKIP_EMBEDDINGS" == false ]]; then
            stages_to_run+=("embeddings")
        fi
    fi
    
    # Always run directory creation first (unless resuming from later stage)
    if [[ -z "$RESUME_FROM" ]] || [[ "$RESUME_FROM" == "docx" ]]; then
        stage_1_create_directory
    fi
    
    # Run stages in order
    for stage in "${stages_to_run[@]}"; do
        case $stage in
            docx)
                stage_2_convert_docx
                ;;
            markdown)
                stage_3_convert_to_markdown
                ;;
            segment)
                stage_4_segment_legal_code
                ;;
            embeddings)
                stage_5_create_embeddings
                ;;
        esac
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_success "Pipeline completed successfully in ${duration}s"
    log "Pipeline completed successfully in ${duration}s"
}

print_summary() {
    echo
    print_success "Pipeline Summary for $JURISDICTION_NAME:"
    echo
    
    local jurisdiction_dir="data/laws/$JURISDICTION_NAME"
    
    # Show directory structure
    if [[ -d "$jurisdiction_dir" ]]; then
        echo "📁 Directory Structure:"
        find "$jurisdiction_dir" -type f -name "*.txt" -o -name "*.md" -o -name "*.parquet" | sort | while read -r file; do
            local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
            local relative_path=${file#$jurisdiction_dir/}
            echo "   📄 $relative_path (${size} bytes)"
        done
        echo
    fi
    
    # Show database info
    if [[ -d "$CHROMA_DB_PATH" ]] && [[ "$NO_SHARED_CHROMA" == false ]]; then
        echo "🗄️  Database:"
        echo "   📍 Path: $CHROMA_DB_PATH"
        echo "   🏷️  Collection: $COLLECTION_NAME"
        echo
    fi
    
    echo "📋 Log File: $LOG_FILE"
    echo
    
    if [[ "$DRY_RUN" == false ]]; then
        print_success "Ready for search! Use the retrieve.py module to query the legal code."
    fi
}

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Trap signals for graceful cleanup
trap 'print_warning "Pipeline interrupted"; exit 130' INT TERM

# Run main function with all arguments
main "$@"