# legiscope

Automated analysis of municipal codes for legal epidemiology.

`legiscope` implements a retrieval and query pipeline for extracting information
from municipal codes for research purposes. It aims to:

1. preserve the structure of the source documents for precise segmentation
   and accurate citations, despite differences in source formats;
2. support enhancements such as query rewriting, incorporating cross-referenced
   information from other sources, and output verification;
3. make it easy to systematically apply queries over a large number of
   jurisdictions, and collect results in a uniform format suitable for further
   analysis; and
4. enable tweaking and experimentation.

## Getting started

Developed on MacOS and Linux. It's possible some changes would be needed to run
on Windows.

### Environment Setup

This project uses `uv` for dependency management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on MacOS, with homebrew
brew install uv

# Set up the development environment
make env

# run scripts with uv
uv run python foo.py

# open notebooks
cd notebooks
uv run marimo edit

# Alternatively, activate the environment
#source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Model Configuration

The project uses environment variables for model configuration.

The main client types are:

- **Fast Client** (`Config.get_fast_client()`): Uses configured fast model
- **Powerful Client** (`Config.get_powerful_client()`): Uses configured powerful model
- **Embedding Client** (`get_embedding_client()`): Uses configured embedding provider

Models are automatically selected based on your `.env` configuration.

#### Environment Variables

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings:

   ```bash
   # Example: Use OpenAI
   LEGISCOPE_LLM_PROVIDER=openai
   LEGISCOPE_FAST_MODEL=gpt-4.1-mini
   LEGISCOPE_POWERFUL_MODEL=gpt-4.1
   OPENAI_API_KEY=XXXXXX
   ```

3. Load environment variables:

   ```bash
   export $(cat .env | grep -v '^#' | xargs)
   ```

### Example .env Configurations

**Local models only** (e.g., for hacking on the plane):

```bash
LEGISCOPE_LLM_PROVIDER=ollama
LEGISCOPE_FAST_MODEL=gemma3:4b
LEGISCOPE_POWERFUL_MODEL=gemma3:12b

LEGISCOPE_EMBEDDING_PROVIDER=ollama
LEGISCOPE_EMBEDDING_MODEL=embeddinggemma
LEGISCOPE_COLLECTION_NAME=legal_code_ollama
```

**Mistral LLMs and embedding models** (for development; only one API key needed):

```bash
LEGISCOPE_LLM_PROVIDER=mistral
LEGISCOPE_FAST_MODEL=mistral-small-2506
LEGISCOPE_POWERFUL_MODEL=mistral-medium-2508

LEGISCOPE_EMBEDDING_PROVIDER=mistral
LEGISCOPE_EMBEDDING_MODEL=mistral-embed
LEGISCOPE_COLLECTION_NAME=legal_code_mistral

MISTRAL_API_KEY=XXXXXX
```

**OpenAI LLMs and Mistral embeddings** (two API keys needed):

```bash
LEGISCOPE_LLM_PROVIDER=openai
LEGISCOPE_FAST_MODEL=gpt-4.1-mini
LEGISCOPE_POWERFUL_MODEL=gpt-4.1

LEGISCOPE_EMBEDDING_PROVIDER=mistral
LEGISCOPE_EMBEDDING_MODEL=mistral-embed
LEGISCOPE_COLLECTION_NAME=legal_code_mistral

OPENAI_API_KEY=XXXXXX
MISTRAL_API_KEY=XXXXXX
```

## Development

### Running Tests

```bash
make test
```

### Code Quality

```bash
# Run linting and formatting checks
make lint

# Format code
make format

# Fix linting issues
make fix
```

## Usage

### Processing Municipal Codes

To process a new municipal code from DOCX files to searchable embeddings:

```bash
# Basic usage (without queries)
./scripts/pipeline.sh NY "New York"

# With queries
./scripts/pipeline.sh NY "New York" data/queries/example_queries.txt

# Using Makefile
make pipeline STATE=NY MUNICIPALITY="New York"
make pipeline STATE=NY MUNICIPALITY="New York" QUERIES=data/queries/example_queries.txt
```

The pipeline performs these steps automatically:

1. Creates directory structure for the jurisdiction
2. Converts DOCX files to plain text (if present)
3. Converts text to structured Markdown with headings
4. Segments the code into searchable sections
5. Generates embeddings for semantic search
6. Runs queries against the legal code (optional, if queries file provided)

### Running Queries

The pipeline can optionally run a batch of queries against the processed legal code:

```bash
# Run pipeline with queries
./scripts/pipeline.sh CA LosAngeles data/queries/test_queries.csv
```

You can also run the query script directly for more control over processing options:

```bash
# Full configuration example (maximum control)
source .venv/bin/activate
python scripts/run_queries.py \
    --queries-path "data/queries/test_queries.csv" \
    --jurisdiction-id "CA-LosAngeles" \
    --n-results 10 \
    --use-hyde False \
    --filter-relevance True \
    --relevance-threshold 0.5 \
    --validate-supporting-passages True \
    --output "data/output/CA-LosAngeles/test_results.csv"

# Minimal example (using defaults)
source .venv/bin/activate
python scripts/run_queries.py \
    --queries-path "data/queries/test_queries.csv" \
    --jurisdiction-id "CA-LosAngeles"
```

**Script Arguments (run_queries.py):**

- `--queries-path`: Path to queries CSV file (required)
- `--jurisdiction-id`: Target jurisdiction ID (e.g., "CA-LosAngeles") (required)
- `--collection-name`: ChromaDB collection name (defaults to env var)
- `--output`: Output CSV file path
- `--n-results`: Number of results to retrieve per query (default: 10)
- `--use-hyde`: Enable HYDE query rewriting (default: False)
- `--filter-relevance`: Enable LLM-based relevance filtering (default: True)
- `--relevance-threshold`: Confidence threshold for filtering (default: 0.5)
- `--validate-supporting-passages`: Validate LLM citations against text (default: True)

**Query File Format:**

- CSV format with a "question" column
- One query per row

**Example queries file (queries.csv):**

```csv
question
"Are there restrictions on selling drug paraphernalia in this jurisdiction?"
"What are the parking regulations for residential areas?"
"Do I need a permit to operate a home-based business?"
```

Query results are saved to `data/output/{JURISDICTION}/query_results.csv` and include:

- Short answer to each query
- Detailed legal reasoning
- Citations and supporting passages
- Confidence scores
- Processing metrics

## Scripts and Modules

### Scripts

- `scripts/pipeline.sh` - Simple jurisdiction processing workflow automation
- `scripts/create_jurisdiction.py` - Create jurisdiction directory structure
- `scripts/convert_docx.sh` - Convert DOCX files to plain text using pandoc
- `scripts/convert_to_markdown.py` - Convert legal text to structured Markdown
- `scripts/segment_legal_code.py` - Segment Markdown into sections and segments
- `scripts/create_embeddings.py` - Generate embeddings and populate ChromaDB
- `scripts/run_queries.py` - Run batch queries against legal code database

### Notebooks

- `demo_query.py` - Interactive Marimo notebook demonstrating section-level retrieval with drug paraphernalia query

### Source Modules

- `utils.py` - Core utilities including LLM client and directory functions
- `llm_config.py` - Centralized LLM configuration using instructor's provider abstraction
- `convert.py` - Text conversion utilities and LLM response models
- `segment.py` - Text segmentation and hierarchical section processing
- `embeddings.py` - Embedding generation and ChromaDB management
- `retrieve.py` - Information retrieval with HYDE query rewriting and section-level search
- `query.py` - Legal query processing with structured responses and batch query execution

## Data Directory Structure

The project organizes municipal code data in a structured hierarchy:

```{txt}
data/
├── laws/                           # Municipal code data
│   └── {state}-{municipality}/     # Jurisdiction-specific directories
│       ├── raw/                    # Original source files (DOCX, PDF, etc.)
│       ├── processed/              # Processed text files and intermediate results
│       │   ├── code.txt            # Converted plain text
│       │   └── code.md             # Structured markdown
│       └── tables/                 # Structured data tables and exports
│           ├── sections.parquet    # Section-level data
│           ├── segments.parquet    # Segment-level data
│           └── embeddings.parquet  # Generated embeddings
├── chroma_db/                      # ChromaDB vector database
├── queries/                        # Query templates and examples
│   └── example_queries.txt         # Example legal queries
├── output/                         # LLM query output
│   └── {state}-{municipality}/     # Jurisdiction-specific directories
│       └── query_results.parquet   # Query results (if queries were run)
└── monqcle_data/                   # Human-annotated MonQcle data
        └── monqcle_data.csv        # Data file of human-answered legal questions
```

### Project Structure

```txt
.
├── src/
│   └── legiscope/       # Main package source code
│       ├── llm_config.py    # LLM configuration and client management
│       ├── convert.py   # Conversion utilities and response models
│       ├── utils.py     # Core utility functions
│       ├── embeddings.py # Embedding generation and ChromaDB management
│       ├── retrieve.py   # Information retrieval with HYDE and section-level search
│       ├── segment.py   # Text segmentation utilities
│       ├── query.py     # Legal query processing with structured responses
│       └── eval.py      # Evaluation and benchmarking logic
├── tests/               # Test files
├── scripts/             # Utility scripts
│   ├── pipeline.sh          # End-to-end processing pipeline
│   ├── benchmark_pipeline.py # Benchmarking workflow
│   ├── run_queries.py       # Batch query execution
│   └── ...
├── notebooks/           # Interactive notebooks
│   └── query_demo.py    # Demo notebook
├── docs/                # Documentation
│   ├── BENCHMARKING.md      # Benchmarking guide
│   └── VALIDATION_EXAMPLE.md # Validation guide
├── data/                # Data directory (not tracked by git)
├── pyproject.toml       # Project configuration and dependencies
├── Makefile            # Development commands
├── AGENTS.md           # Detailed development documentation
└── CONTRIBUTING.md     # Contribution guidelines
```

## Documentation

Additional documentation is available in the `docs/` directory:

- [Benchmarking Workflow](docs/BENCHMARKING.md) - Guide to running the RAG evaluation pipeline against MonQcle data
- [Supporting Passages Validation](docs/VALIDATION_EXAMPLE.md) - Guide to automatic validation of LLM-generated supporting passages

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on development setup, code style, commit conventions, and pull requests.

Instructions for the bots: [AGENTS.md](AGENTS.md).
