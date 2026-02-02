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

Model and provider settings are managed in two YAML files tracked by DVC:

- **`params.yaml`** — Hyperparameters: LLM provider, model names, temperature, embedding settings, etc.
- **`config.yaml`** — Infrastructure settings: data directory path, ChromaDB location.

Secrets (API keys) are kept in `.env` and are **not** tracked.

#### Setup

1. Copy the example environment file and add your API keys:

   ```bash
   cp .env.example .env
   # Edit .env to set OPENAI_API_KEY and/or MISTRAL_API_KEY
   ```

2. Edit `params.yaml` to select your provider and models:

   ```yaml
   llm:
     default_provider: "mistral"   # or "openai", "ollama"
   embeddings:
     default_provider: "ollama"    # or "mistral"
   ```

The main client types are:

- **Fast Client** (`Config.get_fast_client()`): Uses configured fast model
- **Powerful Client** (`Config.get_powerful_client()`): Uses configured powerful model
- **Embedding Client** (`get_embedding_client()`): Uses configured embedding provider

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

The pipeline uses [DVC](https://dvc.org/) to manage reproducible stages:
**parse → segment → embed → index**.

#### Step 1: Initialize a jurisdiction (one-time setup)

Initialization is not a DVC stage — run it directly:

```bash
python -m legiscope.pipeline.init \
    --state CA --locality LosAngeles \
    --code-slug municipal-code --name "LA Municipal Code"
# Creates: data/laws/CA/LosAngeles/municipal-code/raw/
```

After initialization, place your raw files (DOCX, TXT, etc.) in the `raw/` directory.

For state-level codes, use `--locality State` (or omit it):

```bash
python -m legiscope.pipeline.init \
    --state CA --code-slug penal-code --name "CA Penal Code"
```

#### Step 2: Run the pipeline

Use the wrapper script, which calls `dvc exp run` with the right `-S` flags:

```bash
./scripts/dvc_repro.sh --state CA --locality LosAngeles --code-slug municipal-code
```

Or call DVC directly:

```bash
dvc exp run \
    -S jurisdiction.state=CA \
    -S jurisdiction.locality=LosAngeles \
    -S jurisdiction.code_slug=municipal-code
```

Run a single stage:

```bash
./scripts/dvc_repro.sh --state CA --locality LosAngeles --code-slug municipal-code \
    --stage segment
```

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

### Pipeline Package (DVC stages)

- `legiscope.pipeline.init` — Initialize jurisdiction (not a DVC stage; run directly)
- `legiscope.pipeline.parse` — Convert raw files to structured Markdown
- `legiscope.pipeline.segment` — Segment Markdown into sections and segments
- `legiscope.pipeline.embed` — Generate embedding vectors
- `legiscope.pipeline.index` — Build ChromaDB search index

### Scripts

- `scripts/dvc_repro.sh` — Wrapper around `dvc exp run` for running the pipeline
- `scripts/run_queries.py` — Run batch queries against legal code database
- `scripts/benchmark_pipeline.py` — Benchmarking workflow
- `scripts/convert_docx.sh` — Convert DOCX files to plain text using pandoc
- `scripts/pipeline_init.sh` — *(deprecated: use `legiscope.pipeline.init`)*
- `scripts/pipeline_parse.sh` — *(deprecated: use `dvc_repro.sh`)*
- `scripts/pipeline_process.sh` — *(deprecated: use `dvc_repro.sh`)*
- `scripts/pipeline_query.sh` — *(deprecated: use `run_queries.py`)*

### Notebooks

- `demo_query.py` - Interactive Marimo notebook demonstrating section-level retrieval with drug paraphernalia query

### Source Modules

- `config.py` — Infrastructure configuration (data directory, ChromaDB path)
- `params.py` — DVC params.yaml loader
- `llm_config.py` — Centralized LLM configuration using instructor's provider abstraction
- `utils.py` — Core utilities including LLM client and directory functions
- `convert.py` — Text conversion utilities and LLM response models
- `segment.py` — Text segmentation and hierarchical section processing
- `embeddings.py` — Embedding generation and ChromaDB management
- `retrieve.py` — Information retrieval with HYDE query rewriting and section-level search
- `query.py` — Legal query processing with structured responses and batch query execution

## Data Directory Structure

The project organizes municipal code data in a structured hierarchy:

```txt
data/
├── jurisdictions.parquet           # Registry of all jurisdictions
├── codes.parquet                   # Registry of all legal codes
├── laws/                           # Municipal code data
│   └── {STATE}/{Locality}/{code-slug}/
│       ├── raw/                    # Original source files (DOCX, PDF, etc.)
│       ├── code.md                 # Structured Markdown
│       ├── sections.parquet        # Section hierarchy
│       ├── segments.parquet        # Text segments
│       ├── embeddings.parquet      # Embedding vectors
│       ├── relations.parquet       # Intra-code relations
│       └── external_references.parquet
├── chroma_db/                      # ChromaDB vector database
├── queries/                        # Query templates and examples
├── output/                         # LLM query output
│   └── {STATE}-{Locality}/
│       └── query_results.csv
└── monqcle_data/                   # Human-annotated MonQcle data
```

### Project Structure

```txt
.
├── src/
│   └── legiscope/           # Main package source code
│       ├── pipeline/        # DVC stage modules (parse, segment, embed, index, init)
│       ├── config.py        # Infrastructure configuration (config.yaml)
│       ├── params.py        # DVC params.yaml loader
│       ├── llm_config.py    # LLM configuration and client management
│       ├── models.py        # Data models (JurisdictionRef, CodeRef, schema constants)
│       ├── convert.py       # Conversion utilities and response models
│       ├── utils.py         # Core utility functions
│       ├── embeddings.py    # Embedding generation and ChromaDB management
│       ├── retrieve.py      # Information retrieval with HYDE and section-level search
│       ├── segment.py       # Text segmentation utilities
│       ├── query.py         # Legal query processing with structured responses
│       └── eval.py          # Evaluation and benchmarking logic
├── tests/                   # Test files
├── scripts/                 # Utility scripts
│   ├── dvc_repro.sh             # DVC pipeline wrapper
│   ├── benchmark_pipeline.py    # Benchmarking workflow
│   ├── run_queries.py           # Batch query execution
│   └── ...
├── notebooks/               # Interactive notebooks
├── docs/                    # Documentation
├── config.yaml              # Infrastructure settings (data dir, ChromaDB path)
├── params.yaml              # DVC parameters (provider, models, jurisdiction)
├── dvc.yaml                 # DVC pipeline definition
├── data/                    # Data directory (not tracked by git)
├── pyproject.toml           # Project configuration and dependencies
├── Makefile                 # Development commands
├── AGENTS.md                # Detailed development documentation
└── CONTRIBUTING.md          # Contribution guidelines
```

## Documentation

Additional documentation is available in the `docs/` directory:

- [Benchmarking Workflow](docs/BENCHMARKING.md) - Guide to running the RAG evaluation pipeline against MonQcle data
- [Supporting Passages Validation](docs/VALIDATION_EXAMPLE.md) - Guide to automatic validation of LLM-generated supporting passages

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on development setup, code style, commit conventions, and pull requests.

Instructions for the bots: [AGENTS.md](AGENTS.md).
