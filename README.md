# legiscope

Automated analysis of local codes for legal epidemiology.

`legiscope` implements a retrieval and query pipeline for extracting information
from local codes for research purposes. It aims to:

1. preserve the structure of the source documents for precise segmentation
   and accurate citations, despite differences in source formats;
2. support enhancements such as query rewriting, incorporating cross-referenced
   information from other sources, and output verification;
3. make it easy to systematically apply queries over a large number of
   jurisdictions, and collect results in a uniform format suitable for further
   analysis; and
4. enable tweaking and experimentation.

## Quick Start

The preprocessing pipeline uses [DVC](https://dvc.org/) to track parameters and
manage reproducible stages. All settings live in `params.yaml`; edit it once and
every command picks up the values.

### Prerequisites

- Python environment ready: `make env` (installs DVC and all dependencies)
- API keys in `.env` (copy from `.env.example`)
- Provider/model choices set in `params.yaml` (`llm.default_provider`,
  `embeddings.default_provider`)
- For Ollama: server running (`ollama serve`) and models pulled

_See [Configuration Files](#configuration-files) for details._

### Preprocess a New Jurisdiction

1. **Configure `params.yaml`** with your jurisdiction:

   ```yaml
   jurisdiction:
     state: CA
     locality: LosAngeles
     code_slug: municipal-code
     code_name: "LA Municipal Code"
   ```

2. **Initialize** (one-time):

   ```bash
   # Reads jurisdiction from params.yaml
   uv run python scripts/init.py
   ```

3. **Place raw files** in `data/laws/CA/LosAngeles/municipal-code/raw/`
   (DOCX, TXT, etc.). Convert DOCX first if needed:

   ```bash
   ./scripts/convert_docx.sh path/to/file.docx
   ```

4. **Run the pipeline** (parse → segment → embed → index → benchmark):

   ```bash
   ./scripts/dvc_repro.sh
   ```

| Stage     | Key outputs                                  |
|-----------|----------------------------------------------|
| parse     | `code.md`, `headings.parquet`                |
| segment   | `sections.parquet`, `segments.parquet`, `relations.parquet` |
| embed     | `embeddings.parquet`                         |
| index     | ChromaDB collection (in `data/chroma_db/`)   |
| benchmark | `benchmark_results.csv`, `benchmark_metrics.json` (in `data/output/`) |

### Query

Querying can be run standalone or as part of the benchmark DVC stage:

```bash
# Standalone query execution
uv run python scripts/run_queries.py

# Or run the benchmark stage (includes querying + evaluation)
./scripts/dvc_repro.sh --stage benchmark
```

_See [Running Queries](#running-queries) for all options._

### Experiments

`params.yaml` is the single source of truth for the current jurisdiction,
retrieval, and query settings. All commands read from it by default. Edit it,
then `dvc exp run` picks up the changes and reruns affected stages automatically.
Use `-S` flags to create one-off overrides without editing the file.

```bash
# Run with current params.yaml settings
dvc exp run

# One-off override (does not modify params.yaml)
dvc exp run \
    -S retrieval.hyde.enabled=true \
   -S retrieval.relevance_filter.enabled=true \
   -S retrieval.debug=true
```

Compare experiments with `dvc exp show` / `dvc exp diff <exp-name>`.

### Storing and Sharing Results

- `dvc push` / `dvc pull` — share processed artifacts via a configured remote.
- Query output (`data/output/`) is not DVC-tracked by default; commit to git
  or add to DVC as needed.
- `dvc exp push` / `dvc exp pull` — share experiment state.

_See [DVC Remote Storage](#dvc-remote-storage) for setup._

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
#source .venv/bin/activate
```

### Configuration Files

- **`params.yaml`** — All pipeline parameters: jurisdiction, LLM provider/models, embedding settings, retrieval/query tuning. Tracked by DVC.
- **`config.yaml`** — Infrastructure: data directory path, ChromaDB location.
- **`.env`** — API keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`). Not tracked. Copy from `.env.example`.

## Development

### Running Tests

```bash
make test
```

### Code Quality

```bash
# Run linting and formatting checks
make lint

# Run type checks
make typecheck

# Format code
make format

# Fix linting issues
make fix
```

## Usage

All commands read jurisdiction and settings from `params.yaml`. Edit the file
to switch jurisdictions or tune retrieval/query settings. For one-off DVC
overrides use `-S` flags (see below).

### CLI Overrides

All scripts read jurisdiction and settings from `params.yaml`. Edit the file
to change jurisdictions. For one-off DVC overrides, use `-S` flags directly:

```bash
# Run a single pipeline stage
./scripts/dvc_repro.sh --stage segment

# Override params for a DVC experiment (does not modify params.yaml)
dvc exp run -S jurisdiction.state=CA -S jurisdiction.locality=LosAngeles \
    -S jurisdiction.code_slug=municipal-code
```

For state-level codes, set `locality` to `State` in `params.yaml`.

### Running Queries

```bash
uv run python scripts/run_queries.py
```

Queries are read from the default path configured in `config.yaml`
(`paths.default_queries_file`) and expected to include a `question` column.
Results are saved to
`data/output/{JURISDICTION}/query_results.csv` with answers, citations,
confidence scores, and processing metrics.

<details>
<summary>Full query CLI options</summary>

This script currently takes no command-line arguments.

Jurisdiction and retrieval/query settings (HYDE, relevance filtering, etc.) are
read from `params.yaml`; paths are read from `config.yaml`.

</details>

## Scripts and Modules

### DVC Stage Scripts

- `scripts/init.py` — Initialize jurisdiction (not a DVC stage; run directly)
- `scripts/parse.py` — Convert raw files to structured Markdown
- `scripts/segment.py` — Segment Markdown into sections and segments
- `scripts/embed.py` — Generate embedding vectors
- `scripts/index.py` — Build ChromaDB search index
- `coep/scripts/benchmark_pipeline.py` — Benchmark RAG answers against MonQcle ground truth

### Other Scripts

- `scripts/dvc_repro.sh` — Wrapper around `dvc exp run` for running the pipeline
- `scripts/run_queries.py` — Run batch queries against legal code database
- `scripts/convert_docx.sh` — Convert DOCX files to plain text using pandoc

### Notebooks

- `query_demo.py` - Interactive Marimo notebook demonstrating section-level retrieval with drug paraphernalia query

### Source Modules

- `config.py` — Infrastructure configuration (data directory, ChromaDB path)
- `params.py` — DVC params.yaml loader
- `llm_config.py` — Centralized LLM configuration using instructor's provider abstraction
- `utils.py` — Core utilities including LLM client and directory functions
- `parse/convert.py` — Text conversion utilities and LLM response models
- `segment.py` — Text segmentation and hierarchical section processing
- `embeddings.py` — Embedding generation and ChromaDB management
- `retrieve.py` — Information retrieval with HYDE query rewriting and section-level search
- `query.py` — Legal query processing with structured responses and batch query execution

## Data Directory Structure

The default data location is `data/` (set in `config.yaml`) and can be overridden
with the `LEGISCOPE_DATA_DIR` environment variable.

The project organizes local code data in a structured hierarchy:

```txt
data/
├── jurisdictions.parquet           # Registry of all jurisdictions
├── codes.parquet                   # Registry of all legal codes
├── laws/                           # Legal code data
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
└── ...
```

### DVC Remote Storage

DVC can push/pull data to a remote store so that collaborators and CI systems
share processed artifacts without re-running the pipeline.

#### Setup

1. Choose a storage backend (S3, GCS, SSH, local path, etc.).
2. Add the remote:

   ```bash
   dvc remote add -d myremote gs://my-bucket/legiscope
   ```

3. Push data after running the pipeline:

   ```bash
   dvc push
   ```

4. Pull data on another machine:

   ```bash
   dvc pull
   ```

See the [DVC remote storage docs](https://dvc.org/doc/user-guide/data-management/remote-storage) for full configuration options.

### Project Structure

```txt
.
├── src/
│   └── legiscope/           # Main package source code
│       ├── parse/           # Parse stage: raw text → structured Markdown
│       │   ├── convert.py       # Conversion utilities and response models
│       │   ├── scan.py          # LLM heading scanning and verification
│       │   ├── headings.py      # Heading models and pattern helpers
│       │   └── elements.py      # Raw text element splitting
│       ├── pipeline/        # DVC stage modules (parse, segment, embed, index, init)
│       ├── config.py        # Infrastructure configuration (config.yaml)
│       ├── params.py        # DVC params.yaml loader
│       ├── llm_config.py    # LLM configuration and client management
│       ├── models.py        # Data models (JurisdictionRef, CodeRef, schema constants)
│       ├── utils.py         # Core utility functions
│       ├── embeddings.py    # Embedding generation and ChromaDB management
│       ├── retrieve.py      # Information retrieval with HYDE and section-level search
│       ├── segment.py       # Text segmentation utilities
│       └── query.py         # Legal query processing with structured responses
├── tests/                   # Test files
├── scripts/                 # Utility scripts
│   ├── dvc_repro.sh             # DVC pipeline wrapper
│   ├── run_queries.py           # Batch query execution
│   └── ...
├── coep/
│   ├── __init__.py
│   ├── src/
│   │   ├── eval.py              # COEP-specific evaluation logic
│   │   └── query.py             # COEP-specific query preprocessing
│   ├── tests/
│   │   ├── test_eval.py
│   │   └── test_query_adjustments.py
│   ├── scripts/
│   │   └── benchmark_pipeline.py # COEP benchmarking workflow
│   ├── docs/
│   │   └── BENCHMARKING.md      # COEP benchmarking guide
│   └── data/
│       └── monqcle_data/        # COEP MonQcle data
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

Additional documentation is available in `docs/` and `coep/docs/`:

- [COEP Benchmarking Workflow](coep/docs/BENCHMARKING.md) - Guide to running the COEP RAG evaluation pipeline against MonQcle data
- [Supporting Passages Validation](docs/VALIDATION_EXAMPLE.md) - Guide to automatic validation of LLM-generated supporting passages

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on development setup, code style, commit conventions, and pull requests.

Instructions for the bots: [AGENTS.md](AGENTS.md).
