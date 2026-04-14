# Agents

This document contains information about available commands, workflows, and development practices for legiscope project.

## Environment Setup

This project uses `uv` for dependency management and Python environment handling.

### Initial Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
make env
```

Or manually:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### LLM Provider Configuration

The project supports OpenAI, Mistral, and Ollama as LLM providers. Configuration is managed via `params.yaml` (tracked by DVC) and `config.yaml` (infrastructure settings).

#### Environment Variables

- `OPENAI_API_KEY`: Required when using OpenAI provider (secret)
- `MISTRAL_API_KEY`: Required when using Mistral provider (secret)
- `OPENROUTER_API_KEY`: Required when using OpenRouter embedding provider (secret)
- `LEGISCOPE_DATA_DIR`: Override the root data directory path (infrastructure)

All hyperparameters (provider, model names, temperature, retrieval/query settings, etc.) are configured in `params.yaml`.

#### Code-based Configuration

You can configure the LLM provider directly in code using the `Config` class:

```python
from legiscope.llm_config import Config
from legiscope.utils import LLMConfig

# Get fast client for quick tasks
fast_client = Config.get_fast_client()      # Model determined by params.yaml
powerful_client = Config.get_powerful_client()  # Model determined by params.yaml

# Create reusable LLM configuration
llm_config = LLMConfig(
    client=Config.get_fast_client(),
    temperature=0.0,
    max_retries=3
)

# Get appropriate models for different tasks
fast_model = Config.get_fast_model()        # For quick tasks
powerful_model = Config.get_powerful_model() # For complex reasoning
```

#### Using Config Objects

The library uses config objects for cleaner, more maintainable API:

```python
from legiscope.models import JurisdictionRef, CodeRef
from legiscope.utils import LLMConfig
from legiscope.retrieve import SectionRetrievalSettings, retrieve_sections
from legiscope.query import QuerySettings, BatchQuerySettings
from legiscope.query import query_legal_documents, run_queries
from legiscope.llm_config import Config
import chromadb

# Setup
jurisdiction = JurisdictionRef(state="IL", locality="WindyCity")
code_ref = CodeRef(jurisdiction=jurisdiction, code_slug="municipal-code")

chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
collection = chroma_client.get_collection("legal_code_ollama_embeddinggemma")

# Example 1: Retrieve sections with settings
retrieval_settings = SectionRetrievalSettings(
    jurisdiction_id=code_ref.jurisdiction_id,
    n_results=10,
    use_hyde=True,
    hyde_client=Config.get_fast_client()
)
results = retrieve_sections(
    collection=collection,
    sections_parquet_path=str(code_ref.full_data_dir / "sections.parquet"),
    query_text="What are the parking regulations?",
    settings=retrieval_settings
)

# Example 2: Query with LLM analysis
llm_config = LLMConfig(client=Config.get_powerful_client(), temperature=0.0)
query_settings = QuerySettings(
    llm=llm_config,
    filter_relevance=True,
    relevance_threshold=0.7
)
response = query_legal_documents(
    retrieval_results=results,
    query="What are the parking regulations?",
    settings=query_settings
)

# Example 3: Batch queries
batch_settings = BatchQuerySettings(
    llm=llm_config,
    use_hyde=True,
    filter_relevance=True
)
results_df = run_queries(
    collection=collection,
    sections_parquet_path=str(code_ref.full_data_dir / "sections.parquet"),
    queries=["Query 1", "Query 2", "Query 3"],
    jurisdiction_id=code_ref.jurisdiction_id,
    settings=batch_settings
)
```

#### Available Models by Provider

Models are configured per-provider in `params.yaml` under `llm.providers`. Current defaults:

**OpenAI Provider:**
- Fast model: `Qwen/Qwen3.5-27B` (via vLLM on HPC)
- Powerful model: `Qwen/Qwen3.5-27B`

**Mistral Provider:**
- Fast model: `mistral-small-2506`
- Powerful model: `mistral-large-2512`

**Ollama Provider:**
- Fast model: `gemma3:4b`
- Powerful model: `gemma3:4b`
- Requires Ollama server running locally

#### Example Setup

```bash
# Set API keys as secrets (in .env or environment)
export OPENAI_API_KEY=your_openai_key
export OPENROUTER_API_KEY=your_openrouter_key

# To change the default provider, edit params.yaml:
#   llm.default_provider: "openai"  (or "mistral", "ollama")

# For Ollama (local), ensure server is running: ollama serve
```

### Embedding Model Configuration

The project supports multiple embedding models for generating text embeddings. The default provider is OpenRouter.

#### Current Configuration

**OpenRouter with qwen/qwen3-embedding-8b (default)**
- Uses OpenRouter's OpenAI-compatible API
- Model: `qwen/qwen3-embedding-8b`
- Client: `get_embedding_client("openrouter")`
- Requires: `OPENROUTER_API_KEY` environment variable

#### Alternative Configuration

**Ollama with embeddinggemma**
- Uses local Ollama server
- Model: `embeddinggemma`
- Client: `get_embedding_client("ollama")`
- Requires: Ollama server running locally (`ollama serve`)

#### Switching Between Embedding Models

To switch between embedding providers, edit `params.yaml`:
```yaml
embeddings:
  default_provider: "openrouter"  # or "ollama"
```

#### Retrieval & Query Configuration

Retrieval settings (HYDE, relevance filtering) and query settings (model tier,
passage validation) are all in `params.yaml`. CLI scripts (`scripts/run_queries.py`,
`coep/scripts/benchmark_pipeline.py`) read these settings from `params.yaml`
and path settings from `config.yaml`.

```yaml
retrieval:
    n_results: 20
  distance_metric: l2        # ChromaDB HNSW distance (l2, cosine, ip)
  debug: true               # Enable consolidated stage debug CSVs
  hyde:
    enabled: false            # uses fast model
  relevance_filter:
    enabled: true             # uses fast model
    threshold: 0.7

query:                        # uses powerful model
  validation:
    enabled: true
    exact_match_threshold: 1.0
    fuzzy_match_threshold: 0.9
```

Project code can also inject per-query guidance with
`BatchQuerySettings.retrieval_guidance_provider`. The provider receives a
`RetrievalGuidanceRequest` containing the base query, `variable_name`, and
metadata, and can return `RetrievalGuidance` with separate fields for:

- retrieval-time query shaping (`retrieval_query`, `retrieval_instructions`)
- relevance-filter prompt shaping (`relevance_instructions`, `anchor_terms`)
- completion-time coding hints (`completion_instructions`, `shared_context`)

The COEP benchmark uses this hook in `coep/src/retrieval_guidance.py` to keep
drug-paraphernalia family logic out of the generic RAG core.

Use the embedding interface:
   ```python
   from legiscope.embeddings import get_embedding_client, get_embeddings

   # For OpenRouter (default)
   client = get_embedding_client("openrouter")
   embeddings = get_embeddings(client, ["text1", "text2"])

   # For Ollama
   client = get_embedding_client("ollama")
   embeddings = get_embeddings(client, ["text1", "text2"])
   ```

#### Usage Examples

```python
from legiscope.embeddings import get_embedding_client, get_embeddings, EmbeddingConfig

# Get embedding client for specific provider
client = get_embedding_client("openrouter")  # or "ollama"

# Generate embeddings (auto-detects model)
texts = ["Legal text 1", "Legal text 2"]
embeddings = get_embeddings(client, texts)

# Or specify model explicitly
embeddings = get_embeddings(client, texts, model="qwen/qwen3-embedding-8b", provider="openrouter")

# Using with EmbeddingConfig
config = EmbeddingConfig(provider="openrouter")  # Uses default qwen3-embedding-8b model
client = get_embedding_client(config.provider)
```

## Development Commands

### Testing

```bash
# Run all tests
make test
# Or manually:
pytest

# Run tests with coverage
make test-cov
# Or manually:
pytest --cov=src/legiscope --cov-report=html

# Run specific test file
pytest tests/test_llm_config.py
```

### Linting and Formatting

```bash
# Run linting and formatting checks
make lint
# Or manually:
ruff check src/ tests/
ruff format --check src/ tests/

# Run type checks separately
make typecheck
# Or manually:
basedpyright src/

# Format code
make format
# Or manually:
ruff format src/ tests/

# Fix linting issues
make fix
# Or manually:
ruff check --fix src/ tests/
```

### Environment Management

```bash
# Create/refresh environment
make env

# Clean environment
make clean-env

# Show installed packages
make list
# Or manually:
uv pip list
```

### Pipeline Commands

The pipeline is managed by DVC with five stages: **parse → segment → embed → index → benchmark**.

The `index` and `benchmark` stages both depend on the `embed` stage. The
`benchmark` stage additionally depends on `index` (it reads from the ChromaDB
index that `index` creates).

#### Initialize a jurisdiction (one-time, not a DVC stage)

Reads jurisdiction from `params.yaml`:

```bash
# Uses jurisdiction.state/locality/code_slug/code_name from params.yaml
uv run python scripts/init.py

# Override code type or jurisdiction display name
uv run python scripts/init.py --code-type zoning
uv run python scripts/init.py --jurisdiction-name "City of LA"
```

#### Run the pipeline

```bash
# Using the wrapper script (recommended) — reads params.yaml:
./scripts/dvc_repro.sh

# Run a single stage:
./scripts/dvc_repro.sh --stage segment

# Or calling DVC directly with one-off overrides:
dvc exp run \
    -S jurisdiction.state=CA \
    -S jurisdiction.locality=LosAngeles \
    -S jurisdiction.code_slug=municipal-code
```

> **Note:** The legacy `make init/parse/process/query` targets still work but are
> deprecated. Use the DVC workflow above instead.

### Benchmarking

The project handles benchmarking against MonQcle data using an LLM-as-judge approach.
See `coep/docs/BENCHMARKING.md` for full documentation.

Benchmarking is a DVC stage (`benchmark`). It runs automatically as part of
`dvc repro` / `./scripts/dvc_repro.sh`, or can be run individually:

```bash
# Run benchmark as part of the full pipeline
./scripts/dvc_repro.sh

# Run only the benchmark stage (requires embed stage outputs to exist)
./scripts/dvc_repro.sh --stage benchmark

# Or run the script directly (standalone, outside DVC)
uv run python coep/scripts/benchmark_pipeline.py

# Dev/debug run with limited queries (set `retrieval.debug: true` in `params.yaml`)
uv run python coep/scripts/benchmark_pipeline.py --test-limit 5
```

DVC tracks benchmark metrics in `benchmark_metrics.json` and results in
`benchmark_results.csv` under `data/output/{STATE}-{Locality}/`.
When debug is enabled, benchmarking also writes exactly three consolidated
CSV artifacts under `data/output/{STATE}-{Locality}/debug/`:

- `retrieval_stage_<timestamp>.csv`
- `relevance_stage_<timestamp>.csv`
- `query_stage_<timestamp>.csv`

Each file contains one row per question.

### Advanced Query Execution

All settings (jurisdiction, retrieval, query, paths) are read from
`params.yaml` and `config.yaml`:

```bash
# Zero args — paths resolved from config.yaml, settings from params.yaml
uv run python scripts/run_queries.py
```

## Project Structure

```
.
├── src/
│   └── legiscope/       # Main package source code
│       ├── parse/           # Parse stage: raw text → structured Markdown
│       │   ├── convert.py       # Markdown conversion and frontmatter
│       │   ├── scan.py          # LLM heading scanning, regex refinement, and normalized markdown prefixes
│       │   ├── headings.py      # Heading models and pattern helpers
│       │   ├── elements.py      # Raw text element splitting
│       │   └── find_code_start.py  # Locate start of code proper
│       ├── config.py        # Infrastructure configuration (config.yaml)
│       ├── params.py        # DVC params.yaml loader
│       ├── models.py        # Data models (JurisdictionRef, CodeRef, schema constants)
│       ├── llm_config.py    # LLM configuration and client management
│       ├── utils.py         # Core utility functions
│       ├── embeddings.py    # Embedding generation and ChromaDB management
│       ├── retrieve.py      # Information retrieval with HYDE and section-level search
│       ├── retrieval_guidance.py # Project-agnostic stage-specific query guidance hooks
│       ├── segment.py       # Text segmentation utilities
│       └── query.py         # Legal query processing with structured responses and consolidated debug exports
├── tests/               # Test files
├── scripts/             # DVC stage entry-point scripts and utilities
│   ├── dvc_repro.sh           # DVC pipeline wrapper (primary interface)
│   ├── dvc_python.sh          # Shared Python runner for DVC stages
│   ├── init.py                # Initialize jurisdiction (not a DVC stage)
│   ├── parse.py               # Parse raw files to Markdown
│   ├── segment.py             # Segment Markdown into sections
│   ├── embed.py               # Generate embedding vectors
│   ├── index.py               # Build ChromaDB search index
│   ├── run_queries.py         # Batch query execution
│   └── convert_docx.sh        # Convert DOCX to TXT
├── coep/
│   ├── __init__.py
│   ├── src/
│   │   ├── eval.py            # COEP evaluation logic
│   │   ├── query.py           # COEP query preprocessing
│   │   └── retrieval_guidance.py # COEP variable-family guidance provider
│   ├── tests/
│   │   ├── test_eval.py
│   │   └── test_query_adjustments.py
│   ├── scripts/
│   │   └── benchmark_pipeline.py  # COEP benchmarking (DVC benchmark stage)
│   ├── docs/
│   │   └── BENCHMARKING.md        # COEP benchmark docs
│   └── data/
│       └── monqcle_data/          # COEP MonQcle data
├── data/                # Data directory (not tracked by git)
│   ├── jurisdictions.parquet  # Registry of all jurisdictions
│   ├── codes.parquet          # Registry of all legal codes
│   └── laws/                  # Per-code data directories
│       └── {STATE}/{Locality}/{code-slug}/
│           ├── raw/               # Raw source files
│           ├── code.md            # Structured Markdown
│           ├── sections.parquet   # Section hierarchy
│           ├── segments.parquet   # Text segments
│           ├── embeddings.parquet # Embedding vectors
│           ├── relations.parquet  # Intra-code relations
│           └── external_references.parquet
├── config.yaml          # Infrastructure settings (data dir, ChromaDB path)
├── params.yaml          # DVC parameters (provider, models, jurisdiction)
├── dvc.yaml             # DVC pipeline definition (5 stages + validate)
├── pyproject.toml       # Project configuration and dependencies
├── Makefile            # Development commands
├── AGENTS.md           # This file
└── CONTRIBUTING.md     # Contribution guidelines
```

## Key Dependencies

- `dvc`: Pipeline orchestration and experiment tracking
- `openai`: OpenAI API client for LLM and OpenRouter embeddings
- `mistralai`: Mistral API client for language models
- `ollama`: Ollama client for local LLM inference
- `instructor`: AI-powered function calls and structured outputs
- `pytest`: Testing framework
- `ruff`: Fast Python linter and formatter
- `black`: Code formatter
- `chromadb`: Vector database for embeddings
- `duckdb`: Analytical database
- `marimo`: Reactive notebooks
- `python-dotenv`: Environment variable management
