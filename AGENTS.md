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
- `LEGISCOPE_DATA_DIR`: Override the root data directory path (infrastructure)

All hyperparameters (provider, model names, temperature, retrieval/query settings, etc.) are configured in `params.yaml`.

#### Code-based Configuration

You can configure the LLM provider directly in code using the `Config` class:

```python
from legiscope.llm_config import Config
from legiscope.utils import LLMConfig

# Get fast client for quick tasks
fast_client = Config.get_fast_client()      # Uses mistral-medium-latest (default)
powerful_client = Config.get_powerful_client()  # Uses magistral-medium-latest

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

**OpenAI Provider:**
- Fast model: `gpt-4.1-mini` (for HYDE, relevance assessment, etc.)
- Powerful model: `gpt-4.1` (for complex legal analysis)

**Mistral Provider:**
- Fast model: `mistral-small-2506` (for quick tasks)
- Powerful model: `mistral-large-2512` (for complex reasoning)

**Ollama Provider:**
- Fast model: `qwen3:8b` (for quick local tasks)
- Powerful model: `qwen3:30b` (for complex local reasoning)
- Requires Ollama server running locally

#### Example Setup

```bash
# Set API keys as secrets (in .env or environment)
export OPENAI_API_KEY=your_openai_key
export MISTRAL_API_KEY=your_mistral_key

# To change the default provider, edit params.yaml:
#   llm.default_provider: "mistral"  (or "openai", "ollama")

# For Ollama (local), ensure server is running: ollama serve
```

### Embedding Model Configuration

The project supports multiple embedding models for generating text embeddings. The default provider is Ollama.

#### Current Configuration

**Ollama with embeddinggemma (default)**
- Uses local Ollama server
- Model: `embeddinggemma`
- Client: `get_embedding_client("ollama")`
- Requires: Ollama server running locally (`ollama serve`)

#### Alternative Configuration

**Mistral with mistral-embed**
- Uses Mistral's API
- Model: `mistral-embed`
- Client: `get_embedding_client("mistral")`
- Requires: `MISTRAL_API_KEY` environment variable

#### Switching Between Embedding Models

To switch between embedding providers, edit `params.yaml`:
```yaml
embeddings:
  default_provider: "ollama"  # or "mistral"
```

#### Retrieval & Query Configuration

Retrieval settings (HYDE, relevance filtering) and query settings (model tier,
passage validation) are all in `params.yaml`. CLI scripts (`scripts/run_queries.py`,
`coep/scripts/benchmark_pipeline.py`) read these as defaults; CLI flags override them.

```yaml
retrieval:
  n_results: 10
  distance_metric: l2        # ChromaDB HNSW distance (l2, cosine, ip)
  hyde:
    enabled: false            # uses fast model
  relevance_filter:
    enabled: false            # uses fast model
    threshold: 0.5

query:                        # uses powerful model
  validation:
    enabled: true
    exact_match_threshold: 1.0
    fuzzy_match_threshold: 0.9
```

Use the embedding interface:
   ```python
   from legiscope.embeddings import get_embedding_client, get_embeddings

   # For Mistral (default)
   client = get_embedding_client("mistral")
   embeddings = get_embeddings(client, ["text1", "text2"])

   # For Ollama
   client = get_embedding_client("ollama")
   embeddings = get_embeddings(client, ["text1", "text2"])
   ```

#### Usage Examples

```python
from legiscope.embeddings import get_embedding_client, get_embeddings, EmbeddingConfig

# Get embedding client for specific provider
client = get_embedding_client("mistral")  # or "ollama"

# Generate embeddings (auto-detects model)
texts = ["Legal text 1", "Legal text 2"]
embeddings = get_embeddings(client, texts)

# Or specify model explicitly
embeddings = get_embeddings(client, texts, model="mistral-embed", provider="mistral")

# Using with EmbeddingConfig
config = EmbeddingConfig(provider="ollama")  # Uses default embeddinggemma model
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

The pipeline is managed by DVC with four stages: **parse → segment → embed → index**.

#### Initialize a jurisdiction (one-time, not a DVC stage)

```bash
python -m legiscope.pipeline.init \
    --state CA --locality LosAngeles \
    --code-slug municipal-code --name "LA Municipal Code"

# For state-level codes:
python -m legiscope.pipeline.init \
    --state CA --code-slug penal-code --name "CA Penal Code"
```

#### Run the pipeline

```bash
# Using the wrapper script (recommended):
./scripts/dvc_repro.sh --state CA --locality LosAngeles --code-slug municipal-code

# Or calling DVC directly:
dvc exp run \
    -S jurisdiction.state=CA \
    -S jurisdiction.locality=LosAngeles \
    -S jurisdiction.code_slug=municipal-code

# Run a single stage:
./scripts/dvc_repro.sh --state CA --locality LosAngeles --code-slug municipal-code \
    --stage segment
```

> **Note:** The legacy `make init/parse/process/query` targets still work but are
> deprecated. Use the DVC workflow above instead.

### Benchmarking

The project handles benchmarking against MonQcle data using an LLM-as-judge approach.
See `coep/docs/BENCHMARKING.md` for full documentation.

```bash
# Run benchmarking pipeline
uv run python coep/scripts/benchmark_pipeline.py \
    --queries-path data/queries/drug_paraphernalia_queries_clean.csv \
    --monqcle-path coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv \
    --series-title DPL_2025_Consolidated \
    --jurisdiction-id CA-LosAngeles \
    --output data/output/CA-LosAngeles/benchmark_results.csv \
    --n-results 10 \
    --use-hyde False \
    --filter-relevance False \
    --relevance-threshold 0.5 \
    --validate-supporting-passages False \
    --test-limit 5 \
    --debug
```

### Advanced Query Execution

For granular control over query execution (HYDE, Relevance Filtering), run the script directly:

```bash
uv run python scripts/run_queries.py \
    --state CA \
    --locality LosAngeles \
    --code-slug municipal-code \
    --queries-path "data/queries/test_queries.csv" \
    --n-results 10 \
    --use-hyde False \
    --filter-relevance False \
    --relevance-threshold 0.5 \
    --validate-supporting-passages False \
    --output "data/output/test_results.csv"
```

## Project Structure

```
.
├── src/
│   └── legiscope/       # Main package source code
│       ├── pipeline/        # DVC stage modules
│       │   ├── init.py      # Initialize jurisdiction (not a DVC stage)
│       │   ├── parse.py     # Parse raw files to Markdown
│       │   ├── segment.py   # Segment Markdown into sections
│       │   ├── embed.py     # Generate embedding vectors
│       │   └── index.py     # Build ChromaDB search index
│       ├── config.py        # Infrastructure configuration (config.yaml)
│       ├── params.py        # DVC params.yaml loader
│       ├── models.py        # Data models (JurisdictionRef, CodeRef, schema constants)
│       ├── llm_config.py    # LLM configuration and client management
│       ├── convert.py       # Conversion utilities and response models
│       ├── utils.py         # Core utility functions
│       ├── embeddings.py    # Embedding generation and ChromaDB management
│       ├── retrieve.py      # Information retrieval with HYDE and section-level search
│       ├── segment.py       # Text segmentation utilities
│       └── query.py         # Legal query processing with structured responses
├── tests/               # Test files
├── scripts/             # Utility scripts
│   ├── dvc_repro.sh           # DVC pipeline wrapper (primary interface)
│   ├── run_queries.py         # Batch query execution
│   ├── pipeline_init.sh       # (deprecated) Initialize jurisdiction
│   ├── pipeline_parse.sh      # (deprecated) Parse raw files to Markdown
│   ├── pipeline_process.sh    # (deprecated) Create embeddings and index
│   ├── pipeline_query.sh      # (deprecated) Run queries
│   ├── create_jurisdiction.py # (deprecated) Register jurisdiction
│   ├── convert_to_markdown.py # (deprecated) Convert raw text to Markdown
│   ├── segment_legal_code.py  # (deprecated) Segment Markdown
│   ├── create_embeddings.py   # (deprecated) Generate embeddings
│   ├── build_chroma_index.py  # (deprecated) Build ChromaDB index
│   └── ...
├── coep/
│   ├── __init__.py
│   ├── src/
│   │   ├── eval.py            # COEP evaluation logic
│   │   └── query.py           # COEP query preprocessing
│   ├── tests/
│   │   ├── test_eval.py
│   │   └── test_query_adjustments.py
│   ├── scripts/
│   │   └── benchmark_pipeline.py  # COEP benchmarking workflow
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
├── dvc.yaml             # DVC pipeline definition
├── pyproject.toml       # Project configuration and dependencies
├── Makefile            # Development commands
├── AGENTS.md           # This file
└── CONTRIBUTING.md     # Contribution guidelines
```

## Key Dependencies

- `dvc`: Pipeline orchestration and experiment tracking
- `openai`: OpenAI API client for embeddings and language models
- `mistralai`: Mistral API client for embeddings and language models
- `ollama`: Ollama client for local LLM inference
- `instructor`: AI-powered function calls and structured outputs
- `pytest`: Testing framework
- `ruff`: Fast Python linter and formatter
- `black`: Code formatter
- `chromadb`: Vector database for embeddings
- `duckdb`: Analytical database
- `marimo`: Reactive notebooks
- `python-dotenv`: Environment variable management
