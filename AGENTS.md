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

The project supports both OpenAI and Mistral as LLM providers. The default provider is Mistral.

#### Environment Variables

- `LEGISCOPE_LLM_PROVIDER`: Set to "openai" or "mistral" (default) to select LLM provider
- `LEGISCOPE_FAST_MODEL`: Override fast model selection
- `LEGISCOPE_POWERFUL_MODEL`: Override powerful model selection
- `OPENAI_API_KEY`: Required when using OpenAI provider
- `MISTRAL_API_KEY`: Required when using Mistral provider

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
jurisdiction = JurisdictionRef(state="IL", municipality="WindyCity")
code_ref = CodeRef(jurisdiction=jurisdiction, code_slug="municipal-code")

chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
collection = chroma_client.get_collection("legal_code_all")

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
- Powerful model: `mistral-medium-2508` (for complex reasoning)

**Ollama Provider:**
- Fast model: `ministral-3` (for quick local tasks)
- Powerful model: `ministral-3:14b` (for complex local reasoning)
- Requires Ollama server running locally

#### Example Setup

```bash
# For OpenAI
export LEGISCOPE_LLM_PROVIDER=openai
export OPENAI_API_KEY=your_openai_key

# For Mistral (default)
export LEGISCOPE_LLM_PROVIDER=mistral
export MISTRAL_API_KEY=your_mistral_key

# For Ollama (local)
export LEGISCOPE_LLM_PROVIDER=ollama
# Requires Ollama server running: ollama serve
```

### Embedding Model Configuration

The project supports multiple embedding models for generating text embeddings. The default provider is Mistral.

#### Current Configuration

**Mistral with mistral-embed (default)**
- Uses Mistral's API
- Model: `mistral-embed`
- Client: `get_embedding_client("mistral")`
- Requires: `MISTRAL_API_KEY` environment variable

#### Alternative Configuration

**Ollama with embeddinggemma**
- Uses local Ollama server
- Model: `embeddinggemma`
- Client: `get_embedding_client("ollama")`
- Auto-detected provider and model

#### Switching Between Embedding Models

To switch between embedding providers:

1. Set your environment variables:
   ```bash
   # For Mistral (default)
   export LEGISCOPE_EMBEDDING_PROVIDER=mistral
   export MISTRAL_API_KEY=your_mistral_key

   # For Ollama
   export LEGISCOPE_EMBEDDING_PROVIDER=ollama
   ```

2. Use the embedding interface:
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
config = EmbeddingConfig(provider="mistral")  # Uses default mistral-embed model
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
# Run all linting and formatting checks
make lint
# Or manually:
ruff check src/ tests/
ruff format --check src/ tests/

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

The pipeline is split into 4 independent stages:

```bash
# Stage 1: Initialize (create directory structure)
make init STATE=CA MUNICIPALITY=LosAngeles CODE_SLUG=municipal-code

# Stage 2: Parse (convert raw files to Markdown)
make parse STATE=CA MUNICIPALITY=LosAngeles CODE_SLUG=municipal-code

# Stage 3: Process (create embeddings and build index)
make process STATE=CA MUNICIPALITY=LosAngeles CODE_SLUG=municipal-code

# Stage 4: Query (run queries, optional)
make query STATE=CA MUNICIPALITY=LosAngeles CODE_SLUG=municipal-code QUERIES=data/queries/example.csv

# For state-level codes, omit MUNICIPALITY:
make init STATE=CA CODE_SLUG=penal-code
make parse STATE=CA CODE_SLUG=penal-code
make process STATE=CA CODE_SLUG=penal-code
```

### Benchmarking

The project handles benchmarking against MonQcle data using an LLM-as-judge approach.
See `BENCHMARKING.md` for full documentation.

```bash
# Run benchmarking pipeline
uv run python scripts/benchmark_pipeline.py \
    --queries-path data/queries/drug_paraphernalia_queries_clean.csv \
    --monqcle-path data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv \
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
    --municipality LosAngeles \
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
│       ├── models.py    # Data models (JurisdictionRef, CodeRef, schema constants)
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
│   ├── pipeline_init.sh       # Stage 1: Initialize jurisdiction
│   ├── pipeline_parse.sh      # Stage 2: Parse raw files to Markdown
│   ├── pipeline_process.sh    # Stage 3: Create embeddings and index
│   ├── pipeline_query.sh      # Stage 4: Run queries
│   ├── create_jurisdiction.py # Register jurisdiction and create directory structure
│   ├── convert_to_markdown.py # Convert raw text to structured Markdown
│   ├── segment_legal_code.py  # Segment Markdown into sections and segments
│   ├── create_embeddings.py   # Generate embeddings (Parquet, no ChromaDB)
│   ├── build_chroma_index.py  # Build ChromaDB index from embeddings
│   ├── run_queries.py         # Batch query execution
│   ├── benchmark_pipeline.py  # Benchmarking workflow
│   └── ...
├── data/                # Data directory (not tracked by git)
│   ├── jurisdictions.parquet  # Registry of all jurisdictions
│   ├── codes.parquet          # Registry of all legal codes
│   └── laws/                  # Per-code data directories
│       └── {STATE}/{Municipality}/{code-slug}/
│           ├── raw/               # Raw source files
│           ├── code.md            # Structured Markdown
│           ├── sections.parquet   # Section hierarchy
│           ├── segments.parquet   # Text segments
│           ├── embeddings.parquet # Embedding vectors
│           ├── relations.parquet  # Intra-code relations
│           └── external_references.parquet
├── pyproject.toml       # Project configuration and dependencies
├── Makefile            # Development commands
├── AGENTS.md           # This file
└── CONTRIBUTING.md     # Contribution guidelines
```

## Key Dependencies

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
