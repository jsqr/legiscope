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
    temperature=0.1,
    max_retries=3
)

# Get appropriate models for different tasks
fast_model = Config.get_fast_model()        # For quick tasks
powerful_model = Config.get_powerful_model() # For complex reasoning
```

#### Using Config Objects

The library uses config objects for cleaner, more maintainable API:

```python
from legiscope.utils import LLMConfig
from legiscope.retrieve import SectionRetrievalSettings, retrieve_sections
from legiscope.query import QuerySettings, BatchQuerySettings
from legiscope.query import query_legal_documents, run_queries
from legiscope.llm_config import Config
import chromadb

# Setup
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
collection = chroma_client.get_collection("legal_code_all")

# Example 1: Retrieve sections with settings
retrieval_settings = SectionRetrievalSettings(
    jurisdiction_id="IL-WindyCity",
    n_results=10,
    use_hyde=True,
    hyde_client=Config.get_fast_client()
)
results = retrieve_sections(
    collection=collection,
    sections_parquet_path="./data/laws/IL-WindyCity/tables/sections.parquet",
    query_text="What are the parking regulations?",
    settings=retrieval_settings
)

# Example 2: Query with LLM analysis
llm_config = LLMConfig(client=Config.get_powerful_client(), temperature=0.1)
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
    sections_parquet_path="./data/laws/IL-WindyCity/tables/sections.parquet",
    queries=["Query 1", "Query 2", "Query 3"],
    jurisdiction_id="IL-WindyCity",
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
make test
```

Or manually:

```bash
pytest
pytest --cov=src/legiscope --cov-report=html
```

### Linting and Formatting

```bash
make lint
```

Or manually:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
ruff format src/ tests/
ruff check --fix src/ tests/
```

### Environment Management

```bash
make env
make clean-env
make list
```

Or manually:

```bash
uv pip list
```

### Pipeline Commands

```bash
# Run complete pipeline for specific jurisdiction
make pipeline STATE=CA MUNICIPALITY="San Francisco"

# Or manually:
./scripts/pipeline.sh california "San Francisco"
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

## Project Structure

```
.
├── src/
│   └── legiscope/       # Main package source code
├── tests/               # Test files
├── notebooks/           # Jupyter notebooks for analysis
│   ├── demo_nb.py       # Demo notebook
│   └── README.md
├── scripts/             # Utility scripts
│   ├── pipeline.sh      # Complete processing pipeline
│   └── ...
├── .env.example         # Environment variables template
├── pyproject.toml       # Project configuration and dependencies
├── Makefile            # Development commands
└── AGENTS.md           # This file
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
