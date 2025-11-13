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

The project supports both OpenAI and Mistral as LLM providers:

#### Environment Variables

- `LLM_PROVIDER`: Set to "openai" (default) or "mistral" to select LLM provider
- `OPENAI_API_KEY`: Required when using OpenAI provider
- `MISTRAL_API_KEY`: Required when using Mistral provider

#### Example Setup

```bash
# For OpenAI (default)
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_openai_key

# For Mistral
export LLM_PROVIDER=mistral
export MISTRAL_API_KEY=your_mistral_key
```

### Embedding Model Configuration

The project supports multiple embedding models for generating text embeddings:

#### Current Configuration

**Ollama with embeddinggemma (default)**
- Uses local Ollama server
- Model: `embeddinggemma`
- Client: `get_embedding_client("ollama")`
- Auto-detected provider and model

#### Alternative Configuration

**Mistral with mistral-embed**
- Uses Mistral's API
- Model: `mistral-embed`
- Requires: `MISTRAL_API_KEY` environment variable
- Client: `get_embedding_client("mistral")`

#### Switching Between Embedding Models

To switch from Ollama to Mistral embeddings:

1. Set your Mistral API key:
   ```bash
   export MISTRAL_API_KEY=your_mistral_key
   ```

2. Use the new embedding interface:
   ```python
   from legiscope.embeddings import get_embedding_client, get_embeddings
   
   # For Ollama (default)
   client = get_embedding_client("ollama")
   embeddings = get_embeddings(client, ["text1", "text2"])
   
   # For Mistral
   client = get_embedding_client("mistral")
   embeddings = get_embeddings(client, ["text1", "text2"])
   ```

#### Usage Examples

```python
from legiscope.embeddings import get_embedding_client, get_embeddings, EmbeddingConfig

# Get embedding client for specific provider
client = get_embedding_client("ollama")  # or "mistral"

# Generate embeddings (auto-detects model)
texts = ["Legal text 1", "Legal text 2"]
embeddings = get_embeddings(client, texts)

# Or specify model explicitly
embeddings = get_embeddings(client, texts, model="embeddinggemma", provider="ollama")

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
├── scripts/             # Utility scripts
├── pyproject.toml       # Project configuration and dependencies
├── Makefile            # Development commands
└── AGENTS.md           # This file
```

## Key Dependencies

- `openai`: OpenAI API client for embeddings and language models
- `instructor`: AI-powered function calls and structured outputs
- `pytest`: Testing framework
- `ruff`: Fast Python linter and formatter

