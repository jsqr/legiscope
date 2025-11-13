# legiscope

Automated analysis of municipal codes for legal epidemiology.

## Getting started

Developed on MacOS and Linux. No idea what, if anything, works on Windows.

### Environment Setup

This project uses `uv` for dependency management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on MacOS
brew install uv

# Set up the development environment
make env

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Model Configuration

The project uses a simplified model configuration approach with instructor's provider abstraction. Three main client types are configured:

- **Default Client** (`Config.get_default_client()`): Uses `gpt-4.1-mini` for general-purpose tasks
- **Big Client** (`Config.get_big_client()`): Uses `gpt-4.1` for complex reasoning tasks  
- **Embedding Client** (`get_embedding_client()`): Uses `ollama` with `embeddinggemma` model by default

#### Switching Models

To switch models, simply uncomment the desired alternative in `src/legiscope/model_config.py`:

```python
# In get_default_client():
# return instructor.from_provider("openai:gpt-4o")           # More powerful
# return instructor.from_provider("openai:gpt-4o-mini")     # Faster
# return instructor.from_provider("mistral:mistral-medium-latest")  # Mistral

# In get_big_client():
# return instructor.from_provider("openai:o1-preview")       # Advanced reasoning
# return instructor.from_provider("openai:o1-mini")          # Lightweight reasoning
# return instructor.from_provider("mistral:magistral-medium-latest")  # Mistral

# For embeddings, use the new interface:
# from legiscope.embeddings import get_embedding_client
# client = get_embedding_client("ollama")  # or "mistral"
```

#### Environment Variables

Set your API keys in the environment:

```bash
# For OpenAI models (default)
export OPENAI_API_KEY=your_openai_key

# For Mistral models
export MISTRAL_API_KEY=your_mistral_key

# Optional: Override default provider
export LLM_PROVIDER=openai  # or "mistral"
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
# Basic usage
./scripts/pipeline.sh NY "New York"

# Another example
./scripts/pipeline.sh CA LosAngeles
```

The pipeline performs these steps automatically:
1. Creates directory structure for the jurisdiction
2. Converts DOCX files to plain text (if present)
3. Converts text to structured Markdown with headings
4. Segments the code into searchable sections
5. Generates embeddings for semantic search

### Using Different Models

```python
from legiscope.model_config import Config
from legiscope.utils import ask
from pydantic import BaseModel

class LegalAnalysis(BaseModel):
    summary: str
    relevant_sections: list[str]

# Use default client for general tasks
default_client = Config.get_default_client()
result = ask(
    client=default_client,
    prompt="Analyze this legal text...",
    response_model=LegalAnalysis
)

# Use big client for complex reasoning
big_client = Config.get_big_client()
complex_result = ask(
    client=big_client,
    prompt="Perform deep legal analysis...",
    response_model=LegalAnalysis
)

# Use embedding client for semantic search
from legiscope.embeddings import get_embedding_client, create_embeddings_df
embedding_client = get_embedding_client("ollama")  # or "mistral"
embeddings_df = create_embeddings_df(segments_df, embedding_client)
```

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
- `model_config.py` - Centralized model configuration using instructor's provider abstraction
- `convert.py` - Text conversion utilities and LLM response models
- `utils.py` - Core utilities including LLM client and directory functions
- `embeddings.py` - Embedding generation and ChromaDB management
- `retrieve.py` - Information retrieval with HYDE query rewriting and section-level search
- `segment.py` - Text segmentation and hierarchical section processing

## Data Directory Structure

The project organizes municipal code data in a structured hierarchy:

```
data/
├── laws/                           # Municipal code data
│   └── {state}-{municipality}/     # Jurisdiction-specific directories
│       ├── raw/                    # Original source files (DOCX, PDF, etc.)
│       ├── processed/              # Processed text files and intermediate results
│       └── tables/                 # Structured data tables and exports
└── queries/                        # Database queries and search templates
```

### Project Structure

```
.
├── src/
│   └── legiscope/       # Main package source code
│       ├── model_config.py    # Model configuration and client management
│       ├── convert.py   # Conversion utilities and response models
│       ├── utils.py     # Core utility functions (ask function, directory creation)
│       ├── embeddings.py # Embedding generation and ChromaDB management
│       ├── retrieve.py   # Information retrieval with HYDE and section-level search
│       └── segment.py   # Text segmentation utilities
├── tests/               # Test files (123 tests including HYDE functionality)
├── scripts/             # Utility scripts
├── data/                # Data directory (not tracked by git)
├── pyproject.toml       # Project configuration and dependencies
├── Makefile            # Development commands
└── AGENTS.md           # Detailed development documentation
```

Instructions for the bots: [AGENTS.md](AGENTS.md).
