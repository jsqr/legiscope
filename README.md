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

# Or just run with uv
# uv run python foo.py
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

### Example Configurations

**OpenAI for language models:**
```bash
LEGISCOPE_LLM_PROVIDER=openai
LEGISCOPE_FAST_MODEL=gpt-4.1-mini
LEGISCOPE_POWERFUL_MODEL=gpt-4.1
```

**Ollama for local embeddings:**
```bash
LEGISCOPE_EMBEDDING_PROVIDER=ollama
LEGISCOPE_EMBEDDING_MODEL=embeddinggemma
LEGISCOPE_COLLECTION_NAME=legal_code_ollama
```

**Mistral for LLMs and embeddings:**
```bash
LEGISCOPE_LLM_PROVIDER=mistral
LEGISCOPE_FAST_MODEL=mistral-medium-latest
LEGISCOPE_POWERFUL_MODEL=magistral-medium-latest

LEGISCOPE_EMBEDDING_PROVIDER=mistral
LEGISCOPE_EMBEDDING_MODEL=mistral-embed
LEGISCOPE_COLLECTION_NAME=legal_code_mistral
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
- `llm_config.py` - Centralized LLM configuration using instructor's provider abstraction
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
│       ├── llm_config.py    # LLM configuration and client management
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
