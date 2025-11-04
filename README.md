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

## Scripts and Modules

### Scripts
- `scripts/pipeline.sh` - Complete jurisdiction processing workflow automation
- `scripts/create_jurisdiction.py` - Create jurisdiction directory structure
- `scripts/convert_docx.sh` - Convert DOCX files to plain text using pandoc
- `scripts/convert_to_markdown.py` - Convert legal text to structured Markdown
- `scripts/segment_legal_code.py` - Segment Markdown into sections and segments
- `scripts/create_embeddings.py` - Generate embeddings and populate ChromaDB
- `scripts/example_query.py` - Example script demonstrating query functionality

### Notebooks
- `demo_query.py` - Interactive Marimo notebook demonstrating section-level retrieval with drug paraphernalia query

### Source Modules
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
