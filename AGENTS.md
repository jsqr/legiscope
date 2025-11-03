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

