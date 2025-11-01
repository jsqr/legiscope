# Agents

This document contains information about available commands, workflows, and development practices for the legiscope project.

## Environment Setup

This project uses `uv` for dependency management and Python environment handling.

### Initial Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
make env

# Or manually:
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
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
pytest tests/test_code.py
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
- `marvin`: AI-powered function calls and structured outputs
- `psycopg[binary]`: PostgreSQL database adapter
- `pytest`: Testing framework
- `ruff`: Fast Python linter and formatter

## Database Setup

The project requires a PostgreSQL database with the `vector` extension. See README.md for detailed setup instructions.

## Common Workflows

### Adding New Dependencies

```bash
# Add production dependency
uv pip add package_name

# Add development dependency
uv pip add --dev package_name

# Update pyproject.toml manually if preferred
```

### Running the Full Pipeline

1. Set up database (see README.md)
2. Activate environment: `source .venv/bin/activate`
3. Run tests: `make test`
4. Process municipal codes using notebooks or scripts

### Code Quality Standards

- Use `ruff` for both linting and formatting
- All tests must pass before committing
- Follow existing code patterns and naming conventions
- Add type hints where appropriate
- Include docstrings for public functions and classes

## Testing Strategy

- Unit tests for core parsing and text processing functions
- Integration tests for database operations
- Test data and examples included in `tests/test_code.py`
- Use pytest fixtures for database setup when needed

## Performance Considerations

- Embeddings are batched to respect API limits
- Database operations use connection pooling
- Large text documents are chunked for processing
- Vector similarity search uses materialized views