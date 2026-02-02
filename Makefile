.PHONY: help env clean-env test test-cov lint format fix list clean install init parse process query dvc-repro

# Default target
help:
	@echo "Available commands:"
	@echo "  env        - Create/refresh virtual environment and install dependencies"
	@echo "  clean-env  - Remove virtual environment"
	@echo "  test       - Run tests"
	@echo "  test-cov   - Run tests with coverage report"
	@echo "  lint       - Run linting checks"
	@echo "  format     - Format code"
	@echo "  fix        - Fix linting issues"
	@echo "  list       - Show installed packages"
	@echo "  clean      - Clean build artifacts"
	@echo "  install    - Install package in development mode"
	@echo ""
	@echo "Pipeline (DVC — preferred):"
	@echo "  dvc-repro  - Show DVC pipeline usage"
	@echo ""
	@echo "Pipeline (legacy — deprecated, use DVC instead):"
	@echo "  init       - Initialize jurisdiction directory structure"
	@echo "  parse      - Convert raw files to structured Markdown"
	@echo "  process    - Create embeddings and build search index"
	@echo "  query      - Run queries against processed codes"

# Environment management
env:
	@echo "Syncing dependencies with uv..."
	@uv sync
	@echo "Environment setup complete!"

clean-env:
	@echo "Removing virtual environment..."
	@rm -rf .venv
	@echo "Virtual environment removed."

# Testing
test:
	@echo "Running tests..."
	@uv run pytest

test-cov:
	@echo "Running tests with coverage..."
	@uv run pytest --cov=src/legiscope --cov-report=html --cov-report=term

# Code quality
lint:
	@echo "Running linting checks..."
	@uv run ruff check src/ tests/
	@echo "Checking formatting..."
	@uv run ruff format --check src/ tests/
	@echo "Running type checks..."
	@uv run basedpyright src/

format:
	@echo "Formatting code..."
	@uv run ruff format src/ tests/

fix:
	@echo "Fixing linting issues..."
	@uv run ruff check --fix src/ tests/

# Utilities
list:
	@echo "Installed packages:"
	@uv pip list

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "Clean complete."

install:
	@echo "Installing package in development mode..."
	@uv sync
	@echo "Installation complete."

# DVC pipeline (preferred interface)
dvc-repro:
	@echo "Run the DVC pipeline with:"
	@echo ""
	@echo "  ./scripts/dvc_repro.sh --state STATE --locality LOCALITY --code-slug SLUG"
	@echo ""
	@echo "Or call DVC directly:"
	@echo ""
	@echo "  dvc exp run -S jurisdiction.state=STATE -S jurisdiction.locality=LOCALITY -S jurisdiction.code_slug=SLUG"
	@echo ""
	@echo "Initialize a new jurisdiction first:"
	@echo ""
	@echo "  python -m legiscope.pipeline.init --state STATE --locality LOCALITY --code-slug SLUG --name 'Display Name'"
	@echo ""
	@echo "See ./scripts/dvc_repro.sh --help for full options."

# Legacy pipeline stages (deprecated — use DVC workflow above)
init:
	@if [ -z "$(STATE)" ] || [ -z "$(CODE_SLUG)" ]; then \
		echo "Usage: make init STATE=CA CODE_SLUG=municipal-code [LOCALITY=LosAngeles]"; \
		echo "  Omit LOCALITY for state-level codes."; \
		exit 1; \
	fi
	@./scripts/pipeline_init.sh "$(STATE)" "$(or $(LOCALITY),-)" "$(CODE_SLUG)"

parse:
	@if [ -z "$(STATE)" ] || [ -z "$(CODE_SLUG)" ]; then \
		echo "Usage: make parse STATE=CA CODE_SLUG=municipal-code [LOCALITY=LosAngeles]"; \
		echo "  Omit LOCALITY for state-level codes."; \
		exit 1; \
	fi
	@./scripts/pipeline_parse.sh "$(STATE)" "$(or $(LOCALITY),-)" "$(CODE_SLUG)"

process:
	@if [ -z "$(STATE)" ] || [ -z "$(CODE_SLUG)" ]; then \
		echo "Usage: make process STATE=CA CODE_SLUG=municipal-code [LOCALITY=LosAngeles]"; \
		echo "  Omit LOCALITY for state-level codes."; \
		exit 1; \
	fi
	@./scripts/pipeline_process.sh "$(STATE)" "$(or $(LOCALITY),-)" "$(CODE_SLUG)"

query:
	@if [ -z "$(STATE)" ] || [ -z "$(CODE_SLUG)" ] || [ -z "$(QUERIES)" ]; then \
		echo "Usage: make query STATE=CA CODE_SLUG=municipal-code QUERIES=path/to/queries.csv [LOCALITY=LosAngeles]"; \
		echo "  Omit LOCALITY for state-level codes."; \
		exit 1; \
	fi
	@./scripts/pipeline_query.sh "$(STATE)" "$(or $(LOCALITY),-)" "$(CODE_SLUG)" "$(QUERIES)"
