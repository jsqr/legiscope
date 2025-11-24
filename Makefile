.PHONY: help env clean-env test test-cov lint format fix list clean install pipeline

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
	@echo "  pipeline   - Run complete jurisdiction processing pipeline"

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

# Pipeline
pipeline:
	@if [ -z "$(STATE)" ] || [ -z "$(MUNICIPALITY)" ]; then \
		echo "Usage: make pipeline STATE=NY MUNICIPALITY=\"New York\" [QUERIES=path/to/queries.txt]"; \
		echo "Example: make pipeline STATE=CA MUNICIPALITY=LosAngeles"; \
		echo "Example with queries: make pipeline STATE=CA MUNICIPALITY=LosAngeles QUERIES=data/queries/example_queries.txt"; \
		exit 1; \
	fi
	@echo "Running complete pipeline for $(STATE)-$(MUNICIPALITY)..."
	@if [ -n "$(QUERIES)" ]; then \
		./scripts/pipeline.sh "$(STATE)" $(MUNICIPALITY) $(QUERIES); \
	else \
		./scripts/pipeline.sh "$(STATE)" $(MUNICIPALITY); \
	fi

