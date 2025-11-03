.PHONY: help env clean-env test test-cov lint format fix list clean install

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

# Environment management
env:
	@if [ ! -d ".venv" ]; then \
		uv venv; \
	fi
	@echo "Activating virtual environment and installing dependencies..."
	@source .venv/bin/activate && uv pip install -e ".[dev]"
	@echo "Environment setup complete!"

clean-env:
	@echo "Removing virtual environment..."
	@rm -rf .venv
	@echo "Virtual environment removed."

# Testing
test:
	@echo "Running tests..."
	@source .venv/bin/activate && pytest

test-cov:
	@echo "Running tests with coverage..."
	@source .venv/bin/activate && pytest --cov=src/legiscope --cov-report=html --cov-report=term

# Code quality
lint:
	@echo "Running linting checks..."
	@source .venv/bin/activate && ruff check src/ tests/
	@echo "Checking formatting..."
	@source .venv/bin/activate && ruff format --check src/ tests/

format:
	@echo "Formatting code..."
	@source .venv/bin/activate && ruff format src/ tests/

fix:
	@echo "Fixing linting issues..."
	@source .venv/bin/activate && ruff check --fix src/ tests/

# Utilities
list:
	@echo "Installed packages:"
	@source .venv/bin/activate && uv pip list

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
	@source .venv/bin/activate && uv pip install -e ".[dev]"
	@echo "Installation complete."

