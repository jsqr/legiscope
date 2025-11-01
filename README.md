# legiscope

Automated analysis of municipal codes for legal epidemiology.

## Getting started

### Environment Setup

This project uses `uv` for dependency management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up the development environment
make env

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Setting up a local postgres database

#### 1. Install postgres.

* MacOS: I find [Postgres.app](https://www.postgresql.org/download/macosx/) is an easy
way to get set up for local development instances on the Mac.

* Linux, etc.: [This site](https://www.postgresql.org/download/) gives instructions for
different distros, and other operating systems. You will also need to install the
`vector` extension.

#### 2. Start psql and create a user

```sh
psql -U postgres
```

then

```sql
CREATE USER muni WITH PASSWORD 'muni';
```

#### 2. Create a database and set up permissions

```sql
CREATE DATABASE muni;
```

then

```sh
psql -U postgres -d muni
```

then

```sql
CREATE EXTENSION IF NOT EXISTS vector;
GRANT CREATE ON SCHEMA public TO muni;
```

#### 3. Run the initialization script

```sh
. scripts/reset.sh localhost muni muni
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

## Municipal Code Analysis

This project uses [Marimo](https://marimo.io/) notebooks for interactive municipal code analysis.

### Getting Started with Marimo

Run the analysis notebook:
```bash
uv run marimo edit notebooks/template-workflow.py
```

This will open the notebook in your browser at `http://localhost:2718`

### Key Features

- **Interactive UI Elements**: Text areas for queries, sliders for result limits, buttons for actions
- **Reactive Cells**: Automatic updates when inputs change
- **Real-time Results**: Query results update as you type
- **Separate Interfaces**: Dedicated interfaces for semantic, full-text, and hybrid queries
- **Interactive Reports**: Generate and upload reports with button controls

### Usage Workflow

1. **Data Preparation**: Create a `data/<jurisdiction>` directory with DOCX files containing the municipal code, then run `scripts/convert_docx.sh` to convert them to a single text file
2. **Configure**: Update the `heading_examples` dictionary in the notebook with your jurisdiction's heading patterns
3. **Process**: The notebook automatically parses and processes the municipal code
4. **Query**: Use the interactive query interfaces to search the code
5. **Report**: Generate and upload reports using the interactive buttons

### Creating Jurisdiction-Specific Notebooks

To create a version for a new jurisdiction:

```bash
cp notebooks/template-workflow.py notebooks/<jurisdiction>.py
uv run marimo edit notebooks/<jurisdiction>.py
```

Then update the jurisdiction-specific parameters in the notebook.

### Project Structure

```
.
├── src/
│   └── legiscope/       # Main package source code
├── tests/               # Test files
├── notebooks/           # Marimo notebooks for analysis
├── scripts/             # Utility scripts
├── pyproject.toml       # Project configuration and dependencies
├── Makefile            # Development commands
└── AGENTS.md           # Detailed development documentation
```

Instructions for the bots: [AGENTS.md](AGENTS.md).
