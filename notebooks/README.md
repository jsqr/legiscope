# Notebooks

This directory contains Marimo notebooks demonstrating legiscope functions.

## query_demo.py

Interactive notebook demonstrating a real-world legal query about drug paraphernalia laws.

### Usage

```bash
# Run the notebook using uv (recommended)
uv run marimo edit notebooks/query_demo.py

# Or install marimo, activate the environment, and run directly
```

### Requirements

- Processed jurisdiction data in `data/laws/` directory
- ChromaDB collection populated with embeddings
- Sections parquet file available for the target jurisdiction
