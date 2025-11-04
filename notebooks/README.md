# Notebooks

This directory contains Marimo notebooks demonstrating legiscope functions.

## demo_query.py

Interactive notebook demonstrating a real-world legal query about drug paraphernalia laws.

### Usage

```bash
# Run the notebook using uv (recommended)
uv run marimo edit notebooks/demo_query.py

# Or install marimo and run directly
pip install marimo
marimo edit notebooks/demo_query.py
```

### Requirements

- Processed jurisdiction data in `data/laws/` directory
- ChromaDB collection populated with embeddings
- Sections parquet file available for the target jurisdiction
