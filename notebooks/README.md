# Notebooks

This directory contains Marimo notebooks demonstrating legiscope functionality.

## demo_query.py

Interactive notebook demonstrating the new `retrieve_sections` function with a real-world legal query about drug paraphernalia laws.

### Features

- **Interactive Controls**: Adjust jurisdiction, query text, and search parameters
- **HYDE Integration**: Optional LLM-powered query rewriting
- **Rich Display**: Formatted results using Marimo's markdown capabilities
- **Error Handling**: Graceful handling of missing data and configuration issues

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

### Query Example

The notebook demonstrates searching for:
> "Does jurisdiction have any laws restricting the sale of drug paraphernalia?"

This showcases how section-level retrieval provides broader legal context compared to segment-level search, making it ideal for comprehensive legal research.