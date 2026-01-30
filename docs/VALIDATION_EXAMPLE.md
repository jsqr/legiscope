# Supporting Passages Validation

## Overview

The `query.py` module now includes automatic validation of `supporting_passages` in `LegalQueryResponse` to guard against LLM hallucination or distortion.

## How It Works

When `query_legal_documents()` generates a response, it automatically validates each supporting passage against the retrieved text using:

1. **Text Normalization**: Collapses whitespace (tabs, newlines) and converts smart quotes to standard ASCII to ensure robust matching.
2. **Exact substring matching** (fast path) - checks if the passage appears exactly in the retrieved sections (both raw and normalized).
3. **Fuzzy matching** (fallback) - uses `rapidfuzz` (C++ accelerated) to find the best matching alignment and detect near-misses or distortions.

## Warning Levels

### No Match Found
```
WARNING - Supporting passage 1 NOT FOUND in retrieved text (best similarity: 0.24):
  Passage: This text does not exist in the retrieved documents...
```

### Close Match (Possible Distortion)
```
WARNING - Supporting passage 2 has close match (similarity: 0.95) but not exact - possible LLM distortion:
  LLM passage: No person should sell drug paraphernalia items...
  Best match:  No person shall sell drug paraphernalia...
```

### Summary Warning
```
WARNING - HALLUCINATION WARNING: 2/5 supporting passages not found in retrieved documents.
The LLM may have distorted or fabricated some supporting text.
```

## Configuration

You can adjust the thresholds in `_validate_supporting_passages()`:

```python
_validate_supporting_passages(
    response,
    sections,
    exact_match_threshold=1.0,   # 1.0 = exact match required
    fuzzy_match_threshold=0.9     # 0.9 = warn if similarity >= 0.9 but < 1.0
)
```

## Example Usage

```python
from legiscope.query import QuerySettings, query_legal_documents
from legiscope.utils import LLMConfig
from legiscope.llm_config import Config

# Create query settings
llm_config = LLMConfig(client=Config.get_fast_client())
settings = QuerySettings(
    llm=llm_config,
    filter_relevance=True,
    relevance_threshold=0.7
)

# Process query - validation happens automatically
response, scores = query_legal_documents(
    retrieval_results=results,
    query="Are there parking restrictions?",
    settings=settings
)

# Check logs for any hallucination warnings
# Warnings are logged via loguru at WARNING level
```

## What Gets Validated

The validation checks all text from:
- `section.body_text` - Full section text
- `segment.segment_text` - Individual matching segments

This ensures passages must actually appear somewhere in the retrieved documents.

## Performance

The validation uses optimized matching:
- **Pre-computed Normalization**: Text is normalized once to handle formatting differences.
- **Rapidfuzz**: Uses C++ implementation for string matching, which is orders of magnitude faster than Python's standard library.
- **Fast Path**: Skips fuzzy matching if exact match found.
- **Minimal Overhead**: Even with fuzzy matching, the C++ acceleration keeps this step extremely fast.
