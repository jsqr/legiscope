# Supporting Passages Validation

## Overview

The `query.py` module now includes automatic validation of `supporting_passages` in `LegalQueryResponse` to guard against LLM hallucination or distortion.

## How It Works

When `query_legal_documents()` generates a response, it automatically validates each supporting passage against the retrieved text using:

1. **Exact substring matching** (fast path) - checks if the passage appears exactly in the retrieved sections
2. **Fuzzy matching** (fallback) - uses `difflib.SequenceMatcher` to detect near-misses or distortions

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
from legiscope.query import QueryConfig, query_legal_documents
from legiscope.utils import LLMConfig
from legiscope.llm_config import Config

# Create query config
llm_config = LLMConfig(client=Config.get_fast_client())
config = QueryConfig(
    llm=llm_config,
    query="Are there parking restrictions?",
    retrieval_results=results,
)

# Process query - validation happens automatically
response = query_legal_documents(config)

# Check logs for any hallucination warnings
# Warnings are logged via loguru at WARNING level
```

## What Gets Validated

The validation checks all text from:
- `section.body_text` - Full section text
- `segment.segment_text` - Individual matching segments

This ensures passages must actually appear somewhere in the retrieved documents.

## Performance

The validation uses optimized fuzzy matching:
- Skips fuzzy matching if exact match found (fast path)
- Only checks substrings within ±20% of passage length
- Early exits when good match found
- Minimal overhead for typical legal documents
