# Config-Based API Guide

## Quick Start

The legiscope library now uses config objects for cleaner, more maintainable code. Instead of passing many parameters to functions, you create configuration objects that encapsulate all settings.

---

## Config Classes

### LLMConfig
Configure LLM behavior (client, model, temperature, retries).

```python
from legiscope.utils import LLMConfig
from legiscope.llm_config import Config

# Minimal - uses defaults
llm_config = LLMConfig(client=Config.get_fast_client())

# Custom settings
llm_config = LLMConfig(
    client=Config.get_powerful_client(),
    model="gpt-4",
    temperature=0.2,
    max_retries=5
)
```

---

### RetrievalConfig
Configure document retrieval (search, filtering, HYDE, embeddings).

```python
from legiscope.retrieve import RetrievalConfig, retrieve_segments

# Basic retrieval
config = RetrievalConfig(
    collection=chroma_collection,
    query_text="parking regulations",
    jurisdiction_id="IL-WindyCity"
)
results = retrieve_segments(config)

# With HYDE rewriting
config = RetrievalConfig(
    collection=chroma_collection,
    query_text="where can I park my car",
    use_hyde=True,
    hyde_client=Config.get_fast_client(),
    n_results=20
)
results = retrieve_segments(config)
```

**Key Parameters**:
- `collection` (required): ChromaDB collection
- `query_text` (required): Search query
- `n_results`: Number of results (default: 10)
- `jurisdiction_id`: Filter by jurisdiction
- `use_hyde`: Enable HYDE rewriting (default: False)
- `hyde_client`: Required if use_hyde=True
- `hyde_model`: Model for HYDE (default: fast model)

---

### SectionRetrievalConfig
Extended retrieval config for section-level results.

```python
from legiscope.retrieve import SectionRetrievalConfig, retrieve_sections

config = SectionRetrievalConfig(
    collection=chroma_collection,
    query_text="parking regulations",
    sections_parquet_path="./data/sections.parquet",
    jurisdiction_id="IL-WindyCity",
    n_results=10,
    use_hyde=True,
    hyde_client=Config.get_fast_client()
)
results = retrieve_sections(config)
```

**Additional Parameters** (inherits all from RetrievalConfig):
- `sections_parquet_path` (required): Path to sections.parquet file

---

### QueryConfig
Configure LLM-powered query processing.

```python
from legiscope.utils import LLMConfig
from legiscope.query import QueryConfig, query_legal_documents

llm_config = LLMConfig(client=Config.get_powerful_client())

config = QueryConfig(
    llm=llm_config,
    query="Are there parking restrictions?",
    retrieval_results=results,
    filter_relevance=True,
    relevance_threshold=0.7
)
response = query_legal_documents(config)
```

**Key Parameters**:
- `llm` (required): LLMConfig object
- `query` (required): User's legal question
- `retrieval_results` (required): Results from retrieve_sections()
- `filter_relevance`: Filter sections before LLM (default: False)
- `relevance_threshold`: Min confidence for filtering (default: 0.5)
- `filter_llm`: Separate LLM for filtering (default: uses main llm)

---

### BatchQueryConfig
Configure batch query processing.

```python
from legiscope.query import BatchQueryConfig, run_queries

# Minimal - uses default LLM
config = BatchQueryConfig(
    queries=["Query 1", "Query 2", "Query 3"],
    jurisdiction_id="IL-WindyCity",
    sections_parquet_path="./data/sections.parquet",
    collection=chroma_collection
)
results_df = run_queries(config)

# Full customization
llm_config = LLMConfig(
    client=Config.get_powerful_client(),
    model="gpt-4",
    temperature=0.1
)

config = BatchQueryConfig(
    queries=queries,
    jurisdiction_id="IL-WindyCity",
    sections_parquet_path="./data/sections.parquet",
    collection=chroma_collection,
    llm=llm_config,
    n_results=20,
    use_hyde=True,
    filter_relevance=True,
    relevance_threshold=0.8
)
results_df = run_queries(config)
```

**Key Parameters**:
- `queries` (required): List of questions
- `jurisdiction_id` (required): Jurisdiction to search
- `sections_parquet_path` (required): Path to sections file
- `collection` (required): ChromaDB collection
- `llm`: LLMConfig (default: creates fast client)
- `n_results`: Results per query (default: 10)
- `use_hyde`: Enable HYDE rewriting (default: False)
- `filter_relevance`: Filter sections (default: False)
- `relevance_threshold`: Min confidence (default: 0.5)

---

## Common Patterns

### Pattern 1: Simple Workflow
```python
from legiscope.llm_config import Config
from legiscope.retrieve import SectionRetrievalConfig, retrieve_sections
from legiscope.utils import LLMConfig
from legiscope.query import QueryConfig, query_legal_documents
import chromadb

# Setup
collection = chromadb.PersistentClient("./data/chroma_db").get_collection("legal_code_all")

# Step 1: Retrieve
retrieval_config = SectionRetrievalConfig(
    collection=collection,
    query_text="parking regulations",
    sections_parquet_path="./data/sections.parquet",
    jurisdiction_id="IL-WindyCity"
)
results = retrieve_sections(retrieval_config)

# Step 2: Query
llm_config = LLMConfig(client=Config.get_powerful_client())
query_config = QueryConfig(
    llm=llm_config,
    query="What are the parking regulations?",
    retrieval_results=results
)
response = query_legal_documents(query_config)

print(response.short_answer)
```

---

### Pattern 2: Batch Processing
```python
from legiscope.query import BatchQueryConfig, run_queries

queries = [
    "Are there parking restrictions?",
    "Do I need a business license?",
    "What are the noise ordinances?"
]

config = BatchQueryConfig(
    queries=queries,
    jurisdiction_id="IL-WindyCity",
    sections_parquet_path="./data/sections.parquet",
    collection=collection,
    use_hyde=True,
    filter_relevance=True
)

results_df = run_queries(config)
print(results_df.select(["query", "short_answer", "confidence"]))
```

---

### Pattern 3: Config Reuse
```python
from dataclasses import replace

# Create base configuration
base_config = BatchQueryConfig(
    queries=[],  # Will override per batch
    jurisdiction_id="IL-WindyCity",
    sections_parquet_path="./data/sections.parquet",
    collection=collection,
    use_hyde=True,
    filter_relevance=True
)

# Process multiple batches efficiently
all_results = []
for query_batch in query_batches:
    # Create new config with different queries
    batch_config = replace(base_config, queries=query_batch)
    results = run_queries(batch_config)
    all_results.append(results)

# Combine results
import polars as pl
combined_df = pl.concat(all_results)
```

---

### Pattern 4: Different LLMs for Different Tasks
```python
# Fast LLM for retrieval/filtering
fast_llm = LLMConfig(client=Config.get_fast_client())

# Powerful LLM for final analysis
powerful_llm = LLMConfig(
    client=Config.get_powerful_client(),
    temperature=0.1
)

# Use fast LLM for filtering, powerful for analysis
query_config = QueryConfig(
    llm=powerful_llm,
    query="complex legal question",
    retrieval_results=results,
    filter_relevance=True,
    filter_llm=fast_llm  # Use fast model for filtering
)
```

---

## Parameter Name Changes

| Old Name | New Name | Reason |
|----------|----------|--------|
| `rewrite` | `use_hyde` | Clearer intent |
| `rewrite_client` | `hyde_client` | Consistency |
| `rewrite_model` | `hyde_model` | Consistency |

---

## Benefits

### 1. Cleaner Function Signatures
```python
# Before: 13 parameters, hard to remember order
run_queries(client, queries, jur_id, path, coll, None, 0.1, 3, 10, True, True, 0.7, None)

# After: 1 parameter, self-documenting
run_queries(config)
```

### 2. Better IDE Support
- Autocomplete shows all available options
- Type checking catches errors earlier
- Inline documentation for each field

### 3. Config Reusability
```python
# Create once
standard_config = BatchQueryConfig(...)

# Reuse many times
results1 = run_queries(standard_config)
results2 = run_queries(standard_config)

# Modify for specific case
strict_config = replace(standard_config, relevance_threshold=0.9)
```

### 4. Easier Testing
```python
# Easy to create test configs
test_config = RetrievalConfig(
    collection=mock_collection,
    query_text="test"
)

# Easy to verify
assert test_config.n_results == 10  # Default
```

### 5. Future-Proof
Adding new parameters doesn't break existing code:
```python
@dataclass
class LLMConfig:
    client: Instructor
    model: str | None = None
    temperature: float = 0.1
    max_retries: int = 3
    timeout: int | None = None  # NEW! Add with default
    # All existing code continues to work
```

---

## Validation

All configs validate in `__post_init__`:

```python
config = RetrievalConfig(
    collection=collection,
    query_text="",  # ❌ Empty
    use_hyde=True   # ❌ Missing hyde_client
)
# Raises ValueError immediately with clear message
```

Validation errors happen at config creation time, not during function execution, making debugging easier.

---

## Next Steps

1. Use the new config-based API for all new code
2. Gradually migrate remaining old test patterns (30 tests marked as skipped)
3. Enjoy cleaner, more maintainable code!

For complete examples, see `AGENTS.md` and the test files:
- `tests/test_utils_config.py`
- `tests/test_retrieve_config.py`
- `tests/test_query_config.py`
