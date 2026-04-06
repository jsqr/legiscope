# Manual Validation Checklist

These steps require real LLM/embedding service calls and cannot be automated in unit tests.
Run them against a local or staging environment after all unit tests pass.

## MV-0: Set-up
1. Create the jurisdiction data folders:

```bash
# Edit params.yaml to set jurisdiction.state=IL, locality=WindyCity,
# code_slug=municipal-code, code_name="IL-WindyCity Municipal Code", then:
uv run python scripts/init.py
```

2. Paste the Chicago municipal code .docx file into the data/laws/IL/WindyCity/municipal-code/raw folder

3. Convert .docx to .txt file:

```bash
./scripts/convert_docx.sh "data/laws/IL/WindyCity/municipal-code/raw"
```


## MV-1: Full DVC Pipeline Run

Run the complete pipeline on the jurisdiction configured in `params.yaml`:

```bash
# Edit params.yaml to set jurisdiction.state=IL, locality=WindyCity,
# code_slug=municipal-code, code_name="IL-WindyCity Municipal Code", then:
./scripts/dvc_repro.sh --force
```

**Verify:**
- [ ] All 5 stages complete without error (parse, segment, embed, index, benchmark)
- [ ] `data/laws/IL/WindyCity/municipal-code/code.md` is generated
- [ ] `sections.parquet`, `segments.parquet`, `embeddings.parquet` exist
- [ ] Log output shows segment/embedding counts

## MV-2: Incremental Indexing (Fresh Build)

Delete ChromaDB and rebuild from embeddings:

```bash
rm -rf data/chroma_db
uv run python scripts/index.py --state IL --locality WindyCity --code-slug municipal-code
```

**Verify:**
- [ ] Index created successfully
- [ ] Segment count in ChromaDB matches rows in `embeddings.parquet`
- [ ] Log shows "Adding N new segments"

## MV-3: Re-run Same Jurisdiction (No Duplicates)

Run index again without deleting ChromaDB:

```bash
uv run python scripts/index.py --state IL --locality WindyCity --code-slug municipal-code
```

**Verify:**
- [ ] No new segments added
- [ ] Log says "All N segments from IL:WindyCity:municipal-code already indexed"
- [ ] ChromaDB count unchanged

## MV-4: Per-Code Parameter Override

Create a per-code params.yaml with a smaller token limit:

```bash
cat > data/laws/IL/WindyCity/municipal-code/params.yaml << 'EOF'
segmentation:
  token_limit: 128
EOF
```

Run segmentation:

```bash
uv run python scripts/segment.py --state IL --locality WindyCity --code-slug municipal-code
```

**Verify:**
- [ ] More segments produced compared to default `token_limit: 1024`
- [ ] Log confirms the smaller token_limit was used
- [ ] Clean up: `rm data/laws/IL/WindyCity/municipal-code/params.yaml`

## MV-5: DVC Experiment with `-S` Flag

Override a parameter via DVC experiment:

```bash
dvc exp run -S segmentation.token_limit=128
```

**Verify:**
- [ ] Experiment runs successfully
- [ ] `dvc exp show` lists the experiment with `segmentation.token_limit=128`
- [ ] Different segment count compared to default

## MV-6: Error Handling — Missing Raw Files

Temporarily set a throwaway jurisdiction in `params.yaml`, initialize it,
then attempt to parse without placing raw files:

```bash
# Edit params.yaml to set jurisdiction.state=TEST, locality=TestCity,
# code_slug=test-code, code_name="Test Code", then:
uv run python scripts/init.py

./scripts/dvc_repro.sh --stage parse --force
```

**Verify:**
- [ ] Parse step fails with a clear error message about missing raw files
- [ ] Error is actionable (tells user what to do)
- [ ] Clean up: `rm -rf data/laws/TEST`

## MV-7: Cross-Jurisdiction Query

If multiple jurisdictions are indexed, verify cross-jurisdiction queries:

```python
import chromadb

client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("legal_code_embeddinggemma")

# Query single jurisdiction
results = collection.query(
    query_texts=["lead paint regulations"],
    n_results=5,
    where={"jurisdiction_id": "IL-WindyCity"},
)
print(f"Single jurisdiction: {len(results['ids'][0])} results")

# Query multiple jurisdictions (if available)
results = collection.query(
    query_texts=["lead paint regulations"],
    n_results=5,
    where={
        "$or": [
            {"jurisdiction_id": "IL-WindyCity"},
            {"jurisdiction_id": "IL"},
        ]
    },
)
print(f"Cross-jurisdiction: {len(results['ids'][0])} results")
```

**Verify:**
- [ ] Single-jurisdiction query returns relevant results
- [ ] Cross-jurisdiction query includes results from both jurisdictions
- [ ] Results have correct metadata (jurisdiction_id, code_id)
