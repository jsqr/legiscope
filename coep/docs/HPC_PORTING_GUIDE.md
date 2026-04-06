# Legiscope HPC Porting Guide

This document provides everything an LLM or developer needs to deploy legiscope
on an institutional HPC cluster, run the full pipeline + benchmarking for 50
municipal jurisdictions, and manage input/output data.

---

## 1. System Overview

**Legiscope** is a legal epidemiology research tool developed by the Cerda Lab at
NYU Langone Health. It is a Python-based RAG (Retrieval-Augmented Generation)
pipeline for automated coding of municipal legal codes. It parses raw text from
city ordinances, segments them, generates embeddings, indexes them in a vector
database (ChromaDB), and uses LLM inference to answer structured research
questions (e.g., drug paraphernalia law questions). Answers are benchmarked
against human-labeled MonQcle ground truth using an LLM-as-a-judge strategy.

The target HPC environment is **NYU Langone BigPurple** (cerdalab allocation).

### Single-Job Architecture

Since benchmarking is a DVC stage that depends on the index stage, the entire
pipeline — data ingest through benchmarking — runs as a single `dvc repro`
command within one SLURM job:

```
validate → parse → segment → embed → index → benchmark
```

Each SLURM job starts a vLLM server, runs `dvc repro` (which executes all
stages end-to-end), and produces both the vector index and benchmark results.
The OpenRouter API is used for embeddings (embed stage); vLLM handles all LLM
calls (parse heading scanning, benchmark querying, and LLM-as-judge evaluation).

To **re-run benchmarking only** (e.g., with different retrieval settings or a
different model), submit a lighter job that runs `dvc repro benchmark`. DVC
will skip all upstream stages since their outputs already exist.

### What the system does end-to-end

```
DOCX files  ──►  TXT conversion  ──►  DVC Pipeline (parse → segment → embed → index → benchmark)
                                            │
                                            ▼
                                     ChromaDB vector index + benchmark results (CSV + metrics JSON)
```

### Key technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Package manager | `uv` (local), Conda (BigPurple) | Python dependency resolution |
| Python | 3.12.x (exact) | Runtime |
| Pipeline orchestration | DVC | Reproducible stages, parameter tracking, remote storage |
| Vector database | ChromaDB | Local persistent embedding index |
| LLM providers | OpenAI (vLLM via API), Ollama (local dev) | Parsing headings, HYDE, relevance filtering, query answering, evaluation |
| LLM inference (HPC) | vLLM | Self-hosted OpenAI-compatible API server on BigPurple |
| Embeddings | OpenRouter API `qwen/qwen3-embedding-8b` | Cloud embedding generation (embed stage only; not needed for benchmark-only re-runs) |
| Embeddings (local dev alt.) | Ollama `embeddinggemma` | Alternative local embedding provider |
| Data processing | Polars, Parquet | Tabular data (sections, segments, embeddings, results) |
| Structured outputs | Instructor + Pydantic | Typed LLM responses |
| DVC remote | GCS (`gs://coep-muni`) | Artifact storage |

---

## 2. Repository Structure (Key Files)

```
legiscope/
├── src/legiscope/               # Core Python package
│   ├── config.py                # Reads config.yaml (paths, infra settings)
│   ├── params.py                # Reads params.yaml (DVC-tracked hyperparameters)
│   ├── llm_config.py            # LLM client factory (Mistral/OpenAI/Ollama via instructor)
│   ├── models.py                # JurisdictionRef, CodeRef dataclasses + Parquet schemas
│   ├── embeddings.py            # Embedding generation + ChromaDB collection management
│   ├── retrieve.py              # Vector search with optional HYDE and relevance filtering
│   ├── query.py                 # RAG query engine (single + batch)
│   ├── segment.py               # Text segmentation
│   └── parse/                   # Raw text → structured Markdown (heading scanning, etc.)
│
├── coep/                        # Benchmark evaluation module
│   ├── src/eval.py              # LLM-as-a-judge Evaluator class
│   ├── src/query.py             # Drug paraphernalia query adjustments
│   ├── scripts/benchmark_pipeline.py  # Main benchmark workflow script
│   └── data/monqcle_data/       # MonQcle ground truth CSV
│
├── scripts/                     # Entry-point scripts
│   ├── init.py                  # Initialize jurisdiction registries + directories
│   ├── parse.py                 # DVC parse stage
│   ├── segment.py               # DVC segment stage
│   ├── embed.py                 # DVC embed stage
│   ├── index.py                 # DVC index stage
│   ├── run_queries.py           # Standalone batch query runner
│   ├── convert_docx.sh          # DOCX → TXT conversion (uses textutil or pandoc)
│   ├── dvc_repro.sh             # DVC pipeline wrapper
│   └── dvc_python.sh            # Shared Python runner for DVC stages
│
├── config.yaml                  # Infrastructure config (paths, ChromaDB, logging)
├── params.yaml                  # DVC parameters (jurisdiction, LLM, embeddings, retrieval, query)
├── dvc.yaml                     # DVC pipeline definition (5 stages + validate)
├── pyproject.toml               # Python project config (dependencies, build)
├── .dvc/config                  # DVC remote: gs://coep-muni
├── .env.example                 # Template for API keys
└── Makefile                     # Dev commands (env, test, lint, etc.)
```

---

## 3. Configuration System

### 3.1 `params.yaml` — Pipeline Parameters (DVC-tracked)

This file controls **all** pipeline behavior. Key sections:

```yaml
jurisdiction:
  state: CA                    # Two-letter state code
  locality: LosAngeles         # PascalCase city name (no spaces)
  code_slug: municipal-code    # URL-friendly code identifier
  code_name: "LA Municipal Code"

llm:
  default_provider: mistral    # "mistral" | "openai" | "ollama"
  providers:
    mistral:
      fast: mistral-small-2506       # Used for HYDE, relevance filtering
      powerful: mistral-large-2512   # Used for query answering, evaluation
    openai:
      fast: gpt-4.1-mini
      powerful: gpt-4.1
    ollama:
      fast: qwen3:8b
      powerful: qwen3:30b
  temperature: 0.0
  max_retries: 3
  timeout: 300

embeddings:
  default_provider: openrouter  # "openrouter" | "ollama"
  chroma_batch_size: 100
  providers:
    openrouter:
      model: qwen/qwen3-embedding-8b
      batch_size: 100
    ollama:
      model: embeddinggemma
      batch_size: 1

retrieval:
  n_results: 10
  hyde:
    enabled: false
  relevance_filter:
    enabled: false
    threshold: 0.5
  debug: false

query:
  validation:
    enabled: true

benchmark:
  series_title: "DPL_2025_Consolidated"
```

### 3.2 `config.yaml` — Infrastructure Settings (not DVC-tracked)

```yaml
paths:
  data_dir: "data"                    # Override via LEGISCOPE_DATA_DIR env var
  laws_dir: "laws"                    # Under data_dir
  chroma_db_dir: "chroma_db"          # Under data_dir
  queries_dir: "queries"
  output_dir: "output"
  default_queries_file: "drug_paraphernalia_queries_clean.csv"
  monqcle_report: "coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv"

database:
  chromadb:
    default_collection: "legal_code"
    model_suffix: true                # Collection name: legal_code_<embedding_model>
```

### 3.3 `.env` — Secrets

```bash
OPENAI_API_KEY=your-key-here            # For vLLM (any string) or OpenAI cloud
OPENROUTER_API_KEY=your-key-here        # For OpenRouter embeddings
# LEGISCOPE_DATA_DIR=/custom/data/path  # Optional override
```

### 3.4 `.dvc/config` — DVC Remote

Currently configured to use Google Cloud Storage:
```
[core]
    remote = gcs
['remote "gcs"']
    url = gs://coep-muni
```

---

## 4. Pipeline Stages (DVC)

The processing pipeline has 6 stages defined in `dvc.yaml`. Each stage is
parameterized by `params.yaml` jurisdiction settings and produces outputs under
`data/laws/{STATE}/{Locality}/{code-slug}/`.

### Stage Flow

```
validate → parse → segment → embed → index → benchmark
```

The `benchmark` stage depends on `index` (it reads from the ChromaDB index)
and on `segment` outputs (`sections.parquet`).

### Stage Details

| Stage | Script | Inputs | Outputs | LLM Calls? |
|-------|--------|--------|---------|-------------|
| **validate** | inline bash | `src/legiscope/__init__.py` | `tmp/validate_import.ok` | No |
| **parse** | `scripts/parse.py` | `raw/` directory (TXT files) | `code.md`, `headings.parquet` | Yes (heading scanning) |
| **segment** | `scripts/segment.py` | `code.md`, `headings.parquet` | `sections.parquet`, `segments.parquet`, `relations.parquet`, `external_references.parquet` | No |
| **embed** | `scripts/embed.py` | `sections.parquet`, `segments.parquet` | `embeddings.parquet` | No (embedding model only) |
| **index** | `scripts/index.py` | `embeddings.parquet` | ChromaDB collection (side effect) | No |
| **benchmark** | `coep/scripts/benchmark_pipeline.py` | `sections.parquet`, `embeddings.parquet`, queries CSV, MonQcle CSV | `benchmark_results.csv`, `benchmark_metrics.json` | Yes (query + evaluation) |

### Data Directory Layout

For a jurisdiction like CA-LosAngeles with code-slug `municipal-code`:

```
data/laws/CA/LosAngeles/municipal-code/
├── raw/                          # Input: original DOCX (and/or TXT) source files
│   └── los_angeles-ca-1.docx     #   Naming convention: {city}-{state}-{number}.docx
├── code.txt                      # Intermediate: DOCX→TXT conversion output
├── code.md                       # Parse output: structured Markdown
├── headings.parquet              # Parse output: heading hierarchy
├── sections.parquet              # Segment output: section-level chunks
├── segments.parquet              # Segment output: finer text segments
├── relations.parquet             # Segment output: intra-code references
├── external_references.parquet   # Segment output: external references
└── embeddings.parquet            # Embed output: embedding vectors
```

**DOCX file naming convention**: `{city}-{state}-{number}.docx` (e.g.,
`los_angeles-ca-1.docx`, `chicago-il-1.docx`). One DOCX per jurisdiction.

### Running the Pipeline

```bash
# 1. Set jurisdiction in params.yaml
# 2. Initialize (one-time per jurisdiction)
python scripts/init.py                     # or: uv run python scripts/init.py (local)

# 3. Place DOCX (or TXT) files in data/laws/{STATE}/{Locality}/{code-slug}/raw/
# 4. Run all stages
./scripts/dvc_repro.sh

# Or run a single stage
./scripts/dvc_repro.sh --stage parse
```

---

## 5. Benchmark Pipeline

The benchmark evaluates RAG-generated answers against MonQcle human-labeled
ground truth using an LLM-as-a-judge approach. It is the `benchmark` DVC stage
and runs automatically as part of `dvc repro`.

### Script: `coep/scripts/benchmark_pipeline.py`

### Workflow

1. **Load queries** from `data/queries/drug_paraphernalia_queries_clean.csv`
   (CSV with `question` and `variable_name` columns)
2. **Load MonQcle data** from `coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv`,
   filter to target jurisdiction
3. **Run RAG pipeline**: query ChromaDB → retrieve sections → LLM generates answers
4. **Join** generated answers with ground truth by `variable_name`
5. **Evaluate** using LLM-as-a-judge (powerful model scores 0-10)
6. **Output** CSV with scores, reasoning, accuracy labels

### Inputs

| Input | Path | Notes |
|-------|------|-------|
| Queries | `data/queries/drug_paraphernalia_queries_clean.csv` | Same for all jurisdictions |
| MonQcle ground truth | `coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv` | **One file with data for all cities**; filtered by jurisdiction ID at runtime |
| ChromaDB index | `data/chroma_db/` | Must be pre-built via DVC pipeline |
| Sections parquet | `data/laws/{STATE}/{Locality}/{code-slug}/sections.parquet` | Pre-built via DVC pipeline |

### Outputs

| Output | Path |
|--------|------|
| Benchmark results (DVC-tracked) | `data/output/{STATE}-{Locality}/benchmark_results.csv` |
| Benchmark results (timestamped copy) | `data/output/{STATE}-{Locality}/benchmark_results_{timestamp}.csv` |
| Benchmark metrics (DVC metrics) | `data/output/{STATE}-{Locality}/benchmark_metrics.json` |
| Debug artifacts (optional) | `data/output/{STATE}-{Locality}/debug/` |

### Running

The benchmark runs as part of the DVC pipeline:

```bash
# Run as part of full pipeline
dvc repro
# Or: ./scripts/dvc_repro.sh

# Run only the benchmark stage
dvc repro benchmark
# Or: ./scripts/dvc_repro.sh --stage benchmark

# Run standalone (outside DVC)
python coep/scripts/benchmark_pipeline.py --state CA --locality LosAngeles --code-slug municipal-code

# Test with limited queries
python coep/scripts/benchmark_pipeline.py --test-limit 5
```

---

## 6. DOCX-to-TXT Conversion

The source municipal codes arrive as Word documents. The conversion script
(`scripts/convert_docx.sh`) converts all `.docx` files in a raw directory to a
single concatenated `.txt` file.

### Converter priority

1. `textutil` (macOS native — fast, but **not available on Linux/HPC**)
2. `pandoc` (cross-platform fallback — install via package manager)

### On HPC (Linux)

You will need `pandoc` installed:
```bash
# Module load if available
module load pandoc

# Or install via conda/mamba
conda install -c conda-forge pandoc

# Or download binary
# https://pandoc.org/installing.html
```

### Usage

```bash
./scripts/convert_docx.sh data/laws/CA/LosAngeles/municipal-code/raw
```

This produces `data/laws/CA/LosAngeles/municipal-code/code.txt` (one level above
`raw/`, in the code directory itself).

**No file move is needed.** The parse stage (`convert_to_markdown()`) looks for
input text in this order:
1. `{code_dir}/code.txt` ← the `convert_docx.sh` output lands here
2. `{code_dir}/raw/code.txt`
3. First `*.txt` in `{code_dir}/raw/`

So the conversion output is found automatically at step 1.

---

## 7. Transferring the 50 Word Documents to the HPC

You have 50 `.docx` files (one per city) on your local machine that need to
reach the HPC filesystem before any pipeline work begins.

### 7.1 Recommended Directory Layout on HPC

The pipeline expects each DOCX to live inside its jurisdiction's `raw/`
directory within the existing `data/laws/` hierarchy:

```
data/laws/
├── CA/
│   └── LosAngeles/
│       └── municipal-code/
│           └── raw/
│               └── los_angeles-ca-1.docx
├── TX/
│   └── Houston/
│       └── municipal-code/
│           └── raw/
│               └── houston-tx-1.docx
├── IL/
│   └── Chicago/
│       └── municipal-code/
│           └── raw/
│               └── chicago-il-1.docx
...                           # 50 total
```

**DOCX naming convention**: `{city}-{state}-{number}.docx` (lowercase with
underscores/hyphens, e.g., `los_angeles-ca-1.docx`, `chicago-il-1.docx`).
The exact filename doesn't affect conversion — `convert_docx.sh` processes
every `*.docx` it finds in the `raw/` directory.

The simplest approach is to organize all 50 DOCX files into this structure
**on your local machine first**, then upload the entire `data/laws/` tree.
The `scripts/init.py` step will create the directory structure if it doesn't
already exist, so alternatively you can upload DOCX files to a flat staging
area and let the `run_jurisdiction.sh` wrapper copy them into place.

### 7.2 Transfer Methods

#### Option A: `scp` / `rsync` from your local machine (simplest)

```bash
# From your LOCAL machine, upload the pre-organized data/laws/ tree:
rsync -avz --progress data/laws/ \
    <netid>@<hpc-host>:/path/to/legiscope/data/laws/

# Or upload a flat staging folder if you haven't pre-organized:
rsync -avz --progress ~/legiscope-input/ \
    <netid>@<hpc-host>:~/legiscope-input/

# Or with scp:
scp -r data/laws/ <netid>@<hpc-host>:/path/to/legiscope/data/laws/
```

Replace `<netid>` with your HPC username and `<hpc-host>` with the login node
hostname (e.g., `greene.hpc.nyu.edu`).

If you need to go through a gateway/bastion host (common at NYU):
```bash
rsync -avz -e "ssh -J <netid>@gw.hpc.nyu.edu" \
    ~/legiscope-input/ \
    <netid>@<hpc-host>:~/legiscope-input/
```

#### Option B: `sftp` interactive session

```bash
sftp <netid>@<hpc-host>
sftp> mkdir legiscope-input
sftp> put -r ~/legiscope-input/*
sftp> quit
```

#### Option C: Globus (recommended for large transfers)

Many institutional HPCs, including NYU Greene, support
[Globus](https://www.globus.org/) for reliable, resumable, high-speed file
transfers. This is best if the 50 DOCX files are collectively large or you have
an unreliable connection.

1. Log into [app.globus.org](https://app.globus.org) with your institutional SSO
2. Set **source endpoint** to "Globus Connect Personal" on your laptop
   (install the Globus Connect Personal agent first)
3. Set **destination endpoint** to your HPC's Globus endpoint (e.g.,
   `nyu#greene` — ask your HPC admins for the exact name)
4. Navigate to the source directory containing your DOCX folders
5. Navigate to `~/legiscope-input/` on the destination
6. Select all 50 folders and click **Start**

Globus handles retries, checksums, and parallelism automatically.

#### Option D: Google Cloud Storage round-trip

If the DOCX files are already in a GCS bucket (or you want to stage them there):

```bash
# From local machine — upload to GCS
gsutil -m cp -r ~/legiscope-input/ gs://coep-muni/input-docx/

# On HPC — download from GCS
module load google-cloud-sdk
gsutil -m cp -r gs://coep-muni/input-docx/ ~/legiscope-input/
```

This requires GCS authentication on both ends but gives you a durable backup
of the original source documents.

#### Option E: Git LFS (not recommended)

Storing 50 DOCX files in the git repo via Git LFS is possible but inadvisable —
binary files bloat the repo and these are input data, not code artifacts.

### 7.3 Verifying the Transfer

After uploading, confirm all 50 jurisdictions are present:

```bash
# On HPC — if you uploaded directly into data/laws/:
find data/laws/ -name "*.docx" | wc -l     # Should be ≥ 50
find data/laws/ -name "*.docx" | head -10   # Spot-check paths

# Or if using a staging area:
find ~/legiscope-input/ -name "*.docx" | wc -l
```

### 7.4 Connecting Input Files to the Pipeline

There are two approaches:

**Approach A: Pre-place DOCX files directly** (recommended)

Organize DOCX files into the `data/laws/{STATE}/{Locality}/{code-slug}/raw/`
structure before uploading. The `run_jurisdiction.sh` wrapper then skips the
copy step since the file is already in place.

**Approach B: Use a flat staging directory**

Put all DOCX files in a staging directory (one subdir per jurisdiction), then
have the `jurisdictions.tsv` manifest (Section 9.1) `DOCX_DIR` column point at
each:

```
# jurisdictions.tsv (DOCX_DIR column points to staging location)
CA	LosAngeles	municipal-code	LA Municipal Code	/home/<netid>/legiscope-input/LosAngeles
IL	Chicago	municipal-code	Chicago Municipal Code	/home/<netid>/legiscope-input/Chicago
...
```

The `run_jurisdiction.sh` wrapper (Section 9.2) copies the DOCX files from
this staging directory into `data/laws/{STATE}/{Locality}/{code-slug}/raw/`,
converts them to TXT, and runs the DVC pipeline.

**If you pre-placed the files (Approach A)**, set `DOCX_DIR` to the `raw/`
directory itself:
```
CA	LosAngeles	municipal-code	LA Municipal Code	data/laws/CA/LosAngeles/municipal-code/raw
```

### 7.5 Alternative: Pre-convert to TXT Locally

If you want to avoid installing `pandoc` on the HPC, you can convert all DOCX
files on your local Mac (which has `textutil`) before uploading:

```bash
# On your LOCAL Mac
for dir in ~/legiscope-input/*/; do
    city=$(basename "$dir")
    for f in "$dir"*.docx; do
        [ -f "$f" ] && textutil -convert txt -output "${f%.docx}.txt" "$f"
    done
done

# Then upload only the .txt files
rsync -avz --include='*/' --include='*.txt' --exclude='*' \
    ~/legiscope-input/ <netid>@<hpc-host>:~/legiscope-input/
```

The `run_jurisdiction.sh` wrapper already handles both `.docx` and `.txt` inputs.

---

## 8. HPC Deployment Strategy

### 8.1 Environment Setup on HPC (Conda)

BigPurple login nodes have limited resources — do NOT run `pip install` or heavy
computation directly on login nodes (processes will be killed). Use SLURM batch
jobs for package installation.

> **Note:** Local development uses `uv` for dependency management (see
> `AGENTS.md`). On BigPurple, use a single **conda** environment with
> `pip install -e .` instead — `uv` is not needed on the HPC.

```bash
# On BigPurple login node (bigpurple-ln2 or bigpurple-ln3):
# Create conda environment (one-time)
conda create -p /gpfs/home/$USER/conda_envs/legiscope_env python=3.12 pip -y

# Clone repo to lab storage (backed up, primary working location)
cd /gpfs/data/cerdalab
git clone https://github.com/jsqr/legiscope.git
cd legiscope

# Submit a SLURM job for package installation (don't install on login node!)
# --- install_packages.sh ---
# #!/bin/bash
# #SBATCH --job-name=legiscope-install
# #SBATCH --partition=cpu_short
# #SBATCH --mem=32G
# #SBATCH --cpus-per-task=4
# #SBATCH --time=02:00:00
# #SBATCH --output=logs/install_%j.out
# set -e
# source ~/.bashrc
# conda activate /gpfs/home/$USER/conda_envs/legiscope_env
# cd /gpfs/data/cerdalab/legiscope
# # If using vLLM, install it FIRST (it pins its own torch version):
# pip install vllm
# # If the above fails to find a wheel, try: pip install vllm --no-build-isolation
# # Then install project dependencies:
# pip install -e .
# ---
# sbatch install_packages.sh

# Copy and configure .env
cp .env.example .env
# Edit .env with API keys or vLLM connection settings
```

**Important BigPurple notes:**
- `source ~/.bashrc` is required in SLURM scripts before `conda activate`
- BigPurple loads `anaconda3/gpu/2025.06` by default; this does not conflict
  with the project conda environment if you explicitly activate it
- vLLM pins its own torch version (e.g., `torch==2.9.0`); install vLLM first,
  then other dependencies, to avoid version conflicts
- The DVC pipeline wrapper scripts (`dvc_python.sh`, `dvc_repro.sh`) look for
  `.venv/bin/python` first and fall back to system `python`. With conda
  activated, the fallback to `python` is the correct behavior — no script
  changes are needed

### 8.2 Embedding Provider: OpenRouter API

Embeddings are generated via the **OpenRouter API** (OpenAI-compatible), which
runs in the cloud and requires no local GPU resources. This is the default
provider and works on both local dev and BigPurple. Set `params.yaml`:

```yaml
embeddings:
  default_provider: openrouter
```

Ensure `OPENROUTER_API_KEY` is set in `.env` (loaded by the SLURM script).
The OpenRouter API is only called during the **embed** stage — benchmark-only
re-runs do not require it.

### 8.3 LLM Provider: Self-Hosted vLLM vs. Cloud API

The project supports two approaches for LLM inference on HPC. **On BigPurple,
the vLLM approach is used** (for all DVC stages requiring LLM calls). The cloud API option
is documented for reference.

#### Cloud API (Mistral or OpenAI) — Alternative

Use cloud LLM APIs by setting API keys in `.env`. No GPU needed for LLM calls.
This is the approach used in local development.

```yaml
# params.yaml (local dev only — not used on BigPurple)
llm:
  default_provider: mistral  # or "openai"
```

#### Self-Hosted vLLM — BigPurple Approach (USED)

On BigPurple, the project self-hosts models via **vLLM**, which exposes an
OpenAI-compatible API server on `localhost:8000`. This eliminates cloud API
costs and allows using any HuggingFace model.

vLLM is started as a background process in the SLURM job, then the pipeline
connects to it via the `openai` provider with `OPENAI_BASE_URL` pointing at
the local server.

**How it works:**

1. SLURM job starts vLLM server in the background
2. Server loads model weights and exposes `http://localhost:8000/v1`
3. Environment variables tell the `openai` Python client to connect locally:
   ```bash
   export OPENAI_BASE_URL=http://localhost:8000/v1
   export OPENAI_API_KEY=<any-string>  # vLLM accepts any key
   ```
4. Set `params.yaml` to use the `openai` provider with model names matching
   vLLM's `--served-model-name`:
   ```yaml
   llm:
     default_provider: openai
     providers:
       openai:
         fast: "powerful-model"    # Must match --served-model-name
         powerful: "powerful-model"
   ```

**⚠️ Instructor Mode Compatibility:**

The openai provider's instructor mode has been changed from
`instructor.Mode.RESPONSES_TOOLS` to `instructor.Mode.JSON` in
`src/legiscope/llm_config.py` for vLLM compatibility. `RESPONSES_TOOLS`
uses OpenAI's Responses API, which vLLM does not implement. `JSON` mode
works by injecting a JSON schema into the system prompt and parsing the
response — universally supported by any OpenAI-compatible server.

```python
# In src/legiscope/llm_config.py (already changed):
mode_map = {
    "openai": instructor.Mode.JSON,          # Changed for vLLM compatibility
    "mistral": instructor.Mode.MISTRAL_TOOLS,
    "ollama": None,
}
```

**vLLM Server Launch (confirmed working on V100-16GB):**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --api-key "$API_KEY" \
    --served-model-name "powerful-model" \
    --download-dir /gpfs/scratch/$USER/hf_cache \
    --dtype float16 \
    --enforce-eager
```

Key flags:
- `--dtype float16`: Required on V100 (no bfloat16 support)
- `--enforce-eager`: Skip CUDA graph compilation, faster startup (~4 min vs ~8 min)
- `--served-model-name`: Name that appears in the API; must match `params.yaml`
- `--download-dir`: Point to scratch for large model weights
- `--tensor-parallel-size N`: Add for multi-GPU models (must match `--gres=gpu:N`)

Server takes approximately 240 seconds to be ready on V100.

**Verified capabilities (all tested and passing with vLLM):**
- Basic chat completions (non-streaming and streaming)
- Structured JSON output via `extra_body` `guided_json`
- Pydantic structured output via `response_format`
- Instructor library integration with `instructor.Mode.JSON`
- Multi-turn conversations

**LLM provider on BigPurple**: Use the `openai` provider pointed at the local
vLLM server. Embeddings use the OpenRouter API separately via
`embeddings.default_provider: openrouter`.

#### Complete `params.yaml` Diff for BigPurple (vLLM + OpenRouter Embeddings)

Apply these changes to `params.yaml` before your first run. Everything else
(retrieval, query, benchmark settings) can stay at defaults.

```yaml
# LLM: use OpenAI provider pointed at local vLLM server
llm:
  default_provider: openai
  providers:
    openai:
      fast: "powerful-model"            # must match vLLM --served-model-name
      powerful: "powerful-model"        # must match vLLM --served-model-name

# Embeddings: OpenRouter API (default, no change needed)
embeddings:
  default_provider: openrouter
```

The SLURM scripts (Section 9.5) set `OPENAI_BASE_URL` and `OPENAI_API_KEY`
as environment variables so the `openai` provider connects to the local vLLM
server. `OPENROUTER_API_KEY` is loaded from `.env` for the embed stage.

### 8.4 Data Storage Strategy

#### BigPurple Storage Tiers

BigPurple has three storage tiers with different characteristics:

| Tier | Path | Backed Up? | Notes |
|------|------|-----------|-------|
| **Home** | `/gpfs/home/$USER/` | Yes | Small quota — conda envs and `.bashrc` only |
| **Lab** | `/gpfs/data/cerdalab/legiscope/` | Yes | Primary working location — git repo, data, results |
| **Scratch** | `/gpfs/scratch/$USER/` | No | Fast SSD, periodically purged — model weights, temp files |

**Recommended layout on BigPurple:**

```
/gpfs/data/cerdalab/legiscope/          # Git repo (lab storage, backed up)
├── data/laws/{STATE}/{Locality}/...    # Pipeline inputs + outputs (actual repo structure)
├── data/chroma_db/                     # ChromaDB vector database
├── data/output/                        # Benchmark results
└── logs/                               # SLURM job logs

/gpfs/scratch/$USER/
├── hf_cache/                           # HuggingFace model weights (large, re-downloadable)
│   └── models--Qwen--Qwen2.5-3B-Instruct/
└── tmp/                                # Temp files for pip builds
```

If scratch is purged, re-download model weights via a SLURM batch job using
`huggingface_hub.snapshot_download()`.

#### Option A: Store everything on HPC lab storage (RECOMMENDED)

- All intermediate files (parquet, Markdown, ChromaDB) stay in the local
  `data/` directory on the HPC filesystem
- Final benchmark CSVs also written to `data/output/`
- DVC still tracks file hashes in `dvc.lock` (committed to git) for
  reproducibility — you can verify which data state produced which results
- **Pros**: Zero setup, no auth complexity, fastest I/O (local filesystem),
  lab storage is already backed up by BigPurple IT, no per-run cost
- **Cons**: Data only accessible from BigPurple; collaborators must log in

#### Option B: Use DVC + GCS for artifact management

- The repo includes a DVC remote configured at `gs://coep-muni` (in `.dvc/config`)
- Run `dvc push` after each jurisdiction to upload artifacts to GCS
- **Pros**: Artifacts accessible from any machine (`dvc pull` from laptop or
  other HPC), cloud-grade durability, enables collaboration without BigPurple
  access
- **Cons**: Requires GCS service account key on BigPurple, network I/O per
  `dvc push` (can be slow for large embeddings), GCS storage + egress costs,
  auth tokens can expire

Add GCS later when you need cross-machine access or collaboration — at that
point it's just `dvc push` after successful runs.

#### GCS Authentication on HPC (if using Option B)

```bash
# Option 1: Service account key
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Option 2: gcloud CLI
module load google-cloud-sdk  # or install
gcloud auth application-default login
```

### 8.5 Single-Jurisdiction vs. Batch Processing

**Recommendation: Run the same script 50 times (one SLURM job per jurisdiction).**

Reasons:
- The system is architecturally built for one jurisdiction at a time
  (`params.yaml` targets a single jurisdiction, DVC stages reference
  `${jurisdiction.state}/${jurisdiction.locality}/${jurisdiction.code_slug}`)
- ChromaDB collection naming includes the embedding model but not the
  jurisdiction — running multiple jurisdictions sequentially in one process
  risks index conflicts
- SLURM array jobs provide natural parallelism, fault isolation, and easier
  retry of failed jurisdictions
- Each jurisdiction's pipeline is independent (no shared state between cities)

**Alternative (sequential loop in one job)** is possible but less robust —
one failure stops all remaining jurisdictions.

---

## 9. SLURM Execution Plan

### 9.1 Jurisdiction Manifest

Create a TSV file listing all 50 jurisdictions. Use PascalCase locality names
(matching the `data/laws/` directory names) and the established DOCX naming
convention:

```
# jurisdictions.tsv
# STATE	LOCALITY	CODE_SLUG	CODE_NAME	DOCX_DIR
CA	LosAngeles	municipal-code	LA Municipal Code	data/laws/CA/LosAngeles/municipal-code/raw
TX	Houston	municipal-code	Houston Municipal Code	data/laws/TX/Houston/municipal-code/raw
IL	Chicago	municipal-code	Chicago Municipal Code	data/laws/IL/Chicago/municipal-code/raw
...
```

If you pre-placed the DOCX files into the `raw/` directories (Section 7.4,
Approach A), `DOCX_DIR` matches `raw/` directly and no copy is needed.

### 9.2 Per-Jurisdiction Wrapper Script

A single wrapper script handles the full pipeline (ingest + benchmark) for one
jurisdiction. To re-run only the benchmark stage with different settings, pass
`--benchmark-only`.

```bash
#!/bin/bash
# run_jurisdiction.sh — Run full DVC pipeline for a single jurisdiction
# Usage: ./run_jurisdiction.sh CA LosAngeles municipal-code "LA Municipal Code" /path/to/docx/dir
#        ./run_jurisdiction.sh --benchmark-only CA LosAngeles municipal-code "LA Municipal Code"

set -euo pipefail

BENCHMARK_ONLY=false
if [ "${1:-}" = "--benchmark-only" ]; then
    BENCHMARK_ONLY=true
    shift
fi

STATE="$1"
LOCALITY="$2"
CODE_SLUG="$3"
CODE_NAME="$4"
DOCX_DIR="${5:-}"   # Optional; not needed for --benchmark-only

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Activate conda environment (conda must be initialized via ~/.bashrc)
source ~/.bashrc
conda activate /gpfs/home/$USER/conda_envs/legiscope_env

# Step 1: Update params.yaml for this jurisdiction (sed preserves comments)
sed -i'' \
    -e "s/^  state: .*/  state: $STATE/" \
    -e "s/^  locality: .*/  locality: $LOCALITY/" \
    -e "s/^  code_slug: .*/  code_slug: $CODE_SLUG/" \
    -e "s/^  code_name: .*/  code_name: $CODE_NAME/" \
    params.yaml

if [ "$BENCHMARK_ONLY" = true ]; then
    # Re-run benchmark stage only (upstream stages are skipped by DVC)
    INDEX_DIR="data/chroma_db"
    if [ ! -d "$INDEX_DIR" ]; then
        echo "ERROR: ChromaDB index not found at $INDEX_DIR"
        echo "Run the full pipeline first to build the vector index."
        exit 1
    fi
    dvc repro benchmark
    echo "Benchmark re-run completed: ${STATE}-${LOCALITY}"
    exit 0
fi

# Step 2: Initialize jurisdiction (creates raw/ dir + registry entries)
python scripts/init.py

# Step 3: Ensure DOCX/TXT files are in place and convert
RAW_DIR="data/laws/${STATE}/${LOCALITY}/${CODE_SLUG}/raw"
mkdir -p "$RAW_DIR"

# Copy source files into raw/ if they aren't already there
if [ -n "$DOCX_DIR" ] && [ "$(realpath "$DOCX_DIR")" != "$(realpath "$RAW_DIR")" ]; then
    if ls "$DOCX_DIR"/*.docx 1>/dev/null 2>&1; then
        cp "$DOCX_DIR"/*.docx "$RAW_DIR/"
    elif ls "$DOCX_DIR"/*.txt 1>/dev/null 2>&1; then
        cp "$DOCX_DIR"/*.txt "$RAW_DIR/"
    fi
fi

# Convert DOCX → TXT if DOCX files present in raw/
if ls "$RAW_DIR"/*.docx 1>/dev/null 2>&1; then
    ./scripts/convert_docx.sh "$RAW_DIR"
fi

# Step 4: Run full DVC pipeline (parse → segment → embed → index → benchmark)
# Requires: vLLM running (for parse + benchmark stages) + OPENROUTER_API_KEY set (for embed stage)
./scripts/dvc_repro.sh

echo "Pipeline completed: ${STATE}-${LOCALITY} — vector index and benchmark results built"
```

### 9.3 SLURM Array Job Script (All Jurisdictions)

This array job runs the full pipeline for all 50 jurisdictions. Each
array task processes one jurisdiction end-to-end (ingest through benchmark).

```bash
#!/bin/bash
#SBATCH --job-name=legiscope-pipeline
#SBATCH --output=logs/pipeline_%A_%a.out
#SBATCH --error=logs/pipeline_%A_%a.err
#SBATCH --array=0-49                    # 50 jurisdictions (0-indexed)
#SBATCH --time=04:00:00                 # Wall time per jurisdiction (adjust based on code size)
#SBATCH --mem=16G                       # Memory per job
#SBATCH --cpus-per-task=4               # CPU cores
#SBATCH --partition=<your-partition>    # Your HPC partition name
#
# GPU (required if running vLLM locally for LLM stages — parse + benchmark):
# #SBATCH --gres=gpu:1
#
# Email notifications (optional):
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your-email@nyu.edu

set -euo pipefail

# ── Load modules ──────────────────────────────────────────────────
# module load pandoc       # If needed for DOCX conversion

# ── Project setup ─────────────────────────────────────────────────
PROJECT_DIR="/path/to/legiscope"   # CHANGE THIS to your HPC project path
cd "$PROJECT_DIR"

source ~/.bashrc
conda activate /gpfs/home/$USER/conda_envs/legiscope_env

# Load environment variables (API keys — OPENROUTER_API_KEY needed for embeddings)
set -a
source .env
set +a

# Create log directory
mkdir -p logs

# ── Read jurisdiction from manifest ───────────────────────────────
MANIFEST="jurisdictions.tsv"
# Skip header line, get line for this array task
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")

STATE=$(echo "$LINE" | cut -f1)
LOCALITY=$(echo "$LINE" | cut -f2)
CODE_SLUG=$(echo "$LINE" | cut -f3)
CODE_NAME=$(echo "$LINE" | cut -f4)
DOCX_DIR=$(echo "$LINE" | cut -f5)

echo "=== Pipeline: ${STATE}-${LOCALITY} (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Code: ${CODE_SLUG} (${CODE_NAME})"
echo "DOCX source: ${DOCX_DIR}"

# ── Create per-task working copy (avoids params.yaml race condition) ──
WORK_DIR="$TMPDIR/legiscope_${SLURM_ARRAY_TASK_ID}"
echo "Creating working copy in $WORK_DIR"
mkdir -p "$WORK_DIR"
cp -r "$PROJECT_DIR"/. "$WORK_DIR"/

# Resolve DOCX_DIR to absolute path BEFORE changing to working copy
if [ -n "$DOCX_DIR" ]; then
    DOCX_DIR="$(cd "$PROJECT_DIR" && realpath "$DOCX_DIR")"
fi
cd "$WORK_DIR"

# ── Run full pipeline (ingest + benchmark) ────────────────────────
./run_jurisdiction.sh "$STATE" "$LOCALITY" "$CODE_SLUG" "$CODE_NAME" "$DOCX_DIR"

# ── Copy results back to shared project directory ─────────────────
mkdir -p "$PROJECT_DIR/data/output"
cp -r data/output/* "$PROJECT_DIR/data/output/" 2>/dev/null || true
mkdir -p "$PROJECT_DIR/data/laws"
cp -r data/laws/* "$PROJECT_DIR/data/laws/" 2>/dev/null || true

echo "=== Pipeline Completed: ${STATE}-${LOCALITY} ==="
```

### 9.4 Single Test Job (CA-LosAngeles)

For initial testing, submit a single job (no array):

```bash
#!/bin/bash
#SBATCH --job-name=legiscope-test
#SBATCH --output=logs/test_pipeline.out
#SBATCH --error=logs/test_pipeline.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=<your-partition>

set -euo pipefail

PROJECT_DIR="/path/to/legiscope"
cd "$PROJECT_DIR"

source ~/.bashrc
conda activate /gpfs/home/$USER/conda_envs/legiscope_env

set -a
source .env
set +a

mkdir -p logs

# Full pipeline: ingest + benchmark for CA-LosAngeles (DOCX pre-placed in raw/)
./run_jurisdiction.sh CA LosAngeles municipal-code "LA Municipal Code" data/laws/CA/LosAngeles/municipal-code/raw
```

To re-run only the benchmark with different settings (after the full pipeline
has completed at least once):

```bash
# Edit params.yaml retrieval/query settings, then:
./run_jurisdiction.sh --benchmark-only CA LosAngeles municipal-code "LA Municipal Code"
```

### 9.5 BigPurple: SLURM Scripts

On BigPurple, each job starts a vLLM server and runs the full pipeline.
A GPU is required for the vLLM server. The OpenRouter API is used for embeddings
(embed stage).

#### Full Pipeline (ingest + benchmark)

This job starts a vLLM server for LLM calls (parse stage heading scanning,
benchmark querying and evaluation) and uses the OpenRouter API for embeddings
(embed stage).

```bash
#!/bin/bash
#SBATCH --job-name=legiscope-pipeline
#SBATCH --partition=gpu4_short          # Or gpu4_medium for larger codes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1                    # For vLLM
#SBATCH --output=/gpfs/data/cerdalab/legiscope/logs/pipeline_%j.out
#SBATCH --error=/gpfs/data/cerdalab/legiscope/logs/pipeline_%j.err

set -euo pipefail

source ~/.bashrc
conda activate /gpfs/home/$USER/conda_envs/legiscope_env

export HF_HOME=/gpfs/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/$USER/hf_cache

cd /gpfs/data/cerdalab/legiscope

# ── Load environment (OPENROUTER_API_KEY needed for embeddings) ──────
set -a
source .env
set +a

# ── Generate API key and start vLLM server ────────────────────────
API_KEY="legiscope-key-${SLURM_JOB_ID}"

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --api-key "$API_KEY" \
    --served-model-name "powerful-model" \
    --download-dir /gpfs/scratch/$USER/hf_cache \
    --dtype float16 \
    --enforce-eager &

VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

# ── Wait for server health ────────────────────────────────────────
echo "Waiting for vLLM server (PID $VLLM_PID)..."
TIMEOUT=600
ELAPSED=0
while ! curl -s http://localhost:8000/health >/dev/null 2>&1; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died"
        exit 1
    fi
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: vLLM server did not start within ${TIMEOUT}s"
        exit 1
    fi
    sleep 15
    ELAPSED=$((ELAPSED + 15))
    echo "  ... waiting (${ELAPSED}s / ${TIMEOUT}s)"
done
echo "vLLM server ready after ${ELAPSED}s"

# ── Configure LLM client connection (vLLM) ───────────────────────
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY="$API_KEY"

# Note: OPENROUTER_API_KEY is already loaded from .env above.
# vLLM handles LLM calls (parse + benchmark); OpenRouter API handles embeddings (embed stage).

# ── Run full DVC pipeline ─────────────────────────────────────────
echo "=== Full Pipeline: $(date) ==="
echo "Git: $(git --no-pager log --oneline -1)"

dvc repro

echo "=== Pipeline completed: $(date) ==="
# Server killed automatically by trap
```

#### Benchmark-Only Re-run (optional)

To re-run only the benchmark stage with different settings (e.g., different
model, retrieval params), submit this lighter job. DVC skips all upstream
stages since their outputs already exist.

```bash
#!/bin/bash
#SBATCH --job-name=legiscope-benchmark
#SBATCH --partition=gpu4_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/data/cerdalab/legiscope/logs/benchmark_%j.out
#SBATCH --error=/gpfs/data/cerdalab/legiscope/logs/benchmark_%j.err

set -euo pipefail

source ~/.bashrc
conda activate /gpfs/home/$USER/conda_envs/legiscope_env

export HF_HOME=/gpfs/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/$USER/hf_cache

cd /gpfs/data/cerdalab/legiscope

# ── Generate API key (unique per job) ─────────────────────────────
API_KEY="legiscope-key-${SLURM_JOB_ID}"

# ── Start vLLM server in background ──────────────────────────────
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --api-key "$API_KEY" \
    --served-model-name "powerful-model" \
    --download-dir /gpfs/scratch/$USER/hf_cache \
    --dtype float16 \
    --enforce-eager &

VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

# ── Wait for server health ────────────────────────────────────────
echo "Waiting for vLLM server (PID $VLLM_PID)..."
TIMEOUT=600
ELAPSED=0
while ! curl -s http://localhost:8000/health >/dev/null 2>&1; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died"
        exit 1
    fi
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: vLLM server did not start within ${TIMEOUT}s"
        exit 1
    fi
    sleep 15
    ELAPSED=$((ELAPSED + 15))
    echo "  ... waiting (${ELAPSED}s / ${TIMEOUT}s)"
done
echo "vLLM server ready after ${ELAPSED}s"

# ── Configure client connection ──────────────────────────────────
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY="$API_KEY"

# ── Run benchmark only (DVC stage) ───────────────────────────────
echo "=== Benchmark re-run: $(date) ==="
dvc repro benchmark

echo "=== Benchmark completed: $(date) ==="
# Server killed automatically by trap
```

#### Submitting Jobs

```bash
# Full pipeline (first run)
sbatch scripts/slurm_pipeline.sh

# Benchmark-only re-run (after full pipeline has completed)
# Edit params.yaml retrieval/query settings first, then:
sbatch scripts/slurm_benchmark.sh
```

For multi-GPU models (e.g., Qwen3.5-32B on 4x V100), change `--gres=gpu:4`
and add `--tensor-parallel-size 4` to the vLLM launch command.

---

## 10. Important Considerations

### 10.1 SLURM Array Job Concurrency & params.yaml

**Critical issue**: `params.yaml` is a single shared file. `run_jurisdiction.sh`
modifies it via `sed` to set the current jurisdiction. If multiple array tasks
run simultaneously, concurrent `sed` writes will race.

**Solution — per-task working copies** (used in Section 9.3):

Each array task copies the repo to a task-specific directory and works there.
Since DOCX files are stored in a separate staging directory (not inside the
repo), the copy is lightweight — just code, configs, and DVC metadata. Each
task then copies its single DOCX into the working copy's `raw/` directory,
runs the full pipeline, and copies results back.

```bash
WORK_DIR="$TMPDIR/legiscope_${SLURM_ARRAY_TASK_ID}"
cp -r "$PROJECT_DIR" "$WORK_DIR"
cd "$WORK_DIR"
# ... run_jurisdiction.sh copies one DOCX, runs init + dvc repro ...
# Copy results back (ChromaDB is consumed during the job and doesn't need to persist)
mkdir -p "$PROJECT_DIR/data/output"
cp -r data/output/* "$PROJECT_DIR/data/output/"
mkdir -p "$PROJECT_DIR/data/laws"
cp -r data/laws/* "$PROJECT_DIR/data/laws/"
```

This also solves ChromaDB concurrency (Section 10.2) — each task gets its own
isolated ChromaDB instance.

**Alternative — sequential execution**: Set `--array=0-49%1` to limit to 1
concurrent task. Safe but slow (50x wall time). No working copies needed.

### 10.2 ChromaDB Concurrency & Collection Design

ChromaDB uses SQLite under the hood. Multiple concurrent processes writing to
the same ChromaDB directory will cause corruption. The working-copy approach
(10.1 solution 1) solves this — each task gets its own ChromaDB instance.

**Collection naming**: Collections are named `legal_code_{provider}_{model}`
(e.g., `legal_code_openrouter_qwen_qwen3-embedding-8b`) with no jurisdiction
component. All jurisdictions share one collection, and retrieval filters by
`jurisdiction_id` in segment metadata. This is intentional — it simplifies
collection management and the working-copies approach means each SLURM task
builds its own isolated, single-jurisdiction ChromaDB anyway.

**Re-running a single jurisdiction**: The index stage is incremental — it skips
segments whose `segment_id` already exists in the collection. To fully re-index
one jurisdiction, delete its working copy's `data/chroma_db/` directory before
re-running.

### 10.3 Rate Limiting

If running multiple array jobs simultaneously, they will all
call the OpenRouter API for embeddings in parallel. This may hit rate limits.

- Use `--array=0-49%5` to limit to 5 concurrent jobs
- The code already has `max_retries: 3` configured
- Add exponential backoff if needed (the `instructor` library handles some retries)

Benchmark-only re-runs use only vLLM (local), so rate limiting is not a
concern for those jobs.

### 10.4 Ollama on HPC

Ollama is not used on BigPurple. LLM inference uses vLLM (self-hosted via the
`openai` provider), and embeddings use the OpenRouter API (cloud). The Ollama
provider is only for local development.

### 10.5 Network Access

The HPC compute nodes need outbound HTTPS access to:
- `openrouter.ai` (**required** — OpenRouter API for embeddings)
- `huggingface.co` (**required** — downloading model weights for vLLM)
- `storage.googleapis.com` (only if using GCS/DVC push)

Check with your HPC admins if compute nodes have internet access. If not, you
may need to run on login nodes or a special partition with network access.

### 10.6 V100 GPU Constraints (BigPurple)

BigPurple uses Tesla V100-SXM2 GPUs (compute capability 7.0). This imposes
specific constraints on vLLM and model selection:

| Constraint | Impact |
|-----------|---------|
| **No bfloat16** | Must use `--dtype float16` in vLLM |
| **No FP8 quantization** | FP8 requires Ada/Hopper (SM89/SM90); V100 is SM70 |
| **No FlashAttention-2** | Requires compute capability >= 8.0; vLLM auto-falls back to TRITON_ATTN (no user action needed) |
| **KV cache in FP16** | Regardless of model weight quantization |
| **Supported quantization** | FP16 (native, best quality), AWQ 4-bit, GPTQ 4-bit |

**Performance tip:** Use `--enforce-eager` to skip CUDA graph compilation,
reducing vLLM startup time from ~8 min to ~4 min on V100 (slight inference
speed tradeoff).

**VRAM verification:** BigPurple has both V100-SXM2-16GB and potentially
V100-SXM2-32GB variants. Run `nvidia-smi` in an interactive GPU session to
confirm actual VRAM per GPU before committing to a model strategy:

```bash
srun --partition=gpu4_dev --gres=gpu:1 --mem=48G --cpus-per-task=8 \
     --time=04:00:00 --pty bash
# Once on GPU node:
nvidia-smi
```

NVLink provides ~300 GB/s bidirectional bandwidth between GPUs on the same node,
enabling efficient tensor parallelism for multi-GPU model hosting.

---

## 11. Step-by-Step: Initial Test (CA-LosAngeles)

1. **Clone repo on HPC**:
   ```bash
   git clone https://github.com/jsqr/legiscope.git
   cd legiscope
   ```

2. **Set up environment**:
   ```bash
   # Create conda env (one-time)
   conda create -p /gpfs/home/$USER/conda_envs/legiscope_env python=3.12 pip -y

   # Activate and install (via SLURM job — see Section 8.1)
   source ~/.bashrc
   conda activate /gpfs/home/$USER/conda_envs/legiscope_env
   # pip install -e .          # Run via SLURM, not on login node!
   # pip install vllm   # add --no-build-isolation only if wheel install fails
   ```

3. **Configure**:
   ```bash
   cp .env.example .env
   # Edit .env: add OPENROUTER_API_KEY (for embeddings)
   # OPENAI_API_KEY is set dynamically by SLURM scripts (vLLM server key)

   # Edit params.yaml:
   #   llm.default_provider: "openai"            # vLLM via OpenAI-compatible API
   #   llm.providers.openai.fast: "powerful-model"
   #   llm.providers.openai.powerful: "powerful-model"
   #   embeddings.default_provider: "openrouter"  # OpenRouter API for embeddings
   #   jurisdiction: set to CA-LosAngeles
   ```

4. **Install pandoc** (for DOCX conversion on Linux):
   ```bash
   module load pandoc   # or: conda install -c conda-forge pandoc
   ```

5. **Upload your DOCX file to the HPC** (see Section 7 for full details):
   ```bash
   # From your LOCAL machine — upload the LA DOCX directly into place:
   scp /path/to/los_angeles-ca-1.docx \
       <netid>@<hpc-host>:/path/to/legiscope/data/laws/CA/LosAngeles/municipal-code/raw/
   ```

6. **Prepare input data**:
   ```bash
   # Ensure the directory exists (init.py also creates it, but just in case):
   mkdir -p data/laws/CA/LosAngeles/municipal-code/raw

   # Verify the DOCX is in place:
   ls data/laws/CA/LosAngeles/municipal-code/raw/
   # Should show: los_angeles-ca-1.docx (or similar)

   # Convert DOCX → TXT (output: data/laws/CA/LosAngeles/municipal-code/code.txt)
   ./scripts/convert_docx.sh data/laws/CA/LosAngeles/municipal-code/raw
   ```

7. **Prepare MonQcle and queries**:
   ```bash
   # The MonQcle CSV (ground truth for all 50 cities) goes here:
   # coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv

   # Query file:
   # data/queries/drug_paraphernalia_queries_clean.csv
   # (Both should be in the repo already or copied from your source)
   ```

8. **Run the full pipeline** (ingest + benchmark, one-time per jurisdiction):
   ```bash
   python scripts/init.py
   # On BigPurple, submit as SLURM job (starts vLLM + runs dvc repro):
   sbatch scripts/slurm_pipeline.sh
   # Or run interactively in a GPU dev session (see Section 15 for srun command)
   ```

9. **Re-run benchmark only** (optional, with different settings):
   ```bash
   # On BigPurple, submit benchmark-only SLURM job:
   sbatch scripts/slurm_benchmark.sh
   # Results in: data/output/CA-LosAngeles/benchmark_results.csv
   ```

10. **Verify results**:
    ```bash
    # Check output exists
    ls data/output/CA-LosAngeles/
    # Quick peek at scores
    python -c "
    import polars as pl
    import glob
    f = sorted(glob.glob('data/output/CA-LosAngeles/benchmark_results_*.csv'))[-1]
    df = pl.read_csv(f)
    print(f'Average score: {df[\"eval_score\"].mean():.2f}')
    print(f'Correct: {df.filter(pl.col(\"eval_label\")==\"Correct\").height}')
    print(f'Total: {df.height}')
    "
    ```

---

## 12. MonQcle Ground Truth (All 50 Cities)

The MonQcle Standard Report CSV
(`coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv`) contains
data for **487 jurisdictions** in wide format (one row per city per series).
The file has a BOM-encoded header; the key columns are:

| Column | Example |
|--------|---------|
| `name` | `"Los Angeles, Los Angeles County, California, United States"` |
| `series_title` | `DPL_2025_Consolidated` |
| `dp_law`, `dp_type`, ... | Variable columns (melted to long format) |

The benchmark pipeline automatically:

1. Maps the `jurisdiction_id` (e.g., `CA-LosAngeles`) to the full MonQcle
   locality name using `jurisdiction_id_to_monqcle_name()` in `coep/src/eval.py`
2. Filters the CSV to the matching row + series title
3. Melts to long format (one row per variable)
4. Joins with RAG query results by `variable_name`

**You only need one MonQcle CSV file for all 50 jurisdictions.** Each
benchmark job filters to its own jurisdiction at runtime.

### CRITICAL: Expanding the Jurisdiction Mapping

Currently, `jurisdiction_id_to_monqcle_name()` only has **one mapping**:

```python
mapping = {
    "CA-LosAngeles": "Los Angeles, Los Angeles County, California, United States",
    # Add more as jurisdictions expand
}
```

**Before running the 50-city batch, you must add all 50 jurisdictions to this
mapping.** The key is the internal `jurisdiction_id` (`{STATE}-{Locality}`,
e.g., `TX-Houston`) and the value is the full MonQcle `name` column value
(e.g., `"Houston, Harris County, Texas, United States"`).

You can generate this mapping by cross-referencing your `jurisdictions.tsv`
with the MonQcle CSV:

```python
import csv
with open('coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv',
          encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    names = sorted({row['name'] for row in reader})
    for name in names:
        print(name)
```

---

## 13. Post-Processing: Aggregating Results

After all 50 jobs complete, aggregate the per-jurisdiction benchmark CSVs:

```python
import polars as pl
from pathlib import Path

output_dir = Path("data/output")
all_results = []

for jurisdiction_dir in sorted(output_dir.iterdir()):
    if not jurisdiction_dir.is_dir():
        continue
    csvs = sorted(jurisdiction_dir.glob("benchmark_results_*.csv"))
    if csvs:
        df = pl.read_csv(str(csvs[-1]))  # Latest results
        all_results.append(df)

if all_results:
    combined = pl.concat(all_results)
    combined.write_csv("data/output/all_jurisdictions_benchmark.csv")

    # Summary statistics
    summary = combined.group_by("jurisdiction_id").agg([
        pl.col("eval_score").mean().alias("avg_score"),
        (pl.col("eval_label") == "Correct").sum().alias("correct"),
        pl.len().alias("total"),
    ])
    print(summary.sort("avg_score", descending=True))
```

---

## 14. Quick Reference: Key Commands

| Task | Command |
|------|---------|
| Set up environment | `conda activate legiscope_env` (HPC) or `make env` (local) |
| Convert DOCX | `./scripts/convert_docx.sh data/laws/STATE/Locality/slug/raw` |
| Initialize jurisdiction | `python scripts/init.py` |
| Run DVC pipeline | `./scripts/dvc_repro.sh` |
| Run benchmark (DVC stage) | `dvc repro benchmark` |
| Run benchmark (standalone) | `python coep/scripts/benchmark_pipeline.py` |
| Run benchmark (test) | `python coep/scripts/benchmark_pipeline.py --test-limit 5` |
| Run queries only | `python scripts/run_queries.py` |
| Push to GCS | `dvc push` |
| Pull from GCS | `dvc pull` |
| Run tests | `make test` |
| Check errors | `make lint` |
| **BigPurple: Full pipeline** | `sbatch scripts/slurm_pipeline.sh` |
| **BigPurple: Benchmark only** | `sbatch scripts/slurm_benchmark.sh` |

---

## 15. BigPurple Cluster Hardware Reference

### Node Types

| Type | Nodes | CPUs | RAM | GPUs | Notes |
|------|-------|------|-----|------|-------|
| Compute | cn-0001 to cn-0054 | 40 (2×20 cores) | 384 GB | 0 | General CPU work |
| Fat | fn-0001 to fn-0004 | 40 (2×20 cores) | 1536 GB | 0 | Memory-intensive tasks |
| GPU4 | gn-0001 to gn-0025 | 40 (2×20 cores) | 384 GB | 4× V100 | NVLink GPU-to-GPU |
| GPU8 | gpu-0001 to gpu-0007 | 40 (2×20 cores) | 768 GB | 8× V100 | NVLink GPU-to-GPU |

### Partitions

| Partition | Time Limit | Nodes | Use Case |
|-----------|-----------|-------|----------|
| **cpu_dev** | 4 hours | cn-0002 to cn-0005 | Development/testing |
| **cpu_short** | 12 hours | cn-0004 to cn-0049 | Short CPU jobs |
| **cpu_medium** | 5 days | cn-0004 to cn-0049 | DVC pipeline (if CPU-only) |
| **cpu_long** | 28 days | cn-0035 to cn-0052 | Long-running CPU work |
| **fn_short** | 12 hours | fn-0001 to fn-0003 | High-memory tasks |
| **fn_medium** | 5 days | fn-0001 to fn-0003 | High-memory tasks |
| **fn_long** | 28 days | fn-0002 to fn-0004 | High-memory tasks |
| **gpu4_dev** | 4 hours | gn-0002 to gn-0005 | Interactive GPU dev (porting, testing) |
| **gpu4_short** | 12 hours | gn-0004 to gn-0022 | Production query/benchmark runs |
| **gpu4_medium** | 5 days | gn-0004 to gn-0022 | Multi-jurisdiction batch runs |
| **gpu4_long** | 28 days | gn-0020 to gn-0025 | Extended production runs |
| **gpu8_dev** | 4 hours | gpu-0001 | 8-GPU development |
| **gpu8_short** | 12 hours | gpu-0001 to gpu-0005 | Large model inference |
| **gpu8_medium** | 5 days | gpu-0001 to gpu-0005 | Large model production |
| **gpu8_long** | 28 days | gpu-0004 to gpu-0007 | Extended large model runs |

### Resource Limits

- **400 cores** per user maximum across all running jobs
- **1.9 TB memory** per user maximum across all running jobs
- GPUs must be requested explicitly with `--gres=gpu:N`
- Default partition is "prod" which has **no resources** — always specify
  `--partition` explicitly

### Scheduling Guidance

- **Prefer gpu4_ partitions** (25 nodes) over gpu8_ (7 nodes) — faster scheduling
- Use `gpu4_dev` / `gpu8_dev` for interactive development sessions
- Use `gpu4_short` / `gpu4_medium` for production batch runs

---

## 16. Model Strategy (BigPurple vLLM)

### Single-Model Design

The codebase has two model slots in `params.yaml` — `fast` and `powerful` —
but for vLLM deployment, **both point to the same self-hosted model**. This
is the simplest approach: one vLLM server, one model, all LLM calls go to it.

Set both slots to the same `--served-model-name` used in the vLLM launch:

```yaml
# params.yaml
llm:
  default_provider: openai
  providers:
    openai:
      fast: "powerful-model"      # Must match vLLM --served-model-name
      powerful: "powerful-model"   # Same model for all calls
```

This works because:
- All LLM calls route through `Config.get_fast_client()` or
  `Config.get_powerful_client()` — no code bypasses this
- `instructor.from_provider("openai/powerful-model")` creates an OpenAI client
  that reads `OPENAI_BASE_URL` from the environment → connects to local vLLM
- Embeddings are completely independent (`embeddings.default_provider`) and
  don't touch the LLM provider
- The only code change needed is in `src/legiscope/llm_config.py`: the
  instructor mode for the `openai` provider must be `instructor.Mode.JSON`
  (not `RESPONSES_TOOLS`, which uses OpenAI's Responses API that vLLM
  doesn't support). This change has already been made.

Once model benchmarking is complete, a two-model approach can be re-introduced
by hosting two vLLM servers on different ports or by splitting the fast model
to a smaller locally-hosted model.

### Confirmed Working Configuration (V100-16GB, Single GPU)

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| VRAM usage | ~5.8 GB weights, ~7 GB free for KV cache |
| Max concurrent requests | ~49 at 4096 context length |
| Server startup | ~240 seconds |

This is the current model for initial pipeline porting and testing. Both
`fast` and `powerful` slots point to the same model as a temporary
simplification.

### Model Testing Plan

Once the pipeline is ported, evaluate models in order. Use the first model
that meets accuracy requirements — earlier options are more practical
operationally.

**Fast model candidate:**
- **Qwen3.5-9B** in FP16 on single V100-32GB (if 32 GB confirmed)
- ~18 GB weight VRAM
- Recommended `--max-model-len 32768`

**Powerful model candidates (in evaluation order):**

| Option | Model | GPUs | VRAM | Quantization | Notes |
|--------|-------|------|------|-------------|-------|
| 1 (preferred) | Qwen3.5-32B | 4× V100 | ~64 GB | FP16 (none) | Full precision, 128K native context, fast gpu4 scheduling |
| 2 | Qwen3-32B | 4× V100 | ~64 GB | FP16 | Thinking mode (`/think`), adds output token overhead |
| 3 | Llama 3.3 70B | 4× V100 | ~38 GB | AWQ 4-bit | Largest dense model on 4× V100; tests if 70B adds value |
| 4 (escalation) | Qwen3-235B-A22B | 8× V100 | ~118 GB | AWQ 4-bit | MoE (128 experts, 8 active, ~22B active). Only if Options 1–3 leave an accuracy gap |

**Decision rule:** If a model scores within 2–3% accuracy of the next-larger
option on the legal coding benchmark, stop and use the smaller model. Staying
on 4× V100 with FP16 provides faster scheduling and simpler operations.

### Context Length Guidance

The RAG pipeline retrieves relevant text chunks via ChromaDB, so individual
inference calls typically use 4K–16K tokens (system prompt + question +
retrieved chunks + structured output). Recommended production
`--max-model-len`: **16384**. Reserve 32K for thinking-mode passes or unusually
long legal sections.

### Downloading Models

Download model weights to scratch via a SLURM batch job (not on login nodes):

```python
# download_model.py
import os
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir=f"/gpfs/scratch/{os.environ['USER']}/hf_cache"
)
```

```bash
# Submit as SLURM job:
#SBATCH --partition=cpu_short --mem=32G --time=02:00:00
python download_model.py
```

---

## 17. Troubleshooting

### vLLM Issues

| Problem | Solution |
|---------|----------|
| **CUDA out of memory** | Reduce `--max-model-len`, lower `--gpu-memory-utilization`, increase `--tensor-parallel-size` (use more GPUs), or use a smaller/quantized model |
| **Connection refused (localhost:8000)** | Server takes ~240s to start on V100. Check if server process is alive. Check vLLM logs. The health check loop in SLURM templates handles this automatically |
| **vLLM tries to build from source** | Ensure torch and vLLM versions are compatible. Working: `torch==2.9.0+cu128` with `vllm==0.11.2`. Install vLLM first (`pip install vllm`), let it pin torch. If it still builds from source, try `pip install vllm --no-build-isolation` |
| **Quantized model not loading** | Verify model was downloaded in correct format (AWQ/GPTQ). Check model's `config.json` for quantization method |
| **instructor structured output fails** | Verify `instructor.Mode.JSON` is set for the openai provider in `llm_config.py` (changed from `RESPONSES_TOOLS`, which requires OpenAI's Responses API that vLLM doesn't support) |

### BigPurple Issues

| Problem | Solution |
|---------|----------|
| **pip install killed on login node** | Always run `pip install` via SLURM batch jobs with ≥32 GB memory |
| **conda stdlib errors** (e.g., "No module named urllib.parse") | Delete and recreate environment: `conda remove -p /path/to/env --all -y` then `conda create -p /path/to/env python=3.12 pip -y` |
| **Model weights missing from scratch** | Scratch is periodically purged. Re-download via SLURM job using `huggingface_hub.snapshot_download()` |
| **Job pending too long** | Check `squeue -u $USER -l` REASON column. Try a different partition. Use `scontrol update JobId=JOBID Partition=NEW_PARTITION` |
| **Default partition error** | Never submit without `--partition`. The default "prod" has no resources |
| **GPU VRAM uncertainty** | Run `nvidia-smi` in interactive GPU session to verify GPU model and VRAM |

---

## 18. BigPurple Commands Reference

### SLURM

```bash
sbatch script.sh                           # Submit batch job
squeue -u $USER                            # Check job status
scancel <job_id>                           # Cancel a job
sinfo -o "%P %G %l %a"                    # Show partitions with GPU info
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,MaxRSS  # Job resource usage

# Interactive sessions
srun --partition=gpu4_dev --gres=gpu:1 --mem=48G --cpus-per-task=8 \
     --time=04:00:00 --pty bash            # Interactive GPU session
srun --partition=cpu_short --mem=16G --cpus-per-task=4 \
     --time=02:00:00 --pty bash            # Interactive CPU session

# Log monitoring
tail -f /gpfs/data/cerdalab/legiscope/logs/<logfile>.out   # Watch job output
cat $(ls -t /gpfs/data/cerdalab/legiscope/logs/*.out | head -1)  # Read most recent log
```

### Model Management

```bash
du -sh /gpfs/scratch/$USER/hf_cache/models--*            # Check downloaded models
ls /gpfs/scratch/$USER/hf_cache/models--*/snapshots/     # Verify model weights
du -sh /gpfs/scratch/$USER/                              # Check scratch usage
nvidia-smi                                               # GPU info (GPU nodes only)
```

### Daily Workflow

```bash
# On local machine:
git commit -am "message" && git push

# On BigPurple:
cd /gpfs/data/cerdalab/legiscope
git pull

# Full pipeline (ingest + benchmark):
sbatch scripts/slurm_pipeline.sh

# Benchmark-only re-run (with different settings):
sbatch scripts/slurm_benchmark.sh
```
