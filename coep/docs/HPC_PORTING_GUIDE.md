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

The entire pipeline — data ingest through benchmarking — runs as a single
`dvc exp run` command within one SLURM job:

```
validate → parse → segment → embed → index → benchmark
```

Each SLURM job starts a vLLM server, runs `dvc exp run` (which executes all
stages end-to-end and records the run as a DVC experiment), and produces both
the vector index and benchmark results. DVC experiments capture the
`params.yaml`, `dvc.lock`, and metrics for each run as lightweight Git
references under `refs/exps/` — no manual branches needed.

The OpenRouter API is used for embeddings (embed stage); vLLM handles all LLM
calls (parse heading scanning, benchmark querying, and LLM-as-judge evaluation).

To **re-run benchmarking only** (e.g., with different retrieval settings or a
different model), first rebuild the shared ChromaDB index with
`bash coep/scripts/HPC_scripts/rebuild_index.sh --clean`, then submit a lighter
job that runs `dvc exp run -f benchmark`.

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
| Experiment tracking | DVC experiments (`dvc exp`) | Per-jurisdiction params + metrics tracking via Git refs |

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
│   ├── retrieval_guidance.py    # Project-agnostic retrieval/relevance/completion guidance hooks
│   ├── query.py                 # RAG query engine (single + batch)
│   ├── segment.py               # Text segmentation
│   └── parse/                   # Raw text → structured Markdown (heading scanning, etc.)
│
├── coep/                        # Benchmark evaluation module
│   ├── src/eval.py              # LLM-as-a-judge Evaluator class
│   ├── src/query.py             # Drug paraphernalia query adjustments
│   ├── src/retrieval_guidance.py # COEP variable-family guidance provider
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
  state: PA                    # Two-letter state code
  locality: Philadelphia         # PascalCase city name (no spaces)
  code_slug: municipal-code    # URL-friendly code identifier
  code_name: "Philadelphia Municipal Code"

llm:
  default_provider: openai    # "mistral" | "openai" | "ollama"
  providers:
    mistral:
      fast: mistral-small-2506       # Used for HYDE, relevance filtering
      powerful: mistral-large-2512   # Used for query answering, evaluation
    openai:
      fast: Qwen/Qwen3.5-27B
      powerful: Qwen/Qwen3.5-27B
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
  n_results: 20
  hyde:
    enabled: false
  relevance_filter:
    enabled: true
    threshold: 0.7
  debug: true

segmentation:
  embedding_model_token_limit: 1024
  llm_context_limit: 32768

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
  default_queries_file: "DPL_queries_with_context.csv"
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
MISTRAL_API_KEY=your-key-here           # For Mistral LLMs or embedding models
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
and on `segment` outputs. Query-time retrieval still receives the canonical
`sections.parquet` path, but it automatically prefers sibling `chunks.parquet`
when chunk metadata is present in indexed segment rows.

### Stage Details

| Stage | Script | Inputs | Outputs | LLM Calls? |
|-------|--------|--------|---------|-------------|
| **validate** | inline bash | `src/legiscope/__init__.py` | `tmp/validate_import.ok` | No |
| **parse** | `scripts/parse.py` | `raw/` directory (TXT files) | `code.md`, `headings.parquet`, `regions.parquet` | Yes (heading scanning, regex refinement) |
| **segment** | `scripts/segment.py` | `code.md`, `headings.parquet`, `regions.parquet` (preferred) | `sections.parquet`, `chunks.parquet`, `segments.parquet`, `relations.parquet`, `external_references.parquet` | No |
| **embed** | `scripts/embed.py` | `sections.parquet`, `segments.parquet` | `embeddings.parquet` | No (embedding model only) |
| **index** | `scripts/index.py` | `embeddings.parquet` | ChromaDB collection (side effect) | No |
| **benchmark** | `coep/scripts/benchmark_pipeline.py` | `sections.parquet`, `embeddings.parquet`, queries CSV, MonQcle CSV | `benchmark_results.csv`, `benchmark_metrics.json` | Yes (query + evaluation) |

Parse-time post-processing is intentionally conservative. After the heading scan
returns candidate levels, `src/legiscope/parse/scan.py` can reorder explicit
levels based on outline coverage, regenerate heading regexes from the example
heading text, and then reset `markdown_prefix` from the normalized level order.
This prevents stale model-supplied prefixes from surviving into `code.md`.
The parse stage also persists `code_start` source/output coordinates in
frontmatter and emits `regions.parquet` so downstream section/chunk logic can
separate TOC, boilerplate, legal intro, annotation, and substantive body text.
During segmentation, `segment_legal_code()` prefers `regions.parquet` to keep
only canonical regions in `sections.parquet`; if regions are missing, it falls
back to `code_start.output_line` from frontmatter. The same stage also derives
`chunks.parquet` from the canonical section tree plus chunkable non-canonical
regions, sizing chunks from `segmentation.llm_context_limit` so completion-time
retrieval can fit several chunks in one LLM request.
If validation later shows repeated ambiguous region cases, a narrow LLM
verification step may be considered, but the current classifier stays
rule-based for determinism and lower parse cost.

### Data Directory Layout

For a jurisdiction like PA-Philadelphia with code-slug `municipal-code`:

```
data/laws/PA/Philadelphia/municipal-code/
├── raw/                          # Input: original DOCX (and/or TXT) source files
│   └── philadelphia-pa-1.docx     #   Naming convention: {city}-{state}-{number}.docx
├── code.txt                      # Intermediate: DOCX→TXT conversion output
├── code.md                       # Parse output: structured Markdown + code_start frontmatter
├── headings.parquet              # Parse output: heading hierarchy
├── regions.parquet               # Parse output: deterministic region roles
├── sections.parquet              # Segment output: canonical section hierarchy
├── chunks.parquet                # Segment output: retrieval-oriented chunks
├── segments.parquet              # Segment output: embedding/search segments derived from chunks
├── relations.parquet             # Segment output: intra-code references
├── external_references.parquet   # Segment output: external references
└── embeddings.parquet            # Embed output: embedding vectors
```

**DOCX file naming convention**: `{city}-{state}-{number}.docx` (e.g.,
`philadelphia-pa-1.docx`, `chicago-il-1.docx`). One DOCX per jurisdiction.

### Running the Pipeline

```bash
# 1. Set jurisdiction in params.yaml (or pass via -S flags)
# 2. Initialize (one-time per jurisdiction)
python scripts/init.py                     # or: uv run python scripts/init.py (local)

# 3. Place DOCX (or TXT) files in data/laws/{STATE}/{Locality}/{code-slug}/raw/
# 4. Run all stages as a DVC experiment
dvc exp run

# Or with jurisdiction overrides (no need to edit params.yaml):
dvc exp run \
    -S jurisdiction.state=PA \
    -S jurisdiction.locality=Philadelphia \
    -S jurisdiction.code_slug=municipal-code

# Run a single stage
./scripts/dvc_repro.sh --stage parse
```

---

## 5. Benchmark Pipeline

The benchmark evaluates RAG-generated answers against MonQcle human-labeled
ground truth using an LLM-as-a-judge approach. It is the `benchmark` DVC stage
and runs automatically as part of `dvc exp run`.

### Script: `coep/scripts/benchmark_pipeline.py`

### Workflow

1. **Load queries** from `data/queries/DPL_queries_with_context.csv`
  (structured CSV with `variable_name`, `query_text`, `coding_instructions`,
  `response_options`, and optional context fields such as `prepend_text`)
2. **Load MonQcle data** from `coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv`,
   filter to target jurisdiction
3. **Construct stage-specific prompts**: `coep/src/query.py` builds the
  completion-oriented `question`, while `coep/src/retrieval_guidance.py`
  derives retrieval, relevance, and completion guidance from
  `variable_name` plus query metadata
4. **Run RAG pipeline**: query ChromaDB → retrieve sections → LLM generates answers
5. **Join** generated answers with ground truth by `variable_name`
6. **Evaluate** using LLM-as-a-judge (powerful model scores 0-10)
7. **Output** CSV with scores, reasoning, accuracy labels

### Inputs

| Input | Path | Notes |
|-------|------|-------|
| Queries | `data/queries/DPL_queries_with_context.csv` | Same for all jurisdictions |
| MonQcle ground truth | `coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv` | **One file with data for all cities**; filtered by jurisdiction ID at runtime |
| ChromaDB index | `data/chroma_db/` | Must be pre-built via DVC pipeline |
| Sections parquet | `data/laws/{STATE}/{Locality}/{code-slug}/sections.parquet` | Pre-built via DVC pipeline |

### Outputs

| Output | Path |
|--------|------|
| Benchmark results (DVC-tracked) | `data/output/{STATE}-{Locality}/benchmark_results.csv` |
| Benchmark results (timestamped copy) | `data/output/{STATE}-{Locality}/benchmark_results_{timestamp}.csv` |
| Benchmark metrics (DVC metrics) | `data/output/{STATE}-{Locality}/benchmark_metrics.json` |
| Debug artifacts (optional) | `data/output/{STATE}-{Locality}/debug/` containing `retrieval_stage_<timestamp>.csv`, `relevance_stage_<timestamp>.csv`, and `query_stage_<timestamp>.csv` |

### Benchmark Query Construction

The benchmark no longer relies on one giant prompt string for every stage.
Instead, it uses three related representations of each query:

- a retrieval-time query optimized for semantic search,
- a relevance-assessment prompt enriched with variable-family instructions and
  anchor terms, and
- a completion-time query that preserves the coding instructions needed for the
  final answer.

This is implemented through the core `RetrievalGuidanceProvider` hook in
`src/legiscope/retrieval_guidance.py` and the COEP provider in
`coep/src/retrieval_guidance.py`.

`prepend_text` remains useful, but only as metadata. It is preserved as shared
context for guidance generation rather than being blindly prepended to every
stage's prompt text.

### Benchmark Debug Artifacts

When `retrieval.debug` is enabled, benchmark runs emit exactly three CSV debug
artifacts with one row per query:

- `retrieval_stage_<timestamp>.csv` records the retrieval query, guidance
  fields, counts, and compact summaries of retrieved sections and segments.
- `relevance_stage_<timestamp>.csv` records the relevance prompt setup and the
  per-section relevance assessments used to keep or discard sections.
- `query_stage_<timestamp>.csv` records the completion query, final prompts,
  model response fields, validation scores, and benchmark ground truth columns
  once the pipeline enriches the file after joining against MonQcle data.

### Running

The benchmark runs as part of the DVC pipeline:

```bash
# Run as part of full pipeline (as DVC experiment)
dvc exp run

# Run only the benchmark stage
dvc exp run -f benchmark

# Run standalone (outside DVC)
python coep/scripts/benchmark_pipeline.py --state PA --locality Philadelphia --code-slug municipal-code

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
./scripts/convert_docx.sh data/laws/PA/Philadelphia/municipal-code/raw
```

This produces `data/laws/PA/Philadelphia/municipal-code/code.txt` (one level above
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

### 7.1 Recommended Layout: Flat Staging Directory

Upload all 50 DOCX files into a single flat directory on the HPC, using the
naming convention from Section 9.2:

```
/gpfs/data/cerdalab/LegalAI/docx_sources/
  PA_Philadelphia.docx
  IL_Chicago.docx
  TX_Houston.docx
  ...                           # 50 total
```

**Filename format**: `STATE_Locality.docx` — see Section 9.2 for details.

The dispatcher script (`slurm_dispatch.sh`) will parse each filename to extract
jurisdiction metadata, and each SLURM job (`slurm_jurisdiction.sh`) will
automatically copy its DOCX file into the correct
`data/laws/{STATE}/{Locality}/{code-slug}/raw/` directory, convert it, and run
the pipeline. No pre-organization into nested directories is needed.

### 7.2 Transfer Methods

#### Option A: `scp` / `rsync` from your local machine (simplest)

```bash
# From your LOCAL machine, upload the flat DOCX staging folder:
rsync -avz --progress ~/legiscope-docx/ \
    <netid>@bigpurple.nyumc.org:/gpfs/data/cerdalab/LegalAI/docx_sources/

# Or with scp:
scp ~/legiscope-docx/*.docx <netid>@bigpurple.nyumc.org:/gpfs/data/cerdalab/LegalAI/docx_sources/
```

Replace `<netid>` with your HPC username.

If you need to go through a gateway/bastion host (common at NYU):
```bash
rsync -avz -e "ssh -J <netid>@gw.hpc.nyu.edu" \
    ~/legiscope-docx/ \
    <netid>@bigpurple.nyumc.org:/gpfs/data/cerdalab/LegalAI/docx_sources/
```

#### Option B: `sftp` interactive session

```bash
sftp <netid>@bigpurple.nyumc.org
sftp> mkdir /gpfs/data/cerdalab/LegalAI/docx_sources
sftp> put ~/legiscope-docx/*.docx /gpfs/data/cerdalab/LegalAI/docx_sources/
sftp> quit
```

#### Option C: Globus (recommended for large transfers)

Many institutional HPCs, including NYU BigPurple, support
[Globus](https://www.globus.org/) for reliable, resumable, high-speed file
transfers. This is best if the 50 DOCX files are collectively large or you have
an unreliable connection.

1. Log into [app.globus.org](https://app.globus.org) with your institutional SSO
2. Set **source endpoint** to "Globus Connect Personal" on your laptop
   (install the Globus Connect Personal agent first)
3. Set **destination endpoint** to your HPC's Globus endpoint (e.g.,
   `nyu#bigpurple` — ask your HPC admins for the exact name)
4. Navigate to the source directory containing your DOCX files
5. Navigate to `/gpfs/data/cerdalab/LegalAI/docx_sources/` on the destination
6. Select all 50 files and click **Start**

Globus handles retries, checksums, and parallelism automatically.

#### Option D: Google Cloud Storage round-trip

If the DOCX files are already in a GCS bucket (or you want to stage them there):

```bash
# From local machine — upload to GCS
gsutil -m cp ~/legiscope-docx/*.docx gs://coep-muni/input-docx/

# On HPC — download from GCS
module load google-cloud-sdk
mkdir -p /gpfs/data/cerdalab/LegalAI/docx_sources
gsutil -m cp gs://coep-muni/input-docx/*.docx /gpfs/data/cerdalab/LegalAI/docx_sources/
```

This requires GCS authentication on both ends but gives you a durable backup
of the original source documents.

#### Option E: Git LFS (not recommended)

Storing 50 DOCX files in the git repo via Git LFS is possible but inadvisable —
binary files bloat the repo and these are input data, not code artifacts.

### 7.3 Verifying the Transfer

After uploading, confirm all 50 DOCX files are present:

```bash
# On HPC:
ls /gpfs/data/cerdalab/LegalAI/docx_sources/*.docx | wc -l   # Should be 50
ls /gpfs/data/cerdalab/LegalAI/docx_sources/                 # Spot-check filenames
```

### 7.4 How Input Files Reach the Pipeline

The `slurm_dispatch.sh` script (Section 9.4) scans the staging directory,
parses each `STATE_Locality.docx` filename, and submits a SLURM job per file.
Each `slurm_jurisdiction.sh` job:

1. Runs `init.py` to create the `data/laws/{STATE}/{Locality}/{code-slug}/raw/`
   directory and register the jurisdiction
2. Copies the DOCX file into the `raw/` directory
3. Runs `convert_docx.sh` to convert DOCX → TXT
4. Runs the full DVC pipeline

No manual file organization is needed — just name the files correctly and run
the dispatcher.

### 7.5 Alternative: Pre-convert to TXT Locally

If you want to avoid installing `pandoc` on the HPC, you can convert all DOCX
files on your local Mac (which has `textutil`) before uploading:

```bash
# On your LOCAL Mac — convert all DOCX to TXT in place
cd ~/legiscope-docx
for f in *.docx; do
    [ -f "$f" ] && textutil -convert txt "$f"
done

# Upload the TXT files (same naming convention: STATE_Locality.txt)
rsync -avz --include='*.txt' --exclude='*' \
    ~/legiscope-docx/ <netid>@bigpurple.nyumc.org:/gpfs/data/cerdalab/LegalAI/docx_sources/
```

The `slurm_jurisdiction.sh` job already handles both `.docx` and `.txt` inputs.

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
# Login to BigPurple:
ssh $USER@bigpurple.nyumc.org

# Conda should already be available via ~/.bashrc after `conda init`
source ~/.bashrc

# Create conda environment (one-time):
conda create -p ~/conda_envs/legiscope_env python=3.12 pip -y

# Clone repo to lab storage (backed up, primary working location)
cd /gpfs/data/cerdalab/LegalAI
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
# conda activate ~/conda_envs/legiscope_env
# cd /gpfs/data/cerdalab/LegalAI/legiscope
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
- Do not add an Anaconda module-load step to these workflows; the provided
  SLURM scripts intentionally rely on `source ~/.bashrc` plus explicit
  `conda activate`, which matches the validated BigPurple setup used here
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

The project supports two approaches for LLM inference on HPC. 
**On BigPurple, the vLLM approach is used** (for all DVC stages requiring LLM calls).
The cloud API option is documented for reference.

#### Cloud API (Mistral or OpenAI) — Alternative

Use cloud LLM APIs by setting API keys in `.env` and setting params.yaml. 
No GPU needed for LLM calls.

```yaml
# params.yaml (local dev only — not used on BigPurple)
llm:
  default_provider: mistral  # or "openai"
```

#### Self-Hosted vLLM — BigPurple Approach (USED)

On BigPurple, the project self-hosts models via **vLLM**, which exposes an
OpenAI-compatible API server on `localhost` (dynamic port). This eliminates cloud API
costs and allows using any HuggingFace model.

vLLM is started as a background process in the SLURM job, then the pipeline
connects to it via the `openai` provider with `OPENAI_BASE_URL` pointing at
the local server.

**How it works:**

1. SLURM job starts vLLM server in the background on a **dynamic port**
   (avoids conflicts when multiple jobs share a node)
2. Server loads model weights and exposes `http://localhost:${VLLM_PORT}/v1`
3. Environment variables tell the `openai` Python client to connect locally:
   ```bash
   export OPENAI_BASE_URL=http://localhost:${VLLM_PORT}/v1
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

**vLLM Server Launch (Qwen3.5-27B on 8x V100-16GB):**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-27B \
    --host 0.0.0.0 \
    --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
    --api-key "$API_KEY" \
  --served-model-name "Qwen/Qwen3.5-27B" \
    --download-dir /gpfs/scratch/$USER/hf_cache \
  --tensor-parallel-size 8 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --language-model-only \
    --dtype float16 \
    --enforce-eager
```

Key flags:
- Qwen3.5 is natively supported in the validated vLLM 0.19.0 build (no `--model-impl` or `--trust-remote-code` needed)
- `--dtype float16`: Required on V100 (no bfloat16 support)
- `--enforce-eager`: Skip CUDA graph compilation, faster startup (~4 min vs ~8 min)
- `--served-model-name`: Name that appears in the API; must match `params.yaml`
- `--download-dir`: Point to scratch for large model weights
- `--tensor-parallel-size 8`: Required on current BigPurple hardware because 4x nodes were confirmed to have 16 GB GPUs
- `--language-model-only`: Skip the vision stack and preserve VRAM for KV cache in this text-only pipeline

Qwen3.5-27B weights are roughly 54 GB in FP16 before runtime overhead, so the
current deployment target is an 8x V100-16GB node with tensor parallelism and a
conservative `--max-model-len 16384`.

**(Recommended): Verify basic LLM capabilities:**
- Basic chat completions (non-streaming and streaming)
- Structured JSON output via raw HTTP top-level `structured_outputs` or OpenAI-client `extra_body`
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

The SLURM scripts (Section 9.6) set `OPENAI_BASE_URL` and `OPENAI_API_KEY`
as environment variables so the `openai` provider connects to the local vLLM
server. `OPENROUTER_API_KEY` is loaded from `.env` for the embed stage.

### 8.4 Data Storage Strategy

#### BigPurple Storage Tiers

BigPurple has three storage tiers with different characteristics:

| Tier | Path | Backed Up? | Notes |
|------|------|-----------|-------|
| **Home** | `/gpfs/home/$USER/` | Yes | Small quota — conda envs and `.bashrc` only |
| **Lab** | `/gpfs/data/cerdalab/LegalAI/legiscope/` | Yes | Primary working location — git repo, data, results |
| **Scratch** | `/gpfs/scratch/$USER/` | No | Fast SSD, periodically purged — model weights, temp files |

**Recommended layout on BigPurple:**

```
/gpfs/data/cerdalab/LegalAI/legiscope/          # Git repo (lab storage, backed up)
├── data/laws/{STATE}/{Locality}/...    # Pipeline inputs + outputs (actual repo structure)
├── data/chroma_db/                     # ChromaDB vector database
├── data/output/                        # Benchmark results
└── logs/                               # SLURM job logs

/gpfs/scratch/$USER/
├── hf_cache/                           # HuggingFace model weights (large, re-downloadable)
│   └── models--Qwen--Qwen3.5-27B/
└── tmp/                                # Temp files for pip builds
```

If scratch is purged, re-download model weights via a SLURM batch job using
`huggingface_hub.snapshot_download()`.

#### Data Storage: HPC Lab Storage + DVC Experiments (RECOMMENDED)

All data files (parquet, Markdown, ChromaDB, benchmark CSVs) stay on the
HPC lab filesystem. **DVC experiments** handle reproducibility tracking:

- Each `dvc exp run` creates a lightweight Git reference under `refs/exps/`
  that captures `params.yaml`, `dvc.lock`, and metrics for that run
- No manual branches needed — 50 jurisdiction runs produce 50 experiments,
  all viewable in a single `dvc exp show` table
- Experiments are pushed to GitHub via `dvc exp push origin`, making params
  and metrics (but not data files) accessible from any machine
- Data files remain on BigPurple lab storage (backed up by IT)

**Pros**: Zero cloud storage setup, fastest I/O (local filesystem), lab
storage is already backed up, `dvc exp show` gives a comparison table of
all 50 jurisdictions' params and metrics in one view.

**Cons**: Data files only accessible from BigPurple (use `rsync` to pull
results to your laptop).

#### Tracking Results Across Jurisdictions

```bash
# View all experiments (params + metrics side-by-side)
dvc exp show

# Push experiments to GitHub (lightweight Git refs, not data files)
dvc exp push origin

# On your laptop — pull experiment metadata from GitHub
dvc exp pull origin
dvc exp show  # See all 50 runs with params + metrics

# Promote a specific experiment to a real branch (if needed)
dvc exp branch <exp-name> results/PA-Philadelphia
```

#### What stays on `main`

- Code changes, `dvc.yaml`, `config.yaml`, HPC scripts
- `dvc.lock` is **not** committed on `main` — each experiment captures its
  own `dvc.lock` in its experiment ref
- `params.yaml` on `main` can hold a "default" jurisdiction or the last-run
  jurisdiction; experiments override it via `-S` flags

#### Pulling Results to Your Laptop

To get benchmark CSVs and metrics off BigPurple:

```bash
# From your LOCAL machine:
rsync -avz <netid>@bigpurple.nyumc.org:/gpfs/data/cerdalab/LegalAI/legiscope/data/output/ \
    ./data/output/
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

Two scripts handle the full batch workflow:

1. **`coep/scripts/HPC_scripts/slurm_dispatch.sh`** — **Login-node** script (thin loop). Iterates
   over DOCX files, parses jurisdiction metadata from each filename, and calls
   `sbatch` for each one. Does **no** setup or pipeline work.
2. **`coep/scripts/HPC_scripts/slurm_jurisdiction.sh`** — **SLURM job** (self-contained). Receives
   `STATE`, `LOCALITY`, `DOCX_PATH` as env vars and handles *everything* else:
   `init.py`, DOCX copy + conversion, `params.yaml` editing, vLLM startup,
   DVC pipeline, experiment push, and result copying.

### 9.1 Responsibility Split: Dispatcher vs. SLURM Job

For each of the 50 jurisdictions, these steps must happen:

| Step | Where | Why |
|------|-------|-----|
| Iterate DOCX files | **Dispatcher** (login node) | Only place that sees all 50 files |
| Parse `STATE`/`Locality` from filename | **Dispatcher** (login node) | Trivial string parsing; provides env vars for `sbatch` |
| `init.py` (create dirs + registries) | **SLURM job** (compute node) | Writes to `data/laws/`; must happen inside the isolated working copy to avoid race conditions |
| Copy DOCX → `raw/` directory | **SLURM job** (compute node) | Each job operates on its own `$TMPDIR` working copy; dispatcher can't write there |
| `convert_docx.sh` (DOCX → TXT) | **SLURM job** (compute node) | Writes `code.txt` into the working copy |
| Edit `params.yaml` jurisdiction metadata | **SLURM job** (compute node) | Each job edits its own isolated `params.yaml` via `sed` — the whole reason for working-copy isolation |
| `dvc_repro.sh` (full pipeline) | **SLURM job** (compute node) | GPU-bound; requires vLLM server running locally |
| Push DVC experiment | **SLURM job** (compute node) | Captures the exact params + metrics for this run |
| Copy results back | **SLURM job** (compute node) | Writes pipeline outputs to shared project directory |

**Why the dispatcher stays thin (Option A, not Option B):**

- **Race conditions**: If the dispatcher ran `init.py` or edited `params.yaml`
  for all 50 jurisdictions before jobs start, each call would overwrite the last,
  and concurrent jobs would read stale metadata.
- **Working copy boundary**: The dispatcher can't write into `$TMPDIR` — that
  directory doesn't exist until SLURM allocates the compute node.
- **Error recovery**: If a job fails, just resubmit the one `sbatch` command.
  No dispatcher-side cleanup needed.

### 9.2 DOCX Naming Convention

Place all 50 DOCX files in a single folder on the HPC (e.g.,
`/gpfs/data/cerdalab/LegalAI/docx_sources/`). Name each file using this
convention:

```
STATE_Locality.docx
```

Where:
- **STATE** — 2-letter state code (e.g., `PA`, `CA`, `IL`)
- **Locality** — PascalCase city name matching directory convention (e.g.,
  `Philadelphia`, `Chicago`, `Houston`)

The `code_slug` defaults to `municipal-code`. To specify a different code type:

```
STATE_Locality_code-slug.docx
```

Examples:

| Filename | STATE | LOCALITY | CODE_SLUG |
|----------|-------|----------|-----------|
| `PA_Philadelphia.docx` | PA | Philadelphia | municipal-code |
| `CA_LosAngeles.docx` | CA | LosAngeles | municipal-code |
| `IL_Chicago_zoning-code.docx` | IL | Chicago | zoning-code |

The display name (`code_name`) is derived automatically as
`"{Locality} Municipal Code"`.

### 9.3 Single-Jurisdiction SLURM Script (`slurm_jurisdiction.sh`)

The script `coep/scripts/HPC_scripts/slurm_jurisdiction.sh` is a
self-contained SLURM job that runs the complete pipeline for one jurisdiction
on BigPurple. It:

1. Creates an isolated per-job working copy of the repo (avoids `params.yaml`
   and ChromaDB race conditions with concurrent jobs)
2. Edits `params.yaml` in the working copy and runs `init.py`
3. Copies the DOCX file into `raw/` and converts to TXT
4. Starts a vLLM server on a dynamic port (avoids port conflicts when
   sharing nodes)
5. Runs the full DVC pipeline via `./scripts/dvc_repro.sh`
6. Pushes the DVC experiment to GitHub
7. Copies results back to the shared project directory

**Required environment variables** (set by the dispatcher or `--export`):

| Variable | Description | Example |
|----------|-------------|---------|
| `STATE` | 2-letter state code | `PA` |
| `LOCALITY` | PascalCase city name | `Philadelphia` |
| `DOCX_PATH` | Absolute path to DOCX file | `/gpfs/data/cerdalab/LegalAI/docx_sources/PA_Philadelphia.docx` |

**Optional environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CODE_SLUG` | `municipal-code` | Code slug |
| `CODE_NAME` | `{Locality} Municipal Code` | Display name |

The full script is at `coep/scripts/HPC_scripts/slurm_jurisdiction.sh`. Key
implementation details:

- **Working copy isolation**: Each job rsyncs the repo to `$TMPDIR` (excluding
  generated data), giving it its own `params.yaml`, ChromaDB, and DVC
  workspace. This is what makes concurrent execution safe.
- **Dynamic vLLM port**: Uses Python's `socket` module to find a free port,
  avoiding conflicts when multiple jobs share a compute node.
- **Result copying**: After the pipeline completes, results (`data/output/`
  and `data/laws/`) are copied back to the shared project directory. Shared
  registry files are intentionally excluded to avoid concurrent overwrite races.
  There is currently no built-in central registry merge or locking step in this
  workflow.

To submit a single jurisdiction manually:

```bash
sbatch \
    --export="ALL,STATE=PA,LOCALITY=Philadelphia,DOCX_PATH=/gpfs/data/cerdalab/LegalAI/docx_sources/PA_Philadelphia.docx" \
  coep/scripts/HPC_scripts/slurm_jurisdiction.sh
```

### 9.4 Dispatcher Script (`slurm_dispatch.sh`)

The dispatcher script `coep/scripts/HPC_scripts/slurm_dispatch.sh` scans a
directory of DOCX files, parses jurisdiction metadata from each filename
(Section 9.2), and submits a separate `slurm_jurisdiction.sh` job for each one.

Usage:

```bash
# Preview what would be submitted (no jobs actually submitted):
./coep/scripts/HPC_scripts/slurm_dispatch.sh --dry-run /gpfs/data/cerdalab/LegalAI/docx_sources

# Submit all jurisdictions:
./coep/scripts/HPC_scripts/slurm_dispatch.sh /gpfs/data/cerdalab/LegalAI/docx_sources
```

The dispatcher runs on the **login node** (no GPU needed). It:

1. Scans the DOCX directory for `*.docx` files
2. Parses each filename into `STATE`, `LOCALITY`, and optional `CODE_SLUG`
3. Calls `sbatch --export=... coep/scripts/HPC_scripts/slurm_jurisdiction.sh` for each file
4. Reports how many jobs were submitted (and any files that couldn't be parsed)

The dispatcher does **not** override the SLURM job name. Submitted jobs keep
the fixed name from `slurm_jurisdiction.sh` (`legiscope-jurisdiction`), which
is what the monitoring and aggregation commands in Section 13 filter on.

SLURM handles GPU scheduling — if only 5 GPUs are available, SLURM queues the
remaining jobs until resources free up. You can monitor all submitted jobs
with `squeue -u $USER`.

**Rate limiting note**: If many jobs run concurrently, they all call the
OpenRouter API for embeddings in parallel. If you hit rate limits, submit in
smaller batches or add `--array=0-49%5` style throttling by modifying the
SLURM script (see Section 10.3).

### 9.5 Testing with a Single Jurisdiction

Before running all 50 jurisdictions, test with one:

```bash
# Place a test DOCX file (e.g., PA_Philadelphia.docx) in your DOCX source folder
# Submit a single job:
sbatch \
    --export="ALL,STATE=PA,LOCALITY=Philadelphia,DOCX_PATH=/gpfs/data/cerdalab/LegalAI/docx_sources/PA_Philadelphia.docx" \
  coep/scripts/HPC_scripts/slurm_jurisdiction.sh

# Monitor the job:
squeue -u $USER
tail -f /gpfs/data/cerdalab/LegalAI/legiscope/logs/jurisdiction_*.out

# After completion, verify results:
ls data/output/PA-Philadelphia/
dvc exp show
```

Or use the dispatcher with a directory containing just one DOCX:

```bash
./coep/scripts/HPC_scripts/slurm_dispatch.sh --dry-run /gpfs/data/cerdalab/LegalAI/docx_sources
./coep/scripts/HPC_scripts/slurm_dispatch.sh /gpfs/data/cerdalab/LegalAI/docx_sources
```

### 9.6 Benchmark-Only Re-run

To re-run only the benchmark stage with different settings (e.g., different
model, retrieval params) after the full pipeline has completed at least once,
you must first rebuild the shared ChromaDB index (since per-job indexes are
ephemeral — see Section 10.2):

```bash
bash coep/scripts/HPC_scripts/rebuild_index.sh --clean
```

Then use `slurm_jurisdiction.sh` with a modified `params.yaml`. Since the
benchmark runs inside the working copy, you can edit retrieval/query settings
in the shared `params.yaml` and the working copy will pick them up.

Alternatively, use `coep/scripts/HPC_scripts/slurm_benchmark.sh` — a lighter
SLURM job that starts vLLM and runs only the benchmark stage. See the script
for full details on prerequisites (rebuild_index, etc.).

#### Submitting Jobs

```bash
# Run all 50 jurisdictions (dispatcher submits one SLURM job per DOCX):
./coep/scripts/HPC_scripts/slurm_dispatch.sh /gpfs/data/cerdalab/LegalAI/docx_sources

# Run a single jurisdiction manually:
sbatch \
    --export="ALL,STATE=PA,LOCALITY=Philadelphia,DOCX_PATH=/gpfs/data/cerdalab/LegalAI/docx_sources/PA_Philadelphia.docx" \
  coep/scripts/HPC_scripts/slurm_jurisdiction.sh

# Benchmark-only re-run (after full pipeline has completed):
# NOTE: Requires rebuild_index.sh --clean first (see Section 13.5)
sbatch coep/scripts/HPC_scripts/slurm_benchmark.sh
```

For the current 27B deployment on BigPurple, request `--gres=gpu:8` and use
`--tensor-parallel-size 8` in the vLLM launch command.

---

## 10. Important Considerations

### 10.1 Concurrent Jobs & params.yaml

**Critical issue**: `params.yaml` is a single shared file. If multiple SLURM
jobs run simultaneously from the same directory, they could conflict when
writing to it.

**Solution — per-job working copies** (built into `slurm_jurisdiction.sh`):

Each SLURM job rsyncs the repo to `$TMPDIR`, giving it an isolated
`params.yaml`, ChromaDB, and DVC workspace. The job edits `params.yaml` via
`sed` for `init.py`, then `dvc_repro.sh` passes `-S` flags to `dvc exp run`.
No race conditions since each job works in its own directory.

This also solves ChromaDB concurrency (Section 10.2) — each job gets its own
isolated ChromaDB instance.

**Alternative — sequential execution**: Submit jobs one at a time, or use
SLURM job dependencies (`--dependency=afterok:JOBID`). Safe but slow.

### 10.2 ChromaDB Concurrency & Collection Design

ChromaDB uses SQLite under the hood. Multiple concurrent processes writing to
the same ChromaDB directory will cause corruption.

#### Per-Job Isolation (During Pipeline Runs)

The working-copy approach (Section 10.1) solves this: each SLURM job rsyncs
the repo to `$TMPDIR` **excluding** `data/chroma_db/`, so the `index` stage
creates a **brand-new, single-jurisdiction ChromaDB** in the working directory.
The benchmark and query stages then run against this isolated database, which
contains only the current jurisdiction's embeddings. No cross-jurisdiction
contamination is possible during pipeline execution.

The per-job ChromaDB is **ephemeral** — it lives in `$TMPDIR` and is
automatically deleted when the SLURM job ends. It is intentionally **not**
copied back to the shared project directory (this avoids corruption from
concurrent writes).

#### What Gets Copied Back

Each job copies these artifacts back to the shared project directory:

- `data/laws/{STATE}/{Locality}/` — pipeline outputs including
  `embeddings.parquet`, `sections.parquet`, etc.
- `data/output/{STATE}-{Locality}/` — benchmark results and metrics

Each job does **not** copy back `data/jurisdictions.parquet` or
`data/codes.parquet`. Those registry files are shared global state; copying a
per-job working-copy version back to the shared filesystem would create a
last-writer-wins race under concurrent SLURM runs. If you need to update those
registries on HPC, do it centrally in a serialized step rather than as part of
every batch job.

At present, that centralized handling is only a recommendation: this HPC
workflow does **not** yet implement an automatic merge, file lock, or separate
registry-update job for these parquet files.

The `embeddings.parquet` files persist permanently and can be used to rebuild
a shared ChromaDB index at any time.

#### Rebuilding a Shared Index (Post-Job, Optional)

After all 50 jobs complete, there is **no shared ChromaDB index** on disk — each
job's index was discarded with its `$TMPDIR`. If you need a unified index for
ad-hoc queries or re-running benchmarks from the shared project directory, use
the rebuild script:

```bash
bash coep/scripts/HPC_scripts/rebuild_index.sh --clean
```

This iterates over all `embeddings.parquet` files in `data/laws/` and calls
`scripts/index.py` for each, building a single ChromaDB collection containing
all 50 jurisdictions. See Section 13.5 for details.

#### Jurisdiction Filtering at Query Time

**Collection naming**: Collections are named `legal_code_{provider}_{model}`
(e.g., `legal_code_openrouter_qwen_qwen3-embedding-8b`) with no jurisdiction
component. All jurisdictions share one collection after rebuild. Each embedding
is tagged with `jurisdiction_id` metadata (e.g., `PA-Philadelphia`).

**Retrieval filtering**: Both `run_queries()` and `benchmark_pipeline.py`
**always** pass `jurisdiction_id` to `SectionRetrievalSettings`, which applies
a ChromaDB `where` filter: `{"jurisdiction_id": "PA-Philadelphia"}`. This
ensures that only embeddings from the target jurisdiction are returned —
cross-jurisdiction contamination is not possible through the standard pipeline.

The `run_queries()` function enforces this with a hard validation:
```python
if not jurisdiction_id or not jurisdiction_id.strip():
    raise ValueError("jurisdiction_id cannot be empty")
```

#### Re-running a Single Jurisdiction

The index stage is incremental — it skips segments whose `segment_id` already
exists in the collection. To fully re-index one jurisdiction, either:

- Use `--clean` with `rebuild_index.sh` to wipe and rebuild everything, or
- Delete the shared `data/chroma_db/` directory before re-running the job

### 10.3 Rate Limiting

If running multiple array jobs simultaneously, they will all
call the OpenRouter API for embeddings in parallel. This may hit rate limits.

- Use `--array=0-49%5` to limit to 5 concurrent jobs
- The code already has `max_retries: 3` configured
- Add exponential backoff if needed (the `instructor` library handles some retries)

Benchmark-only re-runs use only vLLM (local), so rate limiting is not a
concern for those jobs.

### 10.4 Network Access

The HPC compute nodes need outbound HTTPS access to:
- `openrouter.ai` (**required** — OpenRouter API for embeddings)
- `huggingface.co` (**required** — downloading model weights for vLLM)
- `storage.googleapis.com` (only if using GCS/DVC push)

Check with your HPC admins if compute nodes have internet access. If not, you
may need to run on login nodes or a special partition with network access.

### 10.5 V100 GPU Constraints (BigPurple)

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

## 11. Step-by-Step: Initial Test (PA-Philadelphia)

### One-Time Setup

These steps only need to be done once, before your first SLURM submission.

If you want to automate the setup and transfer flow, use the helper scripts in
this repo:

```bash
# On BigPurple: clone/update repo, create .env, create required directories,
# and verify whether the expected inputs are present.
bash coep/scripts/HPC_scripts/bootstrap_bigpurple.sh

# On your local machine: sync the active query CSV, MonQcle CSV, and DOCX files.
./coep/scripts/HPC_scripts/sync_bigpurple_inputs.sh --netid <netid> --docx-dir ~/legiscope-docx

# On your local machine: pull benchmark CSV + metrics back from BigPurple.
./coep/scripts/HPC_scripts/pull_bigpurple_results.sh --netid <netid> --jurisdiction PA-Philadelphia --open
```

1. **Login and initialize your shell environment**:
   ```bash
   ssh $USER@bigpurple.nyumc.org
  source ~/.bashrc
   ```

2. **Clone repo on HPC**:
   ```bash
   cd /gpfs/data/cerdalab/LegalAI
   git clone https://github.com/jsqr/legiscope.git
   cd legiscope
   ```

3. **Set up environment**:
   ```bash
   # Create conda env (one-time)
   conda create -p ~/conda_envs/legiscope_env python=3.12 pip -y

   # Activate (both forms work — the tilde expands to /gpfs/home/$USER):
   conda activate ~/conda_envs/legiscope_env

   # Install dependencies via a SLURM job (not on login node — see Section 8.1):
   #   pip install vllm
   #   pip install -e .
   ```

4. **Configure**:
   ```bash
   cp .env.example .env
   # Edit .env: add OPENROUTER_API_KEY (for embeddings)
   # OPENAI_API_KEY is set dynamically by SLURM scripts (vLLM server key)

   # Edit params.yaml:
   #   llm.default_provider: "openai"            # vLLM via OpenAI-compatible API
   #   llm.providers.openai.fast: "powerful-model"
   #   llm.providers.openai.powerful: "powerful-model"
   #   embeddings.default_provider: "openrouter"  # OpenRouter API for embeddings
   ```

5. **Install pandoc** (required for DOCX conversion — `slurm_jurisdiction.sh` needs it):
   ```bash
   # Install into the conda env so it's always available in SLURM jobs:
   conda activate ~/conda_envs/legiscope_env
   conda install -c conda-forge pandoc -y
   # The SLURM script also tries `module load pandoc` as a fallback.
   ```

6. **Upload your DOCX file to the HPC** (see Section 7 for full details):
   ```bash
   # From your LOCAL machine — upload the Philadelphia DOCX to the staging directory:
   scp /path/to/PA_Philadelphia.docx \
       <netid>@bigpurple.nyumc.org:/gpfs/data/cerdalab/LegalAI/docx_sources/
   ```

7. **Upload MonQcle and query data** (gitignored — not in the repo):
   ```bash
   # On HPC: create the directories
   mkdir -p data/queries
   mkdir -p coep/data/monqcle_data

   # From your LOCAL machine: upload the active query file and MonQcle report
   scp data/queries/DPL_queries_with_context.csv \
       <netid>@bigpurple.nyumc.org:/gpfs/data/cerdalab/LegalAI/legiscope/data/queries/

   scp coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv \
       <netid>@bigpurple.nyumc.org:/gpfs/data/cerdalab/LegalAI/legiscope/coep/data/monqcle_data/
   ```

### Run the Pipeline

Everything from here is handled by `slurm_jurisdiction.sh` — it creates an
isolated working copy, sets `params.yaml`, runs `init.py`, copies and converts
the DOCX, starts vLLM, runs the full DVC pipeline, pushes the experiment, and
copies jurisdiction outputs and benchmark results back.

8. **Submit the SLURM job**:
   ```bash
   sbatch --export="ALL,STATE=PA,LOCALITY=Philadelphia,DOCX_PATH=/gpfs/data/cerdalab/LegalAI/docx_sources/PA_Philadelphia.docx" \
     coep/scripts/HPC_scripts/slurm_jurisdiction.sh
   ```

9. **Monitor** (see Section 13 for more):
   ```bash
   # Check job status
   squeue -u $USER

   # Tail the log
   tail -f logs/jurisdiction_<JOBID>.out
   ```

### After the Job Completes

10. **Verify results**:
   ```bash
   ls data/output/PA-Philadelphia/
   dvc exp show

   # Quick peek at scores
   python -c "
   import polars as pl
   import glob
   f = sorted(glob.glob('data/output/PA-Philadelphia/benchmark_results_*.csv'))[-1]
   df = pl.read_csv(f)
   print(f'Average score: {df[\"eval_score\"].mean():.2f}')
   print(f'Correct: {df.filter(pl.col(\"eval_label\")==\"Correct\").height}')
   print(f'Total: {df.height}')
   "
   ```

11. **Re-run benchmark only** (optional, with different settings):
    ```bash
    # First, rebuild shared ChromaDB index (per-job indexes are ephemeral):
    bash coep/scripts/HPC_scripts/rebuild_index.sh --clean

    # Submit benchmark-only SLURM job:
    sbatch coep/scripts/HPC_scripts/slurm_benchmark.sh
    # Results in: data/output/PA-Philadelphia/benchmark_results.csv
    ```

---

## 12. MonQcle Ground Truth (All 50 Cities)

The MonQcle Standard Report CSV
(`coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv`) contains
data for **487 jurisdictions** in wide format (one row per city per series).
The file has a BOM-encoded header; the key columns are:

| Column | Example |
|--------|---------|
| `name` | `"Philadelphia, Philadelphia County, Pennsylvania, United States"` |
| `series_title` | `DPL_2025_Consolidated` |
| `dp_law`, `dp_type`, ... | Variable columns (melted to long format) |

The benchmark pipeline automatically:

1. Maps the `jurisdiction_id` (e.g., `PA-Philadelphia`) to the full MonQcle
   locality name using `jurisdiction_id_to_monqcle_name()` in `coep/src/eval.py`
2. Filters the CSV to the matching row + series title
3. Melts to long format (one row per variable)
4. Joins with RAG query results by `variable_name`

**You only need one MonQcle CSV file for all 50 jurisdictions.** Each
benchmark job filters to its own jurisdiction at runtime.

### CRITICAL: Maintaining the Jurisdiction Mapping

`jurisdiction_id_to_monqcle_name()` now includes the current 50-city batch.
If you add new jurisdictions beyond that set, extend the mapping in
`coep/src/eval.py` before running the benchmark.

```python
mapping = {
    "PA-Philadelphia": "Philadelphia, Philadelphia County, Pennsylvania, United States",
  # ... current 50-city batch omitted here for brevity ...
}
```

The key is the internal `jurisdiction_id` (`{STATE}-{Locality}`,
e.g., `PA-Philadelphia`) and the value is the full MonQcle `name` column value
(e.g., `"Philadelphia, Philadelphia County, Pennsylvania, United States"`).

You can generate this mapping by cross-referencing your DOCX filenames
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

## 13. Post-Processing: Monitoring & Aggregating Results

### 13.1 Monitoring Running Jobs

Check job status while the batch is running:

```bash
# Jobs submitted via slurm_dispatch.sh inherit the fixed name
# `legiscope-jurisdiction` from slurm_jurisdiction.sh.

# How many jobs are still queued / running?
squeue -u $USER -n legiscope-jurisdiction

# Quick failure check — any jobs that already failed?
sacct -u $USER --name=legiscope-jurisdiction -s FAILED \
      --format=JobID,JobName,State,ExitCode,Elapsed

# Full status of today's jobs
sacct -u $USER --name=legiscope-jurisdiction --starttime=today \
      --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
```

### 13.2 Inspecting Failures

```bash
# Which log files contain errors?
grep -l "ERROR\|FAILED\|Traceback" /gpfs/data/cerdalab/LegalAI/legiscope/logs/jurisdiction_*.err

# Show the last few lines of every job's stdout (quick health check)
tail -n 3 /gpfs/data/cerdalab/LegalAI/legiscope/logs/jurisdiction_*.out

# Read a specific job's error log by job ID
cat /gpfs/data/cerdalab/LegalAI/legiscope/logs/jurisdiction_<JOBID>.err

# Or inspect the most recent error log
cat "$(ls -t /gpfs/data/cerdalab/LegalAI/legiscope/logs/jurisdiction_*.err | head -1)"
```

### 13.3 DVC Experiment Comparison

DVC tracks `benchmark_metrics.json` as metrics for each experiment. Compare
all jurisdiction runs at once:

```bash
# Interactive table (terminal)
dvc exp show

# Export to CSV for further analysis
dvc exp show --csv > all_experiments.csv
```

### 13.4 Aggregation Script

After all jobs complete, use
`coep/scripts/HPC_scripts/aggregate_results.py` to collect per-jurisdiction
metrics and benchmark CSVs into a single report:

```bash
# Basic usage — reads output dir from config.yaml
python coep/scripts/HPC_scripts/aggregate_results.py

# Cross-check against expected jurisdictions from DOCX staging dir
python coep/scripts/HPC_scripts/aggregate_results.py \
    --docx-dir /gpfs/data/cerdalab/LegalAI/docx_sources

# Also query SLURM for failed jobs
python coep/scripts/HPC_scripts/aggregate_results.py \
    --docx-dir /gpfs/data/cerdalab/LegalAI/docx_sources \
    --check-slurm

# Specify a custom output directory
python coep/scripts/HPC_scripts/aggregate_results.py \
    --output-dir /gpfs/data/cerdalab/LegalAI/legiscope/data/output
```

The script produces:

| Output File | Contents |
|-------------|----------|
| `all_jurisdictions_metrics.csv` | Per-jurisdiction accuracy, scores, and counts (ranked) |
| `all_jurisdictions_benchmark.csv` | All per-query results concatenated into one CSV |
| Terminal report | Formatted summary with per-jurisdiction accuracy and overall stats |

When `--docx-dir` is provided, the script also reports which expected
jurisdictions are missing results (i.e., jobs that failed or are still running).

### 13.5 Rebuilding the Shared ChromaDB Index (Optional)

Each SLURM job builds an **isolated, ephemeral** ChromaDB in `$TMPDIR` that is
discarded when the job ends.  The per-job pipeline results (benchmark scores,
sections, embeddings, etc.) are all copied back to the shared project directory,
but the ChromaDB index is **not** — this avoids corruption from concurrent writes.

If you need a **unified index** for ad-hoc cross-jurisdiction queries after all
jobs finish, rebuild it from the persisted `embeddings.parquet` files:

```bash
# Recommended: wipe the old index and rebuild cleanly
# WARNING: --clean deletes the entire existing ChromaDB at data/chroma_db/
bash coep/scripts/HPC_scripts/rebuild_index.sh --clean

# Or append to existing index (incremental, skips already-indexed segments)
bash coep/scripts/HPC_scripts/rebuild_index.sh
```

The script discovers all `embeddings.parquet` files under `data/laws/` and calls
`scripts/index.py` for each jurisdiction.  It prints a summary of indexed vs
failed jurisdictions.

> **When do you need this?**  Only when you want to query the shared ChromaDB
> interactively or re-run benchmarks from the persistent project directory.
> The per-job benchmark results are already complete without it.

#### Post-Job Checklist

After all 50 SLURM jobs finish:

1. **Always run** — aggregate benchmark results:
   ```bash
  python coep/scripts/HPC_scripts/aggregate_results.py \
     --docx-dir /gpfs/data/cerdalab/LegalAI/docx_sources --check-slurm
   ```

2. **Run only if needed** — rebuild unified ChromaDB for ad-hoc queries:
   ```bash
  bash coep/scripts/HPC_scripts/rebuild_index.sh --clean
   ```

---

## 14. Quick Reference: Key Commands

| Task | Command |
|------|---------|
| Set up environment | `conda activate legiscope_env` (HPC) or `make env` (local) |
| Convert DOCX | `./scripts/convert_docx.sh data/laws/STATE/Locality/slug/raw` |
| Initialize jurisdiction | `python scripts/init.py` |
| Run DVC pipeline (experiment) | `dvc exp run` |
| Run with jurisdiction override | `dvc exp run -S jurisdiction.state=PA -S jurisdiction.locality=Philadelphia` |
| Run benchmark only (experiment) | `dvc exp run -f benchmark` |
| Run benchmark (standalone) | `python coep/scripts/benchmark_pipeline.py` |
| Run benchmark (test) | `python coep/scripts/benchmark_pipeline.py --test-limit 5` |
| Run queries only | `python scripts/run_queries.py` |
| View all experiments | `dvc exp show` |
| Push experiments to GitHub | `dvc exp push origin` |
| Pull experiments from GitHub | `dvc exp pull origin` |
| Promote experiment to branch | `dvc exp branch <exp-name> results/PA-Philadelphia` |
| Run tests | `make test` |
| Check errors | `make lint` |
| **BigPurple: Bootstrap setup** | `bash coep/scripts/HPC_scripts/bootstrap_bigpurple.sh` |
| **BigPurple: All 50 jurisdictions** | `./coep/scripts/HPC_scripts/slurm_dispatch.sh /gpfs/data/cerdalab/LegalAI/docx_sources` |
| **BigPurple: Single jurisdiction** | `sbatch --export=ALL,STATE=PA,LOCALITY=Philadelphia,DOCX_PATH=/gpfs/data/cerdalab/LegalAI/docx_sources/PA_Philadelphia.docx coep/scripts/HPC_scripts/slurm_jurisdiction.sh` |
| **BigPurple: Dry run** | `./coep/scripts/HPC_scripts/slurm_dispatch.sh --dry-run /gpfs/data/cerdalab/LegalAI/docx_sources` |
| **BigPurple: Benchmark only** | `bash coep/scripts/HPC_scripts/rebuild_index.sh --clean && sbatch coep/scripts/HPC_scripts/slurm_benchmark.sh` |
| **BigPurple: Check failures** | `sacct -u $USER --name=legiscope-jurisdiction -s FAILED` |
| **BigPurple: Aggregate results** | `python coep/scripts/HPC_scripts/aggregate_results.py --docx-dir /gpfs/data/cerdalab/LegalAI/docx_sources --check-slurm` |
| **BigPurple: Rebuild shared index** | `bash coep/scripts/HPC_scripts/rebuild_index.sh --clean` |
| **Local: Sync inputs to HPC** | `./coep/scripts/HPC_scripts/sync_bigpurple_inputs.sh --netid <netid> --docx-dir ~/legiscope-docx` |
| **Local: Pull one jurisdiction's benchmark CSV** | `./coep/scripts/HPC_scripts/pull_bigpurple_results.sh --netid <netid> --jurisdiction PA-Philadelphia --open` |

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

### Current Deployment Target (V100-16GB, 8 GPU)

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen3.5-27B` |
| Node class | `gpu8_short` / `gpu8_medium` |
| GPU topology | 8× V100-16GB with tensor parallel size 8 |
| Starting max model len | 16384 |
| Launch mode | `--language-model-only --dtype float16 --enforce-eager` |

Both `fast` and `powerful` slots now point to the same 27B model so the current
single-server HPC workflow can satisfy every legiscope LLM call without running
separate fast and powerful vLLM servers.

### Model Testing Plan

Once the pipeline is ported, evaluate models in order. Use the first model
that meets accuracy requirements — earlier options are more practical
operationally.

**Current production model:**
- **Qwen3.5-27B** in FP16 on 8× V100-16GB
- Recommended `--max-model-len 16384` to start; test `32768` only after smoke-test success

**Larger fallback / comparison candidates (in evaluation order):**

| Option | Model | GPUs | VRAM | Quantization | Notes |
|--------|-------|------|------|-------------|-------|
| 1 | Qwen3.5-32B | 8× V100-16GB or 4× V100-32GB | ~64 GB | FP16 (none) | Nearest larger dense Qwen option |
| 2 | Qwen3-32B | 8× V100-16GB or 4× V100-32GB | ~64 GB | FP16 | Thinking mode adds output-token overhead |
| 3 | Llama 3.3 70B | 8× V100 | ~38 GB | AWQ 4-bit | Larger dense comparison if 27B underperforms |
| 4 (escalation) | Qwen3-235B-A22B | 8× V100 | ~118 GB | AWQ 4-bit | MoE escalation only if dense models leave an accuracy gap |

**Decision rule:** If the current 27B model meets accuracy requirements, keep it.
Moving larger than 27B will increase queue friction and startup cost on
BigPurple without helping the rest of the pipeline.

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
  "Qwen/Qwen3.5-27B",
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
| **Connection refused (localhost)** | Server takes ~240s to start on V100. Check if server process is alive. Check vLLM logs. The health check loop in SLURM scripts handles this automatically |
| **vLLM tries to build from source** | On BigPurple this is expected because glibc 2.28 rejects the official vLLM 0.19.0 wheel. Use GCC 11.2.0, pin `torch==2.10.0`, `torchvision==0.25.0`, `torchaudio==2.10.0`, clear cached local vLLM wheels, then install with `pip install "vllm==0.19.0" --no-build-isolation --no-cache-dir --no-binary vllm --force-reinstall`. Verified working combo: vLLM 0.19.0 with PyTorch 2.10.0+cu128 |
| **Quantized model not loading** | Verify model was downloaded in correct format (AWQ/GPTQ). Check model's `config.json` for quantization method |
| **instructor structured output fails** | Verify `instructor.Mode.JSON` is set for the openai provider in `llm_config.py` (changed from `RESPONSES_TOOLS`, which requires OpenAI's Responses API that vLLM doesn't support) |

### BigPurple Issues

| Problem | Solution |
|---------|----------|
| **pip install killed on login node** | Always run `pip install` via SLURM batch jobs with ≥32 GB memory |
| **conda stdlib errors** (e.g., "No module named urllib.parse") | Delete and recreate environment: `conda remove -p ~/conda_envs/legiscope_env --all -y` then `conda create -p ~/conda_envs/legiscope_env python=3.12 pip -y` |
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
tail -f /gpfs/data/cerdalab/LegalAI/legiscope/logs/<logfile>.out   # Watch job output
cat $(ls -t /gpfs/data/cerdalab/LegalAI/legiscope/logs/*.out | head -1)  # Read most recent log
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
cd /gpfs/data/cerdalab/LegalAI/legiscope
git pull

# Refresh repo + required directories, then verify inputs:
bash coep/scripts/HPC_scripts/bootstrap_bigpurple.sh

# Full pipeline for all jurisdictions — dispatcher submits one SLURM job per DOCX:
./coep/scripts/HPC_scripts/slurm_dispatch.sh /gpfs/data/cerdalab/LegalAI/docx_sources

# Or single jurisdiction:
sbatch --export="ALL,STATE=PA,LOCALITY=Philadelphia,DOCX_PATH=/gpfs/data/cerdalab/LegalAI/docx_sources/PA_Philadelphia.docx" \
  coep/scripts/HPC_scripts/slurm_jurisdiction.sh

# Benchmark-only re-run (with different settings):
# NOTE: Requires rebuild_index.sh --clean first (see Section 13.5)
sbatch coep/scripts/HPC_scripts/slurm_benchmark.sh

# After jobs complete — view results across all jurisdictions:
dvc exp show

# Pull experiment metadata to your laptop:
# (on local machine)
dvc exp pull origin
dvc exp show

# Pull result files to your laptop:
# (on local machine)
rsync -avz <netid>@bigpurple.nyumc.org:/gpfs/data/cerdalab/LegalAI/legiscope/data/output/ ./data/output/
```
