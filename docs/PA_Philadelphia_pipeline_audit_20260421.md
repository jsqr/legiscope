# PA Philadelphia Pipeline Audit - 2026-04-21

## Scope

Audit of the 2026-04-20 full pipeline run for `PA:Philadelphia:municipal-code`.

This document is intentionally planning-oriented. It records observed issues, likely root causes, and suggested fixes or follow-up investigations. It does not prescribe code changes yet.

## Inputs Reviewed

- Output log provided in chat from `jurisdiction_20784313.err`
- `data/laws/PA/Philadelphia/municipal-code/code.md`
- `data/laws/PA/Philadelphia/municipal-code/chunks.parquet`
- `data/laws/PA/Philadelphia/municipal-code/segments.parquet`
- `data/output/PA-Philadelphia/benchmark_results_20260420_191549.csv`
- `data/output/PA-Philadelphia/benchmark_metrics.json`
- `data/output/PA-Philadelphia/debug/query_stage_20260420_191549.csv`
- `data/output/PA-Philadelphia/debug/relevance_stage_20260420_191549.csv`
- `data/output/PA-Philadelphia/debug/retrieval_stage_20260420_191549.csv`

## Audit Status

- Status: in progress
- Current focus: artifact quality, retrieval-unit design, benchmark output quality, and log warnings

## Initial Findings

### 1. Chunk-to-segment expansion is much smaller than intended

Initial artifact counts:

- `chunks.parquet`: 3287 rows
- `segments.parquet`: 6275 rows
- Mean segments per chunk: 1.91
- Median segments per chunk: 1
- 75th percentile segments per chunk: 1
- Max segments per chunk: 11

Interpretation:

- Most chunks produce exactly one embedded segment.
- The current chunk layer is therefore not acting like a materially larger retrieval context unit for most of the corpus.
- This matches the user's concern that retrieved segment count and retrieval-unit count are staying too close.

Likely issue to investigate:

- Either chunk construction is not aggregating enough neighboring context, or the chunk schema is carrying larger context in theory but the practical text body being embedded remains close to section-sized.

Suggested follow-up:

- Compare `chunk_id` to `section_id`, `chunk_part`, `chunk_count`, `token_count`, and source-region boundaries to identify whether chunks are being split by section/region too early.
- Inspect representative high-value chunks from the drug-paraphernalia sections and compare their text span with the produced segments.
- Decide whether the intended retrieval unit should be:
  - one semantic chunk built from adjacent sections/regions, or
  - one segment hit expanded into a stitched context window at retrieval time.

### 2. Benchmark results CSV has obvious schema noise

Initial artifact shape:

- `benchmark_results_20260420_191549.csv`: 21 rows, 41 columns
- Includes a blank column named `""`
- Includes `_duplicated_0` through `_duplicated_10`

Interpretation:

- The benchmark export is currently preserving duplicate source columns instead of resolving them before write.
- This makes the file harder to review manually and raises a risk that downstream analysis picks the wrong copy of a field.

Suggested follow-up:

- Trace the join path that merges generated answers, source query metadata, and ground truth.
- Explicitly define the export contract for `benchmark_results.csv` and drop or rename duplicates before write.
- Separate three categories cleanly:
  - query metadata
  - model output
  - evaluation output

### 3. Evaluation output should add explicit error taxonomy

Requested improvement:

- Add an LLM-generated field that classifies failures into categories such as:
  - retrieval error
  - reasoning error
  - output structuring error
  - other error

Initial planning note:

- This belongs in the LLM-as-a-judge response schema, not as a post hoc heuristic based only on score.
- It should likely be paired with a short free-text rationale and possibly allow multiple labels if the judge sees overlapping failures.

Suggested design direction:

- Add a structured field such as `error_type` or `error_types` to the evaluation model.
- Only populate it when the evaluated answer is imperfect or failed; otherwise return `none`.
- Keep label vocabulary fixed and documented so benchmark trend analysis stays stable over time.

### 4. Log contains multiple actionable warnings or improvement areas

Observed from the provided log:

- Repeated `FutureWarning` messages for deprecated CUDA modules: `cuda.cudart` and `cuda.nvrtc`
- Repeated `UserWarning` messages from vLLM FLA ops about likely tensor format mismatch
- A Polars `DeprecationWarning` for `DataFrame.with_row_count`
- DVC warnings that file hash info was not found for several outputs
- Repeated heading scan timeouts before a valid response was returned
- Repeated hallucination or close-match warnings during supporting-passage validation
- 11 benchmark queries had no ground truth and were excluded from judged evaluation

These need triage into:

- code fixes
- dependency or environment fixes
- expected-but-noisy warnings that should be suppressed or documented

## Open Questions To Resolve During Audit

1. Are chunk rows actually large in text span, with segment splitting happening only for embedding token budget, or are most chunk rows already section-sized?
2. How much of benchmark underperformance is retrieval failure versus relevance-filter over-pruning versus answer-generation distortion?
3. Which benchmark result columns are canonical, and which are artifacts of joins or duplicated source CSV headers?
4. Should missing-ground-truth rows stay in `benchmark_results.csv`, and if so, how should they be represented so users do not confuse them with judged failures?

## Working Notes

### `code.md` header

- Header exists and looks structurally valid.
- `created_at` is present.
- `code_start` metadata is present with `element_id`, `source_line`, and `output_line`.
- Heading patterns inferred for title/article/chapter/section appear plausible for Philadelphia code formatting.

### Immediate next checks

- Inspect representative rows from `chunks.parquet`, `segments.parquet`, and retrieval debug output to verify whether chunks actually carry neighborhood context.
- Inspect benchmark/debug CSV contents to identify redundant columns, evaluation blind spots, and failure patterns by query family.
- Summarize log issues into likely code-path owners and recommended fixes.

## Concrete Artifact Findings

### 5. Current chunk design is bimodal: many tiny auxiliary chunks, plus very large mixed chapter-part chunks

Additional artifact evidence:

- `regions.parquet`: 5536 rows
- `regions.parquet` role counts:
  - `main_body`: 2706
  - `annotation`: 2520
  - `publisher_boilerplate`: 240
  - `legal_intro`: 51
- `chunks.parquet` source kinds:
  - `region`: 2569
  - `section_packed_split`: 466
  - `section_body_split`: 204
  - `section_packed`: 22
  - `section_subtree`: 22

Chunk body length distribution is highly skewed:

- median chunk body length: 5 characters
- 75th percentile chunk body length: 494 characters
- max chunk body length: 26664 characters

Interpretation:

- The chunk corpus is dominated by tiny auxiliary region chunks, mostly annotations and legal intro material.
- The larger operative chunks are often very broad `section_packed_split` chapter parts rather than moderate, local context windows.

Representative issue:

- A retrieved chunk headed `### CHAPTER 39 (Part 7)` contains multiple unrelated section bodies in the same retrieval unit, including `§ 6-310`, `§ 9-625`, `§ 9-1101`, `§ 10-803`, and `§ 10-903`.
- That is larger context, but not the kind of nearby context centered on the matched provision that was intended.
- The retrieval step then groups by `chunk_id`, so a single matched segment expands to the whole broad chapter-part chunk.

Planning recommendation:

- Rework chunking so retrieval units are centered on a matched section or matched segment neighborhood rather than arbitrary packed slices of a large ancestor subtree.
- Two viable approaches:
  - build section-local neighborhood chunks during segmentation, or
  - keep fine-grained segments for retrieval and stitch neighboring sections/segments at retrieval time.
- If auxiliary annotation/legal-intro chunks must remain searchable, consider isolating them from the main operative-text retrieval path or applying a stronger downweight than the current light priority bonus.

### 6. Retrieval-unit inflation is happening at the wrong granularity

Representative mapping evidence:

- `CHAPTER 39 (Part 7)` appears as one retrieval unit but expands to 6 embedded segments.
- Many other large `section_packed_split` chunks expand to 9 or 11 embedded segments.
- At query time, however, the top 10 hits often land in 7-10 distinct chunk IDs, so the grouped retrieval-unit count stays close to the raw segment count.

Interpretation:

- The current system is not failing to create larger text artifacts.
- It is failing to create the right larger artifacts for retrieval: the expansion is broad and heterogeneous, not local and semantically focused.

Suggested fix direction:

- Preserve parent heading context, but center the retrieval unit around the matched section and its immediate neighbors.
- Avoid generic headings like `CHAPTER 39 (Part 7)` as the only retrieval-unit label when the matched legal proposition is actually a specific section such as `§ 9-629`.
- Consider emitting both:
  - a retrieval label rooted at the matched section, and
  - a context payload that includes parent and sibling context.

### 7. Benchmark results are carrying source-query CSV corruption directly into the final artifact

Root-cause evidence:

- `config.default_queries_path()` resolves to `data/queries/DPL_queries_with_context.csv`.
- That CSV header itself contains a blank column plus eleven trailing duplicate columns.
- `load_queries()` currently reads all CSV columns and stores every non-`question` and non-`variable_name` column inside `QueryInput.metadata`.
- `run_queries()` then copies that metadata back into the results DataFrame.
- In the local workspace, the canonical DVC-style output `data/output/PA-Philadelphia/benchmark_results.csv` is currently absent even though the run log says it was written. Only the timestamped copies are present locally.

Interpretation:

- The final benchmark export is noisy because the source query file is noisy and the loading path preserves that noise unchanged.
- This is not just a `benchmark_results.csv` writing issue. The problem starts at query ingestion.

Planning recommendation:

- Clean the query CSV contract at load time:
  - drop blank column names
  - drop all-null columns
  - either reject duplicate columns or normalize them explicitly
- Keep only the structured query fields actually needed downstream.
- Investigate whether the missing canonical `benchmark_results.csv` is a sync issue, an HPC pull-script omission, or an output-materialization problem in the DVC workflow.

Recommended canonical `benchmark_results.csv` shape:

- identifiers and scope:
  - `benchmark_row_id`
  - `jurisdiction_id`
  - `question_number`
  - `variable_name`
- structured query input:
  - `query_text`
  - `prepend_text`
  - `response_options`
  - `coding_instructions`
- generation output:
  - `short_answer`
  - `reasoning`
  - `citations`
  - `supporting_passages`
  - `confidence`
  - `limitations`
- retrieval or validation diagnostics:
  - `retrieval_units_found`
  - `segments_found`
  - `processing_time`
  - `supporting_passage_validation_scores`
- ground truth or judge output:
  - `ground_truth`
  - `ground_truth_citation`
  - `ground_truth_available`
  - `evaluation_status`
  - `eval_score`
  - `eval_label`
  - `eval_reason`
  - proposed new field: `error_type` or `error_types`

Low-value or redundant fields to demote to debug-only output:

- composed `query` string when structured query fields are already present
- `raw_short_answer`
- `sections_found` when retrieval is chunk-based and `retrieval_units_found` is already present
- all blank or `_duplicated_*` columns

### 8. Headline metrics understate the weakest part of the run

Observed metrics:

- processed queries: 21
- scored queries: 10
- unscored queries: 11
- judged accuracy rate: 50.0% (`5 Correct`, `2 Partially Correct`, `3 Incorrect`)
- average judged score: 6.6 / 10

Important caveat:

- All 11 unscored rows also have confidence `0.0` and ended with zero retrieval units after relevance filtering.
- So the current metrics capture only the 10 rows with available ground truth and completely hide the largest concentration of retrieval-stage failures or abstentions.

Planning recommendation:

- Extend benchmark metrics with operational fields such as:
  - `abstained_queries`
  - `zero_confidence_queries`
  - `queries_filtered_to_zero_units`
  - `rows_with_non_exact_supporting_passages`
  - `avg_original_retrieval_units`
  - `avg_filtered_retrieval_units`

### 9. Confidence is severely miscalibrated relative to judged correctness

Artifact evidence:

- Mean confidence across scored rows: `0.955`
- Yet among those same scored rows there are:
  - 3 judged `Incorrect`
  - 2 judged `Partially Correct`

Examples from judged rows:

- `dp_enacted`: confidence `1.0`, judged `Incorrect`
- `dp_effective_dt`: confidence `0.95`, judged `Incorrect`
- `dp_state_fed_combined`: confidence `1.0`, judged `Incorrect`

Interpretation:

- The answer model's confidence field is not currently reliable as a user-facing trust signal.

Planning recommendation:

- Treat model confidence as a raw internal signal unless it is calibrated.
- Consider adding derived confidence adjustments from:
  - retrieval support strength
  - supporting-passage validation quality
  - abstention or conflict signals

### 10. Supporting-passage validation is catching real answer distortion

Artifact evidence:

- Five rows have at least one non-exact supporting-passage validation score below `1.0`.
- Worst examples:
  - `dp_type`: minimum validation score `0.719`
  - `dp_activity`: minimum validation score `0.740`
  - `dp_law`: minimum validation score `0.923`

Interpretation:

- The generation stage is often semantically close but not faithfully quoting retrieved text.
- This is already visible in the log hallucination warnings and is confirmed in the final artifact.

Planning recommendation:

- Promote supporting-passage validation into the benchmark summary.
- Consider either:
  - stricter answer formatting that discourages paraphrased quotations, or
  - post-processing that rewrites near-match citations to exact retrieved spans.

### 11. Relevance filtering is likely the biggest runtime sink

Artifact evidence:

- mean processing time per query: `793.36s`
- median processing time per query: `760.61s`
- average original retrieval units before relevance filtering: `9.29`
- average filtered retrieval units after filtering: `0.86`
- 11 queries end with `filtered_retrieval_unit_count = 0`

Interpretation:

- The pipeline spends substantial time performing expensive relevance judgments on roughly 8-10 candidate units per query, only to discard nearly all of them.
- This is especially costly for queries that end in an abstention anyway.

Planning recommendation:

- Add a cheaper deterministic prefilter before the LLM relevance pass.
- Consider lowering `n_results` for narrowly scoped follow-up questions or using lexical or anchor-based pruning before LLM calls.
- Track relevance-filter cost separately in debug and benchmark metrics.

### 12. Exemption-activity subquestions are a strong candidate for deterministic short-circuiting

Observed pattern:

- The 11 zero-confidence rows are all exemption-activity follow-up variables except the cannabis-specific one that succeeded.
- `dp_exemption` for Philadelphia returned a cannabis-focused exemption answer.
- `dp_exempt_can_activity` succeeded.
- The remaining exemption-activity variables spent full retrieval and relevance-filter time before returning `I cannot answer your question as no relevant legal provisions were found after filtering.`

Important architectural note:

- The COEP retrieval-guidance layer already encodes dependency logic through `_EXEMPTION_DEPENDENCY_LABELS_BY_VARIABLE` and injected `prior_answers` context.
- Right now that dependency is only advisory prompt context, not hard control flow.

Planning recommendation:

- Promote this dependency into deterministic orchestration logic.
- If the earlier `dp_exemption` answer does not include the relevant exemption family label, short-circuit the dependent subquery as `not applicable` or `no qualifying exemption identified` instead of running full retrieval.
- This would likely save most of the runtime currently spent on the 11 abstaining exemption-family rows.

### 13. Log triage and likely ownership

Code-level fixes:

- `coep/scripts/benchmark_pipeline.py`
  - replace `with_row_count` with `with_row_index`
  - extend output schema and metrics
- `coep/src/eval.py`
  - add structured evaluation error taxonomy field(s)
- `src/legiscope/query.py`
  - stop propagating noisy query metadata into benchmark outputs unchanged
  - consider deterministic dependency short-circuiting for follow-up variables
- `src/legiscope/segment.py` and `src/legiscope/retrieve.py`
  - redesign retrieval-unit granularity so it reflects local context rather than broad packed chapter slices

Dependency or environment fixes:

- CUDA deprecation warnings (`cuda.cudart`, `cuda.nvrtc`)
- vLLM tensor-format warning from FLA ops

Operational or infra investigation:

- DVC warnings about missing file hash info despite the outputs being declared in `dvc.yaml`
- repeated `load_params()` debug logging, which suggests either missing memoization or repeated config reloads throughout the run

### 14. Parse-stage quality signals worth preserving in future audits

The parse stage completed, but the log shows:

- heading scan timed out twice before succeeding
- final heading-structure score remained `0.476`
- final iterations still reported `31` errors

Interpretation:

- The final artifacts look usable, but heading inference reached a low-confidence convergence state.
- This may not be the direct cause of the benchmark failures, but it is a meaningful upstream risk indicator and should be preserved in benchmark or audit summaries.

Planning recommendation:

- Persist heading-scan diagnostics as an artifact or summary metric.
- Consider adding a parse-quality gate or warning threshold for exceptionally low heading-structure scores.

### 15. Log output is too repetitive in four specific code paths

The current run log is harder to review than it needs to be because several success-path messages repeat dozens of times in long uninterrupted blocks.

This is primarily a log-throttling issue, not a correctness issue.

#### 15a. `load_params()` repeats source-loading messages excessively

Owning code path:

- `src/legiscope/params.py`

Current behavior:

- `load_params()` logs `Loaded params via dvc.api.params_show()` on every successful call.
- The function is called from many helper surfaces, including module initialization and convenience accessors.
- In a simple local inspection that only resolved the default queries path, the process emitted the same debug message ten times.
- In the provided run log, this message appears in long repeated stretches across parse, embed, index, and benchmark stages.

Interpretation:

- The message is useful once per process or once per source change, but low-value when repeated for every call.
- The deeper issue is that params appear to be reloaded repeatedly with no memoization.

Planning recommendation:

- Add process-local caching for global params loads, with an explicit refresh path only when needed.
- Log the source of params loading only:
  - on first successful load,
  - on fallback from DVC API to direct YAML load,
  - or when a per-code override is merged.
- If repeated visibility is still desired, demote the steady-state success message from `DEBUG` to `TRACE`.

#### 15b. `_process_markdown_elements()` logs heading conversion progress too frequently for large codes

Owning code path:

- `src/legiscope/parse/convert.py`

Current behavior:

- The function logs every 50th heading conversion.
- Philadelphia produced `5410` heading rows in `headings.parquet`.
- At the current threshold, that implies roughly `108` progress messages from this function alone.

Interpretation:

- For large codes, every-50 logging is too granular and creates long blocks of nearly identical progress messages.
- The current message is also low-signal because it only states the element id and heading level for one periodic sample, not a summary of conversion quality.

Planning recommendation:

- Replace the fixed every-50 message with one of these patterns:
  - every 500 or 1000 headings for large files,
  - exponentially spaced milestones,
  - or a single end-of-pass summary with counts by heading level or section type.
- If retaining periodic logs, prefer a message like `Converted 1000 headings so far (level1=..., level2=..., level3=..., level4=...)` over a single sampled element line.

#### 15c. `_generate_embeddings_openrouter()` emits redundant per-call `INFO` logs

Owning code path:

- `src/legiscope/embeddings.py`

Current behavior:

- `_embed_with_fallback()` already emits high-level embedding progress such as:
  - start of embedding job
  - periodic overall progress
  - completion
- But `_generate_embeddings_openrouter()` also logs an `INFO` line at the start of every provider call.
- In this run, `6476` segments were processed with chunk size `100`, so the provider function emitted about `65` near-identical `Processing 100 texts in 1 batches of 100 (OpenRouter)` lines.

Interpretation:

- These lines are operationally redundant because the higher-level embedding loop already reports progress.
- The repeated `INFO` severity makes the log look more eventful than it really is.

Planning recommendation:

- Keep one start message for the whole embedding job in `_embed_with_fallback()`.
- Change `_generate_embeddings_openrouter()` so it:
  - logs batch-level progress only at `DEBUG`, or
  - only logs on retries, rate-limit handling, or partial failures, or
  - logs a start line only when `total_batches > 1` within a single provider call.
- Prefer a single periodic progress stream owned by the outer embedding loop.

#### 15d. `_add_documents_to_collection()` logs every Chroma batch write

Owning code path:

- `src/legiscope/embeddings.py`

Current behavior:

- The function logs a good summary line once: total documents and total batches.
- It then logs one `DEBUG` line per batch.
- For `6476` documents at batch size `100`, that produced `65` repetitive batch-write messages.

Interpretation:

- The summary line is useful.
- Logging every successful batch is usually unnecessary unless debugging a write failure or a hang.

Planning recommendation:

- Reuse the same throttling pattern used for embedding progress.
- Log only:
  - the initial summary,
  - every N batches or every M documents,
  - and the final completion.
- Keep per-batch logging available only at a very verbose level or behind a dedicated debugging flag.

#### 15e. Cross-cutting recommendation for repeat-heavy logs

The project would benefit from a simple logging policy for long-running loops:

- One start message per long-running stage.
- Periodic progress at coarse intervals based on percentage, milestone count, or elapsed work.
- One completion summary per stage.
- Per-item or per-batch logs only for:
  - warnings,
  - retries,
  - threshold crossings,
  - or explicit high-verbosity debugging.

This would materially improve readability of BigPurple stderr logs without losing observability.

## Prioritized Recommendations

### Highest priority

1. Redesign retrieval units around local legal context, not broad packed chapter slices.
2. Add deterministic short-circuiting for dependent exemption-activity questions when the prerequisite exemption family is absent.
3. Clean query-file metadata at load time so `benchmark_results.csv` no longer inherits blank and duplicate columns.

### Medium priority

4. Add evaluation error taxonomy to the judge schema and final benchmark export.
5. Extend benchmark metrics so abstentions, filtered-to-zero failures, and supporting-passage drift are visible.
6. Replace `with_row_count` with `with_row_index` and remove the current Polars deprecation warning.
7. Throttle repeat-heavy success-path logs in `load_params()`, `_process_markdown_elements()`, `_generate_embeddings_openrouter()`, and `_add_documents_to_collection()`.

### Lower priority but still worth addressing

8. Investigate repeated parameter reloads and noisy debug logging.
9. Investigate CUDA and vLLM warning sources on the BigPurple environment.
10. Preserve parse-stage confidence diagnostics in a reusable artifact.