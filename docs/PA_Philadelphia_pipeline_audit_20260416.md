# PA Philadelphia Pipeline Audit (2026-04-16)

## Scope

This audit reviews the pulled BigPurple artifacts for the PA-Philadelphia run stamped `20260416_150837`.
The analysis is artifact-based: benchmark outputs, debug CSVs, code artifacts, parquet outputs, and source code were inspected locally.

Local note: the current OpenRouter Chroma collection in this workspace is empty, so retrieval could not be replayed end-to-end locally without rebuilding the index. The conclusions below come from the persisted debug artifacts and code/parquet inspection.

## Top-level result

- `benchmark_results_20260416_150837.csv` scored 10 benchmark rows.
- 2 rows were correct and 8 were incorrect.
- 21 rows appear in the retrieval, relevance, and query debug CSVs, which means 11 benchmark questions were processed but not scored because usable ground truth was missing or excluded.

## Stage-by-stage findings

### 1. Benchmark and evaluation stage

What is working:

- The benchmark pipeline is wiring together query execution, debug artifact capture, and scoring correctly.

What is wrong:

- Only 10 of the 21 query rows were actually scored, so the headline metric underrepresents what the run processed.
- At least some benchmark labels appear inconsistent with the local source text.

Verified examples:

- The local Philadelphia code notes say the amendment is effective `May 1, 2012`, while the MonQcle row records `05/11/2012` for `dp_effective_dt`.
- The MonQcle row records `dp_activity = Sales, possession with intent to sell, offer for sale`, but its own citation field points to `§ 9-629`, which supports `sell or offer for sale` and does not clearly support `possession with intent to sell` from the local artifact alone.
- The MonQcle row marks `dp_state_fed_reference = 0` while also setting `dp_state_fed = State (ARCHIVE)`.

Impact:

- The current benchmark score mixes true pipeline failures with at least one likely label problem and a few ambiguous semantics.

### 2. Parse and heading-detection stage

What is wrong:

- Body-level structure is not being recovered reliably from the Philadelphia code artifact.
- The parsed heading artifacts omit live body headings that are visibly present in `code.md` and `code.txt`.
- The scan-stage code-start boundary for this run was also wrong before regex generation even began.

Verified examples:

- `code.md` contains body markers for `TITLE 9` and `Chapter 9-600` near the live paraphernalia provisions.
- `headings.parquet` does not contain the body `Chapter 9-600` heading in that region.
- Downstream artifacts still carry stale ancestry such as `TITLE 22 > CHAPTER 39` through the same span.
- The scan log reports `Code starts at element 115`.
- Element-level inspection shows `E114 = TABLE OF CONTENTS`, `E115 = PREAMBLE`, and the following elements (`E116`, `E117`, `E118`, ...) are still TOC-style navigation entries, not the substantive body.
- The earliest clear non-navigation section/body pair appears much later at `E633/E634` (`§ 2-100` plus body text), so the chosen start point is at least 518 elements too early.
- The immediately preceding body heading chain at `E630` / `E632` (`ARTICLE II`, `CHAPTER 1`) is legitimate code structure, but current heuristics still classify those standalone headings as navigation.

Likely root cause:

- The example-based chapter refinement in `src/legiscope/parse/scan.py` only accepts chapter headings shaped like `CHAPTER <digits>...`, which does not cover hyphenated body headings like `Chapter 9-600`.
- `find_code_start` currently allows a TOC `PREAMBLE` element to survive if the verifier accepts it as a boundary candidate.
- `_advance_past_toc_candidate` only looks ahead 500 elements. In this document the first defensible section/body pair starts beyond that window, so the TOC escape logic can fail even when the candidate is obviously too early.
- `_looks_like_body_start_element` only recognizes a section heading when the heading and substantive prose are in the same element. In this file, many true body starts are split across adjacent elements (`§ 2-100` in one element, body text in the next), so valid starts are missed.
- `_is_navigation_element` can also over-classify standalone structural headings as navigation because `ARTICLE` / `CHAPTER` lines without prose look like TOC listings under the current heuristic.

Impact:

- This is not a local typo on one section. The stale ancestry propagates across a large slice of the codebase and poisons downstream `parent_id`, `ancestor_path`, and `context_path` metadata.

Additional scan-loop issue from the log:

- Iteration 1 scored only `0.477` with 45 errors, then iterations 2 through 5 never reached scoring because the LLM call timed out.
- The retry loop kept reusing the same 300-element sample span (`E115` to `E26571`) after timeout because sample size is only reduced for explicit context-length failures, not ordinary timeouts.
- Reconstructed sampling confirms that 161 of the 300 sampled elements are packed into `E115`-`E300`, while the rest are sparsely distributed across the entire document up to `E26571`. That mixes early TOC-heavy material with distant later headings in a single regex-induction prompt.
- The scan diagnostics in the log are also misleading: `heading_like=515` on a 300-element sample is possible because `_sample_diagnostics()` double-counts generic `heading_like` rows.
- In practice this means most later iterations were not fresh attempts. They were near-repeats of the same broad prompt with slightly different feedback text.

### 3. Sectioning and chunk-construction stage

What is wrong:

- Section hierarchy and chunk context are being built from corrupted heading ancestry.

Verified examples:

- The live `§ 9-629` chunk exists in the chunk artifact, but its `context_path` is assigned to the wrong title/chapter lineage.
- Large numbers of downstream sections inherit the same stale pathing.
- The section and chunk artifacts contain duplicated or inconsistently contextualized entries around the paraphernalia region.

Code path:

- `src/legiscope/segment.py` builds parent relationships and context paths from the heading stack. Once the heading layer is wrong, the chunk layer is deterministically wrong.

Impact:

- Retrieval, relevance scoring, and completion all consume misleading structural context even when the operative body text itself is present.

### 4. Embedding and index stage

What is working:

- The pulled parquet artifacts include full `segments.parquet` and `embeddings.parquet` outputs for Philadelphia.
- The parquet artifacts do not show obvious catastrophic loss of the paraphernalia region.

What is wrong:

- Ranking quality is still poor for several benchmark variables.
- The direct operative section is often not the strongest retrieved candidate, especially for activity and cannabis-exemption questions.

Verified examples from debug artifacts:

- Queries targeting paraphernalia activity and exemptions often surfaced zoning/use-table material and unrelated accessory-use fragments ahead of the operative sales ban.
- The direct `§ 9-629` evidence was sometimes present but not ranked strongly enough to dominate the answer.

Local reproducibility caveat:

- The current local Chroma collection for the configured OpenRouter embedding collection is empty, so index integrity could not be replayed interactively here. The audit therefore relies on persisted retrieval-stage debug artifacts from the BigPurple run.

Impact:

- The embed/index stage is not the primary root cause, but ranking quality is still materially contributing to failures.

### 5. Relevance-filter stage

What is wrong:

- Relevance filtering is too brittle for the current legal retrieval behavior.

Code path:

- `src/legiscope/retrieve.py` keeps a section only when `is_relevant` is true and both `relevance_score` and `confidence` are at least the configured threshold.
- The active threshold in `params.yaml` is `0.7`.

Verified impact:

- Relevant sections were dropped even when their reasoning indicated clear usefulness.
- Some cannabis subquestions were reduced to zero sections after filtering.
- For mixed-signal legal contexts, the double-threshold gate is acting like a hard cutoff instead of a mild reranker.

Impact:

- The pipeline is losing useful evidence after retrieval and before completion.

### 6. Query and answer-normalization stage

This is the highest-confidence deterministic bug cluster.

What is wrong:

- The completion schema allows `short_answer` to be arbitrary free text.
- The system prompt asks for a concise answer, but it does not require the model to return the exact declared response option text.
- Post-processing then tries to coerce free-form answers back into benchmark codes, and this coercion is causing wrong final outputs.

Verified examples:

- `dp_enacted`: the raw short answer and reasoning identified `12/21/2011`, but normalization changed the final short answer to `01/23/2007`.
- `dp_type`: the raw answer captured all major paraphernalia families, but normalization collapsed the final short answer to `Other`.
- `dp_penalties`: the final normalized output expanded into many unrelated penalty categories.
- `dp_activity`: the final short answer became `Use`, which does not match the underlying reasoning.

Code path:

- `src/legiscope/query.py` normalizes short answers through `_normalize_structured_short_answer`, `_normalize_multi_select_answer`, `_normalize_single_choice_answer`, and related date helpers.
- `_build_legal_prompts` does not explicitly instruct the model to emit `short_answer` using the benchmark response options exactly.

Impact:

- Even when the model retrieves and reasons correctly, the benchmark-facing answer can be corrupted after generation.

### 7. Query-dependency handling

What is wrong:

- Some benchmark subquestions depend on prior answers, but the query runner treats every row independently.

Verified example:

- The cannabis activity subquestion is only meaningful if a cannabis exemption was selected in the prior exemption question, but `run_queries` loops over independent `QueryInput` objects with no answer carry-forward.

Impact:

- Subquestions drift away from the intended evidence set and become much more sensitive to retrieval noise.

## Root-cause ranking

1. `short_answer` normalization is rewriting correct or mostly correct answers into wrong benchmark outputs.
2. Parse-stage heading detection misses body structure, which corrupts section ancestry and chunk context across a large region.
3. Relevance filtering is over-pruning useful evidence.
4. Retrieval ranking is too semantic-only for these legal coding questions.
5. Some benchmark labels or semantics appear inconsistent with the local source artifact.

## Recommended fix order

### Priority 1: Fix answer coding before touching models

- Change the completion contract so `short_answer` must use the declared response options exactly, instead of relying on free-text normalization afterward.
- Reduce or remove aggressive post-hoc canonicalization for multi-select and single-choice questions.
- Keep raw model output in the final artifacts for auditability.

### Priority 2: Fix heading extraction and rebuild downstream artifacts

- Extend body heading recognition to cover hyphenated chapter numbers such as `Chapter 9-600`.
- Re-run parse, segment, embed, and index after this fix.
- Validate that the `§ 9-629` chunk lands under the correct title/chapter context path.

### Priority 3: Relax relevance gating

- Stop requiring both score and confidence to exceed the threshold simultaneously.
- Prefer a softer rerank or top-k preservation strategy so one uncertain score does not drop clearly useful legal text.

### Priority 4: Improve retrieval for benchmark-coded variables

- Add lexical or hybrid retrieval for exact anchors like `drug paraphernalia`, `medical marijuana`, `§ 9-629`, and `offer for sale`.
- For dependent benchmark subquestions, carry forward prior answer state or prior evidence instead of treating each row as isolated.

### Priority 5: Clean benchmark expectations

- Review at least the Philadelphia labels for `dp_effective_dt`, `dp_activity`, and `dp_state_fed_reference` against the source law and the benchmark instructions.

## Implementation plan

### A. Benchmark reporting and ground-truth traceability

- Update `coep/src/eval.py` so melted MonQcle rows carry `ground_truth_citation` alongside `ground_truth`, including combined-variable expansions when citation source columns exist.
- Update `coep/scripts/benchmark_pipeline.py` so the benchmark output preserves all processed query rows, adds a `ground_truth_available` / evaluation-status indicator, and reports processed-versus-scored counts explicitly in `benchmark_metrics.json`.
- Keep evaluation scoped to rows with usable ground truth, but merge the evaluation columns back onto the full processed result set before writing the timestamped CSV.

### B. Parse-stage boundary recovery

- Patch `src/legiscope/parse/find_code_start.py` to distinguish genuine body heading chains from TOC listings more conservatively, especially for standalone `TITLE` / `ARTICLE` / `CHAPTER` lines.
- Extend code-start refinement so split heading/body starts count as substantive structure when a section heading is immediately followed by real body prose in the next element.
- Replace the fixed 500-element TOC escape window with a document-aware forward scan so very long tables of contents cannot strand the start boundary inside navigation.
- Treat a transition anchor such as `PREAMBLE` as suspicious when it is immediately followed by a long navigation run, instead of accepting it as a valid body boundary by default.

### C. Scan-stage regex induction stability

- Patch `src/legiscope/parse/scan.py` so chapter refinement accepts hyphenated identifiers like `Chapter 9-600` in addition to simple numeric chapters.
- Reduce sample size after ordinary timeout failures, not just explicit context-length errors, so later iterations are materially different from the first failed prompt.
- Tighten sample diagnostics so counts are internally consistent and do not double-count generic `heading_like` elements.

### D. Retrieval and relevance robustness

- Patch `src/legiscope/retrieve.py` so section retrieval can apply a lightweight lexical rerank on top of semantic results using the active query text and any retrieval-guidance anchor terms.
- Over-fetch semantic candidates when lexical reranking is active so exact-anchor sections still have a chance to be boosted.
- Relax relevance retention so useful sections survive when either graded relevance or confidence clears the threshold, and backfill a small number of top relevant sections when the filter would otherwise collapse to zero or near-zero evidence.

### E. Query answer contract and dependency carry-forward

- Patch `src/legiscope/query.py` so prompt construction explicitly requires `short_answer` to use the declared response-option surface form exactly when structured options are available.
- Make normalization conservative: preserve already valid option text, canonicalize dates and simple `Yes, <citation>` cases, and stop coercing ambiguous free text into unrelated benchmark codes.
- Always retain the pre-normalized short answer in result artifacts when structured normalization is applied.
- Carry forward prior structured answers during `run_queries()` so dependent COEP subquestions can see the earlier exemption/activity outputs.
- Extend `coep/src/retrieval_guidance.py` so exemption-activity subquestions receive dependency context from prior answers in both retrieval and completion guidance.

### F. Regression coverage

- Add parse/code-start regression tests for long TOCs, split heading/body starts, and genuine heading chains before first prose.
- Add retrieval tests for lexical reranking and softened relevance retention.
- Add query tests for exact-option prompting, conservative normalization, raw short-answer preservation, and prior-answer carry-forward.
- Add evaluation tests for `ground_truth_citation` propagation and processed-versus-scored benchmark accounting.

## Expected payoff

If only the answer-coding bug is fixed, the score should improve immediately without any model change.
If the parse/heading bug is then fixed and downstream artifacts are rebuilt, retrieval and relevance quality should improve materially on the paraphernalia questions.
The remaining gap after those fixes will be a cleaner measure of true retrieval/prompt quality rather than artifact corruption.