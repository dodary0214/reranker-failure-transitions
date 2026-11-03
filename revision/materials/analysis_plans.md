# Analysis plans

The plans below are reproduced verbatim from the study repository. Later decisions and departures are recorded in `DEVIATIONS.md`, which takes precedence where the two differ (for example, the repeat-label stability check was labelled by the first author, DEV-15). Local file paths are replaced by `<local path>`.


---

# Revision pre-registration (frozen before the revision experiments)

# REVISION_PREREG — JIIS revision (Reviewer 1 and Reviewer 2)

**Status:** FROZEN before any revision experiment ran: no new model inference, retrieval, QASA mapping or
human labelling has been done.
**Frozen on:** 2026-10-29 · **Submission baseline:** commit `6943ebfb22669500b8e4ccb67364cca32fb28ddc`
(tag `jiis-submitted-v1`).
**Governing documents:** `REVISION_EXECUTION_PLAN.md` (the plan), `FREEZE_DECISIONS.md` (8 decisions),
`PLAN_REVIEW.md` (feasibility review). Where this file is more specific, it governs execution.
Any later change is logged in `revision/DEVIATIONS.md` with a timestamp and the reason, **before** the
affected result is computed. A deviation logged after its result was seen is reported as post hoc.

---

## 1. Frozen inputs

The full list (30 files, sha256) is in `prereg/FROZEN_INPUTS.json`. Key items:

| item | file | sha256 |
|---|---|---|
| submitted manuscript (tex) | `maunuscripts/manuscript.tex` | `ebc0ddbc…40ab43` |
| submitted manuscript (pdf) | `maunuscripts/manuscript.pdf` | `3081c021…e872c` |
| canonical results | `reports/FINAL_CANONICAL_RESULTS.json` | `7bcf1f79…edb3dc` |
| GPT-4o primary semantic labels | `outputs/phase5b_expanded/phase5b_expanded_semantic_states.jsonl` | `3ad49e6c…dd74` |
| GPT-5 sensitivity labels (view B) | `judge_robustness/views/secondary/states_view_B.jsonl` | `abe072df…da45f` |
| six-reranker top-10 rankings | `release_v1/data/reranker_outputs/{cross,within}.jsonl` | `2aac0539…0ff1`, `d136ef68…1f75` |
| original adjudication prompts and config | `frozen_inputs/phase5b_config.py` | `420ebe0b…7237` |
| downstream prompts (this revision) | `revision/prereg/downstream_prompts.py` | `c52be894…4618` |
| QASA Q0 script (this revision) | `revision/qasa/scripts/qasa_q0.py` | `5b290091…af63b` |

QASPER IDs (`prereg/QASPER_COHORT_IDS.json`): all 3,719 queries (`b1e15cfb…a043`), comparable Cross
806 (`feeec307…1f2f`), comparable Within 3,180 (`78452f47…3e20`), sha256 over the sorted newline-joined IDs.

Original adjudication prompt hashes (`phase5b_config.prompt_hashes()`): threeway system `8705b154…`,
threeway user `addead56…`, stage-1 system `3070fb5e…`, stage-1 user `1f35f436…`, stage-2 system
`5a83ae9d…`, stage-2 user `9bd1abd0…`, examples `783fa7ed…` (full values in `FROZEN_INPUTS.json`).

QASA source: `github.com/lgresearch/QASA` commit `caaba4d959d3afb012853bc2526e16d619870397`;
`data/testset_answerable_1554_v1.1.json` sha256 `0b236ea5…54b6`.

The Phase 0 builder is `prereg/scripts/build_freeze.py` (deterministic; re-running it reproduces the sample
files byte for byte).

## 2. Models and runtime (`prereg/RUN_MANIFEST.json`)

| role | model | revision |
|---|---|---|
| rerankers (unchanged) | MiniLM-L6, BGE-v2-M3, BGE-v2-Gemma, Qwen3-RR-4B, FIRST, ReasonRank-7B | as `frozen_inputs/configs/rerankers_locked_v4.yaml` |
| retrieval embedding (unchanged) | BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` |
| semantic local judge | mistralai/Mistral-Small-3.2-24B-Instruct-2506 | `95a6d26c4bfb886c58daf9d3f7332c857cb27b43` |
| downstream generator | meta-llama/Llama-3.3-70B-Instruct | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` |
| downstream judge | Qwen/Qwen3.5-27B | `fc05daec18b0a78c049392ed2e771dde82bdf654` |

- GPUs 0 and 1 (H200 NVL) only. No API. New LLMs in BF16, no quantization. Rerankers and the retrieval
  embedding keep their **locked** dtypes (MiniLM/BGE-v2-M3 float32; BGE-M3 served float16 by vLLM as locked).
- Mistral is served by vLLM in mistral format (`tokenizer_mode`, `config_format`, `load_format` = `mistral`).
  Llama uses tensor parallel 2. Mistral and Qwen use one GPU each.
- Weights live in `<local path>` and are deleted after the phase that uses them; revisions are pinned.
- Software versions (torch, vLLM, transformers, Milvus client) are written into `RUN_MANIFEST.json` when the
  environment is built, before the first GPU run, and are not changed during a phase.

## 3. Evidence layers (never mixed)

| layer | content | semantic labels |
|---|---|---|
| A — original primary | QASPER submission results | GPT-4o |
| B — robustness | strict taxonomy, Mistral view, human subset, full pool, depth 20, top-k, joint bootstrap | GPT-4o (A, O, F), Mistral (B, C, E) |
| C — external / practical | QASA replication, downstream generation and judging | Mistral (QASA); GPT-4o primary + Mistral sensitivity (QASPER downstream) |

Cross-dataset comparison is **Mistral-QASPER vs Mistral-QASA** only. Depth comparisons (top-10 vs top-20) are
made inside the Mistral view so that both depths use one judge.

## 4. Shared definitions

- States `G V C X L`; raw `answers` and `substitutable` → V. Primary grouping A = {G,V,C}, H = {X,L}.
- Transition rates, persistence decomposition, conditional subtype switch, D5/DG/DH and HiddenFraction use the
  submitted definitions (`maunuscripts/manuscript.tex` §3, Eqs. 1–8) and the code conventions of
  `audits/manuscript_alignment_v1/scripts/compute_manuscript_values.py`: upper median `sorted(v)[n//2]`,
  ordered pairs for directional rates, unordered pairs for symmetric disagreement.
- **Missing semantic labels in a revision view:** a model unit whose label is `UNRESOLVED_ADJUDICATION` or
  `error` is treated as missing, as Amendment 04 treats invalid reranker outputs: it is dropped from every pair
  that contains it, never imputed, and counted in the report.

## 5. Experiments

### A — Taxonomy sensitivity (R1-1) · CPU
Remap GPT-4o labels to the strict grouping A = {G,V}, H = {C,X,L}; recompute Resolution, Persistence
(same/alternative passage), Transformation, Regression, harmful disagreement DH and HiddenFraction. Under strict
grouping "harmful subtype" has three values {C,X,L}; Transformation is any change among them. Primary and strict
reported side by side. Outputs `revision/exp_a_taxonomy/results/taxonomy_sensitivity.json`,
`…/tables/taxonomy_sensitivity.csv`. Revised manuscript definitions are worded from what the prompts measured:
Complementary = adds material evidence that strengthens the answer when shown with the annotated evidence;
"harmful" = non-answer-supporting selected evidence (analytical shorthand, not downstream harm).

### B — Mistral re-adjudication of the 5,567 selected non-gold identities (R1-2) · 1 GPU
- Inputs: `judge_robustness/manifests/AUDIT_JUDGE_INPUT_PROJECTION.jsonl` and `AUDIT_REFERENCE_ANSWERS.json`
  (question, candidate text, reference answer, gold text chosen by `GOLD_SELECTION_RULE`).
- Procedure: the **original three-stage cascade, prompts verbatim** (`phase5b_config.py`), the same control flow
  as `judge_robustness/scripts/run_secondary.py`: 3-way → stage 1 (only `valid_alt`) → stage 2 (only
  `substitutable_no`); stage-2 `uncertain` → `UNRESOLVED_ADJUDICATION`. Only the transport changes (local vLLM
  OpenAI-compatible endpoint).
- Decoding: temperature 0, `max_tokens` 250, `response_format` json_object (the GPT-4o settings). Retry policy:
  3 attempts on transport error or unparsable JSON, as `phase5b_config.RETRIES`. One pass (greedy decoding is
  deterministic; the GPT-4o verification pass is not replicated).
- The judge never sees reranker identity, any GPT label, transitions or downstream results.
- Agreement GPT-4o↔Mistral and GPT-5↔Mistral: exact five-state agreement, Cohen's κ, acceptable/harmful
  agreement, harmful-common subtype agreement, class-wise confusion. Mistral view: Resolution, Persistence,
  Transformation, Regression, conditional subtype switch, HiddenFraction, Transformation > 0 pair count.
- Outputs `revision/exp_b_mistral/results/qasper_mistral_adjudication.json`, `…/tables/judge_agreement.csv`.

### C — Complete top-10 pool semantics, QASPER (R1-8) · 1 GPU
- Every candidate in the 3,986 comparable top-10 pools. Gold candidates → G deterministically. Non-gold
  candidates → Mistral cascade (identities already judged in B are reused, not re-judged).
- Reference answers for all comparable queries: QASPER v0.3 (`qasper-train-dev-v0.3.tgz`,
  `qasper-test-and-evaluator-v0.3.tgz`) with the frozen `chosen_answer`/`ans_text` rule of
  `build_audit_references.py`. **Gate:** the 2,623 existing reference answers must be reproduced byte for byte;
  otherwise C stops and the mismatch is reported.
- Per query-arm: complete pool state distribution; random-selection expectation (uniform over pool members);
  reranker-selected distribution; random vs observed harmful rate; harmful-subtype marginal; complete-pool
  independence reference `ps(1−pt)+(1−ps)pt` with ps = pt = Lexical-trap share of harmful pool candidates
  (pooled per arm) and a query-conditioned variant (per-query pool share, event-weighted). Compared with the
  selected-only references (48.4/43.0 % marginal; 37.02/32.76 % query-conditioned).
- Output `revision/exp_c_fullpool/results/qasper_full_pool_semantics.json`.

### D — Single-expert human semantic validation (R1-2) · human
- Sample frozen in `prereg/HUMAN_SEMANTIC_SAMPLE.json` (sha256 `9fb9d660…7e82`): 120 non-gold identities,
  15 per arm × GPT-4o state {V, C, X, L}, seed 42.
- Blind packet (`revision/exp_d_human/human_validation/PACKET_BLIND.csv`) in `random.Random(42)` shuffled order,
  showing exactly the cascade's inputs: question, reference answer, gold passage, candidate passage, and the four
  state definitions. Hidden: every model label, reranker identity, transition, stratum.
- Annotator: the first author, forced choice among V/C/X/L. Wording in the paper: "single-expert human
  validation"; never "gold standard", "consensus" or "inter-annotator reliability".
- Human↔GPT-4o, ↔GPT-5, ↔Mistral: exact agreement, macro-F1 (4 classes), per-class precision/recall/F1 with the
  human label as reference, confusion matrix; V↔C, C↔X and X↔L confusions reported separately.
- Low agreement is reported as assessor/taxonomy sensitivity; the sample is never extended.

### E — Candidate depth 10 → 20 (R1-3, R1-4) · GPU
- **Retrieval reconstruction** under `protocol/EXPANDED_RETRIEVAL_STACK_LOCK.json`: Milvus v2.6.23 (repo
  compose file), BGE-M3 at the pinned revision served float16 by vLLM, HNSW (M 16, efConstruction 100,
  cosine), BM25 sparse index (english analyzer, `drop_ratio_build` 0.2), RRF k = 60.
- **Per-channel depth stays 20** for both pools. The locked recipe ties channel depth to `top_k × 2`; following
  it for top-20 would make depth 40 and could reorder the original top-10 through added RRF terms. Holding 20
  makes the fused top-10 a prefix of the fused top-20, so pool depth is the only intervention.
- **Gate (Decision 3):** ordered exact top-10 match (identities and RRF order) and set match against
  `candidate_pool_manifest.json`. Ordered match ≥ 95 % → E1 on the ordered-matched query-arms;
  < 95 % → matched-subset analysis is exploratory/supplementary only. E2 runs either way.
- Rerankers: the six locked models. Pointwise models score all 20. FIRST: one window of 20 with identifiers A–T;
  **before any GPU run**, a tokenizer check confirms that A–T are single tokens in the format the FIRST runner
  reads; failure stops E until the original implementation is rechecked. ReasonRank: one window of 20. No sliding
  windows. Invalid outputs follow Amendment 04 (no repair). Paper wording: "The listwise inference procedure was
  retained, while the candidate window size increased from 10 to 20 as part of the candidate-depth intervention."
- New non-gold rank-1 identities → Mistral cascade. Metrics in the Mistral view: Acc@1, D5, DG, Resolution,
  Persistence, Transformation, Regression, conditional subtype switch, HiddenFraction.
- E1: query-arms gold-containing at both depths (and passing the gate). E2: query-arms that become
  gold-containing only at top-20; report N₂₀/3,719 for Cross and Within against 806 and 3,180.
- Outputs `revision/exp_e_depth20/results/qasper_depth20_{matched,expanded}.json`.

### F — Beyond rank-1 (R1-4) · CPU
From the frozen rankings (`release_v1/data/reranker_outputs`, rank-1 verified identical to canonical for all
23,914 units): Gold@1/3/5; for every unordered pair Overlap@k = |Top_k(A)∩Top_k(B)|/k and Jaccard@k, k ∈ {3, 5};
on pair-queries with different rank-1: mean/median Overlap@3/5, zero-overlap@3/5, shared-gold@3/5. Pools with
fewer than k candidates use k = pool size. Output `revision/exp_f_topk/results/qasper_topk_diagnostics.json`.

### G — QASA Q0 gate (R1-4) · CPU, before any QASA GPU work
- Script `revision/qasa/scripts/qasa_q0.py` (sha256 above). Inventory → arXiv HTML (`https://arxiv.org/html/{id}`,
  3 s between requests) → segmentation → mapping.
- Segmentation: abstract (heading removed) + every `div.ltx_para` outside the bibliography, document order,
  whitespace collapsed, paragraphs < 5 words dropped; paragraph ID = element id.
- Normalization: NFKC → lowercase → keep `[a-z0-9]`. Mapped = exact normalized match, or
  `difflib.SequenceMatcher(autojunk=False).ratio()` ≥ 0.80. Ambiguous = ≥ 2 exact matches, or accepted fuzzy
  match with best − second < 0.05. Gold = unambiguous mapped paragraphs.
- Exclusions: `MISSING_ARXIV_ID` (no title search or manual recovery), `PAPER_UNAVAILABLE`,
  `NO_UNAMBIGUOUS_GOLD_MAPPED`. Question type → added field `DEEP_COMPLEX | SHALLOW | TESTING | UNKNOWN`
  (original field kept; metadata only).
- **GO_QASA** iff ≥ 90 % of attempted queries have ≥ 1 unambiguous mapped gold **and** ambiguous evidence
  ≤ 5 % **and** human audit ≥ 28/30. Audit sample: 30 seed-42 fuzzy-accepted mappings (all fuzzy plus random
  exact fill if fewer than 30), checked by the first author against the source. Otherwise **STOP_QASA_MAPPING**;
  thresholds are never relaxed and no other dataset replaces QASA.
- **Disclosure:** during dataset scouting (before this plan), one QASA paper (arXiv 2210.01241, 9 questions,
  16 evidence paragraphs) was mapped with a prefix-ratio pilot (16/16 ≥ 0.84). No reranker, judge or retrieval
  output on QASA was produced. The frozen rule above was not tuned on it.

### H — QASA external replication (R1-4) · GPU
- Corpus: all segmented paragraphs of available papers. Cross = whole corpus; Within = the question's paper.
  Retrieval: the E recipe in a separate Milvus collection; top-10 primary, top-20 sensitivity (E's window rule).
- Comparable = pool contains ≥ 1 gold (any-valid). Six locked rerankers. Labels: Mistral cascade with reference
  answer = QASA `composition`, gold passage for stage 2 = earliest gold in RRF order. No GPT call.
- Metrics: comparable rate, Acc@1, composition, D5, DG, Resolution, Persistence (same/alt), Transformation,
  Regression, conditional subtype switch, HiddenFraction; Gold@3/5, Overlap@3/5; the same at top-20.
- Outcome labels: REPLICATION / PARTIAL_REPLICATION / NON_REPLICATION, judged against Mistral-QASPER on the
  qualitative patterns of RQ1–RQ3 (nonzero Transformation in most ordered pairs; D5 well above |ΔAcc@1| for
  low-gap pairs; HiddenFraction > 0 across pairs). Reported whatever the outcome.
- Outputs `revision/exp_h_qasa/results/qasa_{cross_top10,within_top10,depth20,topk}.json`.

### I — Downstream generation (R1-6) · 2 GPUs
Unit = unique `(dataset, query, arm, selected rank-1 passage)`; shared across rerankers. Prompt
`downstream_prompts.generator_user` (hash `fe5ceafb…`), temperature 0, `max_tokens` 256, Llama-3.3-70B TP 2.
Input: question and passage only. QASPER primary, QASA external.

### J — Downstream judging (R1-6) · 1 GPU
Qwen3.5-27B, `enable_thinking=False`, temperature 0, `max_tokens` 512, json_object, 3 retries on error / bad
JSON / out-of-schema value (`downstream_prompts.JUDGE_*`, hashes `637a44d5…`, `9764e3c4…`). Input: question,
all reference answers (QASPER: `ans_text` of every non-unanswerable annotation, de-duplicated; QASA:
`composition`), reference evidence (QASPER gold paragraphs; QASA evidence contexts), **the passage the answer was
generated from**, the answer. Abstentions (`is_insufficient`) skip the judge: INCORRECT, SUPPORTED, abstained.
Metrics: P(CORRECT); U = 1·CORRECT + 0.5·PARTIAL; SUPPORTED rate; auxiliary token F1 / EM / ROUGE-L (SQuAD
normalization, max over references) — secondary only.

### K, L, M — State-, transition- and hidden-disagreement-conditioned utility (R1-6, R2) · CPU
As in the plan (§15–17). State source: QASPER GPT-4o (primary) and Mistral (sensitivity); QASA Mistral.
L uses ΔU_q = U_target − U_source per ordered pair; M covers G↔V, G↔C, V↔C and every other pair that binary
projection hides. A small effect is reported as "evidence/behaviour divergence without demonstrated
answer-quality difference".

### N — Human downstream sanity check · human
QASPER 30 units frozen in `prereg/HUMAN_DOWNSTREAM_SAMPLE_QASPER.json` (sha256 `8d6f4d00…5158`; 6 per GPT-4o
state, Exp D identities excluded, seed 42). QASA 10 units: 2 per Mistral state, seed 42, drawn after QASA labels
exist and **before Qwen judging**. Order: labels → sample freeze → Llama generation → packet → Qwen judging.
The human records CORRECT/PARTIAL/INCORRECT and SUPPORTED/UNSUPPORTED without seeing judge output. Exact
agreement and confusion matrix; described as a sanity check of the LLM judge.

### O — Dependence-aware joint bootstrap (R1-7) · CPU
Per arm, resample query IDs with replacement (B = 2,000, `random.Random(42)`); a drawn query brings all its
model units, so every pair and both directions move together. Each replicate recomputes pair-level rates and
their upper medians. 95 % percentile CI (2.5th/97.5th order statistics). Quantities: median Resolution,
Persistence, Transformation, Regression, conditional subtype switch, HiddenFraction, D5 — GPT-4o view (primary),
Mistral view, and QASA where N allows.

### R2 practical selection and R1-5 citations
No scalar reranker score. Incumbent-relative hierarchy: effectiveness (Acc@1, downstream correct rate) →
Regression → Resolution → Persistence/Transformation → D5 / HiddenFraction / top-k overlap, supported with L.
The three suggested citations are reviewed for research object, heterogeneity definition, method and task, and
cited only if directly related; otherwise the response letter explains the difference.

## 6. Execution order and gates

Phase 0 freeze (this file) → Phase 1 CPU: A, F, QASA G (gate) → Phase 2: B, C, D packet → Phase 3: E (gates:
retrieval reconstruction, FIRST identifier check) → Phase 4: H (only if GO_QASA) → Phase 5: N sample freeze,
I, J, K–M, N → Phase 6: O → Phase 7: `REVISION_CANONICAL_RESULTS.json`, `REVISION_RESULTS_SUMMARY.md`,
`RESPONSE_TO_REVIEWERS_EVIDENCE.md`, `SUPPLEMENTARY_RESULTS/`, manuscript, response letter.

## 7. Unexpected results

Handled exactly as plan §24: STOP_QASA_MAPPING without relaxing thresholds; NON/PARTIAL_REPLICATION reported
without changing dataset; low human or Mistral agreement reported as sensitivity without extending samples or
changing judges; top-20 changes reported as depth sensitivity with top-10 primary unchanged; weak downstream
effects narrow the claim.


---

# QASA external-replication decision rule (DEV-11, frozen before any QASA output)

## DEV-11 — QASA external-replication decision rule (FROZEN by the author 2026-10-30, before any QASA retrieval/reranker/label output)

Prereg §5 H states the replication patterns qualitatively. The author froze the operational rule below before any QASA
result existed. It supersedes the proposal logged earlier the same day. Implemented in
`revision/exp_h_qasa/scripts/08_analyze.py`.

**Per arm.** Cross and Within are judged independently and never combined into one score. The paper reports
"QASA Cross: … / QASA Within: …".

- **C1 — harmful-state transformation prevalence.** PASS iff Transformation > 0 in ≥ 16 of the 30 ordered pairs.
- **C2 — low-aggregate-gap semantic disagreement.**
  - Low-gap pair: an unordered pair with |ΔAcc@1| < 2 pp on the pair-valid cohort.
  - For each low-gap pair, compute D5 − |ΔAcc@1| on the same cohort.
  - PASS iff all three hold: ≥ 1 low-gap pair exists; the conventional sample median of D5 − |ΔAcc@1| over the
    low-gap pairs is ≥ 10 pp (for an even count, the mean of the two middle values; full precision); and
    D5 > |ΔAcc@1| in every low-gap pair.
  - No low-gap pair: C2 = NOT_EVALUABLE.
- **C3 — binary projection loss.** PASS iff both hold:
  - HiddenFraction > 0 in 15/15 unordered pairs.
  - The arm median HiddenFraction ≥ 0.5 × the Mistral-QASPER arm median. The reference uses full precision from
    `exp_b_mistral/results/bootstrap_mistral.json` point estimates: Cross 0.5597014925 → threshold 0.2798507463;
    Within 0.4996096799 → threshold 0.2498048400. The script asserts that the same code reproduces these values on
    `states_view_M.jsonl`.

**Arm label** (author clarification 2026-11-02, before any QASA output). NOT_EVALUABLE is neutral: it counts as
neither PASS nor FAIL and cannot rescue a failed criterion.
- REPLICATION = C1, C2 and C3 all evaluable and PASS.
- NON_REPLICATION = every evaluable criterion FAILs (e.g. C1 FAIL + C2 NE + C3 FAIL).
- PARTIAL_REPLICATION = every other case (e.g. C1 PASS + C2 NE + C3 PASS, or any PASS/FAIL mix).

**Reporting.**
- Both arms REPLICATION → "external replication confirmed in both retrieval conditions".
- Exactly one arm REPLICATION → "condition-dependent or partial external replication".
- Differences between the arms are reported as such.
- Mistral-QASPER is used only as the threshold source and as context. Its own C1–C3 status is computed but not used
  for scoring. The earlier proposal to skip criteria that fail in Mistral-QASPER is withdrawn.

**Implementation choices.**
- Medians: C2 uses the conventional sample median (author decision 2026-11-02). The C3 arm median HiddenFraction and
  all reported metric medians keep the manuscript's upper median, sorted(v)[n//2], so that they match the Mistral-QASPER
  reference values.
- Corpus: all 7,311 segmented paragraphs of the 106 available papers, with alttext text (DEV-10) and frozen IDs. This
  includes the 88 short math-only paragraphs.
- Gold = QASA evidence annotation status (any unambiguous mapped evidence paragraph).
- The stage-2 single gold reference = the earliest gold in the top-20 RRF order. It is recorded per query-arm in
  `exp_h_qasa/outputs/STAGE2_GOLD_MANIFEST.json`.
- Depth sensitivity: nested top-10 vs top-20 of the same retrieval result. The top-20 cohort is scored with the same
  C1–C3 as a sensitivity analysis, not as the primary verdict.
- Gate record: the author's audit result (≥ 28/30 → GO_QASA) is written into `q0_mapping_quality_report.json` `gate`.
- After QASA results are seen, the C1–C3 thresholds, the low-gap definition and the aggregation rule are not changed.



---

# Secondary downstream evaluator (Exp N2, frozen before any Gemma judgment)

# Exp N2 — Cross-family secondary downstream judge (DEV-14)

Frozen 2026-11-02, before any Gemma judgment exists. Committed together with DEV-14 and the scripts in `scripts/`.
Replaces Exp N (40-item human downstream sanity check). Exp N status: `NOT_EXECUTED_REPLACED_BY_DEV14`.

## 1. Purpose
Reviewer 1-6 asks whether semantic-state disagreement relates to downstream QA quality. Exp I–M answered this with one
LLM evaluator: Qwen3.5-27B. Exp N2 checks whether those conclusions depend on that evaluator. It re-judges the same
answers with a judge from another model family. It is a sensitivity analysis. Qwen3.5-27B stays the primary judge, and
Gemma results never replace Qwen results.

## 2. Secondary judge
| setting | value |
|---|---|
| model | `google/gemma-3-27b-it`, revision `005ad3404e59d6023443cb575daa05336842228a` |
| inference | local only; vLLM 0.28.0, transformers 5.16.1 (`<local path>`); 1 × H200 (GPU 0), TP 1 |
| precision | BF16; `max_model_len` 16,384; server seed 0 |
| decoding | `downstream_prompts.JUDGE_DECODING` without `chat_template_kwargs`: temperature 0 (greedy, = `do_sample=False`), `max_tokens` 512, `response_format` json_object |
| prompt | `downstream_prompts.JUDGE_SYSTEM` + `judge_user`, byte-identical to the primary run (hashes `637a44d5…18bd`, `9764e3c4…8e1d`; file `c52be894…4618`) |
| retry rule | up to 3 attempts (`JUDGE_RETRIES`) on transport error, invalid JSON or out-of-schema value; then `error`, kept missing and never imputed |
| abstentions | `downstream_prompts.ABSTENTION_RULE`, as in the primary run: an `INSUFFICIENT` answer is not sent to the judge and gets INCORRECT / SUPPORTED |
| output schema | `{"correctness": "CORRECT|PARTIAL|INCORRECT", "faithfulness": "SUPPORTED|UNSUPPORTED", "rationale": "..."}` |
| utility | U = 1 (CORRECT), 0.5 (PARTIAL), 0 (INCORRECT); `downstream_prompts.UTILITY` |

- **Dropped setting:** `enable_thinking=False` is a Qwen-only chat-template switch; Gemma has no thinking mode.
- **System prompt:** Gemma's chat template has no system role. It prepends the system text to the user turn.
- **Manifest:** model revision, runtime versions, prompt hashes, decoding, retry rule and input/output hashes are written
  to `RUN_MANIFEST.json`.

## 3. Population
- **No sampling.** Every generated answer that the primary run judged is re-judged:
  - QASPER: 9,314 units.
  - QASA: 4,958 units.
  - Total: 14,272.
- **Answers:** the existing Llama-3.3-70B answers are reused unchanged. No new generation.
- **Abstentions** (`INSUFFICIENT`) are kept in the population and handled by the abstention rule above.

## 4. Inputs to the judge
- **Gemma sees:** the question, all reference answers, the reference evidence, the passage the answer was generated
  from, and the generated answer. These come from the unit and answer records only.
- **Gemma never sees:** Qwen labels or rationales, GPT-4o or Mistral states, reranker or source/target model,
  transition type, K/L/M results, or any disagreement expectation.
- **How this is enforced:** `judge_gemma.py` reads only the units file and the answers file.

## 5. Evaluator agreement (Qwen vs Gemma)
- **Correctness:** exact agreement, Cohen's κ, linear weighted κ (INCORRECT < PARTIAL < CORRECT), and a 3 × 3
  confusion matrix.
- **Faithfulness:** exact agreement, Cohen's κ, and a 2 × 2 confusion matrix.
- **Groups:**
  - overall
  - QASPER, QASA
  - QASPER Cross, QASPER Within, QASA Cross, QASA Within
- **Populations:**
  - `judge_called`: non-abstained units, the ones either model actually judged. This is the informative population.
  - `all`: every unit. Abstentions carry identical rule labels in both runs, which inflates agreement.
- **Errors:** units with a judge error in either run are excluded and counted.
- **No judge replacement:** low agreement does not lead to replacing the secondary judge.

## 6–8. K, L, M recomputed with Gemma
`revision/exp_k_l_m/scripts/downstream_analysis.py` runs unchanged with the Gemma judgments. The units, answers, state
views, grouping (primary) and joint query bootstrap (B 2,000, seed 42) are the same as in the primary run.

**Views:**
- QASPER, GPT-4o states → Gemma judge
- QASPER, Mistral states → Gemma judge
- QASA, Mistral states → Gemma judge

**What is reported:**
- **K:** U(G), U(V), U(C), U(X), U(L) per arm.
  - K1: whether the ordering G > V > C > X > L holds.
  - K2: U(C) − U(X) with its bootstrap 95 % CI.
- **L:** for Resolution, Regression, Persistence and Transformation: mean and median |ΔU|, positive and negative
  utility-change rates, and correctness flip rates.
- **M:** for each pair group below: mean and median |ΔU| and correctness-disagreement rate.
  - acceptable-internal pairs G↔V, G↔C, V↔C
  - harmful-internal pair X↔L
  - coarse acceptable ↔ harmful flips
- **Fixed inputs:** Gemma results never change semantic labels. Automatic metrics (token F1, EM, ROUGE-L) stay as they
  are and are auxiliary only.

## 9. Primary vs secondary comparison (qualitative; no PASS/FAIL gate)
Each item is evaluated per view × arm (6 cells) for Qwen and for Gemma, with these fixed rules:
- **R1:** the point estimates of U over unique units follow the strict order G > V > C > X > L.
- **R2:** U(C) − U(X) > 0, and the lower bound of its bootstrap 95 % CI is > 0.
- **R3:** pooled mean |ΔU| of Resolution (= Regression by symmetry) > pooled mean |ΔU| of Transformation.
- **R4:** each acceptable-internal pair (G↔V, G↔C, V↔C) has mean |ΔU| greater than both X↔L and Transformation.
  The coarse-flip mean |ΔU| is reported alongside for order-of-magnitude comparison.

An item is *maintained* when Gemma reproduces Qwen's outcome in all 6 cells. Wording:
- **All 4 items maintained:** "evaluator-robust".
- **More than 2 items reversed** (Qwen held, Gemma did not) in more than half of the cells: "downstream conclusions
  are evaluator-dependent".
- **Otherwise:** "evaluator-sensitive but partially robust".

This wording describes the result; it is not a gate. No composite score or new threshold is introduced.

## 10. Prohibited after Gemma output exists
- Changing the prompt and re-running.
- Changing the model or adding a third judge.
- Selecting a subset, or dropping items where Qwen and Gemma disagree.
- Changing the PARTIAL definition or the semantic labels.
- Discarding Qwen results.

Differences are reported as evaluator sensitivity.

## 11. Outputs
- **This folder:** `RUN_MANIFEST.json`, `gemma_judgments.jsonl` (labels only), `judge_agreement.json`,
  `judge_agreement_confusion.csv`, `exp_k_gemma_state_utility.csv`, `exp_l_gemma_transition_utility.csv`,
  `exp_m_gemma_hidden_utility.csv`, `results/*_gemma.json`, `tables/`, and `SECONDARY_JUDGE_SUMMARY.md`.
- **Run dir:** full records with rationales in `$REVISION_RUN_DIR/exp_n2/gemma_judgments_full.jsonl`. Rationales are
  for audit only and are not used in any metric.

## 12. Use in the paper
- **Primary downstream results** stay Qwen3.5-27B.
- **Main text:** if the results are consistent, one or two sentences pointing to Supplementary S8.
- **Supplementary S8:** agreement, confusion matrices, full Gemma K/L/M, and the R1–R4 comparison.
- **Human labels:** no single-author 40-item human downstream annotation is presented as human gold.


---

# Repeat-label stability check (DEV-15; written before the labels were returned)

# Exp D second annotator — design and analysis plan (DEV-15)

Written 2026-11-03, before the packet was sent and before any second-annotator label existed. Freeze this file, the
packet and both scripts (commit) before the packet goes to the annotator.

## Why
Exp D labels are AI-assisted, author-expert-reviewed (DEV-12: Claude Fable 5.1 proposals, first author reviewed all
120, changed 5). Agreement with those labels does not by itself establish correctness. A blind human second judgment of
the contested items tests whether the reviewed label holds where the primary judge (GPT-4o) disagrees with it.

## Items (82)
- **Disagreement items (62):** all Exp D items whose GPT-4o label differs from the expert-reviewed label. Composition by
  reviewed label: X 39, L 12, C 8, V 3; by arm: Cross 30, Within 32.
- **Filler items (20):** agreement items, drawn from the 58 agreement items by proportional stratified sampling on the
  label (V 4, C 3, X 9, L 4; `random.Random(2026)`). They hide which items are contested and give a check on the
  annotator's reading of uncontested items.
- Order shuffled (`random.Random(15)`), new IDs S01–S82. The key (S-id → first-round D-id, role) is kept outside git:
  `$REVISION_RUN_DIR/exp_d_human/SECOND_PACKET_KEY_PRIVATE.json`.

## Blinding
- **Packet content:** the packet shows only the question, the reference answer, the annotated evidence passage and the
  candidate passage. It shows no model label, reviewed label, Claude proposal, stratum, arm or item role.
- **What the annotator is told:** the selection rule is not disclosed. The annotator is told only that items are in
  random order and that there is no expected label distribution.
- **Label definitions:** identical to the first round (`human_validation/LABEL_DEFINITIONS.md`).
- **Reading aid:** machine translation only. No AI system may be asked for a label, for whether the passage answers the
  question, or for which sentence is relevant. Translation use is recorded per item.
- **First author:**
  - does not discuss items with the annotator before all labels are returned;
  - does not see any returned label before the analysis script runs on the complete file.

## Pre-specified analysis (`scripts/analyze_second_annotator.py`)
**Primary.** On the 62 disagreement items, report the share of items where the second annotator's label:
- equals the reviewed label ("supports reviewed");
- equals the GPT-4o label ("supports GPT-4o");
- equals neither.

Each share is reported as k/n with a Wilson 95% interval.

**Breakdowns:**
- the same three shares by confusion type (GPT-4o label → reviewed label);
- on the binary scale (useful V/C vs not useful X/L), restricted to items where GPT-4o and reviewed differ on that
  scale.

**Secondary:**
- agreement with the reviewed label on the 20 fillers;
- second annotator vs GPT-5 and vs Mistral on the 62 items;
- second annotator vs reviewed on all 82 (exact, Cohen's κ, 4×4 confusion). This is descriptive: 62 contested items
  plus fillers is not a random sample.

## Fixed rules
- Every result is reported as computed. There is no threshold and no "pass/fail".
- The reviewed labels are not changed in light of the second annotator. Exp D agreement statistics and every headline
  result stay as they are.
- There is no relabelling after unblinding and no item exclusion other than missing or invalid labels. Those are
  listed, not imputed.
- If the annotator cannot be recruited, nothing is run. The paper keeps the DEV-12 framing and adds a limitation
  statement.

## Wording for SI S3.3 (fill with the computed numbers)
> To test the reviewed labels where they matter most, a second annotator (co-author, blind to all model labels, to the
> reviewed labels, to the AI pre-annotations and to the selection rule) independently labelled the 62 items on which
> GPT-4o disagreed with the reviewed label, mixed with 20 randomly drawn agreement items. The only reading aid was
> machine translation; no AI system was consulted for labels. The second annotator agreed with the reviewed label on
> k/62 (xx %, 95% CI …) of contested items and with GPT-4o on k/62 (…); k/20 fillers matched.

## Frozen inputs (sha256)
- `PACKET_SECOND_BLIND.csv` `c18c24f983259d214eb02550d7375a8a47853873b674384c7f6d44db577473e4`
- `SECOND_PACKET_KEY_PRIVATE.json` `36848a4e90d42a19113b3a68c393fae28573463477baa65ae62c6f4d2d0746fa`
- `INSTRUCTIONS_SECOND_ANNOTATOR.md` `dfcc0a2e414810e8c7bf329187fa40936d16c7a56893bbfe4db9cf3e521267b9`
- `human_validation/LABELS_EXPERT_REVIEWED.csv` `5ef3f622f466252f1f8c7624d6ac2913c6126bad9dce9c720bf5f0b0f11d589f`
