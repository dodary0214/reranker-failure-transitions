# Deviations from REVISION_PREREG.md

> File paths in this log refer to the study repository, not to this release; the released counterparts are listed in `../data/README.md`.

Each entry is written **before** the affected result is computed, unless marked post hoc.

## DEV-01 — Exp E: top-10 re-run control on the revision hardware (addition)

- **Logged:** 2026-10-29, before any reranker, retrieval or depth-20 result exists.
- **Reason:** the frozen top-10 rankings were produced on 2× A100 80GB (`rerankers_locked_v4.yaml`
  `environment.gpus`); the revision runs on H200 NVL with the same locked software versions
  (torch 2.6.0+cu124, transformers 5.16.1, tokenizers 0.23.1, accelerate 1.14.0). Floating-point kernels can
  differ across GPU generations, so a top-20 run on H200 compared with the A100 top-10 would mix hardware drift
  with the depth intervention.
- **Addition:** before the top-20 run, re-run the six rerankers on the frozen top-10 pools of every comparable
  query-arm on H200 with the locked runtime and report rank-1 agreement with the frozen rankings per model.
- **Use:** E1 as preregistered (frozen top-10 vs new top-20) is reported unchanged. As a sensitivity analysis,
  E1 is also reported with the H200 top-10 re-run as the baseline, so that both depths share hardware. The
  re-run never replaces the frozen top-10 results (§1.1 of the plan).
- **Status (updated 2026-11-07):** after the reconstruction gate failed, E1 and E2 are not reported in the paper;
  depth sensitivity is reported only through the nested comparison of DEV-06.

## DEV-02 to DEV-05 — Exp E retrieval reconstruction: environment substitutions

- **Timing:** written into `revision/exp_e_depth20/README.md` at 2026-10-29 10:49Z, after the corpus ingest and
  before any pool was generated or compared. Copied here after the reconstruction gate result was known; no
  retrieval parameter was changed after that result.
- **DEV-02 MinIO image.** `minio/minio:RELEASE.2024-05-28T17-19-04Z` (locked) is no longer published (Docker
  Hub/quay.io absent, dl.min.io 410). Used the Milvus project's mirror of the same release,
  `milvusdb/minio:RELEASE.2024-05-28T17-19-04Z` (digest `sha256:648817f3…5fda33`, vendor label MinIO Inc,
  vcs-ref 433dec61), through an override file that changes only the image; the compose file is unchanged
  (sha256 = lock). MinIO is object storage for Milvus segments and does not take part in scoring.
- **DEV-03 Missing monorepo modules.** `milvus.embedding.VLLMEmbeddings` (lock sha `943b425e…`),
  `_classify_section`, `milvus.backend`, `benchmark_config`, `logging_utils` are not available. Stand-ins in
  `revision/exp_e_depth20/scripts/common.py`: an OpenAI-compatible embedding client (batch 64, L2-normalised);
  `section_role = "unknown"` (no retrieval step reads it). The vendored `create_collection`/`process_paper` run
  unchanged; all 79,965 passage IDs and texts reproduce pid2content exactly and the paragraph-ID hash matches
  the lock.
- **DEV-04 langchain_milvus shim.** langchain-milvus 0.3.3 expects an ORM connection alias that pymilvus 2.6.17
  no longer registers; a shim registers it. Inserted data and the insert call are unchanged.
- **DEV-05 Runtime.** H200 NVL instead of A100; vLLM 0.28.0 as locked, but with torch 2.13.0+cu130 and
  transformers 5.21.1 (the versions vLLM 0.28.0 installs today; the lock does not record the embedding env's
  torch/transformers).

## DEV-06 — Exp E: same-retrieval top-10 vs top-20 comparison (addition, post-gate, approved by the author)

- **Logged:** after the reconstruction gate result was known; before any reranker, judge or depth result exists.
- **Addition:** run the six locked rerankers also on the **regenerated** top-10 pools (the first 10 of each
  regenerated top-20 list, by construction) for every query-arm that is gold-containing at regenerated top-10.
  Compare regenerated top-10 vs top-20 on the query-arms gold-containing at both, in the Mistral view, with the E
  metrics. Both depths then share retrieval run and hardware; only pool depth differs.
- **Status in the paper:** depth sensitivity (Layer B). After the reconstruction gate failed, the paper reports this
  nested comparison (SI S5.2, Table S14) and does not report E1 or E2; E1 was demoted to exploratory by the gate,
  and the nested comparison isolates pool depth more cleanly than E2. The submitted top-10 results stay primary.
  (Status updated 2026-11-07 to match the reported analyses.)

## Reconstruction gate outcome (recorded, not a deviation)

Ordered exact top-10 match on the 3,986 comparable query-arms = 2,472 (62.0 %) < 95 % → per FREEZE_DECISIONS §3,
E1 is **exploratory/supplementary only** (ordered-matched subset: Cross 120, Within 2,352); E2 runs. The rebuilt
stack is deterministic (repeat run 100 % ordered match); Cross mismatch is dominated by HNSW approximation
(`param={}`), consistent with the original stack's own 97.5 % ordered repeatability (Phase-0 audit, n = 400).
Details: `revision/exp_e_depth20/outputs/retrieval_reconstruction_report.json`.

## DEV-07 — Storage policy (operational)

- **Logged:** 2026-10-30. Free space on `/data` rose from 184 GB to 871 GB (another user freed space), so model
  weights are kept after their phase instead of deleted (`RUN_MANIFEST.json` `storage.policy`). Revisions remain
  pinned; this affects no result.

## DEV-08 — vLLM environment: transformers pinned to 5.16.1 (operational)

- **Logged:** 2026-10-30, before any Mistral judgment was produced.
- vLLM 0.28.0 (locked) accepts `transformers>=5.5.3`; pip installed 5.21.1, which renamed
  `PixtralRotaryEmbedding`, so vLLM could not load Mistral-Small-3.2 (Pixtral architecture). Pinned
  transformers 5.16.1, the release current when vLLM 0.28.0 shipped and the version in the reranker lock.
  The Exp E BGE-M3 embedding had already run under 5.21.1 (DEV-05); every judge, generator and later run uses
  5.16.1. Versions: `revision/RUNTIME_ENVIRONMENTS.json`. Software versions were recorded after the first GPU
  run (the embedding), not before it as §2 of the prereg asked.

## DEV-09 — Exp E reranker runtime substitutions (logged before any depth or judge result)

- **Gemma scoring adapter.** `experiments/_gemma_score_adapter.py` (Amendment 02, sha256 d676b0ed…) is not
  available. Stand-in `revision/exp_e_depth20/scripts/gemma_adapter_standin.py` follows the Amendment 02
  contract (FlagEmbedding 1.4.2 `FlagLLMReranker`, `devices=['cuda:0']`, `use_fp16=True`,
  `compute_score(max_length=512, normalize=False)`, pinned snapshot). Byte identity cannot be checked; the R1
  (frozen top-10) agreement with the release rankings is reported as the check.
- **Packages added** at Amendment 02 snapshot versions: FlagEmbedding 1.4.2, peft 0.20.0 (`--no-deps`),
  datasets 5.0.1 (+ deps); core packages held by a constraints file. Python 3.12.3 (lock 3.12.9), driver
  580.178.04 (lock 580.159.03).
- **`run_smoke.py` hash.** The vendored copy hashes to df0e29f6…, while `frozen_inputs/_hashes.json` records
  8230da36… for the original (likely the vendoring path header). Recorded, not enforced; `reasonrank_messages`
  matches the lock description.
- **Concurrency.** ReasonRank runs as 12 single-device shards (original: 2), several processes per GPU; each
  process holds its own single-device model copy (lock device policy), batch 1, decoding unchanged.
  Scheduling change 2026-10-30 04:42Z (`chain_r3_rebalanced.sh`): GPU 0 sat idle after its regen_top10 shards
  finished, so top20 started there (lane A + ReasonRank shards 0-5) while regen_top10 shards 6-11 finished on
  GPU 1; top20 shards 6-11 follow on GPU 1. Process placement only.
- **Observed (DEV-01 control, partial):** float32 encoders reproduce the A100 rankings exactly (MiniLM,
  BGE-v2-M3: 100 % full-ranking identity); bf16 models drift on H200 (Qwen3 rank-1 agreement 98.0 % on 3,581
  units; ReasonRank 89.6 % on 890 units). Depth comparisons therefore use the H200 runs on both sides
  (DEV-01 sensitivity, DEV-06).

## DEV-10 — QASA arXiv paragraph text: formulas extracted twice (found before any audit label)

- **Issue (raised by the first author while reviewing the audit page).** `segment()` uses
  `BeautifulSoup.get_text()` on `div.ltx_para`; arXiv's LaTeXML HTML holds each formula as MathML rendering plus
  a LaTeX `<annotation>`, so both were extracted ("f ( S ) f(S)", "𝒞 S \mathcal{C}_{S}"). QASA evidence carries
  LaTeX only. The frozen matcher therefore scored math-bearing paragraphs low, and the audit page showed noisy text.
- **Sensitivity check** (`revision/qasa/scripts/q0_math_sensitivity.py`, output
  `revision/qasa/outputs/q0_math_sensitivity.json`). Each `<math>` was replaced by its `alttext`, with every
  other frozen rule kept. Outcome transitions:
  - exact → exact: 1,432
  - fuzzy → exact: 789
  - fuzzy → fuzzy: 386
  - none → fuzzy: 2
  - none → none: 203
- **What changed and what did not:**
  - All 2,607 frozen mappings point to the **same paragraph** under alttext (0 changed best IDs).
  - The ambiguous count is unchanged at 8.
  - Gate rates are unchanged (+2 mapped evidence paragraphs, both unmapped in the frozen run).
  - 88 short math-only paragraphs (21 papers) fall below the 5-word filter under alttext. None of them is a
    mapped paragraph.
- **Decision.**
  - The frozen mapping, cohort and gate statistics stay as they are. Q0 is not re-run, because its decisions
    are identical.
  - Display only: `revision/qasa/scripts/build_audit_packet.py` rebuilds `MAPPING_AUDIT_PACKET.csv` from the
    frozen seed-42 sample. Items, order and mapped IDs are the same; the arXiv text is the alttext version.
  - The labelling page was republished (version 2) before any label was saved. The label store was empty when
    checked.
- **Consequence for the audit.** Under alttext, 22 of the 30 frozen audit items are exact normalized matches;
  only 8 are fuzzy. The prereg sample ("seed-42 fuzzy-accepted") was inflated by the extraction artefact, so it
  tests genuine fuzzy matching on 8 items only. This is disclosed alongside the audit result.
- **Downstream.** QASA retrieval and judge passages (Exp H/I) will be built from the alttext segmentation, with
  the frozen paragraph IDs. Text that a model reads should not contain duplicated formulas.

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

## DEV-12 — Exp D and QASA mapping audit labels are AI-assisted (logged 2026-11-02, after labelling, before any agreement was computed)

Prereg §5 D/G specify blind single-expert labels with model labels hidden. What actually happened:
- **Proposals:** Claude Fable 5.1 (`claude-fable-5-1`) proposed a label, a confidence and a step-by-step rationale for
  all 120 Exp D items and all 30 mapping-audit items. It worked from the same blind packets, with no GPT-4o, GPT-5 or
  Mistral labels and no strata.
- **Review:** the first author reviewed every item and chose the final label.
  - Exp D: 5/120 labels changed (D002 X→C, D012 X→C, D035 X→C, D043 C→X, D045 C→V). In every case the new label is the
    one Claude listed as its alternative.
  - Mapping audit: 0/30 changed.
- **Expert–Claude agreement:** 95.8 %.
- **Files:** both workbooks are kept outside git (`$RUN/human_labels_ai_assisted/`; sha256 final `bfd66a81…`,
  proposals `76dfc034…`). Imported by `exp_d_human/scripts/import_labels.py`.
- **Author decision (2026-11-02):** use these labels (option a).
- **Wording in the paper:** "AI-proposed, author-reviewed reference labels" (Claude Fable 5.1 pre-annotations; all items
  reviewed; 5/120 changed). Never "independent" or "blinded human annotation". (Wording updated 2026-11-07.)
- **Roles:** Exp B provides cross-family model robustness; Exp D provides expert review of model-assisted
  pre-annotations.
- **Scope of the numbers:** Exp D agreement statistics are results on the GPT-4o-stratified validation sample, not
  population-level accuracy estimates.
- **Interpretation:** the labels are effectively a fourth judge family (Anthropic) with expert sign-off. Claude is not
  the family of GPT-4o, GPT-5 or Mistral, so the comparison stays cross-family.
- **Sensitivity:** agreement with the unreviewed Claude proposals as the reference was planned but is not reported;
  the paper uses only the author-reviewed labels as the reference (status updated 2026-11-07).
- **Mapping audit:** 30/30 CORRECT, including M21, a version-revised sentence. The GO decision does not depend on M21
  (29/30 ≥ 28).
- **GO_QASA:** approved by the author on 2026-11-02 on all three preregistered gates: mapped-gold query coverage
  94.3 % ≥ 90 %, ambiguous 0.28 % ≤ 5 %, audit 30/30 ≥ 28/30. Recorded in `q0_mapping_quality_report.json` `gate`.

## DEV-13 — Exp H GPU scheduling (operational, logged 2026-11-02 before any QASA reranker output)
At launch, GPU 1 was fully occupied by another user's vLLM processes (≈ 139 GB). Those processes were not
touched. Changes:
- BGE-M3 embedding (runbook step 2a) ran on GPU 0 instead of GPU 1.
- Reranking uses `exp_h_qasa/scripts/chain_qasa_rerank_flex.sh` instead of `chain_qasa_rerank.sh`. Runs, shards,
  code, seeds and models are the same; only the dispatch differs.
  - Lane A runs on GPU 0.
  - The 12 ReasonRank shards come from a queue with at most 6 concurrent shards per GPU.
  - A shard goes to GPU 1 only when GPU 1 has < 20 GiB in use; otherwise it waits for a GPU 0 slot.
- GPUs 2/3 are never used.
- Effect: wall time only (longer if GPU 1 stays busy). Rankings do not depend on which device runs them, beyond the
  H200 nondeterminism already covered by DEV-01.
- Retrieval sanity checks: repeat top-10 is ordered-identical for 1,390/1,390 queries, top-20 has top-10 as its prefix
  for 1,390/1,390, and Within has no candidate from another paper. Reranker units: 2,393 (top-10) / 2,522 (top-20).

## DEV-14 — Replace 40-item human downstream sanity check with cross-family downstream evaluator sensitivity (author decision; logged and committed 2026-11-02 before any Gemma judgment)
- **Not executed:** the human review of Exp N (prereg §5 N; 30 QASPER + 10 QASA units) is not performed.
  Exp N status: `NOT_EXECUTED_REPLACED_BY_DEV14`.
- **Reason:** for specialist scientific-QA answers, a single author's manual judgment of the CORRECT/PARTIAL boundary
  cannot be made reliable enough. A small, uncertain human label set would add measurement error to the downstream
  validation rather than reduce it. Reviewer 1-6 asks whether semantic-state disagreement relates to downstream QA
  quality. It does not ask for human downstream annotation as such.
- **Preserved:** the Exp N packets, keys and sample IDs are kept unchanged and not deleted:
  - `prereg/HUMAN_DOWNSTREAM_SAMPLE_{QASPER,QASA}.json`
  - `exp_n_human_downstream/human_validation/PACKET_BLIND_{QASPER,QASA}.csv`
  - `$REVISION_RUN_DIR/exp_n_human_downstream/PACKET_KEY_PRIVATE_*.json`
- **Primary unchanged:** the primary downstream analysis keeps the Qwen3.5-27B results.
- **Replacement, Exp N2:** `google/gemma-3-27b-it` (revision `005ad340…228a`) is a secondary sensitivity judge from
  another model family. Its results never replace the primary results.
  - Design, settings and comparison rules: `exp_n2_secondary_downstream_judge/PREREG_EXP_N2.md`.
- **Population:** no selection based on Qwen results. Gemma re-judges every unit of the existing downstream
  evaluation: QASPER 9,314 and QASA 4,958, abstentions included.
- **Blinding of the judge:** Gemma cannot see Qwen labels or rationales, semantic states, reranker identity, or
  transition type.
- **No adjustments after results:** whatever the outcome, there is no model change, prompt change, subset reselection,
  or search for an additional judge.
- **Execution record (2026-11-02, after the commit above):** 14,272 judged, 1 Gemma judge error (kept missing), 0
  retries. Nothing in the frozen design was changed.
  - `RUN_MANIFEST.json` additionally notes that vLLM loaded Gemma's `generation_config.json` (top_k 64 / top_p 0.95)
    as server defaults. These have no effect under temperature-0 greedy decoding.
  - Result: `exp_n2_secondary_downstream_judge/SECONDARY_JUDGE_SUMMARY.md` (evaluator-robust).

## DEV-15 — Exp D blind second annotator on the GPT-4o disagreements (addition; logged 2026-11-03 before the packet was sent)
- **What:** a co-author independently labels the 62 Exp D items where GPT-4o differs from the expert-reviewed label,
  mixed with 20 stratified agreement fillers (82 items, shuffled, new IDs S01–S82).
- **Blinding:**
  - The packet shows no model, reviewed or Claude label, no stratum, no arm and no role.
  - The selection rule is not disclosed to the annotator.
  - The only reading aid is machine translation. No AI system is consulted for labels or relevance.
- **Analysis:** pre-specified in `exp_d_human/human_validation_second/ANALYSIS_PLAN.md`. The primary result is the
  share of the 62 items where the second annotator supports the reviewed label vs the GPT-4o label (Wilson 95%).
- **Reviewed labels unchanged:** the result does not change the reviewed labels, the Exp D statistics or any headline
  result. It is reported as computed, with no threshold.
- **Not run:** if the annotator cannot be recruited, nothing is run.
- **Files:**
  - `exp_d_human/scripts/build_second_annotator_packet.py`
  - `exp_d_human/scripts/analyze_second_annotator.py`
  - `exp_d_human/human_validation_second/`
  - key outside git: `$REVISION_RUN_DIR/exp_d_human/SECOND_PACKET_KEY_PRIVATE.json`
- **Execution record (2026-11-03):** the labelled workbook was returned twice.
  - **Final** `$REVISION_RUN_DIR/exp_d_human/second_annotator/LABELS_SECOND_RAW_v2.xlsx`, sha256 `d1ffb727…039e539`.
    It has 82/82 valid labels, and the packet text is unchanged (checked cell by cell).
  - **First** `LABELS_SECOND_RAW.xlsx` (`77cebb96…6044457`) was identical except for an extra scratch sheet that
    repeated S27–S82 with confidence. Its labels matched 56/56. v2 drops that sheet. Labels, notes and every other
    column are identical across the two files.
  - The confidence and translation columns are blank in the final file.
  - Label export: `human_validation_second/LABELS_SECOND_ANNOTATOR.csv`.
  - The packet and plan were not committed before the packet was sent.
  - The pre-specified script ran unchanged (sha256 `34b52f4f…`) and gives the same result on both files:
    `exp_d_human/results/second_annotator_agreement.json`.
- **Annotator (recorded 2026-11-03, after the labels were returned):**
  - **Planned vs actual:** the plan specified a blind second annotation by a co-author. The labels were in fact
    assigned by the first author.
  - **Redefinition:** the check is therefore a first-author repeat labelling (test–retest), not an independent second
    annotation. Paper and SI wording: "an intra-annotator stability check, not an independent second annotation".
  - **Release field name:** `first_author_second_pass` in `revision/data/reference_sample_120.jsonl` of the public
    release.
  - **κ:** the exact κ over the 82 items is 0.5148, i.e. 0.51 at two decimals.

## DEV-16 — Exp C: without-replacement candidate-pool reference for subtype switching (addition, post hoc; code added 2026-11-04, logged 2026-11-07)

- **What:** for each pair–query event in which two rerankers select different harmful passages, a pool with h
  harmful candidates of which l are Lexical traps gives the probability 2l(h−l)/[h(h−1)] that two distinct uniform
  draws differ in subtype; the reference is the mean over the events that define the observed pooled conditional
  subtype switch (Mistral view; pools with h < 2 excluded).
- **Why:** the two selections are necessarily distinct passages, whereas the prespecified 2p(1−p) references
  (pooled per arm; query-conditioned, event-weighted) treat the draws as independent with replacement.
- **Status:** added after the complete-pool labels were obtained. Reported in SI Table S13 together with the
  prespecified references: pooled 49.77 % / 49.88 %, query-conditioned 28.25 % / 30.53 %, without replacement
  32.47 % / 35.24 % (Cross / Within). The observed rate (16.29 % / 27.58 %) is below all three, so the conclusion does
  not depend on the choice of reference.
- **Code:** `revision/reproduce_si.py`, `si_complete_pool` (`S13_complete_pool` keys
  `pool_based_independence_reference`, `prespecified_pooled_reference`, `prespecified_query_conditioned_reference`).
