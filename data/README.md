# Data

All files are JSON Lines. `query_id` (40-hex) is the join key everywhere.
QASPER text is **not** redistributed; passage ids are `<paper_id>_s<section>_p<paragraph>` and
resolve against the public QASPER release (https://allenai.org/data/qasper).

| file | rows | one row per | fields |
|---|---|---|---|
| `qasper_reference/eligible_ids.jsonl` | 3,719 | eligible query (§3.1) | `query_id`, `paper_id`, `n_gold`, `gold_paragraph_ids`, `historical_source_split`, `eligibility` |
| `candidate_pools/{cross,within}.jsonl` | 3,719 each | (query, arm): frozen top-10 pool after RRF (§3.1, Table 1) | `query_id`, `arm`, `candidate_passage_ids` (RRF order; used for pointwise tie-breaking), `candidate_count`, `gold_ids_in_pool`, `gold_in_pool` (comparable query-arm flag), `paper_id`, `pool_hash` |
| `reranker_outputs/{cross,within}.jsonl` | 22,312 / 22,313 | (query, arm, reranker): final ranking (§3.2–3.3) | `query_id`, `arm`, `reranker`, `ranked_passage_ids`, `rank1_passage_id`, `n_candidates` |
| `adjudication/canonical_gpt4o_5state.jsonl` | 23,914 | (query, arm, reranker) in the comparable cohort: five-state label of the rank-1 passage (§3.4) | `query_id`, `arm`, `reranker`, `passage_id`, `state_5`, `judge`, `judge_run` |
| `adjudication/secondary_gpt5_5state.jsonl` | 11,134 | (query, arm, passage) × two GPT-5 passes over the 5,567 unique non-gold identities (§6.3) | as above |
| `adjudication/agreement_only_5state.jsonl` | 19,255 | (query, arm, reranker) whose GPT-4o and GPT-5 labels agree exactly (§6.3, view C) | as above |

`state_5` ∈ {Gold, Valid substitute, Complementary, Contextual, Lexical trap}. Gold is assigned
mechanically when `rank1_passage_id ∈ gold_ids_in_pool`; the remaining labels come from the judge.
Labels are attached to (query, arm, passage) identities and reused across rerankers that select
the same passage (§3.4).

Three (query, arm, reranker) units are absent from `reranker_outputs/`: all are ReasonRank outputs
that omitted or invented a candidate identifier and were recorded as invalid rather than repaired
(§3.2). Two of them fall inside the comparable cohort, which is why the paper reports
23,914 = 806×6 + 3,180×6 − 2 observations.
