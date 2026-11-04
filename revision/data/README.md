# Revision data

All files are JSON Lines. They add to `../../data/` (the submitted release, unchanged); `revision/reproduce_si.py`
reads both. Files that mix several record types carry a `kind` field.

**Text policy.**
- **QASPER:** text is not redistributed. Passage ids are `<paper_id>_s<section>_p<paragraph>`, as in `../../data/`.
- **QASA:** questions, reference answers and evidence sentences are included (QASA is MIT-licensed). arXiv paragraph
  text is not included. QASA passage ids are `<arxiv_id>:<arXiv HTML element id>` and resolve against the arXiv HTML
  versions listed in `qasa.jsonl` (`kind=arxiv_version`).
- **Generated answers:** included. Generated answers may quote source passages.

Five-state labels (`state_5`) use the vocabulary of `../../data/`: Gold, Valid substitute, Complementary, Contextual,
Lexical trap. `state_5 = null` marks a missing label (an invalid judge output). It is never imputed.

| file | rows | one row per | SI |
|---|---|---|---|
| `mistral_5state.jsonl` | 5,567 + 2,130 | selected non-gold (query, arm, passage) identity labelled by Mistral-Small-3.2. `kind=rank1_top10`: the 5,567 rank-1 identities of the submitted analysis; `kind=depth20`: identities first selected in the reconstructed top-10/top-20 runs | S3, S5 |
| `complete_pool_mistral.jsonl` | 28,751 | remaining non-gold candidate of the frozen top-10 pools (`kind=pool_candidate`) | S4 |
| `reference_sample_120.jsonl` | 120 | item of the state-stratified reference sample: GPT-4o, GPT-5 and Mistral labels, the Claude pre-annotation, the author-reviewed reference label, and, for 82 items, the first author's repeat label (`first_author_second_pass`, with `first_author_second_pass_role` = `gpt4o_disagreement` or `agreement_control`) | S3.4 |
| `pools_top20.jsonl` | 7,438 | (query, arm) of the reconstructed retrieval run: top-20 pool, gold flags, and whether the reconstructed top-10 matches the original top-10 (`matches_original_top10_ordered`, `matches_original_top10_set`) | S5 |
| `outputs_top20.jsonl` | 27,192 + 23,832 | (query, arm, reranker) ranking of the reconstructed run: `kind=top20` and the nested `kind=regen_top10` | S5 |
| `qasa.jsonl` | 41,726 | `kind=arxiv_version` (109 papers: version, html sha256), `cohort` (1,554 questions with exclusion reason), `mapping` (2,812 evidence-to-paragraph mappings with scores), `mapping_review` (30 reviewed mappings and verdicts), `query` (1,390 eligible questions with reference answers and evidence), `pool` (top-20 pools), `ranking` (top-10 and top-20 reranker outputs, field `depth`), `label` (Mistral labels of selected non-gold identities) | S7 |
| `downstream.jsonl` | 3 × 14,272 | downstream unit (dataset, query, arm, selected passage): `kind=answer` (Llama-3.3-70B-Instruct answer), `kind=judgment_qwen` (primary evaluator), `kind=judgment_gemma` (secondary evaluator) | S8 |

`reproduce_si.py` reproduces Supplementary Tables S4–S23 and the data behind Figures S1–S2 (Tables S1–S3 are
model, environment and rubric listings). The 81.5% HNSW diagnostic in Supplementary Section S5.1 requires the
original channel-level retrieval lists, which are not included.

Join keys:
- `query_id` + `arm` + `passage_id` for labels;
- `query_id` + `arm` + `reranker` for rankings;
- `unit` = `query_id|arm|passage_id` for downstream rows.

QASA `query_id` values are the QASA question keys.
