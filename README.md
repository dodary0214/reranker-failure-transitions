# Semantic Failure-State Transitions Across Heterogeneous Rerankers in Scientific Retrieval

Data, materials, and a single reproduction script for the paper.

Six rerankers (MiniLM-L6, BGE-reranker-v2-m3, BGE-reranker-v2-gemma, Qwen3-Reranker-4B, FIRST,
ReasonRank-7B) are run over the same frozen 10-candidate pool for 3,719 QASPER queries in two
retrieval conditions (Cross, Within). Each rank-1 passage is assigned one of five semantic states
(Gold, Valid substitute, Complementary, Contextual, Lexical trap), and the paper analyzes
directional state transitions, passage identity, and what coarser evaluation views hide.

```
data/        frozen candidate pools, reranker outputs, adjudication labels   (§3.1–3.4)
materials/   adjudication prompts, judge/reranker configurations, model locks (§3.2, §3.4)
figures/     Figures 1–5 as they appear in the paper
reproduce.py recomputes every table and reported statistic from data/     (§4–6)
```

## Reproduce

```bash
pip install -r requirements.txt
python reproduce.py          # ~1 min
python reproduce.py --ci     # adds 2,000-replicate bootstrap intervals for every ordered pair (~15 min)
```

Outputs are written to `results/`:

| file | paper |
|---|---|
| `table3_rank1_outcomes.csv` | Table 3, Figure 2 |
| `transitions_ordered_pairs.csv` | Figure 3, Figure 4, Table 4 (rates, identity decomposition, conditional subtype-switch rate) |
| `statistical_tests.csv` | §3.6 / §5.1 McNemar, Bowker or permutation, Holm-adjusted p-values |
| `projection_unordered_pairs.csv` | Figure 5, §6.1–6.2 (D5, DG, DH, |ΔAcc@1|, hidden fraction, low-gap flag) |
| `table5a_adjudicator_agreement.csv` | Table 5(a) |
| `summary.json` | every headline number quoted in the text, per arm |

`summary.json` reproduces the manuscript exactly for all counts, medians, agreement statistics,
Holm-significant test counts, and Table 5(b). Bootstrap interval bounds in Table 4(b) agree to
within 0.5 percentage points; exact bounds depend on the random-number stream of the original run.

## What is not here

* **QASPER text.** Obtain it from https://allenai.org/data/qasper; all files reference passages by id.
* **Model weights and judge calls.** `materials/models.json` pins the HuggingFace revision of each
  reranker and `materials/reranker_templates.yaml` gives the exact input template and ranking
  derivation; `materials/adjudication_prompts.md` gives the verbatim judge prompts. Re-running
  reranking or adjudication requires the corresponding GPU/API access and is not needed to
  reproduce the paper's numbers.

## Design conventions (from the paper)

* Cross and Within share 798 comparable queries and are analyzed separately, never pooled.
* Pair-level medians over the 30 ordered pairs per arm use the upper median (16th sorted value; §3.6).
* The GPT-5 re-adjudication is a sensitivity analysis: it never overwrites canonical labels and no
  majority vote is formed (§3.4, §6.3).

## License and citation

Code: MIT. Data and documentation: CC BY 4.0 (see `LICENSE`). QASPER and the model weights carry
their own terms. Cite the paper; machine-readable metadata is in `CITATION.cff`.
