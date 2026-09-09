#!/usr/bin/env python3
"""Reproduce the tables of
"Semantic Failure-State Transitions Across Heterogeneous Rerankers in Scientific Retrieval"
from three inputs: frozen candidate pools, reranker outputs, and semantic adjudication labels.

Usage:  python reproduce.py            # writes results/*.csv and results/summary.json
Requires: Python 3.10+, numpy, scipy.
"""
import json, csv, itertools, collections, os, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

MODELS = ["minilm_l6", "bge_v2_m3", "gemma", "qwen3_rr_4b", "first", "reasonrank"]
NAMES = {"minilm_l6": "MiniLM-L6", "bge_v2_m3": "BGE-v2-M3", "gemma": "BGE-v2-Gemma",
         "qwen3_rr_4b": "Qwen3-RR-4B", "first": "FIRST", "reasonrank": "ReasonRank"}
ARMS = ["cross", "within"]
STATES = ["Gold", "Valid substitute", "Complementary", "Contextual", "Lexical trap"]
ACC = {"Gold", "Valid substitute", "Complementary"}
HARM = {"Contextual", "Lexical trap"}
N_BOOT, SEED, LOW_GAP = 2000, 42, 0.02


def jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def upper_median(v):
    """Median convention of the paper for 30 ordered-pair values: 16th value after sorting."""
    s = sorted(v)
    return s[len(s) // 2]


# ---------------------------------------------------------------- inputs
pools = {(d["arm"], d["query_id"]): d for a in ARMS for d in jsonl(DATA / f"candidate_pools/{a}.jsonl")}
outputs = {(d["arm"], d["query_id"], d["reranker"]): d["rank1_passage_id"]
           for a in ARMS for d in jsonl(DATA / f"reranker_outputs/{a}.jsonl")}


def load_view(name):
    """state[(arm, query, reranker)] -> five-state label for the given adjudication view."""
    return {(d["arm"], d["query_id"], d["reranker"]): d["state_5"] for d in jsonl(DATA / f"adjudication/{name}.jsonl")}


VIEWS = {"canonical": load_view("canonical_gpt4o_5state"),
         "gpt5": {k: v for k, v in load_view("secondary_gpt5_5state").items()},  # pass1 rows override below
         "agreement_only": load_view("agreement_only_5state")}
# secondary file holds pass1 and pass2; keep pass1 as the alternative-label view
_sec = collections.defaultdict(dict)
for d in jsonl(DATA / "adjudication/secondary_gpt5_5state.jsonl"):
    _sec[d["judge_run"]][(d["arm"], d["query_id"], d["passage_id"])] = d["state_5"]

comparable = {a: sorted(q for (arm, q), d in pools.items() if arm == a and d["gold_in_pool"]) for a in ARMS}


def state_of(view, arm, q, m):
    """Five-state label of the rank-1 passage; Gold is mechanical, non-gold looked up in the view."""
    key = (arm, q, m)
    if key not in outputs:
        return None
    pid = outputs[key]
    if pid in pools[(arm, q)]["gold_ids_in_pool"]:
        return "Gold"
    if view == "gpt5":
        return _sec["pass1"].get((arm, q, pid))
    return VIEWS[view].get(key)


# ---------------------------------------------------------------- Table 3 / RQ1
def table3(view="canonical"):
    rows = []
    for a in ARMS:
        for m in MODELS:
            st = [state_of(view, a, q, m) for q in comparable[a]]
            st = [s for s in st if s]
            n = len(st)
            comp = {s: sum(x == s for x in st) / n for s in STATES}
            rows.append({"arm": a, "model": NAMES[m], "N": n, "Acc@1": comp["Gold"],
                         "acceptable": sum(comp[s] for s in ACC), "harmful": sum(comp[s] for s in HARM), **comp})
    return rows


# ---------------------------------------------------------------- RQ2 transitions
def pair_records(view, a, m, m2):
    recs = []
    for q in comparable[a]:
        s1, s2 = state_of(view, a, q, m), state_of(view, a, q, m2)
        if s1 is None or s2 is None:
            continue
        recs.append((s1, s2, outputs[(a, q, m)], outputs[(a, q, m2)]))
    return recs


def rates(recs):
    nH = sum(s1 in HARM for s1, *_ in recs)
    nA = len(recs) - nH
    res = sum(s1 in HARM and s2 in ACC for s1, s2, *_ in recs)
    same = sum(s1 in HARM and s1 == s2 and o1 == o2 for s1, s2, o1, o2 in recs)
    alt = sum(s1 in HARM and s1 == s2 and o1 != o2 for s1, s2, o1, o2 in recs)
    trans = sum(s1 in HARM and s2 in HARM and s1 != s2 for s1, s2, *_ in recs)
    reg = sum(s1 in ACC and s2 in HARM for s1, s2, *_ in recs)
    r = dict(n=len(recs), nH=nH, nA=nA, resolved=res / nH, persistent=(same + alt) / nH,
             pers_same=same / nH, pers_alt=alt / nH, transformed=trans / nH, regressed=reg / nA,
             cnt_same=same, cnt_alt=alt, cnt_trans=trans,
             cs=trans / (alt + trans) if alt + trans else None)
    # directional subtype movements (Fig. 4): denominators are source lexical traps / source contextual
    nL = sum(s1 == "Lexical trap" for s1, *_ in recs); nX = sum(s1 == "Contextual" for s1, *_ in recs)
    r["lt_to_ctx"] = sum(s1 == "Lexical trap" and s2 == "Contextual" for s1, s2, *_ in recs) / nL
    r["ctx_to_lt"] = sum(s1 == "Contextual" and s2 == "Lexical trap" for s1, s2, *_ in recs) / nX
    return r


def bootstrap(recs, rng):
    recs = np.array([(s1 in HARM, s2 in ACC, s1 == s2, o1 == o2, s2 in HARM, s1 in ACC)
                     for s1, s2, o1, o2 in recs], dtype=bool)
    out = collections.defaultdict(list)
    for _ in range(N_BOOT):
        b = recs[rng.integers(0, len(recs), len(recs))]
        h, a2, eq, same, h2, a1 = b.T
        nH, nA = h.sum(), a1.sum()
        out["resolved"].append((h & a2).sum() / nH)
        out["persistent"].append((h & eq).sum() / nH)
        out["transformed"].append((h & h2 & ~eq).sum() / nH)
        out["regressed"].append((a1 & h2).sum() / nA)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def transition_table(view="canonical", with_ci=False):
    rng = np.random.default_rng(SEED)
    rows = []
    for a in ARMS:
        for m, m2 in itertools.permutations(MODELS, 2):
            recs = pair_records(view, a, m, m2)
            r = rates(recs)
            row = {"arm": a, "source": NAMES[m], "target": NAMES[m2], **r}
            if with_ci:
                for k, (lo, hi) in bootstrap(recs, rng).items():
                    row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = lo, hi
            rows.append(row)
    return rows


# ---------------------------------------------------------------- statistical tests (Sec. 3.6)
def tests(view="canonical"):
    rng = np.random.default_rng(SEED)
    rows = []
    for a in ARMS:
        for m, m2 in itertools.combinations(MODELS, 2):
            recs = pair_records(view, a, m, m2)
            b = sum(s1 in ACC and s2 in HARM for s1, s2, *_ in recs)
            c = sum(s1 in HARM and s2 in ACC for s1, s2, *_ in recs)
            p_mc = stats.binomtest(b, b + c, 0.5).pvalue if b + c else None
            M = np.zeros((5, 5), int)
            for s1, s2, *_ in recs:
                M[STATES.index(s1), STATES.index(s2)] += 1
            pairs = [(M[i, j], M[j, i]) for i in range(5) for j in range(i + 1, 5) if M[i, j] + M[j, i] > 0]
            sparse = any(x + y < 10 for x, y in pairs)  # evaluation plan: min 10 per off-diagonal pair
            if sparse:  # permutation fallback
                obs = sum((x - y) ** 2 / (x + y) for x, y in pairs); ge = 0
                for _ in range(5000):
                    s = 0.0
                    for x, y in pairs:
                        t = x + y; k = rng.binomial(t, 0.5); s += (k - (t - k)) ** 2 / t
                    ge += s >= obs
                p_bk, method = (ge + 1) / 5001, "permutation"
            else:
                st = sum((x - y) ** 2 / (x + y) for x, y in pairs)
                p_bk, method = stats.chi2.sf(st, len(pairs)), "bowker"
            rows.append({"arm": a, "pair": f"{NAMES[m]}–{NAMES[m2]}", "mcnemar_p": p_mc, "symmetry_p": p_bk, "symmetry_method": method})
    for key in ["mcnemar_p", "symmetry_p"]:
        pv = [r[key] for r in rows]
        order = sorted(range(len(pv)), key=lambda i: pv[i])
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, (len(pv) - rank) * pv[i]); rows[i][key + "_holm"] = min(1.0, run)
    return rows


# ---------------------------------------------------------------- RQ3 projections
def projection_table(view="canonical"):
    rows = []
    for a in ARMS:
        for m, m2 in itertools.combinations(MODELS, 2):
            recs = pair_records(view, a, m, m2); N = len(recs)
            d5 = sum(s1 != s2 for s1, s2, *_ in recs) / N
            dg = sum((s1 == "Gold") != (s2 == "Gold") for s1, s2, *_ in recs) / N
            dh = sum((s1 in HARM) != (s2 in HARM) for s1, s2, *_ in recs) / N
            dacc = (sum(s2 == "Gold" for _, s2, *_ in recs) - sum(s1 == "Gold" for s1, *_ in recs)) / N
            hidden = [(s1, s2) for s1, s2, *_ in recs if s1 != s2 and (s1 in HARM) == (s2 in HARM)]
            aa = sum(s1 in ACC for s1, _ in hidden); gold_nongold = sum("Gold" in (s1, s2) for s1, s2 in hidden)
            rows.append({"arm": a, "pair": f"{NAMES[m]}–{NAMES[m2]}", "N": N, "D5": d5, "DG": dg, "DH": dh,
                         "abs_dAcc1": abs(dacc), "hidden_fraction": (d5 - dh) / d5, "low_gap": abs(dacc) < LOW_GAP,
                         "hidden_n": len(hidden), "hidden_acc_acc": aa, "hidden_gold_nongold": gold_nongold,
                         "hidden_nongold_acc": aa - gold_nongold, "hidden_transformation": len(hidden) - aa})
    return rows


# ---------------------------------------------------------------- rank depth (Sec. 5.4)
def rank_depth():
    ranked = {(d["arm"], d["query_id"], d["reranker"]): d["ranked_passage_ids"] for a in ARMS for d in jsonl(DATA / f"reranker_outputs/{a}.jsonl")}
    out = {}
    for a in ARMS:
        bins = collections.Counter()
        for m, m2 in itertools.combinations(MODELS, 2):
            for q in comparable[a]:
                s1, s2 = state_of("canonical", a, q, m), state_of("canonical", a, q, m2)
                if s1 is None or s2 is None: continue
                for sg, sh, mg, mh in [(s1, s2, m, m2), (s2, s1, m2, m)]:
                    if sg == "Gold" and sh in HARM:
                        gold = pools[(a, q)]["gold_ids_in_pool"]
                        r = min(ranked[(a, q, mh)].index(g) + 1 for g in gold if g in ranked[(a, q, mh)])
                        bins["2" if r == 2 else "3-5" if r <= 5 else "6-10"] += 1
        tot = sum(bins.values()); out[a] = {k: bins[k] / tot for k in ["2", "3-5", "6-10"]}
    return out


# ---------------------------------------------------------------- adjudicator agreement (Table 5a)
def agreement():
    can = {(d["arm"], d["query_id"], d["passage_id"]): d["state_5"] for d in jsonl(DATA / "adjudication/canonical_gpt4o_5state.jsonl") if d["state_5"] != "Gold"}
    p1, p2 = _sec["pass1"], _sec["pass2"]
    keys = sorted(set(can) & set(p1))
    three = lambda s: s if s in HARM else "Other"
    binary = lambda s: "H" if s in HARM else "A"
    def kappa(x, y):
        return stats.cohen_kappa_score(x, y) if hasattr(stats, "cohen_kappa_score") else _kappa(x, y)
    rows = []
    for name, f, (A, B) in [("GPT-4o vs GPT-5 five-state", lambda s: s, (can, p1)), ("GPT-4o vs GPT-5 three-state", three, (can, p1)),
                            ("GPT-4o vs GPT-5 acceptable/harmful", binary, (can, p1)), ("GPT-5 pass1 vs pass2 five-state", lambda s: s, (p1, p2)),
                            ("GPT-5 pass1 vs pass2 three-state", three, (p1, p2))]:
        x = [f(A[k]) for k in keys]; y = [f(B[k]) for k in keys]
        rows.append({"comparison": name, "N": len(keys), "agreement": np.mean([u == v for u, v in zip(x, y)]), "kappa": _kappa(x, y)})
    for name, (A, B) in [("GPT-4o vs GPT-5 harmful-common subtype", (can, p1)), ("GPT-5 pass1 vs pass2 harmful-common subtype", (p1, p2))]:
        hk = [k for k in keys if A[k] in HARM and B[k] in HARM]
        x = [A[k] for k in hk]; y = [B[k] for k in hk]
        rows.append({"comparison": name, "N": len(hk), "agreement": np.mean([u == v for u, v in zip(x, y)]), "kappa": _kappa(x, y)})
    return rows


def _kappa(x, y):
    labs = sorted(set(x) | set(y)); n = len(x)
    po = sum(u == v for u, v in zip(x, y)) / n
    pe = sum((x.count(l) / n) * (y.count(l) / n) for l in labs)
    return (po - pe) / (1 - pe)


# ---------------------------------------------------------------- write everything
def write_csv(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    t3 = table3(); write_csv("table3_rank1_outcomes.csv", t3)
    tr = transition_table(with_ci="--ci" in sys.argv); write_csv("transitions_ordered_pairs.csv", tr)
    pj = projection_table(); write_csv("projection_unordered_pairs.csv", pj)
    ts = tests(); write_csv("statistical_tests.csv", ts)
    ag = agreement(); write_csv("table5a_adjudicator_agreement.csv", ag)
    views = {v: transition_table(v) for v in ["canonical", "gpt5", "agreement_only"]}
    summary = {}
    for a in ARMS:
        T = [r for r in tr if r["arm"] == a]; P = [r for r in pj if r["arm"] == a]
        summary[a] = {
            "median_resolved_persistent_transformed_regressed": [upper_median([r[k] for r in T]) for k in ["resolved", "persistent", "transformed", "regressed"]],
            "harmful_to_harmful_counts_same_alt_trans": [sum(r[k] for r in T) for k in ["cnt_same", "cnt_alt", "cnt_trans"]],
            "median_subtype_switch": upper_median([r["cs"] for r in T]),
            "pooled_subtype_switch": sum(r["cnt_trans"] for r in T) / sum(r["cnt_alt"] + r["cnt_trans"] for r in T),
            "median_D5_DG_DH_absdAcc": [float(np.median([r[k] for r in P])) for k in ["D5", "DG", "DH", "abs_dAcc1"]],
            "median_hidden_fraction": float(np.median([r["hidden_fraction"] for r in P])),
            "low_gap_pairs": [r["pair"] for r in P if r["low_gap"]],
            "hidden_share_acc_acc": sum(r["hidden_acc_acc"] for r in P) / sum(r["hidden_n"] for r in P),
            "hidden_share_gold_nongold": sum(r["hidden_gold_nongold"] for r in P) / sum(r["hidden_n"] for r in P),
            "rank_depth": rank_depth()[a],
            "transformation_median_by_view": {v: upper_median([r["transformed"] for r in views[v] if r["arm"] == a]) for v in views},
            "transformation_nonzero_by_view": {v: sum(r["transformed"] > 0 for r in views[v] if r["arm"] == a) for v in views},
            "holm_significant": {"acceptable_harmful": sum(r["mcnemar_p_holm"] < 0.05 for r in ts if r["arm"] == a),
                                 "five_state": sum(r["symmetry_p_holm"] < 0.05 for r in ts if r["arm"] == a),
                                 "permutation_fallback": sum(r["symmetry_method"] == "permutation" for r in ts if r["arm"] == a)},
        }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
