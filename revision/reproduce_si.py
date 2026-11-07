#!/usr/bin/env python3
"""Reproduce the Supplementary Information tables (S4-S23) and the data behind Figures S1-S2 of
"Semantic Failure-State Transitions Across Heterogeneous Rerankers in Scientific Retrieval".

Reads the submitted release (../data) and the revision additions (revision/data). Writes
revision/results/si_tables.json and prints every SI value it reproduces.

Usage:  python revision/reproduce_si.py            # full run, 2,000-replicate joint bootstraps
        python revision/reproduce_si.py --B 200    # quicker run; interval bounds will differ slightly
Requires: Python 3.10+ (standard library only).
"""
import argparse, itertools, json, math, random, sys
from collections import Counter, defaultdict
from operator import itemgetter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA, RDATA, OUT = HERE.parent / "data", HERE / "data", HERE / "results"

MODELS = ["minilm_l6", "bge_v2_m3", "gemma", "qwen3_rr_4b", "first", "reasonrank"]
NAMES = {"minilm_l6": "MiniLM-L6", "bge_v2_m3": "BGE-v2-M3", "gemma": "BGE-v2-Gemma",
         "qwen3_rr_4b": "Qwen3-RR-4B", "first": "FIRST", "reasonrank": "ReasonRank"}
ARMS = ["cross", "within"]
LETTER = {"Gold": "G", "Valid substitute": "V", "Complementary": "C", "Contextual": "X", "Lexical trap": "L"}
GROUPINGS = {"primary": (frozenset("GVC"), frozenset("XL")), "strict": (frozenset("GV"), frozenset("CXL"))}
B_DEFAULT, SEED = 2000, 42


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def umed(v):
    """Upper median, sorted(v)[n//2] (16th of 30 ordered pairs, 8th of 15 unordered pairs)."""
    s = sorted(v)
    return s[len(s) // 2]


def pct(x, d=2):
    return None if x is None else round(100 * x, d)


# ======================================================================= inputs
def load_inputs():
    I = {}
    pools = {(d["arm"], d["query_id"]): d for a in ARMS for d in jl(DATA / f"candidate_pools/{a}.jsonl")}
    I["pools"] = pools
    I["ranked"] = {(d["arm"], d["query_id"], d["reranker"]): d["ranked_passage_ids"]
                   for a in ARMS for d in jl(DATA / f"reranker_outputs/{a}.jsonl")}
    I["rank1"] = {k: v[0] for k, v in I["ranked"].items()}
    I["comparable"] = {a: sorted(q for (arm, q), d in pools.items() if arm == a and d["gold_in_pool"]) for a in ARMS}
    I["gold"] = {k: set(d["gold_ids_in_pool"]) for k, d in pools.items()}
    can = jl(DATA / "adjudication/canonical_gpt4o_5state.jsonl")
    I["gpt4o"] = {(d["arm"], d["query_id"], d["passage_id"]): LETTER[d["state_5"]] for d in can if d["state_5"] != "Gold"}
    I["gpt5"] = defaultdict(dict)
    for d in jl(DATA / "adjudication/secondary_gpt5_5state.jsonl"):
        I["gpt5"][d["judge_run"]][(d["arm"], d["query_id"], d["passage_id"])] = LETTER[d["state_5"]]
    I["agree_only"] = {(d["arm"], d["query_id"], d["reranker"]): LETTER[d["state_5"]]
                       for d in jl(DATA / "adjudication/agreement_only_5state.jsonl")}
    I["mistral"] = {}          # identity -> letter or None (missing)
    I["mistral_kind"] = {}
    for f in ("mistral_5state.jsonl", "complete_pool_mistral.jsonl"):
        for d in jl(RDATA / f):
            k = (d["arm"], d["query_id"], d["passage_id"]) + ((d["reference_gold_id"],) if d.get("reference_gold_id") else ())
            I["mistral"][k] = LETTER.get(d["state_5"]); I["mistral_kind"][k] = d["kind"]
    return I


def build_units(rank1, gold, queries, label, models=MODELS):
    """units[(arm, q, m)] = (passage, state). Gold is mechanical; a missing label drops the unit (never imputed)."""
    units, missing = {}, 0
    for a in ARMS:
        for q in queries[a]:
            for m in models:
                pid = rank1.get((a, q, m))
                if pid is None:
                    continue
                if pid in gold[(a, q)]:
                    units[(a, q, m)] = (pid, "G"); continue
                s = label(a, q, m, pid)
                if s is None:
                    missing += 1
                else:
                    units[(a, q, m)] = (pid, s)
    return units, missing


# ======================================================================= pair metrics
def arm_queries(units, arm):
    per_q = defaultdict(dict)
    for (ar, q, m), v in units.items():
        if ar == arm:
            per_q[q][m] = v
    return sorted(per_q), per_q


def pair_cell_members(qids, per_q, a, b):
    """{(state_a, state_b, same_passage): [query index, ...]} over queries where both units exist."""
    cells = defaultdict(list)
    for i, q in enumerate(qids):
        d = per_q[q]
        if a in d and b in d:
            (ca, sa), (cb, sb) = d[a], d[b]
            cells[(sa, sb, ca == cb)].append(i)
    return cells


def counts(members):
    return {k: len(v) for k, v in members.items()}


def directional(cells, grouping="primary", reverse=False):
    ACC, HARM = GROUPINGS[grouping]
    c = dict(nH=0, nA=0, res=0, same=0, alt=0, trans=0, reg=0, LX=0, XL=0, nL=0, nX=0)
    for (sa, sb, sc), n in cells.items():
        if reverse:
            sa, sb = sb, sa
        if sa in HARM:
            c["nH"] += n
            if sb in ACC: c["res"] += n
            elif sb == sa: c["same" if sc else "alt"] += n
            else: c["trans"] += n
        elif sa in ACC:
            c["nA"] += n
            if sb in HARM: c["reg"] += n
        c["nL"] += n * (sa == "L"); c["nX"] += n * (sa == "X")
        c["LX"] += n * (sa == "L" and sb == "X"); c["XL"] += n * (sa == "X" and sb == "L")
    return c


def rates(c, empty=None):
    f = lambda n, d: n / d if d else empty
    return dict(resolution=f(c["res"], c["nH"]), persistence=f(c["same"] + c["alt"], c["nH"]),
                transformation=f(c["trans"], c["nH"]), regression=f(c["reg"], c["nA"]),
                conditional_subtype_switch=c["trans"] / (c["alt"] + c["trans"]) if c["alt"] + c["trans"] else None,
                lexical_to_contextual=f(c["LX"], c["nL"]), contextual_to_lexical=f(c["XL"], c["nX"]))


def symmetric(cells, grouping="primary"):
    ACC, HARM = GROUPINGS[grouping]
    N = d5 = dh = dg = 0
    for (sa, sb, sc), n in cells.items():
        N += n; d5 += n * (sa != sb); dh += n * ((sa in HARM) != (sb in HARM)); dg += n * ((sa == "G") != (sb == "G"))
    return dict(N=N, D5=d5 / N, DH=dh / N, DG=dg / N, HiddenFraction=(d5 - dh) / d5 if d5 else None)


def pair_metrics(units, arm, grouping="primary"):
    """ordered[(src, tgt)] = counts + rates; unordered[(a, b)] = symmetric rates."""
    qids, per_q = arm_queries(units, arm)
    ordered, unordered = {}, {}
    for a, b in itertools.combinations(MODELS, 2):
        cells = counts(pair_cell_members(qids, per_q, a, b))
        for s, t, rev in ((a, b, False), (b, a, True)):
            c = directional(cells, grouping, rev)
            ordered[(s, t)] = {**c, **rates(c)}
        unordered[(a, b)] = symmetric(cells, grouping)
    return ordered, unordered


def medians(ordered, unordered):
    out = {}
    for k in ("resolution", "persistence", "transformation", "regression", "conditional_subtype_switch"):
        v = [r[k] for r in ordered.values() if r[k] is not None]
        out[k] = umed(v) if v else None
    for k in ("D5", "HiddenFraction"):
        v = [r[k] for r in unordered.values() if r[k] is not None]
        out[k] = umed(v) if v else None
    out["transformation_gt0"] = sum(r["transformation"] is not None and r["transformation"] > 0 for r in ordered.values())
    out["hiddenfraction_gt0"] = sum(r["HiddenFraction"] is not None and r["HiddenFraction"] > 0 for r in unordered.values())
    return out


QUANT = ["resolution", "persistence", "transformation", "regression", "conditional_subtype_switch", "HiddenFraction", "D5"]


def joint_bootstrap(units, arm, B, grouping="primary", seed=SEED):
    """Query-level joint bootstrap of the pair-level upper medians (fresh random.Random(42) per arm).
    A query drawn w times enters every pair, in both directions, with weight w. Percentile CI = sorted
    replicates s[int(0.025B)], s[int(0.975B)]. Empty rate denominators inside a replicate count as 0;
    undefined conditional switch / HiddenFraction are dropped from that replicate's median."""
    qids, per_q = arm_queries(units, arm)
    n = len(qids)
    members = [pair_cell_members(qids, per_q, a, b) for a, b in itertools.combinations(MODELS, 2)]

    def med(cell_sets):
        vals = {q: [] for q in QUANT}
        for cells in cell_sets:
            for rev in (False, True):
                r = rates(directional(cells, grouping, rev), empty=0.0)
                for q in QUANT[:4]:
                    vals[q].append(r[q])
                if r["conditional_subtype_switch"] is not None:
                    vals["conditional_subtype_switch"].append(r["conditional_subtype_switch"])
            s = symmetric(cells, grouping)
            vals["D5"].append(s["D5"])
            if s["HiddenFraction"] is not None:
                vals["HiddenFraction"].append(s["HiddenFraction"])
        return {q: umed(v) if v else None for q, v in vals.items()}

    point = med([counts(m) for m in members])
    rng = random.Random(seed)
    reps = {q: [] for q in QUANT}
    for _ in range(B):
        w = [0] * n
        for i in rng.choices(range(n), k=n):
            w[i] += 1
        cs = [{k: (w[ix[0]] if len(ix) == 1 else sum(itemgetter(*ix)(w))) for k, ix in m.items()} for m in members]
        r = med(cs)
        for q in QUANT:
            if r[q] is not None:
                reps[q].append(r[q])
    lo, hi = int(B * 0.025), int(B * 0.975)
    return {q: dict(point=point[q], ci_lo=sorted(reps[q])[lo] if len(reps[q]) == B else None,
                    ci_hi=sorted(reps[q])[hi] if len(reps[q]) == B else None) for q in QUANT}


def fmt_ci(d, dig=2):
    return f"{100 * d['point']:.{dig}f} [{100 * d['ci_lo']:.{dig}f}, {100 * d['ci_hi']:.{dig}f}]"


# ======================================================================= agreement helpers
def kappa(x, y, labels=None):
    n = len(x)
    labels = labels or sorted(set(x) | set(y))
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[l] * cy[l] for l in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else None


def binary(s):
    return "H" if s in "XL" else "A"


def agree(x, y):
    return dict(n=len(x), exact=sum(a == b for a, b in zip(x, y)) / len(x), kappa=kappa(x, y),
                binary=sum(binary(a) == binary(b) for a, b in zip(x, y)) / len(x),
                binary_kappa=kappa([binary(a) for a in x], [binary(b) for b in y]))


def confusion(x, y, S="VCXL"):
    return {a: {b: sum(1 for u, v in zip(x, y) if u == a and v == b) for b in S} for a in S}


def macro_f1(ref, pred, S="VCXL"):
    f = []
    for c in S:
        tp = sum(r == c and p == c for r, p in zip(ref, pred)); pp = sum(p == c for p in pred); rp = sum(r == c for r in ref)
        pr, rc = (tp / pp if pp else None), (tp / rp if rp else None)
        f.append(2 * pr * rc / (pr + rc) if pr and rc else 0.0)
    return sum(f) / len(f)


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return dict(k=k, n=n, rate=p, ci_lo=max(0.0, c - h), ci_hi=min(1.0, c + h))


# ======================================================================= QASPER: S4-S12, S23, Fig. S1
def qasper_views(I):
    lab = {
        "gpt4o": lambda a, q, m, p: I["gpt4o"].get((a, q, p)),
        "gpt5": lambda a, q, m, p: I["gpt5"]["pass1"].get((a, q, p)),
        "agreement_only": lambda a, q, m, p: I["agree_only"].get((a, q, m)),
        "mistral": lambda a, q, m, p: I["mistral"].get((a, q, p)),
    }
    return {k: build_units(I["rank1"], I["gold"], I["comparable"], f) for k, f in lab.items()}


def si_qasper(I, V, B, R):
    gpt4o, _ = V["gpt4o"]
    # S4 taxonomy sensitivity
    s4 = {}
    for a in ARMS:
        for g in ("primary", "strict"):
            s4[f"{a}:{g}"] = {k: (v if isinstance(v, int) else pct(v, 1)) for k, v in medians(*pair_metrics(gpt4o, a, g)).items()}
    R["S4_taxonomy_sensitivity"] = s4

    # S5-S7 adjudicator agreement and confusions over the 5,567 non-gold identities
    ids = sorted(I["gpt4o"])
    g4 = [I["gpt4o"][k] for k in ids]; g5 = [I["gpt5"]["pass1"][k] for k in ids]
    mi_ids = [k for k in ids if I["mistral"].get(k)]
    s5 = {"GPT-4o vs GPT-5": agree(g4, g5),
          "GPT-4o vs Mistral": agree([I["gpt4o"][k] for k in mi_ids], [I["mistral"][k] for k in mi_ids]),
          "GPT-5 vs Mistral": agree([I["gpt5"]["pass1"][k] for k in mi_ids], [I["mistral"][k] for k in mi_ids])}
    R["S5_adjudicator_agreement"] = {k: {**v, "exact": pct(v["exact"], 1), "binary": pct(v["binary"], 1),
                                         "kappa": round(v["kappa"], 3), "binary_kappa": round(v["binary_kappa"], 3)}
                                     for k, v in s5.items()}
    R["S6_confusion_gpt4o_gpt5"] = confusion(g4, g5) | {"n": len(ids)}
    R["S7_confusion_gpt4o_mistral"] = confusion([I["gpt4o"][k] for k in mi_ids], [I["mistral"][k] for k in mi_ids]) | {"n": len(mi_ids)}

    # S8 Mistral transitions with joint-bootstrap CIs
    mis, _ = V["mistral"]
    R["S8_mistral_transitions"] = {a: {**{q: fmt_ci(v) for q, v in joint_bootstrap(mis, a, B).items()},
                                       "transformation_gt0": medians(*pair_metrics(mis, a))["transformation_gt0"]}
                                   for a in ARMS}

    # S9 GPT-5 view (B) and agreement-only view (C)
    s9 = {}
    for name in ("gpt4o", "gpt5", "agreement_only"):
        u, _ = V[name]
        for a in ARMS:
            o, un = pair_metrics(u, a)
            m = medians(o, un)
            s9[f"{name}:{a}"] = dict(T_gt0=m["transformation_gt0"], T_median=pct(m["transformation"], 1),
                                     CS_median=pct(m["conditional_subtype_switch"], 1),
                                     effective_pair_N_range=[min(x["N"] for x in un.values()), max(x["N"] for x in un.values())])
    R["S9_gpt5_views"] = s9

    # S23 primary GPT-4o joint bootstrap
    R["S23_bootstrap_gpt4o"] = {a: {q: fmt_ci(v) for q, v in joint_bootstrap(gpt4o, a, B).items()} for a in ARMS}

    # Fig. S1: directional harmful-subtype transformation rates (source-subtype denominators)
    R["FigS1_transformation"] = {a: {f"{NAMES[s]}->{NAMES[t]}": dict(L_to_X=pct(r["lexical_to_contextual"], 1),
                                                                      X_to_L=pct(r["contextual_to_lexical"], 1),
                                                                      n_LX=r["LX"], n_L=r["nL"], n_XL=r["XL"], n_X=r["nX"])
                                     for (s, t), r in pair_metrics(gpt4o, a)[0].items()} for a in ARMS}


def si_reference(R):
    rows = jl(RDATA / "reference_sample_120.jsonl")
    L = lambda r, k: LETTER[r[k]] if r[k] else None
    ref = [L(r, "reference") for r in rows]
    s10 = {}
    for j, k in (("GPT-4o", "gpt4o"), ("GPT-5", "gpt5"), ("Mistral-Small-3.2", "mistral")):
        ok = [(L(r, "reference"), L(r, k)) for r in rows if r[k]]
        x, y = [a for a, _ in ok], [b for _, b in ok]
        a_ = agree(x, y)
        s10[j] = dict(n=len(ok), exact=pct(a_["exact"], 1), kappa=round(a_["kappa"], 2), macro_f1=round(macro_f1(x, y), 2),
                      binary=pct(a_["binary"], 1))
    R["S10_reference_agreement"] = s10 | {"reference_distribution": dict(Counter(ref)),
                                          "changed_from_claude": sum(r["reference"] != r["claude_proposed"] for r in rows)}
    sp = [r for r in rows if r["first_author_second_pass"]]
    dis = [r for r in sp if r["first_author_second_pass_role"] == "gpt4o_disagreement"]
    ctl = [r for r in sp if r["first_author_second_pass_role"] == "agreement_control"]
    g = lambda r, k: L(r, k)
    by = defaultdict(lambda: Counter())
    for r in dis:
        key = "/".join(sorted({g(r, "gpt4o"), g(r, "reference")}, key="VCXL".index))
        s2 = g(r, "first_author_second_pass")
        by[key]["n"] += 1
        by[key]["reference"] += s2 == g(r, "reference"); by[key]["gpt4o"] += s2 == g(r, "gpt4o")
        by[key]["neither"] += s2 not in (g(r, "reference"), g(r, "gpt4o"))
    bdis = [r for r in dis if binary(g(r, "gpt4o")) != binary(g(r, "reference"))]
    all82 = [(g(r, "reference"), g(r, "first_author_second_pass")) for r in sp]
    R["S11_repeat_label_stability"] = {
        "n_items": len(sp), "disagreement_items": len(dis), "controls": len(ctl),
        "supports_reference": wilson(sum(g(r, "first_author_second_pass") == g(r, "reference") for r in dis), len(dis)),
        "matches_gpt4o": sum(g(r, "first_author_second_pass") == g(r, "gpt4o") for r in dis),
        "matches_gpt5": sum(g(r, "first_author_second_pass") == g(r, "gpt5") for r in dis),
        "matches_mistral": sum(r["mistral"] is not None and g(r, "first_author_second_pass") == g(r, "mistral") for r in dis),
        "controls_match_reference": sum(g(r, "first_author_second_pass") == g(r, "reference") for r in ctl),
        "all_82_exact": pct(sum(a == b for a, b in all82) / len(all82), 1), "all_82_kappa": round(kappa(*zip(*all82)), 2),
        "by_boundary": {k: dict(v) for k, v in sorted(by.items())},
        "binary_boundary": dict(n=len(bdis),
                                reference=sum(binary(g(r, "first_author_second_pass")) == binary(g(r, "reference")) for r in bdis),
                                gpt4o=sum(binary(g(r, "first_author_second_pass")) == binary(g(r, "gpt4o")) for r in bdis))}
    R["S12_reference_confusions"] = {j: confusion([L(r, "reference") for r in rows if r[k]], [L(r, k) for r in rows if r[k]])
                                     for j, k in (("GPT-4o", "gpt4o"), ("GPT-5", "gpt5"), ("Mistral-Small-3.2", "mistral"))}
    R["S12_reference_confusions"]["model_VC_where_reference_XL"] = {
        j: sum(L(r, "reference") in "XL" and L(r, k) in "VC" for r in rows if r[k]) for j, k in (("GPT-4o", "gpt4o"), ("GPT-5", "gpt5"), ("Mistral", "mistral"))}
    R["S12_reference_confusions"]["model_XL_where_reference_VC"] = {
        j: sum(L(r, "reference") in "VC" and L(r, k) in "XL" for r in rows if r[k]) for j, k in (("GPT-4o", "gpt4o"), ("GPT-5", "gpt5"), ("Mistral", "mistral"))}


SECTIONS = []  # (name, function(I, V, B, R)), run after the QASPER sections


# ======================================================================= S13-S18: complete pool, depth, top-k, QASA
# ======================================================================= S13 complete pool, S14 depth, S15 top-k
def si_complete_pool(I, V, B, R):
    """S13: every candidate of the frozen top-10 pools in the Mistral view (gold = G; missing labels dropped)."""
    mis, _ = V["mistral"]
    out = {}
    for arm in ARMS:
        pooled, shares, missing = Counter(), [], 0
        for q in I["comparable"][arm]:
            p = I["pools"][(arm, q)]
            gold = set(p["gold_ids_in_pool"]); st = []
            for cid in p["candidate_passage_ids"]:
                if cid in gold:
                    st.append("G"); continue
                s = I["mistral"].get((arm, q, cid))
                if s is None: missing += 1
                else: st.append(s)
            c = Counter(st); pooled.update(c)
            shares.append({s: c[s] / len(st) for s in "GVCXL"})
        rand = {s: sum(x[s] for x in shares) / len(shares) for s in "GVCXL"}
        qids, per_q = arm_queries(mis, arm)
        harm = {m: (lambda v: sum(s in "XL" for s in v) / len(v))([per_q[q][m][1] for q in qids if m in per_q[q]])
                for m in MODELS}
        # pool-based independence reference: for a pool with h harmful candidates of which l are Lexical
        # traps, two distinct uniform draws differ in subtype with probability 2l(h-l)/(h(h-1)).
        # (DEV-16, post hoc). The prespecified references (analysis plan, Exp C) use 2p(1-p) with p the Lexical-trap
        # share of harmful pool candidates, pooled per arm or per query (event-weighted); they are reported alongside.
        comp, lh = {}, {}
        for q in qids:
            p = I["pools"][(arm, q)]; gold = set(p["gold_ids_in_pool"])
            st = [I["mistral"].get((arm, q, c)) for c in p["candidate_passage_ids"] if c not in gold]
            h = sum(x in "XL" for x in st if x); l = sum(x == "L" for x in st if x)
            comp[q] = 2 * l * (h - l) / (h * (h - 1)) if h >= 2 else None
            lh[q] = (l, h)
        ps = pooled["L"] / (pooled["L"] + pooled["X"]); ref_pooled = 2 * ps * (1 - ps)
        alt = trans = 0; ref_sum = 0.0; qc_sum = 0.0; qc_n = 0
        below = {"pooled": 0, "query_conditioned": 0, "without_replacement": 0}
        for a, b in itertools.permutations(MODELS, 2):
            o = t = 0; r = 0.0; o2 = t2 = 0; r2 = 0.0
            for q in qids:
                d = per_q[q]
                if a in d and b in d:
                    (ca, sa), (cb, sb) = d[a], d[b]
                    if sa in "XL" and sb in "XL" and ca != cb:
                        l, h = lh[q]; w = 2 * (l / h) * (1 - l / h)
                        qc_sum += w; qc_n += 1; o2 += 1; t2 += sa != sb; r2 += w
                        if comp[q] is not None:
                            o += 1; t += sa != sb; r += comp[q]
            alt += o - t; trans += t; ref_sum += r
            below["without_replacement"] += (t / o) < (r / o)
            below["query_conditioned"] += (t2 / o2) < (r2 / o2)
            below["pooled"] += (t2 / o2) < ref_pooled
        pairs_below = below["without_replacement"]
        out[arm] = {"pools": len(I["comparable"][arm]), "candidates_labelled": sum(pooled.values()),
                    "candidates_missing_label": missing,
                    "random_candidate_harmful_probability": pct(rand["X"] + rand["L"]),
                    "observed_rank1_harmful_range": [pct(min(harm.values())), pct(max(harm.values()))],
                    "lexical_trap_share_among_harmful_pool_candidates": pct(pooled["L"] / (pooled["L"] + pooled["X"])),
                    "harmful_candidates": pooled["L"] + pooled["X"],
                    "observed_pooled_conditional_switch": f"{trans}/{alt + trans} ({100 * trans / (alt + trans):.2f}%)",
                    "pool_based_independence_reference": pct(ref_sum / (alt + trans)),
                    "prespecified_pooled_reference": pct(ref_pooled),
                    "prespecified_query_conditioned_reference": pct(qc_sum / qc_n),
                    "ordered_pairs_observed_below_each_reference": {k: f"{v}/30" for k, v in below.items()},
                    "ordered_pairs_observed_below_reference": f"{pairs_below}/30"}
    R["S13_complete_pool"] = out


def depth_units(I, kind, pools20, gold):
    """Mistral-view units of a depth-20 ranking set (kind top20 / regen_top10). A label is looked up first under
    the gold-specific key (arm, query, passage, first gold in RRF order) (written when the cascade inputs differed from the
    earlier run), then under the plain identity."""
    units, missing = {}, Counter()
    for d in I["depth_rankings"]:
        if d["kind"] != kind: continue
        arm, q, m = d["arm"], d["query_id"], d["reranker"]
        if d["status"] != "OK": missing["invalid_ranking"] += 1; continue
        cid = d["rank1_passage_id"]
        if cid in gold[q]: units[(arm, q, m)] = (cid, "G"); continue
        g = next((p for p in pools20[(arm, q)]["pre_rerank_order"] if p in gold[q]), None)
        k1, k0 = (arm, q, cid, g), (arm, q, cid)
        key = k1 if k1 in I["mistral"] else k0
        s = I["mistral"].get(key, "unlabelled")
        if s in (None, "unlabelled"): missing["label_error" if s is None else "unlabelled"] += 1
        else: units[(arm, q, m)] = (cid, s)
    return units, missing


def depth_block(units, arm, qs):
    u = {k: v for k, v in units.items() if k[0] == arm and k[1] in qs}
    qids, per_q = arm_queries(u, arm)
    acc = {m: (lambda v: sum(s == "G" for s in v) / len(v))([per_q[q][m][1] for q in qids if m in per_q[q]]) for m in MODELS}
    o, un = pair_metrics(u, arm)
    return {"n_query_arms": len(qs), "acc1": acc, **medians(o, un)}


def si_depth(I, V, B, R):
    pools20 = {(d["arm"], d["query_id"]): d for d in jl(RDATA / "pools_top20.jsonl")}
    I["depth_rankings"] = jl(RDATA / "outputs_top20.jsonl")
    gold = {d["query_id"]: set(d["gold_paragraph_ids"]) for d in jl(DATA / "qasper_reference/eligible_ids.jsonl")}
    fid = [p for p in pools20.values() if p["gold_in_original_top10"]]
    R["S14_reconstruction_fidelity"] = {"ordered_top10_match": sum(p["matches_original_top10_ordered"] for p in fid),
                                        "comparable_query_arms": len(fid),
                                        "rate": pct(sum(p["matches_original_top10_ordered"] for p in fid) / len(fid), 1)}
    u10, m10 = depth_units(I, "regen_top10", pools20, gold)
    u20, m20 = depth_units(I, "top20", pools20, gold)
    out = {"missing_units": {"regen_top10": dict(m10), "top20": dict(m20)}}
    for arm in ARMS:
        qs = {q for (a, q), p in pools20.items() if a == arm and p["gold_in_pool_top10"]}
        b10, b20 = depth_block(u10, arm, qs), depth_block(u20, arm, qs)
        d = {m: 100 * (b10["acc1"][m] - b20["acc1"][m]) for m in MODELS}
        out[arm] = {"n_query_arms": len(qs), "T@10": pct(b10["transformation"]), "T@20": pct(b20["transformation"]),
                    "HF@10": pct(b10["HiddenFraction"], 1), "HF@20": pct(b20["HiddenFraction"], 1),
                    "T_gt0@10": b10["transformation_gt0"], "T_gt0@20": b20["transformation_gt0"],
                    "abs_HF_change_pp": round(abs(100 * (b10["HiddenFraction"] - b20["HiddenFraction"])), 1),
                    "acc1_top10_minus_top20_pp": {NAMES[m]: round(v, 2) for m, v in d.items()},
                    "acc1_drop_range_pp": [round(min(d.values()), 1), round(max(d.values()), 1)]}
    R["S14_depth"] = out


def si_topk(I, V, B, R):
    """S15: top-3/top-5 overlap conditional on different rank-1 selections (GPT-4o units, frozen rankings)."""
    units, _ = V["gpt4o"]
    rank, gold = I["ranked"], I["gold"]
    size = {k: len(p["candidate_passage_ids"]) for k, p in I["pools"].items()}
    out = {}
    for arm in ARMS:
        qall = I["comparable"][arm]
        T = lambda q, m, k: set(rank[(arm, q, m)][:min(k, size[(arm, q)])])
        res = {}
        for k in (3, 5):
            means, nd, zero, sg = [], 0, 0, 0
            for a, b in itertools.combinations(MODELS, 2):
                qs = [q for q in qall if (arm, q, a) in units and (arm, q, b) in units]
                diff = [q for q in qs if rank[(arm, q, a)][0] != rank[(arm, q, b)][0]]
                ov = [len(T(q, a, k) & T(q, b, k)) / min(k, size[(arm, q)]) for q in diff]
                means.append(sum(ov) / len(ov)); nd += len(diff); zero += sum(x == 0 for x in ov)
                sg += sum(bool(T(q, a, k) & T(q, b, k) & gold[(arm, q)]) for q in diff)
            res[f"k{k}"] = {"median_pair_mean_overlap": round(umed(means), 3), "zero_overlap": pct(zero / nd, 3),
                            "shared_gold": pct(sg / nd), "n_rank1_differ_pair_queries": nd}
        out[arm] = res
    R["S15_topk"] = out


# ======================================================================= S16-S18 QASA
def sample_median(v):
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def qasa_block(units, acc, arm, qs):
    """Metrics on one QASA population. acc[(arm, q, m)] = rank-1 is gold, for every valid unit (label-independent)."""
    u = {k: v for k, v in units.items() if k[0] == arm and k[1] in qs}
    qids, per_q = arm_queries(u, arm)
    o, un = pair_metrics(u, arm)
    a1 = {}
    for m in MODELS:
        v = [g for k, g in acc.items() if k[0] == arm and k[1] in qs and k[2] == m]
        a1[m] = sum(v) / len(v)
    pairs = {}
    for a, b in itertools.combinations(MODELS, 2):
        both = [q for q in qids if a in per_q[q] and b in per_q[q]]
        ga = sum(per_q[q][a][1] == "G" for q in both) / len(both); gb = sum(per_q[q][b][1] == "G" for q in both) / len(both)
        pairs[(a, b)] = {"abs_dAcc1": abs(ga - gb), "D5": un[(a, b)]["D5"]}
    return {"acc1": a1, "pairs": pairs, **medians(o, un)}


def si_qasa(I, V, B, R):
    rows = jl(RDATA / "qasa.jsonl")
    kind = lambda k: [r for r in rows if r["kind"] == k]

    # S16 evidence-mapping quality checks
    E, coh = kind("mapping"), kind("cohort")
    attempted = [q for q in coh if q["exclusion"] in (None, "NO_UNAMBIGUOUS_GOLD_MAPPED")]
    ge1 = sum(1 for q in attempted if q["gold_paragraph_ids"])
    rev = kind("mapping_review")
    R["S16_qasa_mapping"] = {"queries_ge1_unambiguous_gold": f"{ge1}/{len(attempted)} ({100 * ge1 / len(attempted):.1f}%)",
                             "ambiguous_evidence": f"{sum(m['ambiguous'] for m in E)}/{len(E)} "
                                                   f"({100 * sum(m['ambiguous'] for m in E) / len(E):.2f}%)",
                             "mapping_review": f"{sum(r['verdict'] == 'CORRECT' for r in rev)}/{len(rev)}",
                             "mapping_review_changed_from_ai": sum(r["verdict"] != r["ai_proposed"] for r in rev),
                             "eligible_mapped_questions": len(kind("query"))}

    # S17-S18 replication
    gold = {r["query_id"]: set(r["gold_paragraph_ids"]) for r in kind("query")}
    lab = {(r["arm"], r["query_id"], r["passage_id"]): LETTER.get(r["state_5"]) for r in kind("label")}
    pools = {(r["arm"], r["query_id"]): r for r in kind("pool")}
    views = {}
    for depth in ("top10", "top20"):
        units, acc, miss = {}, {}, Counter()
        for r in kind("ranking"):
            if r["depth"] != depth: continue
            k = (r["arm"], r["query_id"], r["reranker"])
            if r["status"] != "OK": miss["invalid_ranking"] += 1; continue
            cid = r["rank1_passage_id"]; acc[k] = cid in gold[k[1]]
            if acc[k]: units[k] = (cid, "G"); continue
            s = lab.get((k[0], k[1], cid), "unlabelled")
            if s in (None, "unlabelled"): miss[s or "label_error"] += 1
            else: units[k] = (cid, s)
        views[depth] = (units, acc, dict(miss))
    pop = {k: {a: {q for (ar, q), p in pools.items() if ar == a and f(p)} for a in ARMS} for k, f in (
        ("top10", lambda p: p["gold_in_pool_top10"]), ("top20", lambda p: p["gold_in_pool_top20"]),
        ("matched", lambda p: p["gold_in_pool_top10"] and p["gold_in_pool_top20"]))}
    mis, _ = V["mistral"]
    hf_ref = {a: medians(*pair_metrics(mis, a))["HiddenFraction"] for a in ARMS}
    out, crit = {"missing_units": {d: v[2] for d, v in views.items()}}, {}
    for arm in ARMS:
        u10, a10, _ = views["top10"]
        blk = qasa_block(u10, a10, arm, pop["top10"][arm])
        bs = joint_bootstrap({k: v for k, v in u10.items() if k[1] in pop["top10"][arm]}, arm, B)
        low = {p: v for p, v in blk["pairs"].items() if v["abs_dAcc1"] < 0.02}
        margin = sample_median([v["D5"] - v["abs_dAcc1"] for v in low.values()]) if low else None
        c1 = blk["transformation_gt0"] >= 16
        c2 = bool(low) and margin >= 0.10 and all(v["D5"] > v["abs_dAcc1"] for v in low.values())
        c3 = blk["hiddenfraction_gt0"] == 15 and blk["HiddenFraction"] >= 0.5 * hf_ref[arm]
        out[arm] = {"gold_in_top10_pool": f"{len(pop['top10'][arm])}/{len(gold)} ({100 * len(pop['top10'][arm]) / len(gold):.1f}%)",
                    "acc1_range": [pct(min(blk["acc1"].values()), 1), pct(max(blk["acc1"].values()), 1)],
                    "transformation": fmt_ci(bs["transformation"]), "HiddenFraction": fmt_ci(bs["HiddenFraction"]),
                    "D5": fmt_ci(bs["D5"]), "transformation_gt0": f"{blk['transformation_gt0']}/30",
                    "hiddenfraction_gt0": f"{blk['hiddenfraction_gt0']}/15"}
        crit[arm] = {"C1": f"{blk['transformation_gt0']}/30 ({'PASS' if c1 else 'FAIL'})",
                     "C2": f"{len(low)} low-gap pairs; median excess {100 * margin:.1f} pp ({'PASS' if c2 else 'FAIL'})"
                     if low else "NOT_EVALUABLE",
                     "C3": f"{blk['hiddenfraction_gt0']}/15; median HF {100 * blk['HiddenFraction']:.2f}% vs reference "
                           f"{100 * hf_ref[arm]:.2f}% ({'PASS' if c3 else 'FAIL'})",
                     "conclusion": "Supports replication" if c1 and c2 and c3 else "Does not fully support replication"}
        u20, a20, _ = views["top20"]
        m10 = qasa_block(u10, a10, arm, pop["matched"][arm]); m20 = qasa_block(u20, a20, arm, pop["matched"][arm])
        out[arm]["nested_top10_to_top20"] = {"n_query_arms": len(pop["matched"][arm]),
                                             "transformation": [pct(m10["transformation"]), pct(m20["transformation"])],
                                             "HiddenFraction": [pct(m10["HiddenFraction"], 1), pct(m20["HiddenFraction"], 1)],
                                             "transformation_gt0": [m10["transformation_gt0"], m20["transformation_gt0"]]}
    R["S17_qasa_replication"] = out
    R["S18_qasa_criteria"] = crit


SECTIONS += [("S13", si_complete_pool), ("S14", si_depth), ("S15", si_topk), ("S16-S18", si_qasa)]


# ======================================================================= S19-S22: downstream
# ======================================================================= downstream: S19-S22, Fig. 5 / Fig. S2
UTILITY = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
CORR3 = ["INCORRECT", "PARTIAL", "CORRECT"]


def qasa_units(depth="top10"):
    """QASA Mistral view: comparable query-arms (mapped gold in the top-10 pool), Gold mechanical, else Mistral label."""
    rows = jl(RDATA / "qasa.jsonl")
    gold = {r["query_id"]: set(r["gold_paragraph_ids"]) for r in rows if r["kind"] == "query"}
    comp = {(r["arm"], r["query_id"]) for r in rows if r["kind"] == "pool" and r["gold_in_pool_top10"]}
    lab = {(r["arm"], r["query_id"], r["passage_id"]): LETTER.get(r["state_5"]) for r in rows if r["kind"] == "label"}
    units, missing = {}, 0
    for r in rows:
        if r["kind"] != "ranking" or r["depth"] != depth or (r["arm"], r["query_id"]) not in comp:
            continue
        pid = r["rank1_passage_id"]
        if pid is None or r.get("status") not in (None, "OK"):
            continue
        k = (r["arm"], r["query_id"], r["reranker"])
        if pid in gold[r["query_id"]]:
            units[k] = (pid, "G")
        elif lab.get((r["arm"], r["query_id"], pid)):
            units[k] = (pid, lab[(r["arm"], r["query_id"], pid)])
        else:
            missing += 1
    return units, missing


def load_downstream():
    """{judge: {dataset: {unit: record-or-None}}}; record = (correctness, faithfulness, abstained, U).
    A judgment counts only with a valid correctness and faithfulness label (judge errors stay missing)."""
    D = {"qwen": defaultdict(dict), "gemma": defaultdict(dict)}
    units = defaultdict(dict)
    for r in jl(RDATA / "downstream.jsonl"):
        if r["kind"] == "answer":
            units[r["dataset"]][r["unit"]] = (r["arm"], r["query_id"])
        else:
            j = r["kind"].split("_")[1]
            ok = r["correctness"] in UTILITY and r["faithfulness"] in ("SUPPORTED", "UNSUPPORTED")
            D[j][r["dataset"]][r["unit"]] = (r["correctness"], r["faithfulness"], bool(r["abstained"]),
                                             UTILITY[r["correctness"]]) if ok else None
    return units, D


def _l_cat(sa, sb, same):
    if sa in "XL":
        if sb in "GVC": return "resolution"
        if sb == sa: return "persistence_same" if same else "persistence_alt"
        return "transformation"
    return "regression" if sb in "XL" else "acceptable_retained"


def _m_cat(sa, sb, same):
    s1, s2 = sorted((sa, sb), key="GVCXL".index)
    if sa == sb:
        return f"{s1}<->{s2}", "same_state_same_passage" if same else "same_state_diff_passage"
    return f"{s1}<->{s2}", "hidden" if (sa in "XL") == (sb in "XL") else "visible"


def downstream_view(units, U, J, B):
    """Exps K/L/M on one dataset x state view x judge. Ratio quantities get per-query numerator/denominator
    contributions and the joint query bootstrap (per arm, sorted query ids, random.Random(42), s[int(.025B)],
    s[int(.975B)]; CI null if any replicate is undefined)."""
    out = {}
    for arm in ARMS:
        qids, per_q = arm_queries(units, arm)
        qi = {q: i for i, q in enumerate(qids)}
        n = len(qids)
        Q = defaultdict(lambda: ([0.0] * n, [0.0] * n))
        cnt = Counter()

        def add(name, i, num):
            Q[name][0][i] += num; Q[name][1][i] += 1.0
        unit_state = {}
        for (a, q, m), (cand, s) in units.items():
            unit_state[f"{q}|{a}|{cand}"] = s
        for uid, (a, q) in U.items():                       # K: unique generation units
            if a != arm or q not in qi or uid not in unit_state or not J.get(uid):
                continue
            s = unit_state[uid]
            add(f"K:{s}:mean_U", qi[q], J[uid][3]); cnt[f"K:{s}"] += 1
        for q in qids:
            i, d = qi[q], per_q[q]
            for a, b in itertools.permutations(MODELS, 2):  # L: ordered pairs
                if a not in d or b not in d: continue
                (ca, sa), (cb, sb) = d[a], d[b]
                ja, jb = J.get(f"{q}|{arm}|{ca}"), J.get(f"{q}|{arm}|{cb}")
                if not ja or not jb: continue
                cat = _l_cat(sa, sb, ca == cb)
                add(f"L:{cat}:mean_abs_dU", i, abs(jb[3] - ja[3])); cnt[f"L:{cat}"] += 1
                if sa in "XL" and sb in "XL" and sa != sb:
                    cnt[f"L:{sa}->{sb}"] += 1
            for a, b in itertools.combinations(MODELS, 2):  # M: unordered pairs
                if a not in d or b not in d: continue
                (ca, sa), (cb, sb) = d[a], d[b]
                ja, jb = J.get(f"{q}|{arm}|{ca}"), J.get(f"{q}|{arm}|{cb}")
                if not ja or not jb: continue
                key, cls = _m_cat(sa, sb, ca == cb)
                for nm in [f"M:class={cls}"] + ([f"M:{key}"] if cls in ("hidden", "visible") else []):
                    add(f"{nm}:mean_abs_dU", i, abs(jb[3] - ja[3])); cnt[nm] += 1
        names = sorted(Q)
        sparse = {k: [(i, Q[k][0][i], Q[k][1][i]) for i in range(n) if Q[k][1][i]] for k in names}
        point = {k: sum(x[1] for x in sparse[k]) / sum(x[2] for x in sparse[k]) for k in names}
        point["K:C-X:mean_U"] = point["K:C:mean_U"] - point["K:X:mean_U"]
        reps = {k: [] for k in point}
        rng = random.Random(SEED)
        for _ in range(B):
            w = [0] * n
            for i in rng.choices(range(n), k=n):
                w[i] += 1
            r = {}
            for k in names:
                num = den = 0.0
                for i, a_, b_ in sparse[k]:
                    if w[i]:
                        num += w[i] * a_; den += w[i] * b_
                r[k] = num / den if den else None
            r["K:C-X:mean_U"] = r["K:C:mean_U"] - r["K:X:mean_U"] if None not in (r["K:C:mean_U"], r["K:X:mean_U"]) else None
            for k in reps:
                reps[k].append(r[k])
        lo, hi = int(B * 0.025), int(B * 0.975)
        q_out = {}
        for k in point:
            s = sorted(x for x in reps[k] if x is not None)
            q_out[k] = dict(point=point[k], ci_lo=s[lo] if len(s) == B else None, ci_hi=s[hi] if len(s) == B else None)
        out[arm] = dict(q=q_out, n=dict(cnt))
    return out


DS_VIEWS = [("QASPER/GPT-4o", "qasper", "gpt4o"), ("QASPER/Mistral", "qasper", "mistral"), ("QASA/Mistral", "qasa", "qasa")]
CHANGE = [("Resolution", "L:resolution"), ("Regression", "L:regression"), ("G<->V", "M:G<->V"), ("G<->C", "M:G<->C"),
          ("V<->C", "M:V<->C"), ("X<->L", "M:X<->L"), ("Same-state passage change", "M:class=same_state_diff_passage"),
          ("A->A retained", "L:acceptable_retained")]


def _ci3(d):
    return None if d is None else f"{d['point']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]" if d["ci_lo"] is not None else f"{d['point']:.3f} [n/a]"


def si_downstream(I, V, B, R):
    U, D = load_downstream()
    views = {"gpt4o": V["gpt4o"][0], "mistral": V["mistral"][0], "qasa": qasa_units("top10")[0]}
    res = {}
    for judge in ("qwen", "gemma"):
        for label, ds, vk in DS_VIEWS:
            res[(judge, label)] = downstream_view(views[vk], U[ds], D[judge][ds], B)
    s19, s20, s21, fig, concl = {}, {}, {}, {}, {}
    for judge in ("qwen", "gemma"):
        for label, *_ in DS_VIEWS:
            for arm in ARMS:
                r = res[(judge, label)][arm]; q, n = r["q"], r["n"]
                cond = f"{label} {arm.capitalize()}"
                state = {s: q.get(f"K:{s}:mean_U") for s in "GVCXL"}
                change = {}
                for name, key in CHANGE:
                    nk = key.split(":", 1)[1] if key.startswith("M:class") else key.split(":", 1)[1]
                    cnt_key = key if key.startswith("M:") else key
                    change[name] = dict(value=_ci3(q.get(f"{key}:mean_abs_dU")), n=n.get(cnt_key, 0))
                change["X<->L"]["n_ordered"] = n.get("L:X->L", 0) + n.get("L:L->X", 0)
                cx = q["K:C-X:mean_U"]
                acc = [q.get(f"M:{p}:mean_abs_dU") for p in ("G<->V", "G<->C", "V<->C")]
                xl = q.get("M:X<->L:mean_abs_dU"); tr = q.get("L:transformation:mean_abs_dU")
                res_abs = q["L:resolution:mean_abs_dU"]["point"]
                c4 = dict(
                    ordering_GVCXL=all(state[a]["point"] > state[b]["point"] for a, b in zip("GVCX", "VCXL")),
                    C_minus_X_positive_ci_excludes_0=cx["point"] > 0 and cx["ci_lo"] is not None and cx["ci_lo"] > 0,
                    resolution_gt_transformation=tr is None or res_abs > tr["point"],
                    acceptable_internal_gt_retained_refs=all(
                        a is not None and a["point"] > max(q["M:class=same_state_diff_passage:mean_abs_dU"]["point"],
                                                           q["L:acceptable_retained:mean_abs_dU"]["point"]) for a in acc),
                    acceptable_internal_gt_XL_and_transformation=all(
                        a is not None and a["point"] > (xl["point"] if xl else 0) and a["point"] > (tr["point"] if tr else 0) for a in acc))
                block = dict(state_mean_U={s: _ci3(v) for s, v in state.items()}, C_minus_X=_ci3(cx), change=change)
                fig[f"{judge}:{cond}"] = block
                concl[f"{judge}:{cond}"] = c4
                if judge == "qwen":
                    s19[cond] = dict(block["state_mean_U"], C_minus_X=block["C_minus_X"])
                    s20[cond] = {k: change[k] for k in ("Resolution", "Regression", "G<->V", "G<->C")}
                    s21[cond] = {k: change[k] for k in ("V<->C", "X<->L", "Same-state passage change", "A->A retained")}
    cxs = [res[("qwen", l)][a]["q"]["K:C-X:mean_U"]["point"] for l, *_ in DS_VIEWS for a in ARMS]
    R["S19_state_utility"] = s19 | {"C_minus_X_range": [round(min(cxs), 3), round(max(cxs), 3)]}
    R["S20_change_a"] = s20
    R["S21_change_b"] = s21
    R["Fig5_FigS2_data"] = dict(by_judge_condition=fig, conclusions=concl,
                                conclusions_all_hold={j: {c: all(v[c] for k, v in concl.items() if k.startswith(j))
                                                          for c in next(iter(concl.values()))} for j in ("qwen", "gemma")})


def _kappa_w(pairs, cats, weighted=False):
    n = len(pairs); k = len(cats); ix = {c: i for i, c in enumerate(cats)}
    m = [[0] * k for _ in range(k)]
    for a, b in pairs: m[ix[a]][ix[b]] += 1
    ra = [sum(r) / n for r in m]; cb = [sum(m[i][j] for i in range(k)) / n for j in range(k)]
    w = (lambda i, j: abs(i - j) / (k - 1)) if weighted else (lambda i, j: 0.0 if i == j else 1.0)
    do = sum(w(i, j) * m[i][j] / n for i in range(k) for j in range(k))
    de = sum(w(i, j) * ra[i] * cb[j] for i in range(k) for j in range(k))
    return None if de == 0 else 1 - do / de


def si_evaluator_agreement(I, V, B, R):
    """S22: Qwen vs Gemma on units both judges evaluated (non-abstained, no judge error in either run)."""
    U, D = load_downstream()
    pairs, n_units, gem_err, abst = [], 0, 0, 0
    for ds in ("qasper", "qasa"):
        for uid in U[ds]:
            n_units += 1
            q, g = D["qwen"][ds].get(uid), D["gemma"][ds].get(uid)
            if g is None: gem_err += 1
            if q is None or g is None: continue
            if q[2]: abst += 1; continue
            pairs.append((q, g))
    cp = [(q[0], g[0]) for q, g in pairs]; fp = [(q[1], g[1]) for q, g in pairs]
    n = len(pairs)
    R["S22_qwen_gemma"] = dict(
        units=n_units, judge_errors_missing=gem_err, rule_abstentions_excluded=abst, llm_evaluated_units=n,
        correctness_exact=pct(sum(a == b for a, b in cp) / n, 1), correctness_kappa=round(_kappa_w(cp, CORR3), 3),
        correctness_linear_weighted_kappa=round(_kappa_w(cp, CORR3, True), 3),
        mean_U_qwen=round(sum(UTILITY[a] for a, _ in cp) / n, 3), mean_U_gemma=round(sum(UTILITY[b] for _, b in cp) / n, 3),
        faithfulness_exact=pct(sum(a == b for a, b in fp) / n, 1),
        faithfulness_kappa=round(_kappa_w(fp, ["UNSUPPORTED", "SUPPORTED"]), 3))


SECTIONS += [("downstream", si_downstream), ("evaluator_agreement", si_evaluator_agreement)]


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    I = load_inputs()
    V = qasper_views(I)
    R = {"B": args.B, "seed": SEED, "missing_units": {k: v[1] for k, v in V.items()}}
    si_qasper(I, V, args.B, R)
    si_reference(R)
    for name, fn in SECTIONS:
        fn(I, V, args.B, R)
    json.dump(R, open(OUT / "si_tables.json", "w"), indent=1)
    print(json.dumps(R, indent=1)[:200000])


if __name__ == "__main__":
    main()
