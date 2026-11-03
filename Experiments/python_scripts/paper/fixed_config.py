#!/usr/bin/env python3
"""
Deployable single-configuration results.

A "configuration" is (model, heuristic, algorithm). The virtual best reported in the
paper takes a per-instance minimum over every configuration; that is an oracle. Here
each domain gets ONE configuration, chosen using the selection instance only (the
smallest instance of the domain, which is also what the generate-evaluate-select
filter used), and we then report that configuration's coverage over the full domain.

Three protocols:
  P1  per-domain (model, heuristic, algorithm) fixed on the selection instance
  P2  per-domain (model, algorithm) fixed, per-instance min over that model's
      heuristics  -- still mildly oracular
  P3  (model, algorithm) fixed GLOBALLY in advance, per-domain heuristic chosen on
      the selection instance  -- the most deployable protocol

Tie-breaking on the selection instance is reported explicitly: where the selection
instance cannot separate configurations, we give the coverage range over all tied
configurations, so the number does not hide a lucky pick.
"""
import csv, statistics
from collections import defaultdict

from pathlib import Path as _Path
import sys as _sys
RESULTS_DIR = _Path(__file__).resolve().parents[2] / "results"
def _csv(name="results.csv"):
    """CSV path: first CLI arg if given, else Experiments/results/results.csv."""
    for a in _sys.argv[1:]:
        if a.endswith(".csv"):
            return a
    p = RESULTS_DIR / name
    if not p.exists():
        _sys.exit(f"Result CSV not found: {p}")
    return str(p)


CSV = _csv()

DOMAINS = ["Barman-BDI", "Blocksworld-GTOHP", "Depots", "Robot", "Rover-GTOHP", "Towers",
           "Lamps"]

# All 180 instances over seven domains.
DISK_SIZES = {"Barman-BDI": 20, "Blocksworld-GTOHP": 30, "Depots": 30,
              "Robot": 20, "Rover-GTOHP": 30, "Towers": 20, "Lamps": 30}
DOMAIN_SIZES = dict(DISK_SIZES)
DOMAINS = [d for d in DOMAINS if d in DOMAIN_SIZES]
TOTAL = sum(DOMAIN_SIZES.values())
INSTANCE_INDEX = __import__("re").compile(r"(\d+)\D*$")


def in_set(domain, problem):
    limit = DOMAIN_SIZES.get(domain)
    if limit is None:
        return False
    if limit >= DISK_SIZES.get(domain, limit):
        return True
    m = INSTANCE_INDEX.search(problem)
    if not m:
        _sys.exit(f"cannot read an instance index from {domain} '{problem}'")
    return int(m.group(1)) <= limit
SELECTION = {"Barman-BDI": "pfile01", "Blocksworld-GTOHP": "p01", "Depots": "p01",
             "Robot": "pfile_01_001", "Rover-GTOHP": "p01", "Towers": "pfile_01",
             "Lamps": "pfile01"}
LLM_MODELS = {"ClaudeOpus46", "Gemini3Flash", "Gemini3Pro",
              "Gemini20Flash", "Gemini25Pro"}
ALGOS = ("GBFS", "StandardA", "WA")

# rows[(model, heur, algo, domain, problem)] = (nodes, time, plan_len)
runs = {}
llm_rows = 0
for r in csv.DictReader(open(CSV, newline='', encoding='utf-8')):
    if not r['solution_size'] or not in_set(r['domain_name'], r['problem_name']):
        continue
    name = r['experiment_name']
    model = name.split('_')[0]
    algo = name.split('_', 1)[1] if '_' in name else 'ALL'
    try:
        rec = (int(float(r['expanded_nodes'])), float(r['search_elapsed_time']),
               int(float(r['solution_size'])))
    except (ValueError, TypeError):
        continue
    key = (model, r['heuristic_name'], algo, r['domain_name'], r['problem_name'])
    if key not in runs or rec[0] < runs[key][0]:
        runs[key] = rec
    if model in LLM_MODELS:
        llm_rows += 1

print(f"{llm_rows} LLM runs over {len(set(k[:3] for k in runs if k[0] in LLM_MODELS))} "
      f"(model, heuristic, algorithm) configurations\n")

configs = sorted(set(k[:3] for k in runs if k[0] in LLM_MODELS))
solved = defaultdict(set)          # config -> {(domain, problem)}
for (m, h, a, d, p), _ in runs.items():
    if m in LLM_MODELS:
        solved[(m, h, a)].add((d, p))


def dom_solved(cfg, domain):
    return {p for (d, p) in solved[cfg] if d == domain}


def stats(pairs, getter):
    vals = [getter(x) for x in pairs]
    return statistics.median(vals) if vals else float('nan')


# ---------------------------------------------------------------- P1
print("=" * 96)
print("P1  per-domain (model, heuristic, algorithm) fixed on the selection instance")
print("=" * 96)
print(f"{'Domain':<19}{'sel':<14}{'tied':>5}{'chosen configuration':<42}{'cov':>8}{'range over ties':>17}")
p1_total = 0
p1_lo = p1_hi = 0
p1_choice = {}
for d in DOMAINS:
    sel = SELECTION[d]
    # candidates: configurations that solved the selection instance
    cand = [(m, h, a) for (m, h, a) in configs if (m, h, a, d, sel) in runs]
    if not cand:
        print(f"{d:<19}{sel:<14}{'0':>5}  -- nothing solved the selection instance --")
        continue
    best = min(runs[(m, h, a, d, sel)][0] for (m, h, a) in cand)
    tied = [c for c in cand if runs[(c[0], c[1], c[2], d, sel)][0] == best]
    # deterministic pick: fewest nodes, then fastest, then name
    pick = min(tied, key=lambda c: (runs[(c[0], c[1], c[2], d, sel)][1], c))
    covs = sorted(len(dom_solved(c, d)) for c in tied)
    cov = len(dom_solved(pick, d))
    p1_total += cov
    p1_lo += covs[0]
    p1_hi += covs[-1]
    p1_choice[d] = pick
    print(f"{d:<19}{sel:<14}{len(tied):>5}  {'/'.join(pick):<40}{cov:>3}/{DOMAIN_SIZES[d]:<4}"
          f"{f'{covs[0]}-{covs[-1]}':>17}")
print(f"{'TOTAL':<19}{'':<14}{'':>5}{'':<42}{p1_total:>3}/{TOTAL}   "
      f"(over tied configs: {p1_lo}-{p1_hi})")

# ---------------------------------------------------------------- P2
print()
print("=" * 96)
print("P2  per-domain (model, algorithm) fixed on the selection instance, min over that")
print("    model's heuristics per instance")
print("=" * 96)
ma_solved = defaultdict(set)
ma_best = {}
for (m, h, a, d, p), rec in runs.items():
    if m not in LLM_MODELS:
        continue
    ma_solved[(m, a)].add((d, p))
    k = (m, a, d, p)
    if k not in ma_best or rec[0] < ma_best[k][0]:
        ma_best[k] = rec
print(f"{'Domain':<19}{'tied':>5}{'chosen (model, algo)':<34}{'cov':>8}{'range over ties':>17}")
p2_total = p2_lo = p2_hi = 0
p2_choice = {}
for d in DOMAINS:
    sel = SELECTION[d]
    cand = [(m, a) for m in sorted(LLM_MODELS) for a in ALGOS if (m, a, d, sel) in ma_best]
    if not cand:
        print(f"{d:<19}  -- nothing solved the selection instance --")
        continue
    best = min(ma_best[(m, a, d, sel)][0] for (m, a) in cand)
    tied = [c for c in cand if ma_best[(c[0], c[1], d, sel)][0] == best]
    pick = min(tied, key=lambda c: (ma_best[(c[0], c[1], d, sel)][1], c))
    covs = sorted(len({p for (dd, p) in ma_solved[c] if dd == d}) for c in tied)
    cov = len({p for (dd, p) in ma_solved[pick] if dd == d})
    p2_total += cov; p2_lo += covs[0]; p2_hi += covs[-1]
    p2_choice[d] = pick
    print(f"{d:<19}{len(tied):>5}  {pick[0] + ' / ' + pick[1]:<32}{cov:>3}/{DOMAIN_SIZES[d]:<4}"
          f"{f'{covs[0]}-{covs[-1]}':>17}")
print(f"{'TOTAL':<19}{'':>5}{'':<34}{p2_total:>3}/{TOTAL}   (over tied configs: {p2_lo}-{p2_hi})")

# ---------------------------------------------------------------- P3
print()
print("=" * 96)
print("P3  (model, algorithm) fixed GLOBALLY, per-domain heuristic chosen on the selection")
print("    instance. Ranked by total coverage; the a-priori pick cannot use these columns.")
print("=" * 96)
print(f"{'model / algorithm':<34}{'domains w/ a candidate':>24}{'total cov':>11}{'median nodes':>14}{'median s':>10}")
rank = []
for m in sorted(LLM_MODELS):
    for a in ALGOS:
        tot = 0
        got = 0
        nodes, times = [], []
        for d in DOMAINS:
            sel = SELECTION[d]
            hs = [h for (mm, h, aa, dd, pp) in runs
                  if mm == m and aa == a and dd == d and pp == sel]
            if not hs:
                continue
            got += 1
            pick = min(set(hs), key=lambda h: (runs[(m, h, a, d, sel)][0],
                                               runs[(m, h, a, d, sel)][1], h))
            ps = dom_solved((m, pick, a), d)
            tot += len(ps)
            for p in ps:
                nodes.append(runs[(m, pick, a, d, p)][0])
                times.append(runs[(m, pick, a, d, p)][1])
        if got:
            rank.append((tot, m, a, got, statistics.median(nodes), statistics.median(times)))
for tot, m, a, got, mn, mt in sorted(rank, reverse=True):
    print(f"{m + ' / ' + a:<34}{got:>22}/{len(DOMAINS)}{tot:>9}/{TOTAL}{mn:>14,.0f}{mt:>10.2f}")

# ---------------------------------------------------------------- reference points
print()
print("=" * 96)
print("Reference points (same CSV, same solved rule)")
print("=" * 96)


def vb(pred):
    s = defaultdict(set)
    for (m, h, a, d, p) in runs:
        if pred(m):
            s[d].add(p)
    return s


def show(label, s):
    tot = sum(len(s[d]) for d in DOMAINS)
    print(f"{label:<44}" + "  ".join(f"{d[:4]} {len(s[d]):>2}/{DOMAIN_SIZES[d]:<2}" for d in DOMAINS)
          + f"   total {tot}/{TOTAL}")


show("LLM virtual best (5 models x 3 algos x all heur)", vb(lambda m: m in LLM_MODELS))
show("Claude Opus 4.6 virtual best (3 algos x heur)", vb(lambda m: m == 'ClaudeOpus46'))
show("PANDA RC-Add virtual best (3 algos)", vb(lambda m: m == 'Panda-rc2add'))
show("PANDA RC-FF virtual best (3 algos)", vb(lambda m: m == 'Panda-rc2ff'))
show("PANDA RC-LMC virtual best (3 algos)", vb(lambda m: m == 'Panda-rc2lmc'))
show("Pytrich TDG (algorithms not separable in CSV)", vb(lambda m: m == 'Pytrich-TDG'))
show("Pytrich LMCOUNT (algorithms not separable)", vb(lambda m: m == 'Pytrich-LMCOUNT'))
# PANDA per single algorithm, for a like-for-like fixed comparison
for a in ALGOS:
    s = defaultdict(set)
    for (m, h, aa, d, p) in runs:
        if m == 'Panda-rc2ff' and aa == a:
            s[d].add(p)
    show(f"PANDA RC-FF, {a} only (single fixed config)", s)

# median nodes / time for the P1 and P2 picks, on their own solved sets and shared sets
print()
print("=" * 96)
print("Cost of the chosen configurations vs the oracle, on the instances each solves")
print("=" * 96)
llm_vb_nodes = {}
for (m, h, a, d, p), rec in runs.items():
    if m in LLM_MODELS:
        k = (d, p)
        if k not in llm_vb_nodes or rec[0] < llm_vb_nodes[k][0]:
            llm_vb_nodes[k] = rec
pan_nodes = {}
for (m, h, a, d, p), rec in runs.items():
    if m == 'Panda-rc2ff':
        k = (d, p)
        if k not in pan_nodes or rec[0] < pan_nodes[k][0]:
            pan_nodes[k] = rec

p1_runs = {}
for d, cfg in p1_choice.items():
    for p in dom_solved(cfg, d):
        p1_runs[(d, p)] = runs[(cfg[0], cfg[1], cfg[2], d, p)]

print(f"{'system':<40}{'solved':>8}{'median nodes':>14}{'median s':>10}{'mean s':>9}")
for label, data in (("LLM virtual best (oracle)", llm_vb_nodes),
                    ("P1 per-domain fixed configuration", p1_runs),
                    ("PANDA RC-FF virtual best", pan_nodes)):
    print(f"{label:<40}{len(data):>8}{statistics.median(v[0] for v in data.values()):>14,.0f}"
          f"{statistics.median(v[1] for v in data.values()):>10.2f}"
          f"{statistics.mean(v[1] for v in data.values()):>9.2f}")

shared = set(p1_runs) & set(pan_nodes)
print(f"\nOn the {len(shared)} instances solved by both the P1 fixed configuration and PANDA RC-FF:")
w = sum(1 for k in shared if p1_runs[k][0] < pan_nodes[k][0])
print(f"  P1 expands fewer nodes on {w}/{len(shared)}  "
      f"(median {statistics.median(p1_runs[k][0] for k in shared):,.0f} vs "
      f"{statistics.median(pan_nodes[k][0] for k in shared):,.0f})")
shared_vb = set(llm_vb_nodes) & set(pan_nodes)
w2 = sum(1 for k in shared_vb if llm_vb_nodes[k][0] < pan_nodes[k][0])
print(f"  oracle wins {w2}/{len(shared_vb)} for comparison")
