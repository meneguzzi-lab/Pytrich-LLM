#!/usr/bin/env python3
"""Pair the fixed model-and-algorithm configuration, the fully fixed configuration and the
LLM virtual best against TDG, LMCount and PANDA RC-FF on expanded nodes, per domain.

    python3 fixed_vs_pytrich.py RESULTS.csv

PAIR and TRIPLE below are the picks that fixed_pair_tables.py and fixed_config.py print for
Experiments/results/results.csv; re-run those first on a new snapshot and update them.
Output: wins/shared (percent) at the median per-instance node ratio.
"""
import csv
import statistics
import sys
from collections import defaultdict

CSV = sys.argv[1]
PAIR = {"Barman-BDI": ("Gemini3Flash", "GBFS"), "Blocksworld-GTOHP": ("Gemini3Flash", "GBFS"),
        "Depots": ("Gemini3Flash", "WA"), "Robot": ("ClaudeOpus46", "StandardA"),
        "Rover-GTOHP": ("Gemini3Pro", "GBFS"), "Towers": ("Gemini3Pro", "WA"),
        "Lamps": ("Gemini3Flash", "StandardA")}
TRIPLE = {"Barman-BDI": ("Gemini3Flash", "heuristic_0", "GBFS"),
          "Blocksworld-GTOHP": ("Gemini3Flash", "heuristic_4", "GBFS"),
          "Depots": ("Gemini3Flash", "heuristic_0", "WA"),
          "Robot": ("ClaudeOpus46", "RobotAnthropicOpus46Heuristic12", "GBFS"),
          "Rover-GTOHP": ("Gemini3Pro", "heuristic_0", "GBFS"),
          "Towers": ("Gemini3Pro", "heuristic_0", "WA"),
          "Lamps": ("Gemini3Flash", "heuristic_14", "StandardA")}
LLM = {"ClaudeOpus46", "Gemini3Flash", "Gemini3Pro", "Gemini20Flash", "Gemini25Pro"}

best = defaultdict(dict)   # system -> (domain, problem) -> fewest expanded nodes
for r in csv.DictReader(open(CSV, newline="", encoding="utf-8")):
    if not r["solution_size"]:
        continue
    name = r["experiment_name"]
    m = name.split("_")[0]
    a = name.split("_", 1)[1] if "_" in name else "ALL"
    try:
        n = int(float(r["expanded_nodes"]))
    except (ValueError, TypeError):
        continue
    d, p, h = r["domain_name"], r["problem_name"], r["heuristic_name"]

    def put(system):
        k = (d, p)
        if k not in best[system] or n < best[system][k]:
            best[system][k] = n

    if m in ("Pytrich-TDG", "Pytrich-LMCOUNT", "Panda-rc2ff"):
        put(m)
    if PAIR.get(d) == (m, a):
        put("pair")
    if TRIPLE.get(d) == (m, h, a):
        put("triple")
    if m in LLM:
        put("vb")


def paired(x, ref, dom=None):
    sh = [k for k in set(best[x]) & set(best[ref]) if (dom is None or k[0] == dom) and best[ref][k] > 0]
    if not sh:
        return "--"
    w = sum(1 for k in sh if best[x][k] < best[ref][k])
    med = statistics.median(best[x][k] / best[ref][k] for k in sh)
    return f"{w}/{len(sh)} ({100 * w / len(sh):.0f}%) at {med:.2f}"


for x in ("pair", "triple", "vb"):
    print("==", x, "coverage", len(best[x]))
    for ref in ("Pytrich-TDG", "Pytrich-LMCOUNT", "Panda-rc2ff"):
        print(f"  vs {ref:16} all: {paired(x, ref)}")
        for d in PAIR:
            print(f"      {d:18} {paired(x, ref, d)}")
