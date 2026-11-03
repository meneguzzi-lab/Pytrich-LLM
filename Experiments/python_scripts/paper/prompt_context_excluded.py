#!/usr/bin/env python3
"""
Coverage with the prompt-context instances excluded.

Two benchmark instances per domain -- the smallest and the largest, named in
`config/domains.yaml` as `smallest_instance` / `largest_instance` -- are pasted into
the generation prompt as context. Those same instances are then part of the set the
paper reports coverage over, so prompt construction and the test set overlap. This
recomputes every coverage column with the 14 prompt-context instances removed.

The fixed-pair column still *selects* on the smallest instance -- selection is part
of the method, not of the evaluation -- but that instance no longer *counts* toward
its coverage. That asymmetry is deliberate.

Usage:
    python3 prompt_context_excluded.py [results.csv] [--markdown]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

DOMAINS = ["Barman-BDI", "Blocksworld-GTOHP", "Depots", "Robot", "Rover-GTOHP", "Towers", "Lamps"]

# The paper's instance set (matches generate_ablation_table.py and fixed_config.py).
DISK_SIZES = {"Barman-BDI": 20, "Blocksworld-GTOHP": 30, "Depots": 30,
              "Robot": 20, "Rover-GTOHP": 30, "Towers": 20, "Lamps": 30}
PAPER_SIZES = dict(DISK_SIZES)

# smallest_instance / largest_instance from config/domains.yaml, as they appear in
# the results CSV (no .hddl suffix).
PROMPT_CONTEXT = {
    "Barman-BDI":        ["pfile01", "pfile20"],
    "Blocksworld-GTOHP": ["p01", "p30"],
    "Depots":            ["p01", "p30"],
    "Robot":             ["pfile_01_001", "pfile_50_100"],
    "Rover-GTOHP":       ["p01", "p30"],
    "Towers":            ["pfile_01", "pfile_20"],
    "Lamps":             ["pfile01", "pfile30"],
}
SELECTION = {d: v[0] for d, v in PROMPT_CONTEXT.items()}

# The paper's five main models (the \llmvb{} column of Table 1).
LLM_MODELS = {"ClaudeOpus46", "Gemini3Flash", "Gemini3Pro", "Gemini20Flash", "Gemini25Pro"}
BASELINES = [("PANDA RC-FF", "Panda-rc2ff"), ("PANDA RC-LMCut", "Panda-rc2lmc"),
             ("PANDA LMCount^P", "Panda-lmcountlmtypebid"),
             ("TDG", "Pytrich-TDG"), ("LMCount", "Pytrich-LMCOUNT")]


def paper_instance_sets(rows):
    """domain -> the paper's instance set, taken by rank over the names present.

    Capping by rank rather than by parsing an index out of the name matters twice:
    Robot's `pfile_50_100` has no meaningful trailing index, and the reconstructed
    snapshot carries PANDA runs past the cap (Depots p28/p29) that must be dropped
    rather than counted.
    """
    seen = defaultdict(set)
    for d, p in rows:
        seen[d].add(p)
    sets = {}
    for d in DOMAINS:
        names = sorted(seen.get(d, ()))
        want = PAPER_SIZES[d]
        if len(names) < want:
            # The dataset of record holds solved runs only, so an instance no system
            # solves (Towers pfile_16 to 20, Lamps 20 and 22 to 30) has no row; the
            # denominator stays PAPER_SIZES[d] and the set is every instance present.
            sets[d] = set(names)
        else:
            sets[d] = set(names[:want])
    return sets


def load(path):
    """(runs, paper_sets); runs[(system, heuristic, domain, problem)] = (nodes, time)."""
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
    paper_sets = paper_instance_sets((r['domain_name'], r['problem_name']) for r in rows)
    runs = {}
    for r in rows:
        d, p = r['domain_name'], r['problem_name']
        if p not in paper_sets.get(d, ()) or not r['solution_size']:
            continue
        try:
            rec = (int(float(r['expanded_nodes'])), float(r['search_elapsed_time']))
        except (ValueError, TypeError):
            continue
        key = (r['experiment_name'], r['heuristic_name'], d, p)
        if key not in runs or rec[0] < runs[key][0]:
            runs[key] = rec
    return runs, paper_sets


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    csv_path = args[0] if args else str(RESULTS_DIR / "results.csv")
    if not Path(csv_path).exists():
        sys.exit(f"Result CSV not found: {csv_path}")
    runs, paper_sets = load(csv_path)

    # Which prompt-context instances actually sit inside the paper's evaluated set.
    excluded = {d: [p for p in PROMPT_CONTEXT[d] if p in paper_sets[d]] for d in DOMAINS}

    def covers(pred):
        """domain -> set of solved problems, over runs matching pred(system)."""
        out = defaultdict(set)
        for (sysname, _h, d, p) in runs:
            if pred(sysname):
                out[d].add(p)
        return out

    columns = []
    for label, prefix in BASELINES:
        columns.append((label, covers(lambda s, _p=prefix: s == _p or s.startswith(_p + "_"))))
    columns.append(("LLM-VB (oracle)",
                    covers(lambda s: s.split('_')[0] in LLM_MODELS)))

    # Fixed pair: per-domain (model, algorithm) fixed on the selection instance, with a
    # per-instance min over that model's heuristics. This is fixed_config.py's P2 -- the
    # protocol behind the "pair chosen" column, which names (model, algorithm)
    # rather than a single heuristic. Tie-break: fewest nodes, then fastest, then name.
    ma_solved, ma_best = defaultdict(set), {}
    for (sysname, _h, d, p), rec in runs.items():
        model = sysname.split('_')[0]
        if model not in LLM_MODELS:
            continue
        algo = sysname.split('_', 1)[1] if '_' in sysname else 'ALL'
        ma_solved[(model, algo)].add((d, p))
        k = (model, algo, d, p)
        if k not in ma_best or rec[0] < ma_best[k][0]:
            ma_best[k] = rec
    fixed, picks = defaultdict(set), {}
    for d in DOMAINS:
        sel = SELECTION[d]
        cand = [ma for ma in ma_solved if (ma[0], ma[1], d, sel) in ma_best]
        if not cand:
            continue
        best = min(ma_best[(m, a, d, sel)][0] for (m, a) in cand)
        tied = [c for c in cand if ma_best[(c[0], c[1], d, sel)][0] == best]
        pick = min(tied, key=lambda c: (ma_best[(c[0], c[1], d, sel)][1], c))
        picks[d] = pick
        fixed[d] = {p for (dd, p) in ma_solved[pick] if dd == d}
    columns.append(("LLM fixed pair", fixed))

    md = "--markdown" in sys.argv
    w = sys.stdout.write

    w("# Coverage with prompt-context instances excluded\n\n" if md else "")
    if md:
        w("The generation prompt contains two benchmark instances per domain (the\n"
          "`smallest_instance` and `largest_instance` of `config/domains.yaml`). Those\n"
          "instances are also in the evaluated set. This table recomputes coverage with\n"
          "them removed, to show the reported result does not rest on them.\n\n")
        w(f"Source: `{Path(csv_path).name}`, a single reconstructed snapshot so PANDA and\n"
          "PYTRICH come from one sweep.\n\n")
        w("## Which instances are removed\n\n")
        w("| domain | prompt-context | in paper's set | removed |\n|---|---|---|---|\n")
        for d in DOMAINS:
            allp = ", ".join(f"`{p}`" for p in PROMPT_CONTEXT[d])
            inp = ", ".join(f"`{p}`" for p in excluded[d]) or "—"
            w(f"| {d} | {allp} | {PAPER_SIZES[d]} instances | {inp} |\n")
        n_removed = sum(len(v) for v in excluded.values())
        w(f"\n{n_removed} of 12 prompt-context instances are inside the paper's "
          f"{sum(PAPER_SIZES.values())}. The reduced set is {sum(PAPER_SIZES.values()) - n_removed}.\n\n")
        w("## Coverage\n\n")

    # Default is the reduced coverage alone -- the table is the post-exclusion result,
    # and carrying the pre-exclusion number in every cell only invites the reader to
    # compare against a column that is not the point. --delta restores "before → after",
    # which is what to use when checking these against the paper's totals.
    delta = "--delta" in sys.argv

    # Pipes inside a markdown cell must be escaped or they split the column.
    hdr = ["domain", r"\|I'\|" if md else "|I*|"] + [c[0] for c in columns]
    if delta:
        hdr.insert(1, r"\|I\|" if md else "|I|")
    if md:
        w("| " + " | ".join(hdr) + " |\n")
        w("|" + "---|" * len(hdr) + "\n")
    else:
        w(f"{'domain':<19}" + (f"{'|I|':>5}" if delta else "") + f"{'|I*|':>6}" +
          "".join(f"{c[0]:>18}" for c in columns) + "\n")

    tot_full = tot_red = 0
    tot = [[0, 0] for _ in columns]
    for d in DOMAINS:
        full_n = PAPER_SIZES[d]
        red_n = full_n - len(excluded[d])
        tot_full += full_n
        tot_red += red_n
        cells = []
        for i, (_lab, cov) in enumerate(columns):
            s = cov.get(d, set())
            a = len(s)
            b = len(s - set(excluded[d]))
            tot[i][0] += a
            tot[i][1] += b
            cells.append(f"{a} → {b}" if delta else f"{b}")
        lead = [d] + ([str(full_n)] if delta else []) + [str(red_n)]
        if md:
            w("| " + " | ".join(lead + cells) + " |\n")
        else:
            w(f"{d:<19}" + (f"{full_n:>5}" if delta else "") + f"{red_n:>6}" +
              "".join(f"{c:>18}" for c in cells) + "\n")

    cells = [(f"{a} → {b}" if delta else f"{b}") for a, b in tot]
    if md:
        cells = [f"**{c}**" for c in cells]
        lead = ["**Total**"] + ([f"**{tot_full}**"] if delta else []) + [f"**{tot_red}**"]
        w("| " + " | ".join(lead + cells) + " |\n")
        w("\nEach cell is coverage with the prompt-context instances removed"
          + (", shown as *full set → reduced set*" if delta else "")
          + f". `|I'|` is the reduced denominator ({tot_red} of {tot_full}).\n")
        w("\nFixed pair chosen on the selection instance: " +
          ", ".join(f"{d} = {picks[d][0]}/{picks[d][1]}" for d in DOMAINS if d in picks) + ".\n")
        w("\nSelection still runs on the smallest instance — that is part of the method. "
          "It simply no longer counts toward the reported coverage.\n")
    else:
        w(f"{'TOTAL':<19}" + (f"{tot_full:>5}" if delta else "") + f"{tot_red:>6}" +
          "".join(f"{c:>18}" for c in cells) + "\n")


if __name__ == "__main__":
    main()
