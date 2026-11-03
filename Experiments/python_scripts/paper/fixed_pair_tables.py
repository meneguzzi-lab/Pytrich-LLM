#!/usr/bin/env python3
"""
Coverage / expanded-nodes / search-time tables for a single fixed (model, search
algorithm) pair per domain.

The pair is chosen per domain on the development (selection) instance alone: fewest
expanded nodes, ties broken by time then name. Nothing downstream of that choice looks
at a test result, which is the whole point of the question.

    python3 fixed_pair_tables.py RESULTS.csv OUTDIR

Writes coverage_fixed_pair.md, nodes_fixed_pair.md and runtime_fixed_pair.md.

Counted over all 180 instances of the seven domains.
"""
import csv
import os
import re
import statistics
import sys

DOMS = ["Barman-BDI", "Blocksworld-GTOHP", "Depots", "Robot", "Rover-GTOHP", "Towers",
        "Lamps"]
DISK_SIZES = {"Barman-BDI": 20, "Blocksworld-GTOHP": 30, "Depots": 30,
              "Robot": 20, "Rover-GTOHP": 30, "Towers": 20, "Lamps": 30}
SELECTION = {"Barman-BDI": "pfile01", "Blocksworld-GTOHP": "p01", "Depots": "p01",
             "Robot": "pfile_01_001", "Rover-GTOHP": "p01", "Towers": "pfile_01",
             "Lamps": "pfile01"}
LLM_MODELS = {"ClaudeOpus46", "Gemini3Flash", "Gemini3Pro",
              "Gemini20Flash", "Gemini25Pro"}
ALGOS = ("GBFS", "StandardA", "WA")
INDEX = re.compile(r"(\d+)\D*$")

# label -> (column header, predicate on (model, algo))
BASELINES = [
    ("PANDA / RC-FF",      lambda m, a: m == "Panda-rc2ff"),
    ("PANDA / RC-LMCut",   lambda m, a: m == "Panda-rc2lmc"),
    ("PANDA / RC-Add",      lambda m, a: m == "Panda-rc2add"),
    ("PANDA / LMCount^P",  lambda m, a: m == "Panda-lmcountlmtypebid"),
    ("Pytrich / TDG",      lambda m, a: m == "Pytrich-TDG"),
    ("Pytrich / LMCount",  lambda m, a: m == "Pytrich-LMCOUNT"),
    ("LLM-VB",             lambda m, a: m in LLM_MODELS),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        sys.exit(__doc__)
    csv_path, outdir = args
    sizes = DISK_SIZES
    sizes = {d: n for d, n in sizes.items() if n}
    total = sum(sizes.values())

    def in_set(d, p):
        limit = sizes.get(d)
        if limit is None or limit >= DISK_SIZES.get(d, limit):
            return True
        m = INDEX.search(p)
        if not m:
            sys.exit(f"cannot read an instance index from {d} '{p}'")
        return int(m.group(1)) <= limit

    rows = []
    for r in csv.DictReader(open(csv_path, newline="", encoding="utf-8")):
        if not r["solution_size"]:
            continue
        d, p = r["domain_name"], r["problem_name"]
        if d not in sizes or not in_set(d, p):
            continue
        name = r["experiment_name"]
        model = name.split("_")[0]
        algo = name.split("_", 1)[1] if "_" in name else "ALL"
        try:
            rows.append((model, algo, d, p, int(float(r["expanded_nodes"])),
                         float(r["search_elapsed_time"])))
        except (ValueError, TypeError):
            continue
    if not rows:
        sys.exit(f"no usable rows in {csv_path}")

    # (model, algo) best-over-heuristics per instance, for the selection step
    ma = {}
    for model, algo, d, p, nodes, t in rows:
        if model in LLM_MODELS:
            k = (model, algo, d, p)
            if k not in ma or nodes < ma[k][0]:
                ma[k] = (nodes, t)

    picks = {}
    doms = [d for d in DOMS if d in sizes]
    for d in doms:
        sel = SELECTION[d]
        cand = [(m, a) for m in sorted(LLM_MODELS) for a in ALGOS if (m, a, d, sel) in ma]
        if not cand:
            continue
        best = min(ma[(m, a, d, sel)][0] for m, a in cand)
        tied = [c for c in cand if ma[(c[0], c[1], d, sel)][0] == best]
        picks[d] = min(tied, key=lambda c: (ma[(c[0], c[1], d, sel)][1], c))

    def collect(pred):
        """per instance: the fewest-nodes run admitted by pred, and ITS time.

        Its time, not the fastest time -- taking the minimum of each column separately
        would be two different oracles reported as one configuration.
        """
        out = {}
        for model, algo, d, p, nodes, t in rows:
            if pred(model, algo):
                k = (d, p)
                if k not in out or nodes < out[k][0]:
                    out[k] = (nodes, t)
        return out

    # Drop baseline columns this CSV has no rows for, rather than printing 0 for a
    # system that was never run.
    cols = [(lbl, data) for lbl, data in
            ((lbl, collect(pred)) for lbl, pred in BASELINES) if data]
    dropped = [lbl for lbl, pred in BASELINES
               if lbl not in {l for l, _ in cols}]
    fixed = {}
    for model, algo, d, p, nodes, t in rows:
        if picks.get(d) == (model, algo):
            k = (d, p)
            if k not in fixed or nodes < fixed[k][0]:
                fixed[k] = (nodes, t)
    cols.append(("LLM / fixed pair", fixed))

    def paired(cfg, ref, dom=None, idx=0):
        """(wins, shared, median ratio) on metric `idx`: 0 = nodes, 1 = time.

        The metric must match the table it is printed under, or a node comparison ends
        up captioned as a runtime one.
        """
        sh = [k for k in set(cfg) & set(ref)
              if (dom is None or k[0] == dom) and ref[k][idx] > 0]
        if not sh:
            return None
        w = sum(1 for k in sh if cfg[k][idx] < ref[k][idx])
        return w, len(sh), statistics.median(cfg[k][idx] / ref[k][idx] for k in sh)

    os.makedirs(outdir, exist_ok=True)
    head = "| Domain | \\|I\\| | " + " | ".join(l for l, _ in cols) + " | pair chosen |"
    rule = "|---" * (len(cols) + 3) + "|"
    src = os.path.basename(csv_path)
    repro = (f"<!-- Reproduce: python3 Experiments/python_scripts/paper/fixed_pair_tables.py "
             f"<path to {src}> <outdir> -->")

    def pair_label(d):
        return f"{picks[d][0]} / {picks[d][1]}" if d in picks else "--"

    # ---- coverage -----------------------------------------------------------
    out = [repro, "", f"# Coverage with a single fixed model-search algorithm pair per domain", "",
           "One (model, search algorithm) pair per domain, chosen in advance on the "
           "development instance only -- no post-hoc selection on individual test "
           "instances.", "", head, rule]
    for d in doms:
        # "--" not "0" for a system with no rows at all in this CSV.
        cells = [str(len([1 for (dd, _) in data if dd == d])) if data else "--"
                 for _, data in cols]
        out.append(f"| {d} | {sizes[d]} | " + " | ".join(cells) + f" | {pair_label(d)} |")
    tots = [f"**{len(data)}**" if data else "--" for _, data in cols]
    out.append(f"| **Total** | **{total}** | " + " | ".join(tots) + " | |")
    if dropped:
        out += ["", f"Not shown: {', '.join(dropped)} — no rows in this CSV, so the "
                    "column would have been empty. Absent is not the same as zero."]
    out += ["", f"Source: `{src}`, counted over the {total}-instance set.",
            "Cells are instances solved. Paper columns are a virtual best over GBFS, A* "
            "and WA*; the fixed-pair column is one algorithm per domain, so it is the "
            "only directly deployable column here."]
    open(os.path.join(outdir, "coverage_fixed_pair.md"), "w").write("\n".join(out) + "\n")

    # ---- nodes and time ----------------------------------------------------
    for metric, idx, fname, title, fmt in (
            ("nodes", 0, "nodes_fixed_pair.md",
             "Expanded nodes with a single fixed model-search algorithm pair per domain",
             lambda v: f"{statistics.median(v):,.0f}"),
            ("time", 1, "runtime_fixed_pair.md",
             "Search time with a single fixed model-search algorithm pair per domain",
             lambda v: f"{statistics.median(v):.2f}")):
        out = [repro, "", f"# {title}", "",
               f"Median {'expanded nodes' if metric == 'nodes' else 'search time in seconds'} "
               "over the instances each column solves. The fixed pair is chosen per domain "
               "on the development instance only.", "", head, rule]
        for d in doms:
            cells = []
            for _, data in cols:
                v = [val[idx] for (dd, _), val in data.items() if dd == d]
                cells.append(fmt(v) if v else "--")
            out.append(f"| {d} | {sizes[d]} | " + " | ".join(cells) + f" | {pair_label(d)} |")
        cells = []
        for _, data in cols:
            v = [val[idx] for val in data.values()]
            cells.append(f"**{fmt(v)}**" if v else "--")
        out.append(f"| **All instances** | **{total}** | " + " | ".join(cells) + " | |")
        out += ["", "There is no Total row: medians do not sum. **All instances** is the "
                    "median over every instance the column solves, not a median of the six.",
                "", "**Cells are not like-for-like** -- each median is over the instances "
                    "that column solves, so a column solving fewer instances is measured on "
                    "an easier subset. The paired comparison below controls for that.", ""]
        # By label, not index: a dropped baseline column shifts every index after it,
        # which would silently pair against the wrong system.
        by_label = dict(cols)
        ref = dict(by_label.get("PANDA / RC-FF", {}))
        won = "fewer nodes" if metric == "nodes" else "faster"
        out += ["## Paired against PANDA RC-FF, per instance", "",
                f"| Domain | {won} | median ratio |", "|---|---|---|"]
        for d in doms:
            r = paired(fixed, ref, d, idx)
            out.append(f"| {d} | {r[0]} of {r[1]} ({100*r[0]/r[1]:.0f}%) | {r[2]:.2f}x |"
                       if r else f"| {d} | -- | -- |")
        r = paired(fixed, ref, None, idx)
        v = paired(by_label.get("LLM-VB", {}), ref, None, idx)
        if r:
            out.append(f"| **All** | **{r[0]} of {r[1]} ({100*r[0]/r[1]:.0f}%)** "
                       f"| **{r[2]:.2f}x** |")
        out.append("")
        if v:
            out.append(f"For reference the post-hoc oracle scores {v[0]} of {v[1]} "
                       f"({100*v[0]/v[1]:.0f}%) at {v[2]:.2f}x. Above 1.0 is worse "
                       "than PANDA.")
        else:
            # No RC-FF rows means the reference column is absent, not that the
            # comparison is a tie. Say so rather than printing an empty section.
            out.append("**No PANDA RC-FF rows in this CSV**, so nothing above is paired "
                       "against a baseline.")
        out += ["", f"Source: `{src}`."]
        if metric == "time":
            out += ["", "**Search time only.** It excludes grounding, and excludes "
                        "heuristic construction: the `heuristic_elapsed_time` column is "
                        "zero on every row, so build cost is not recoverable from this "
                        "data. PYTRICH is Python and PANDA is C++, so a ~2x factor is "
                        "present before any heuristic difference enters -- visible in the "
                        "oracle's own ratio."]
        open(os.path.join(outdir, fname), "w").write("\n".join(out) + "\n")

    print(f"wrote 3 tables to {outdir} from {src} ({total}-instance set)")
    print("picks: " + ", ".join(f"{d}={pair_label(d)}" for d in DOMS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
