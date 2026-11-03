#!/usr/bin/env python3
"""
Compare a freshly parsed results CSV with the dataset of record.

    python3 Experiments/python_scripts/compare_with_record.py NEW.csv [--record Experiments/results/results.csv]
    python3 Experiments/python_scripts/compare_with_record.py NEW.csv --record Experiments/results/ablation.csv

Both files hold one row per solved run. Systems are compared on their experiment_name
prefix (the part before "_<algorithm>"), so the Pytrich baselines, which the record stores
without an algorithm suffix, line up with a new sweep that has one.

For every system present in the new file the script reports coverage (instances solved
by any run) in the record and in the new file, the instances solved in only one of the
two, and, on the instances both solve, how many have the same per-instance minimum of
expanded nodes. With --per-run the comparison is made per (system, heuristic, algorithm)
run instead, which is the right view for a partial sweep: a run matches when its node
count equals the record's for that configuration (for the Pytrich baselines, which the
record stores without an algorithm suffix, when it equals one of the record's values on
that instance). Pytrich and LLM runs are deterministic (PYTHONHASHSEED=0) and PANDA runs
are seeded, so node counts agree on every instance both files solve when the same
binaries are used; coverage can differ only for runs that finish close to the 30-minute
limit or the 8 GB memory limit.

A partial new file (one domain, one planner) is compared on what it contains: instances
the record solves but the new file does not run are counted under "only rec".
Exit status is 1 when any shared instance has a different node count or the new file
solves an instance the record does not.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RECORD = os.path.join(HERE, "..", "results", "results.csv")
ALGOS = ("GBFS", "StandardA", "WA")


def system_of(experiment_name):
    name = experiment_name or ""
    for a in ALGOS:
        if name.endswith("_" + a):
            return name[: -len(a) - 1]
    return name


def algo_of(experiment_name):
    name = experiment_name or ""
    for a in ALGOS:
        if name.endswith("_" + a):
            return a
    return ""


def load(path, per_run):
    """key -> {(domain, problem): set of expanded-node counts}.

    key is the system (per-instance minimum over its runs, one value), or with per_run
    the (system, heuristic, algorithm) run; a Pytrich baseline row of the record has
    algorithm "" and keeps every value seen on the instance.
    """
    best = defaultdict(lambda: defaultdict(set))
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("solution_size"):
                continue
            try:
                n = int(float(r["expanded_nodes"]))
            except (TypeError, ValueError):
                continue
            s = system_of(r["experiment_name"])
            key = (s, r.get("heuristic_name", ""), algo_of(r["experiment_name"])) if per_run else s
            best[key][(r["domain_name"], r["problem_name"])].add(n)
    if not per_run:
        for key in best:
            for k in best[key]:
                best[key][k] = {min(best[key][k])}
    return best


def record_entry(rec, key, per_run):
    """The record's values for a key; a per-run key falls back to the unsuffixed system."""
    if key in rec:
        return rec[key]
    if per_run and (key[0], key[1], "") in rec:
        return rec[(key[0], key[1], "")]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new", help="CSV produced by parse_all_logs_all_domains.py")
    ap.add_argument("--record", default=DEFAULT_RECORD, help="CSV of record (default: Experiments/results/results.csv)")
    ap.add_argument("--per-run", action="store_true", help="compare per (system, heuristic, algorithm) run")
    ap.add_argument("--details", action="store_true", help="list every instance that differs")
    args = ap.parse_args()

    rec, new = load(args.record, args.per_run), load(args.new, args.per_run)
    keys = sorted(new, key=str)
    label = "system / heuristic / algorithm" if args.per_run else "system"
    print(f"{label:<60}{'record':>8}{'new':>6}{'only rec':>10}{'only new':>10}{'shared':>8}{'same nodes':>12}")
    problems = 0
    seen = set()
    for key in keys:
        b = new[key]
        a = record_entry(rec, key, args.per_run)
        in_record = a is not None
        a = a or {}
        if in_record:
            seen.add(key if key in rec else (key[0], key[1], ""))
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        shared = sorted(set(a) & set(b))
        differ = [k for k in shared if not (b[k] <= a[k])]
        name = " / ".join(key) if isinstance(key, tuple) else key
        flag = "" if not only_b and not differ else "  <-- differs"
        if not in_record:
            flag = "  (not in record)"
        print(f"{name:<60}{len(a):>8}{len(b):>6}{len(only_a):>10}{len(only_b):>10}{len(shared):>8}{len(shared) - len(differ):>12}{flag}")
        problems += len(only_b) + len(differ)
        if args.details:
            for k in only_b:
                print(f"    solved only in new:  {k[0]} {k[1]} ({min(b[k])} nodes)")
            for k in differ:
                print(f"    nodes differ:        {k[0]} {k[1]} record {sorted(a[k])} new {sorted(b[k])}")
    if not args.per_run:
        missing = sorted(set(rec) - set(new), key=str)
        if missing:
            print(f"\nNot in the new file: {', '.join(missing)}")
    print(f"\n{problems} shared instance(s) with a different node count or solved only in the new file.")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
