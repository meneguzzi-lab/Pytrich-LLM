#!/usr/bin/env python3
"""
Replay Pytrich plans against the grounded goal and report whether each one is genuine.

    python3 Experiments/python_scripts/validate_plans.py --sas-dir htn-benchmarks/sas-files FILE...
    python3 Experiments/python_scripts/validate_plans.py --sas-dir htn-benchmarks/sas-files --out-dir JobArrays/outputs --csv report.csv

FILE is a `.plan` file (one `(action[args])` per line) or a Pytrich `.out` log (the actions after
the `Plan:` line). The domain and problem are read from the log's `Domain :` and
`Problem :` lines, or, for a `.plan` file, from its name `<Domain>_<problem>_<engine>.plan`.
The grounded model is `<sas-dir>/<Domain>-<problem>.psas`.

This checks that every action is applicable in sequence and that the final state contains
the goal bit set, independently of the planner's own goal test. It does not check that the
plan refines the initial task network.
"""
import argparse, csv, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from Pytrich.Grounder.panda_ground import PandaGrounder as PandaGround


def read_plan(path):
    meta = {}
    actions = []
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    if path.endswith(".plan"):
        base = os.path.basename(path)[:-5]
        parts = base.split("_")
        meta["domain"], meta["problem"] = parts[0], "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]
        actions = [l.strip() for l in lines if l.strip().startswith("(")]
        return meta, actions, True
    in_plan = False
    solved = False
    for l in lines:
        if l.startswith("Domain :") or l.startswith("Domain Name"):
            meta["domain"] = l.split(":", 1)[1].strip()
        elif l.startswith("Problem :") or l.startswith("Problem Name"):
            # Logs print the problem file ("p12.hddl", or "pfile01.pddl" in Lamps); the
            # .psas name has no extension.
            meta["problem"] = l.split(":", 1)[1].strip().removesuffix(".hddl").removesuffix(".pddl")
        elif l.startswith("Search Status"):
            solved = "GOAL" in l
        elif l.startswith("Plan:"):
            in_plan = True
        elif in_plan:
            s = l.strip()
            if s.startswith("("):
                actions.append(s)
            elif s:
                in_plan = False
    return meta, actions, solved


def validate(sas_file, actions):
    g = PandaGround(sas_file=sas_file)
    model = g()
    by_name = {op.name: op for op in model.operators}
    state = model.initial_state
    for i, a in enumerate(actions):
        name = a.strip()
        if name.startswith("(") and name.endswith(")"):
            name = name[1:-1]
        op = by_name.get(name)
        if op is None:
            return False, f"step {i}: unknown operator {name}"
        if not op.applicable(state):
            return False, f"step {i}: {name} not applicable"
        state = op.apply(state)
    missing = model.goals & ~state
    if missing:
        n = bin(missing).count("1")
        total = bin(model.goals).count("1")
        return False, f"goal not reached: {n} of {total} goal facts false"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--sas-dir", required=True)
    ap.add_argument("--out-dir", help="validate every *.out under this folder (recursively)")
    ap.add_argument("--csv", help="write one row per file")
    args = ap.parse_args()
    files = list(args.files)
    if args.out_dir:
        for root, _, names in os.walk(args.out_dir):
            files += [os.path.join(root, n) for n in names if n.endswith(".out")]
    rows = []
    bad = 0
    for path in sorted(files):
        meta, actions, solved = read_plan(path)
        if not solved:
            rows.append((path, meta.get("domain"), meta.get("problem"), len(actions), "not solved", ""))
            continue
        sas = os.path.join(args.sas_dir, f"{meta.get('domain')}-{meta.get('problem')}.psas")
        if not os.path.exists(sas):
            rows.append((path, meta.get("domain"), meta.get("problem"), len(actions), "no psas", sas)); continue
        ok, why = validate(sas, actions)
        bad += (not ok)
        rows.append((path, meta["domain"], meta["problem"], len(actions), "valid" if ok else "INVALID", why))
        print(f"{'valid  ' if ok else 'INVALID'} {meta['domain']} {meta['problem']} ({len(actions)} actions) {why if not ok else ''}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file", "domain", "problem", "plan_length", "verdict", "detail"]); w.writerows(rows)
    # Print every verdict: a run where nothing was checked ("no psas") must not read as clean.
    counts = {}
    for r in rows:
        counts[r[4]] = counts.get(r[4], 0) + 1
    print(f"\n{len(rows)} files, {bad} invalid; " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
