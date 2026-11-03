#!/usr/bin/env python3
"""
Task list for the timing run: TDG, LMCount and each model's best GBFS candidate per
domain, on every instance of the domain, under GBFS. This is how
Experiments/results/timing_gbfs.csv was produced.

    python3 JobArrays/timing_task_list.py --output JobArrays/TaskLists/timing_GBFS.txt
    sbatch --array=1-$(wc -l < JobArrays/TaskLists/timing_GBFS.txt) \\
        --export=ALL,OUTPUT_ROOT=JobArrays/outputs-timing \\
        JobArrays/run_experiment.sbatch JobArrays/TaskLists/timing_GBFS.txt

The best candidate of a model in a domain is the one with the most instances solved
under GBFS in the results CSV (ties: lowest candidate index). Task lines have the same
eight fields as generate_task_list.py writes.
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_task_list import (DOMAIN_FILESYSTEM_MAP, NODE_CONFIGS, get_instances,  # noqa: E402
                                get_sas_file_path, strip_problem_extension, load_config,
                                get_file_mode_planners)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["Gemini20Flash", "Gemini25Pro", "Gemini3Flash", "Gemini3Pro", "ClaudeOpus46"]
DOMAINS = ["Barman-BDI", "Blocksworld-GTOHP", "Depots", "Robot", "Rover-GTOHP", "Towers"]
INDEX = re.compile(r"(\d+)\D*$")


def best_candidates(results_csv):
    solved = defaultdict(set)   # (model, domain, candidate index) -> instances
    with open(results_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["solution_size"] or not r["experiment_name"].endswith("_GBFS"):
                continue
            model = r["experiment_name"][:-5]
            if model not in MODELS:
                continue
            m = INDEX.search(r["heuristic_name"])
            if not m:
                continue
            solved[(model, r["domain_name"], int(m.group(1)))].add(r["problem_name"])
    best = {}
    for (model, dom, idx), inst in solved.items():
        cur = best.get((model, dom))
        if cur is None or (len(inst), -idx) > (len(cur[1]), -cur[0]):
            best[(model, dom)] = (idx, inst)
    return {k: v[0] for k, v in best.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(ROOT, "Experiments", "results", "results.csv"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    args = ap.parse_args()
    config = load_config(os.path.join(ROOT, "config", "planners.yaml"))
    file_mode = set(get_file_mode_planners(config))
    best = best_candidates(args.results)
    node = NODE_CONFIGS["GBFS"]
    lines = []
    for dom in args.domains:
        fs = DOMAIN_FILESYSTEM_MAP[dom]
        prefix = fs.split("_")[0].lower()
        for inst in get_instances(dom):
            sas = get_sas_file_path(dom, inst)
            for h, engine in (("TDG()", "Pytrich-TDG_GBFS"), ("LMCOUNT(use_bid=True)", "Pytrich-LMCOUNT_GBFS")):
                lines.append(f"{dom} {inst} Pytrich {h} GBFS {engine} {node} {sas}")
            for model in MODELS:
                if (model, dom) not in best:
                    continue
                idx = best[(model, dom)]
                folder = os.path.join("Pytrich", "Heuristics", "GeneratedHeuristics", fs, model)
                files = [f for f in os.listdir(os.path.join(ROOT, folder)) if f.endswith(f"_heuristic_{idx}.py")]
                if model in file_mode:
                    h = os.path.join(folder, files[0])
                else:
                    h = f"{prefix}_{model.lower()}_heuristic_{idx}()"
                lines.append(f"{dom} {inst} {model} {h} GBFS {model}_GBFS {node} {sas}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} tasks -> {args.output}")
    for (model, dom), idx in sorted(best.items()):
        if dom in args.domains:
            print(f"  {dom:<18} {model:<14} candidate {idx}")


if __name__ == "__main__":
    main()
