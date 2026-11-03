#!/usr/bin/env python3
r"""
Generate the LaTeX coverage table (Table 1) for the NeurIPS 2026 HTN+LLM paper.

Usage:
    python generate_coverage_table.py                        # reads Experiments/results/results.csv
    python generate_coverage_table.py -i other_results.csv   # explicit CSV
    python generate_coverage_table.py --stdout               # print only, no file write

Output: coverage_table.tex (ready to \input{} into results.tex)

Column definitions
------------------
PANDA block  : Panda-rc2ff, Panda-rc2lmc  (virtual best over 3 algorithms each)
Pytrich block: Pytrich-TDG, Pytrich-LMCOUNT  (virtual best over 3 algorithms each)
CO4 column   : ClaudeOpus46  (virtual best over 3 algorithms, single model)
LLM-VB column: virtual best over all LLM models and 3 algorithms

LLM models considered for LLM-VB and CO4:
    ClaudeOpus46, Gemini3Flash, Gemini3Pro,
    Gemini20Flash, Gemini25Pro
    (models with heuristics in only one domain are excluded from LLM-VB)
"""

import argparse
import csv
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(SCRIPT_DIR, "..", "..", "results", "results.csv")

DOMAINS = [
    "Barman-BDI",
    "Blocksworld-GTOHP",
    "Depots",
    "Robot",
    "Rover-GTOHP",
    "Towers",
    "Lamps",
]

DOMAIN_SIZES = {
    "Barman-BDI": 20,
    "Blocksworld-GTOHP": 30,
    "Depots": 30,
    "Robot": 20,
    "Rover-GTOHP": 30,
    "Towers": 20,
    "Lamps": 30,
}

# Table rows drop the benchmark-suite suffixes; the experiments section names the suites.
DISPLAY_NAMES = {"Barman-BDI": "Barman", "Blocksworld-GTOHP": "Blocksworld", "Rover-GTOHP": "Rover"}

# Systems whose experiment_name starts with these prefixes count as that column.
PANDA_PREFIXES = {
    "rc2ff":  "Panda-rc2ff",
    "rc2lmc": "Panda-rc2lmc",
    "rc2add": "Panda-rc2add",
}

PYTRICH_PREFIXES = {
    "TDG":    "Pytrich-TDG",
    "LMCOUNT": "Pytrich-LMCOUNT",
}

# Models included in the LLM virtual best (must have heuristics in >= 2 domains)
LLM_VB_MODELS = {
    "ClaudeOpus46",
    "Gemini3Flash",
    "Gemini3Pro",
    "Gemini20Flash",
    "Gemini25Pro",
}

# Single model shown as CO4 column
CO4_MODEL = "ClaudeOpus46"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify(experiment_name):
    """Return (system_key, model_or_none) for a given experiment_name."""
    name = experiment_name or ""
    # Panda — check lmcount before rc2lmc to avoid prefix collision
    if name.startswith("Panda-lmcount"):
        return "Panda-lmcount", None
    if name.startswith("Panda-rc2ff"):
        return "Panda-rc2ff", None
    if name.startswith("Panda-rc2add"):
        return "Panda-rc2add", None
    if name.startswith("Panda-rc2lmc"):
        return "Panda-rc2lmc", None
    # Pytrich built-ins
    if name.startswith("Pytrich-TDG"):
        return "Pytrich-TDG", None
    if name.startswith("Pytrich-LMCOUNT"):
        return "Pytrich-LMCOUNT", None
    if name.startswith("Pytrich-NOVELTY"):
        return "Pytrich-NOVELTY", None
    # LLM: experiment_name is like "ClaudeOpus46_GBFS"
    for model in LLM_VB_MODELS | {CO4_MODEL}:
        if name.startswith(model):
            return "LLM", model
    return None, None


def load_solved(csv_path):
    """
    Returns solved[(system_or_model, domain)] = set of problem_names.

    system_or_model is one of:
        "Panda-rc2ff", "Panda-rc2lmc", "Panda-rc2add",
        "Pytrich-TDG", "Pytrich-LMCOUNT",
        "ClaudeOpus46", <other LLM model names>
    """
    solved = defaultdict(set)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("experiment_name", "")
            domain = row.get("domain_name", "")
            problem = row.get("problem_name", "")
            sol = row.get("solution_size", "")
            if not (domain and problem and sol):
                continue

            system, model = classify(name)
            if system is None:
                continue

            if system in ("Panda-rc2ff", "Panda-rc2lmc", "Panda-rc2add", "Panda-lmcount",
                          "Pytrich-TDG", "Pytrich-LMCOUNT", "Pytrich-NOVELTY"):
                solved[(system, domain)].add(problem)
            elif system == "LLM":
                solved[(model, domain)].add(problem)

    return solved


SELECTION = {"Barman-BDI": "pfile01", "Blocksworld-GTOHP": "p01", "Depots": "p01",
             "Robot": "pfile_01_001", "Rover-GTOHP": "p01", "Towers": "pfile_01",
             "Lamps": "pfile01"}
ALGOS = ("GBFS", "StandardA", "WA")


def load_fixed_pair(csv_path):
    """
    The fixed configuration: one model and one search algorithm per domain, chosen on
    the selection (smallest) instance alone by the fewest expanded nodes of the model's
    best candidate there, ties broken by time then name. The candidate heuristic stays
    free per instance. Same rule as fixed_pair_tables.py..
    Returns (solved[domain] = set of problems, picks[domain] = (model, algo)).
    """
    best = {}       # (model, algo, domain, problem) -> (nodes, time)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("solution_size"):
                continue
            system, model = classify(row.get("experiment_name", ""))
            if system != "LLM" or model not in LLM_VB_MODELS:
                continue
            name = row["experiment_name"]
            algo = name.split("_", 1)[1] if "_" in name else ""
            try:
                rec = (int(float(row["expanded_nodes"])), float(row["search_elapsed_time"]))
            except (ValueError, TypeError):
                continue
            k = (model, algo, row["domain_name"], row["problem_name"])
            if k not in best or rec[0] < best[k][0]:
                best[k] = rec
    picks, solved = {}, {}
    for domain in DOMAINS:
        sel = SELECTION[domain]
        cand = [(m, a) for m in sorted(LLM_VB_MODELS) for a in ALGOS if (m, a, domain, sel) in best]
        if not cand:
            solved[domain] = set()
            continue
        low = min(best[(m, a, domain, sel)][0] for m, a in cand)
        tied = [c for c in cand if best[(c[0], c[1], domain, sel)][0] == low]
        picks[domain] = min(tied, key=lambda c: (best[(c[0], c[1], domain, sel)][1], c))
        solved[domain] = {p for (m, a, d, p) in best if (m, a, d) == (*picks[domain], domain)}
    return solved, picks


def coverage(solved, system, domain):
    """Number of unique problems solved by `system` in `domain`."""
    return len(solved.get((system, domain), set()))


def vb_coverage(solved, models, domain):
    """Virtual best coverage over a set of models/systems for a domain."""
    union = set()
    for m in models:
        union |= solved.get((m, domain), set())
    return len(union)


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_table(solved, fixed_solved):
    llm_vb_models = LLM_VB_MODELS
    co4 = CO4_MODEL

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \caption{Coverage (problems solved) per domain and in aggregate; every column is the virtual best over GBFS, A* and WA* except \llmfixed{}, which fixes one model and one search algorithm per domain in advance, and bold marks the best coverage in each row.}")
    lines.append(r"  \label{tab:coverage}")
    lines.append(r"  \centering")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \begin{tabular}{lrrrrrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    & & \multicolumn{4}{c}{\panda{}} & \multicolumn{5}{c}{\pytrich{}} \\")
    lines.append(r"    \cmidrule(lr){3-6}\cmidrule(lr){7-11}")
    lines.append(r"    Domain & Probs. & \rcff{} & \rclmcut{} & \rcadd{} & \pandalmcount{} & \tdg{} & \lmcount{} & \coiv{} & \llmfixed{} & \llmvb{} \\")
    lines.append(r"    \midrule")

    total_probs = 0
    totals = defaultdict(int)

    for domain in DOMAINS:
        size = DOMAIN_SIZES[domain]
        total_probs += size

        rc2ff      = coverage(solved, "Panda-rc2ff",    domain)
        rc2lmc     = coverage(solved, "Panda-rc2lmc",   domain)
        rc2add     = coverage(solved, "Panda-rc2add",   domain)
        panda_lmc  = coverage(solved, "Panda-lmcount",  domain)
        tdg        = coverage(solved, "Pytrich-TDG",    domain)
        lmcount    = coverage(solved, "Pytrich-LMCOUNT",domain)
        co4_n      = coverage(solved, co4,              domain)
        llmvb_n    = vb_coverage(solved, llm_vb_models, domain)
        fixed_n    = len(fixed_solved.get(domain, set()))

        # A cell may never exceed the number of instances in the domain. The submitted
        # table shipped Depots = 29 against a denominator of 27 because DOMAIN_SIZES was
        # corrected without regenerating the .tex; fail loudly instead.
        for column, value in (("rc2ff", rc2ff), ("rc2lmc", rc2lmc), ("rc2add", rc2add), ("panda_lmcount", panda_lmc),
                              ("TDG", tdg), ("LMCOUNT", lmcount), ("CO4", co4_n), ("LLM-fixed", fixed_n),
                              ("LLM-VB", llmvb_n)):
            if value > size:
                raise ValueError(
                    f"{domain}/{column}: {value} problems solved but DOMAIN_SIZES says {size}. "
                    f"Either DOMAIN_SIZES is stale or the CSV contains instances outside the suite."
                )

        totals["rc2ff"]     += rc2ff
        totals["rc2lmc"]    += rc2lmc
        totals["rc2add"]    += rc2add
        totals["panda_lmc"] += panda_lmc
        totals["tdg"]       += tdg
        totals["lmcount"]   += lmcount
        totals["co4"]       += co4_n
        totals["fixed"]     += fixed_n
        totals["llmvb"]     += llmvb_n

        row_max = max(rc2ff, rc2lmc, rc2add, panda_lmc, tdg, lmcount, co4_n, fixed_n, llmvb_n)

        def cell(n):
            # Bold the best coverage in the row (ties all bold), as in the Total row.
            if n == row_max:
                return rf"\textbf{{{n}}}"
            return str(n)

        shown = DISPLAY_NAMES.get(domain, domain)
        row = (
            f"    {shown:<22} & {size:<4} & "
            f"{cell(rc2ff):<14} & "
            f"{cell(rc2lmc):<14} & "
            f"{cell(rc2add):<14} & "
            f"{cell(panda_lmc):<14} & "
            f"{cell(tdg):<14} & "
            f"{cell(lmcount):<14} & "
            f"{cell(co4_n):<28} & "
            f"{cell(fixed_n):<14} & "
            f"{cell(llmvb_n)} \\\\"
        )
        lines.append(row)

    lines.append(r"    \midrule")

    def total_cell(key, bold_if_max=True):
        n = totals[key]
        all_vals = list(totals.values())
        if bold_if_max and n == max(all_vals):
            return rf"\textbf{{{n}}}"
        return str(n)

    lines.append(
        f"    {'Total':<22} & {total_probs:<4} & "
        f"{total_cell('rc2ff'):<14} & "
        f"{total_cell('rc2lmc'):<14} & "
        f"{total_cell('rc2add'):<14} & "
        f"{total_cell('panda_lmc'):<14} & "
        f"{total_cell('tdg'):<14} & "
        f"{total_cell('lmcount'):<14} & "
        f"{total_cell('co4'):<28} & "
        f"{total_cell('fixed'):<14} & "
        f"{total_cell('llmvb')} \\\\"
    )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX coverage table")
    parser.add_argument(
        "-i", "--input",
        default=DEFAULT_RESULTS,
        help="Path to results CSV (default: Experiments/results/results.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(SCRIPT_DIR, "coverage_table.tex"),
        help="Output .tex file (default: coverage_table.tex next to this script)",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Print table to stdout instead of writing a file",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: CSV not found: {args.input}")
        return 1

    solved = load_solved(args.input)
    fixed_solved, fixed_picks = load_fixed_pair(args.input)
    print('fixed model and algorithm per domain: ' + ', '.join(f'{d}={m}/{a}' for d, (m, a) in fixed_picks.items()))
    table = generate_table(solved, fixed_solved)

    if args.stdout:
        print(table)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(table)
            f.write("\n")
        print(f"Written: {args.output}")

    # Also print a plain-text summary to stdout for quick checking
    print()
    print("Coverage summary:")
    print(f"  {'Domain':<22} {'rc2ff':>6} {'rc2lmc':>7} {'P-LMC':>6} {'TDG':>5} {'LMCOUNT':>8} {'CO4':>5} {'LLM-VB':>7}")
    print("  " + "-" * 68)
    for domain in DOMAINS:
        size = DOMAIN_SIZES[domain]
        rc2ff     = len(solved.get(("Panda-rc2ff",    domain), set()))
        rc2lmc    = len(solved.get(("Panda-rc2lmc",   domain), set()))
        panda_lmc = len(solved.get(("Panda-lmcount",  domain), set()))
        tdg       = len(solved.get(("Pytrich-TDG",    domain), set()))
        lmcount   = len(solved.get(("Pytrich-LMCOUNT",domain), set()))
        co4_n     = len(solved.get((CO4_MODEL,        domain), set()))
        llmvb_n   = len(set.union(*[solved.get((m, domain), set()) for m in LLM_VB_MODELS]))
        print(f"  {domain:<22} {rc2ff:>6}/{size} {rc2lmc:>6}/{size} {panda_lmc:>5}/{size} {tdg:>4}/{size} {lmcount:>7}/{size} {co4_n:>4}/{size} {llmvb_n:>6}/{size}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
