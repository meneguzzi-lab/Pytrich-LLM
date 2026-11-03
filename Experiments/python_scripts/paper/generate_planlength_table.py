#!/usr/bin/env python3
"""
Generate the LaTeX plan length table for the NeurIPS 2026 HTN+LLM paper.

Usage:
    python generate_planlength_table.py                        # reads Experiments/results/results.csv
    python generate_planlength_table.py -i other_results.csv   # explicit CSV
    python generate_planlength_table.py --stdout               # print only, no file write
    python generate_planlength_table.py --intersection         # restrict to problems solved by all systems

Without --intersection (default):
    Each column reports the median plan length over all problems that system solved,
    using the virtual best solution per problem across all three search algorithms.

With --intersection:
    Each column reports the median plan length restricted to the problems solved by
    ALL six systems (rc2ff, rc2lmc, TDG, LMCOUNT, CO4, LLM-VB).  This makes the
    medians directly comparable across columns.  Output goes to
    planlength_intersection_table.tex by default.

LLM models considered for LLM-VB and CO4:
    ClaudeOpus46, Gemini3Flash, Gemini3Pro,
    Gemini20Flash, Gemini25Pro
    (models with heuristics in only one domain are excluded from LLM-VB)
"""

import argparse
import csv
import os
from collections import defaultdict
import statistics

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

# Canonical system keys used throughout
ALL_SYSTEMS = ["Panda-rc2ff", "Panda-rc2lmc", "Panda-lmcount",
               "Pytrich-TDG", "Pytrich-LMCOUNT", CO4_MODEL, "LLM-VB"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify(experiment_name):
    """Return (system_key, model_or_none) for a given experiment_name."""
    name = experiment_name or ""
    # check lmcount before rc2lmc to avoid prefix collision
    if name.startswith("Panda-lmcount"):
        return "Panda-lmcount", None
    if name.startswith("Panda-rc2ff"):
        return "Panda-rc2ff", None
    if name.startswith("Panda-rc2lmc"):
        return "Panda-rc2lmc", None
    if name.startswith("Pytrich-TDG"):
        return "Pytrich-TDG", None
    if name.startswith("Pytrich-LMCOUNT"):
        return "Pytrich-LMCOUNT", None
    if name.startswith("Pytrich-NOVELTY"):
        return "Pytrich-NOVELTY", None
    for model in LLM_VB_MODELS | {CO4_MODEL}:
        if name.startswith(model):
            return "LLM", model
    return None, None


def load_best_per_problem(csv_path):
    """
    Returns best[(system, domain, problem)] = minimum solution_size across all
    algorithm runs (virtual best per problem).

    Systems are the canonical keys in ALL_SYSTEMS; LLM-VB is the union over
    all five main models.
    """
    # raw[(system_or_model, domain, problem)] = min sol_size seen
    raw = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("experiment_name", "")
            domain = row.get("domain_name", "")
            problem = row.get("problem_name", "")
            sol_size = row.get("solution_size", "")
            if not (domain and problem and sol_size):
                continue
            try:
                sol_size = int(sol_size)
            except ValueError:
                continue

            system, model = classify(name)
            if system is None:
                continue

            if system in ("Panda-rc2ff", "Panda-rc2lmc", "Panda-lmcount",
                          "Pytrich-TDG", "Pytrich-LMCOUNT", "Pytrich-NOVELTY"):
                key = (system, domain, problem)
                raw[key] = min(raw.get(key, sol_size), sol_size)
            elif system == "LLM":
                # Store per-model for CO4, and aggregate for LLM-VB
                key_model = (model, domain, problem)
                raw[key_model] = min(raw.get(key_model, sol_size), sol_size)
                key_vb = ("LLM-VB", domain, problem)
                raw[key_vb] = min(raw.get(key_vb, sol_size), sol_size)

    # Remap CO4_MODEL key to canonical name
    best = {}
    for (sys, domain, prob), val in raw.items():
        if sys == CO4_MODEL:
            best[(CO4_MODEL, domain, prob)] = val
        else:
            best[(sys, domain, prob)] = val
    return best


def solved_problems(best, system, domain):
    """Set of problem names solved by `system` in `domain`."""
    return {prob for (sys, dom, prob), _ in best.items()
            if sys == system and dom == domain}


def median_for_problems(best, system, domain, problems):
    """Median plan length for `system` in `domain` restricted to `problems`."""
    vals = [best[(system, domain, p)] for p in problems
            if (system, domain, p) in best]
    return statistics.median(vals) if vals else None


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_table(best, intersection=False):
    lines = []
    lines.append(r"\begin{table}[t]")

    if intersection:
        caption = (
            r"  \caption{Median plan length (primitive actions) per domain, "
            r"restricted to the problems solved by all six systems. "
            r"Each entry is the virtual best over the three search algorithms "
            r"(GBFS, A*, WA*).}"
        )
        label = r"  \label{tab:planlength-intersection}"
    else:
        caption = (
            r"  \caption{Median plan length (primitive actions) per domain. "
            r"Each entry is the virtual best over the three search algorithms "
            r"(GBFS, A*, WA*) on all problems that system solved.}"
        )
        label = r"  \label{tab:planlength}"

    lines.append(caption)
    lines.append(label)
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lrrrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    & & \multicolumn{3}{c}{\panda{}}"
        r" & \multicolumn{4}{c}{\pytrich{}} \\"
    )
    lines.append(r"    \cmidrule(lr){3-5}\cmidrule(lr){6-9}")
    lines.append(
        r"    Domain & $n$ & \rcff{} & \rclmcut{} & \pandalmcount{}"
        r" & \tdg{} & \lmcount{} & \coiv{} & \llmvb{} \\"
    )
    lines.append(r"    \midrule")

    totals = defaultdict(list)

    for domain in DOMAINS:
        if intersection:
            shared = set.intersection(*[
                solved_problems(best, sys, domain) for sys in ALL_SYSTEMS
            ])
            n = len(shared)
        else:
            shared = None  # each system uses its own solved set

        def get_med(system):
            if intersection:
                return median_for_problems(best, system, domain, shared)
            probs = solved_problems(best, system, domain)
            return median_for_problems(best, system, domain, probs)

        rc2ff_med = get_med("Panda-rc2ff")
        rc2lmc_med = get_med("Panda-rc2lmc")
        panda_lmc_med = get_med("Panda-lmcount")
        tdg_med = get_med("Pytrich-TDG")
        lmcount_med = get_med("Pytrich-LMCOUNT")
        co4_med = get_med(CO4_MODEL)
        llmvb_med = get_med("LLM-VB")

        for key, val in [
            ("rc2ff", rc2ff_med), ("rc2lmc", rc2lmc_med),
            ("panda_lmc", panda_lmc_med),
            ("tdg", tdg_med), ("lmcount", lmcount_med),
            ("co4", co4_med), ("llmvb", llmvb_med),
        ]:
            if val is not None:
                totals[key].append(val)

        def fmt(val):
            return "---" if val is None else str(int(round(val)))

        if intersection:
            n_col = f"{n}"
        else:
            # show min solved count across systems to indicate comparison basis
            counts = [
                len(solved_problems(best, s, domain)) for s in ALL_SYSTEMS
            ]
            n_col = f"{min(counts)}--{max(counts)}"

        row = (
            f"    {domain:<22} & {n_col:<7} & "
            f"{fmt(rc2ff_med):<6} & "
            f"{fmt(rc2lmc_med):<8} & "
            f"{fmt(panda_lmc_med):<8} & "
            f"{fmt(tdg_med):<7} & "
            f"{fmt(lmcount_med):<9} & "
            f"{fmt(co4_med):<7} & "
            f"{fmt(llmvb_med)} \\\\"
        )
        lines.append(row)

    lines.append(r"    \midrule")

    def fmt_total(key):
        vals = totals[key]
        if not vals:
            return "---"
        return str(int(round(statistics.median(vals))))

    lines.append(
        f"    {'Median':<22} & {'':7} & "
        f"{fmt_total('rc2ff'):<6} & "
        f"{fmt_total('rc2lmc'):<8} & "
        f"{fmt_total('panda_lmc'):<8} & "
        f"{fmt_total('tdg'):<7} & "
        f"{fmt_total('lmcount'):<9} & "
        f"{fmt_total('co4'):<7} & "
        f"{fmt_total('llmvb')} \\\\"
    )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX plan length table")
    parser.add_argument(
        "-i", "--input",
        default=DEFAULT_RESULTS,
        help="Path to results CSV (default: Experiments/results/results.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output .tex file (default depends on --intersection flag)",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Print table to stdout instead of writing a file",
    )
    parser.add_argument(
        "--intersection", action="store_true",
        help="Restrict medians to problems solved by all six systems",
    )
    args = parser.parse_args()

    if args.output is None:
        if args.intersection:
            args.output = os.path.join(SCRIPT_DIR, "planlength_intersection_table.tex")
        else:
            args.output = os.path.join(SCRIPT_DIR, "planlength_table.tex")

    if not os.path.isfile(args.input):
        print(f"Error: CSV not found: {args.input}")
        return 1

    best = load_best_per_problem(args.input)
    table = generate_table(best, intersection=args.intersection)

    if args.stdout:
        print(table)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(table)
            f.write("\n")
        print(f"Written: {args.output}")

    # Plain-text summary
    mode = "intersection" if args.intersection else "all solved"
    print(f"\nPlan length summary (median, {mode}):")
    print(
        f"  {'Domain':<22} {'n':>8} {'rc2ff':>6} {'rc2lmc':>8}"
        f" {'P-LMC':>6} {'TDG':>6} {'LMCOUNT':>8} {'CO4':>6} {'LLM-VB':>7}"
    )
    print("  " + "-" * 88)
    for domain in DOMAINS:
        if args.intersection:
            shared = set.intersection(*[
                solved_problems(best, sys, domain) for sys in ALL_SYSTEMS
            ])
            n_str = str(len(shared))
        else:
            counts = [
                len(solved_problems(best, s, domain)) for s in ALL_SYSTEMS
            ]
            shared = None
            n_str = f"{min(counts)}-{max(counts)}"

        def get_med(system):
            if args.intersection:
                return median_for_problems(best, system, domain, shared)
            probs = solved_problems(best, system, domain)
            return median_for_problems(best, system, domain, probs)

        def fmt(v):
            return str(int(round(v))) if v is not None else "---"

        print(f"  {domain:<22} {n_str:>8} "
              f"{fmt(get_med('Panda-rc2ff')):>6} "
              f"{fmt(get_med('Panda-rc2lmc')):>8} "
              f"{fmt(get_med('Panda-lmcount')):>6} "
              f"{fmt(get_med('Pytrich-TDG')):>6} "
              f"{fmt(get_med('Pytrich-LMCOUNT')):>8} "
              f"{fmt(get_med(CO4_MODEL)):>6} "
              f"{fmt(get_med('LLM-VB')):>7}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
