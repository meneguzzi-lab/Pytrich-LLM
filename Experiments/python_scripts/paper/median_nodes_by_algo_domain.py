#!/usr/bin/env python3
"""
Print a table of median expanded nodes broken down by algorithm and domain.

Usage:
    python3 median_nodes_by_algo_domain.py [results.csv] [--latex]

Defaults to Experiments/results/results.csv.
Pass --latex to emit a LaTeX tabular instead of plain text.
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_MODELS = {
    "ClaudeOpus46",
    "Gemini20Flash",
    "Gemini25Pro",
    "Gemini3Flash",
    "Gemini3Pro",
    "ChatGPT4o",
    "ClaudeSonnet45",
    "GPT5",
    "GPT52",
}

# Human-readable display names for LaTeX output
MODEL_DISPLAY = {
    "ClaudeOpus46":     r"\coiv{}",
    "Gemini20Flash":    "Gemini 2.0 Flash",
    "Gemini25Pro":      "Gemini 2.5 Pro",
    "Gemini3Flash":     "Gemini 3 Flash",
    "Gemini3Pro":       "Gemini 3 Pro",
    "ChatGPT4o":        "GPT-4o",
    "ClaudeSonnet45":   "Claude Sonnet~4.5",
    "GPT5":             "GPT-5",
    "GPT52":            "GPT-5.2",
    "Panda-rc2ff":      r"\panda{}~\rcff{}",
    "Panda-rc2lmc":     r"\panda{}~\rclmcut{}",
    "Pytrich-TDG":      r"\tdg{}",
    "Pytrich-LMCOUNT":  r"\lmcount{}",
    "Pytrich-NOVELTY":  "Novelty",
}

ALGO_LABELS = {"GBFS": "GBFS", "WAstar": r"WA*", "Astar": r"A*"}

DOMAIN_ORDER = [
    "Barman-BDI",
    "Blocksworld-GTOHP",
    "Depots",
    "Robot",
    "Rover-GTOHP",
    "Towers",
    "Lamps",
]

DOMAIN_SHORT = {
    "Barman-BDI":        "Barman",
    "Blocksworld-GTOHP": "Blocks.",
    "Depots":            "Depots",
    "Robot":             "Robot",
    "Rover-GTOHP":       "Rover",
    "Towers":            "Towers",
    "Lamps":             "Lamps",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_exp(exp: str):
    """Return (model, algorithm_label) from an experiment_name string."""
    for suffix, label in [("_StandardA", "Astar"), ("_WA", "WAstar"), ("_GBFS", "GBFS")]:
        if exp.endswith(suffix):
            return exp[: -len(suffix)], label
    return exp, "GBFS"


def load(path: str):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def med(data):
    return statistics.median(data) if data else None


def fmt_plain(value):
    if value is None:
        return "    —"
    return f"{value:>6,.0f}"


def fmt_latex(value):
    if value is None:
        return r"\multicolumn{1}{c}{---}"
    return f"{value:,.0f}"


def collect_nodes(rows):
    nodes = defaultdict(list)
    for r in rows:
        if not r.get("expanded_nodes") or not r.get("solution_size"):
            continue
        model, algo = split_exp(r["experiment_name"])
        nodes[(model, algo, r["domain_name"])].append(int(r["expanded_nodes"]))
    return nodes


def vb_nodes(nodes, algo, domains):
    """Virtual best over LLM models: pool all per-problem values then take median."""
    per_domain = defaultdict(list)
    for (m, a, d), vals in nodes.items():
        if a == algo and m in LLM_MODELS:
            per_domain[d].extend(vals)
    return per_domain


# ---------------------------------------------------------------------------
# Plain-text output
# ---------------------------------------------------------------------------

def print_plain(nodes, domains, algos, llm_models, other_models):
    col_w = 12

    def header():
        row = f"{'Model':<28} {'Algo':<6}"
        for d in domains:
            row += f"  {DOMAIN_SHORT[d]:>{col_w}}"
        row += f"  {'Overall':>{col_w}}"
        return row

    def data_row(label, algo, per_domain_fn):
        overall = []
        row = f"  {label:<26} {ALGO_LABELS.get(algo, algo):<6}"
        for d in domains:
            data = per_domain_fn(d)
            overall.extend(data)
            row += f"  {fmt_plain(med(data)):>{col_w}}"
        row += f"  {fmt_plain(med(overall)):>{col_w}}"
        return row

    sep = "-" * (28 + 6 + (col_w + 2) * (len(domains) + 1) + 4)

    print("\nMedian expanded nodes by model, algorithm, and domain")
    print(header())
    print(sep)

    for algo in algos:
        vb = vb_nodes(nodes, algo, domains)
        print(data_row("LLM virtual best", algo, lambda d, vb=vb: vb.get(d, [])))

    print(sep)

    for model in llm_models:
        for algo in algos:
            if not any(nodes.get((model, algo, d)) for d in domains):
                continue
            print(data_row(model, algo, lambda d, m=model, a=algo: nodes.get((m, a, d), [])))

    print(sep)

    for model in other_models:
        for algo in algos + ["unknown"]:
            if not any(nodes.get((model, algo, d)) for d in domains):
                continue
            al = "—" if algo == "unknown" else ALGO_LABELS.get(algo, algo)
            print(data_row(model, al, lambda d, m=model, a=algo: nodes.get((m, a, d), [])))

    print(sep)
    print()


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def print_latex(nodes, domains, algos, llm_models, other_models):
    ncols = len(domains) + 1  # domains + overall
    col_spec = "ll" + "r" * ncols

    def tex_row(display, algo_tex, per_domain_fn, bold=False):
        overall = []
        cells = []
        for d in domains:
            data = per_domain_fn(d)
            overall.extend(data)
            cells.append(fmt_latex(med(data)))
        cells.append(fmt_latex(med(overall)))
        if bold:
            cells = [f"\\textbf{{{c}}}" for c in cells]
            display = f"\\textbf{{{display}}}"
            algo_tex = f"\\textbf{{{algo_tex}}}"
        return f"  {display} & {algo_tex} & " + " & ".join(cells) + r" \\"

    domain_headers = " & ".join(f"\\textbf{{{DOMAIN_SHORT[d]}}}" for d in domains)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \caption{Median expanded nodes by model, algorithm, and domain."
                 r" \llmvb{} pools all LLM models; baselines use their default algorithm.}")
    lines.append(r"  \label{tab:median_nodes}")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")
    lines.append(f"    \\textbf{{Model}} & \\textbf{{Algo}} & {domain_headers} & \\textbf{{Overall}} \\\\")
    lines.append(r"    \midrule")

    # LLM virtual best rows
    for algo in algos:
        vb = vb_nodes(nodes, algo, domains)
        lines.append(tex_row(r"\llmvb{}", ALGO_LABELS[algo],
                              lambda d, vb=vb: vb.get(d, []), bold=False))
    lines.append(r"    \midrule")

    # Per-LLM-model rows
    for model in llm_models:
        display = MODEL_DISPLAY.get(model, model)
        first = True
        for algo in algos:
            if not any(nodes.get((model, algo, d)) for d in domains):
                continue
            bold = (model == "ClaudeOpus46")
            lines.append(tex_row(display if first else "",
                                  ALGO_LABELS[algo],
                                  lambda d, m=model, a=algo: nodes.get((m, a, d), []),
                                  bold=bold))
            first = False
    lines.append(r"    \midrule")

    # Baselines
    for model in other_models:
        display = MODEL_DISPLAY.get(model, model)
        first = True
        for algo in algos + ["unknown"]:
            if not any(nodes.get((model, algo, d)) for d in domains):
                continue
            al = r"\multicolumn{1}{c}{---}" if algo == "unknown" else ALGO_LABELS[algo]
            lines.append(tex_row(display if first else "", al,
                                  lambda d, m=model, a=algo: nodes.get((m, a, d), [])))
            first = False

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    latex = "--latex" in args
    csv_args = [a for a in args if not a.startswith("--")]
    csv_path = csv_args[0] if csv_args else Path(__file__).resolve().parents[2] / "results" / "results.csv"

    rows = load(str(csv_path))
    nodes = collect_nodes(rows)

    domains = DOMAIN_ORDER
    algos = ["GBFS", "WAstar", "Astar"]

    all_models = sorted({m for (m, a, d) in nodes})
    llm_models = [m for m in all_models if m in LLM_MODELS]
    other_models = [m for m in all_models if m not in LLM_MODELS]

    if latex:
        print_latex(nodes, domains, algos, llm_models, other_models)
    else:
        print_plain(nodes, domains, algos, llm_models, other_models)


if __name__ == "__main__":
    main()
