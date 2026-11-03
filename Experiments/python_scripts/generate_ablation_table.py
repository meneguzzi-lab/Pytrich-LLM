#!/usr/bin/env python3
"""
Ablation result tables: domains x (the paper's columns + the three prompt arms).

    # every metric, one PDF each plus a combined multi-page PDF
    python3 generate_ablation_table.py --all-metrics OUTDIR \
        -i Experiments/results/ablation.csv \
        --baseline Experiments/results/results.csv

    # a single metric
    python3 generate_ablation_table.py --metric expanded_nodes out.pdf -i ... --baseline ...

    # layout only, fabricated numbers, coverage table
    python3 generate_ablation_table.py --mock out.pdf

Arms are recognised by the directory suffix in experiment_name (ARM_SUFFIXES). Arm 1
is the reference: arm 2 adds the domain hints, arm 3 adds a worked example from a
different domain. Any arm may be absent -- its column reads "not run" rather than
silently collapsing to zero. An unrecognised experiment_name is reported, not dropped
in silence.

Metrics
    coverage             distinct instances solved by >=1 candidate (higher better)
    expanded_nodes       median over solved runs (lower better)
    solution_size        median plan length over solved runs (lower better)
    nodes_per_second     median throughput (higher better)
    search_elapsed_time  median seconds (lower better)
    winrate              PAIRED against ARM 1: of instances both solve, the share where
                         the arm's best candidate expands fewer nodes than arm 1's.
                         Paired on instance, so it controls for instance difficulty and
                         is much less sensitive to unequal candidate pools than a median
                         or a virtual best. Arms 2 and 3 are each compared only to arm 1,
                         never to each other -- they differ in two inputs at once.
                         Arm columns only; baselines are not paired.

Notes that matter when reading the output
  * Coverage saturates. Most domains sit at or near |I| for nearly every system, so
    coverage cannot discriminate between the arms; the signal is in node counts.
  * The arms are typically GBFS-only while the paper's columns are a virtual best
    over GBFS, A* and WA*. That favours the paper's columns; note it or run all three
    (JobArrays/run_ablation_astar_wastar.sh).
  * Candidate pools are unequal because some generated heuristics crash. The coverage
    cells hold the coverage number alone; pass --candidate-counts to print the pool
    size beside it. Either way the counts are listed in the run summary on stdout, so
    a table without them is never the only record.

experiment_name carries BOTH arm and algorithm, e.g. Gemini3Flash-arm1-no-example_WA.
Heuristic class names are identical across arms, so never group by heuristic_name.

Requires matplotlib (present in .venv, not a planner runtime dependency): run with
.venv/bin/python3.
"""
import argparse
import csv
import os
import random
import re
import statistics
import textwrap

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

random.seed(20260725)

DOMS = ["Barman-BDI", "Blocksworld-GTOHP", "Depots", "Robot", "Rover-GTOHP", "Towers"]

# Instances per domain, every one of which is grounded and runnable.
DISK_SIZES = {
    "Barman-BDI": 20,
    "Blocksworld-GTOHP": 30,
    "Depots": 30,
    "Robot": 20,
    "Rover-GTOHP": 30,
    "Towers": 20,
    "Lamps": 30,
}
DOMAIN_SIZES = dict(DISK_SIZES)

# Candidate counts for --mock only (layout check).
USABLE = {"Towers": (17, 18, 18), "Blocksworld-GTOHP": (16, 15, 16), "Robot": (16, 18, 17),
          "Depots": (18, 16, 17), "Rover-GTOHP": (16, 10, 15), "Barman-BDI": (15, 20, 18)}

PAPER_COLS = ["RC-FF", "RC-LMCut", "LMCount$^P$", "TDG", "LMCount", "LLM-VB"]

# Directory-suffix -> internal arm key. Arm 1 is the reference: arms 2 and 3 each
# change exactly one prompt input relative to it, so every paired comparison is
# against arm 1 and never arm 2 against arm 3.
ARM_SUFFIXES = (
    ("arm1", "-arm1-no-example"),
    ("arm2", "-arm2-no-example-hints"),
    ("arm3", "-arm3-fixed-example"),
)
ARM_ORDER = tuple(a for a, _ in ARM_SUFFIXES)
REFERENCE_ARM = "arm1"
ARM_COLS = ["Arm 1\n(no example)", "Arm 2\n(no ex. + hints)", "Arm 3\n(fixed cross-dom. ex.)"]

BASELINE_SYSTEMS = {
    "RC-FF": "Panda-rc2ff", "RC-LMCut": "Panda-rc2lmc",
    "LMCount$^P$": "Panda-lmcountlmtypebid",
    "TDG": "Pytrich-TDG", "LMCount": "Pytrich-LMCOUNT",
}

# Which planner each of the paper's columns belongs to. Markdown has no spanning
# header cells, so the planner is folded into the column label instead -- otherwise
# a reader cannot tell that RC-FF is PANDA and TDG is Pytrich.
PLANNER_OF = {
    "RC-FF": "PANDA", "RC-LMCut": "PANDA", "LMCount$^P$": "PANDA",
    "TDG": "Pytrich", "LMCount": "Pytrich", "LLM-VB": "LLM (virtual best)",
}
LLM_VB_MODELS = {"ClaudeOpus46", "Gemini3Flash", "Gemini3Pro",
                 "Gemini20Flash", "Gemini25Pro"}

# What each arm actually withholds, stated so a reader of the table alone can tell.
# Exact on the two points that are easy to overstate, both of which would flatter us:
#   * arm 1's section 7 is REPLACED by a blind stub, not removed. Saying "no example"
#     without that implies the model got no interface template either, which would
#     blame arm 1's compile failures on a missing section that was in fact present.
#   * the section 2.5 hints are prior-analysis insights, NOT a description of the
#     domain's predicates and objects. Predicates and object types come from section 1,
#     the domain HDDL, which cannot be ablated and is in every arm. Describing the
#     hints as domain description would understate the knowledge we inject.
# Text is shared by the PDF and markdown renderers so the two cannot drift.
ARM_NOTES = (
    # "Shaded" would be a lie in markdown, which has no cell colour.
    "Each Arm column changes exactly one prompt input relative to Arm 1.",
    "Arm 1 withholds both inputs. No domain-specific heuristic example: prompt section 7 "
    "is replaced by a domain-agnostic stub that returns 0 for every node -- a syntactic "
    "template for the required signatures, with no reasoning to imitate. And no domain "
    "hints: section 2.5 is empty. Arm 2 restores section 2.5; Arm 3 restores section 7 "
    "with a working heuristic from a DIFFERENT domain (Towers for every domain, Snake "
    "for Towers itself).",
    "The section 2.5 hints are insights from our own prior analysis, not a description "
    "of the domain -- for Towers they give the exact optimal cost 2^k-1 for the largest "
    "displaced ring, the largest-to-smallest processing order, and to memoize by bitwise "
    "state. The predicates and object types are in section 1, the domain HDDL, which is "
    "not ablatable and is present in all three arms.",
)

ALGO_SUFFIX = re.compile(r"_(GBFS|StandardA|WA)$")
ALGOS = ("GBFS", "StandardA", "WA")
VB_ALGO = "VirtualBest"
# Metrics computed by pairing an arm against arm 1: pre-formatted strings, no
# baseline columns (baselines are not paired), blank reference-arm column.
PAIRED_METRICS = ("winrate", "noderatio")

METRICS = {
    "coverage":            ("Coverage", "instances solved by >=1 candidate", "higher"),
    "expanded_nodes":      ("Expanded nodes (median)", "over solved runs", "lower"),
    "solution_size":       ("Plan length (median)", "over solved runs", "lower"),
    "nodes_per_second":    ("Nodes/second (median)", "over solved runs", "higher"),
    "search_elapsed_time": ("Search time (median, s)", "over solved runs", "lower"),
    "winrate":             ("Win rate vs Arm 1", "paired per instance, fewer expanded nodes", "higher"),
    "noderatio":           ("Node ratio vs Arm 1 (median)", "paired per instance, arm/arm1", "lower"),
}


def base_name(n):
    return ALGO_SUFFIX.sub("", n)


def algo_of(n):
    """Algorithm from experiment_name, or None if it carries no suffix."""
    m = ALGO_SUFFIX.search(n)
    return m.group(1) if m else None


def instance_set_note():
    """One line naming which instance set every number in the table is over."""
    total = sum(DOMAIN_SIZES[d] for d in DOMS)
    return f"|I| is all {total} instances of these six domains. Lamps is not in this table."


INSTANCE_INDEX = re.compile(r"(\d+)\D*$")


def in_instance_set(r):
    """Is this run on an instance DOMAIN_SIZES counts? (Always, with the full instance set.)"""
    dom = r["domain_name"]
    limit = DOMAIN_SIZES.get(dom)
    if limit is None or limit >= DISK_SIZES.get(dom, limit):
        return True
    m = INSTANCE_INDEX.search(r["problem_name"])
    if not m:
        raise SystemExit(f"cannot read an instance index from {dom} "
                         f"'{r['problem_name']}', so it cannot be placed inside or "
                         f"outside the {limit}-instance set; extend INSTANCE_INDEX.")
    return int(m.group(1)) <= limit


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)
                if r.get("solution_size") and in_instance_set(r)]


def num(r, field):
    try:
        return float(r[field])
    except (KeyError, TypeError, ValueError):
        return None


def agg(rows, metric):
    """domain -> aggregated value for these rows."""
    out = {}
    if metric == "coverage":
        for r in rows:
            out.setdefault(r["domain_name"], set()).add(r["problem_name"])
        return {d: len(v) for d, v in out.items()}
    for r in rows:
        v = num(r, metric)
        if v is not None:
            out.setdefault(r["domain_name"], []).append(v)
    return {d: statistics.median(v) for d, v in out.items() if v}


def best_per_instance(rows):
    """(domain, instance) -> fewest expanded nodes any candidate needed."""
    best = {}
    for r in rows:
        v = num(r, "expanded_nodes")
        if v is None:
            continue
        k = (r["domain_name"], r["problem_name"])
        if k not in best or v < best[k]:
            best[k] = v
    return best


def fmt(v, metric):
    if v is None:
        return "--"
    if metric == "coverage":
        return str(int(v))
    if metric in ("winrate", "noderatio"):
        return v            # already a formatted string
    if metric == "search_elapsed_time":
        return f"{v:,.2f}"
    if metric == "nodes_per_second":
        return f"{v:,.0f}"
    if metric == "solution_size":
        return f"{v:,.1f}"
    return f"{v:,.0f}"


def build_rows(metric, arms, cands, base_cov, have_baseline, present=ARM_ORDER,
               show_cands=False):
    rows = []
    show_paper = metric not in PAIRED_METRICS and have_baseline
    for dom in DOMS:
        n = DOMAIN_SIZES[dom]
        cells = [dom, str(n)]
        if metric in PAIRED_METRICS:
            cells += ["" for _ in PAPER_COLS]
        else:
            cells += [fmt(base_cov.get(c, {}).get(dom), metric) if show_paper else "--"
                      for c in PAPER_COLS]
        for a in ARM_ORDER:
            if a not in present:
                # Distinguish "this arm was not evaluated" from "it scored nothing",
                # so a half-collected sweep cannot be read as a result.
                cells.append("not run")
                continue
            v = arms.get(a, {}).get(dom)
            s = fmt(v, metric)
            if metric == "coverage" and show_cands:
                s += f"  ({cands.get(a, {}).get(dom, 0)} cand.)"
            cells.append(s)
        rows.append(cells)

    # totals / overall row, over the rows shown. DOMAIN_SIZES also holds Lamps, which the
    # arms never ran; summing all of it put 30 Lamps instances and the baselines' Lamps
    # solves into a Total whose rows do not include Lamps.
    tot = ["Total", str(sum(DOMAIN_SIZES[d] for d in DOMS))]
    if metric in PAIRED_METRICS:
        tot += ["" for _ in PAPER_COLS]
    elif metric == "coverage":
        tot += [str(int(sum(base_cov.get(c, {}).get(d, 0) for d in DOMS)))
                if show_paper and base_cov.get(c) else "--"
                for c in PAPER_COLS]
    else:
        # a median of medians is meaningless; say so rather than print a number
        tot += ["" for _ in PAPER_COLS]
    for a in ARM_ORDER:
        if a not in present:
            tot.append("not run")
        elif metric == "coverage":
            s = str(int(sum(arms.get(a, {}).values())))
            if show_cands:
                s += f"  ({sum(cands.get(a, {}).values())} cand.)"
            tot.append(s)
        elif metric in PAIRED_METRICS:
            tot.append(arms.get(a, {}).get("__overall__", "--"))
        else:
            tot.append("")
    rows.append(tot)
    return rows


# PDF layout, in inches and points. The page height is derived from these rather than
# fixed, so a longer footnote grows the page instead of overprinting the table.
TABLE_IN, TITLE_IN, NOTE_PAD_IN = 3.13, 0.54, 0.30
NOTE_PT, NOTE_LEAD, NOTE_WRAP = 8.0, 1.6, 165


def build_notes(metric, mock, have_baseline, show_cands, extra_note):
    """Footnotes for one table, in reading order. Shared by both renderers."""
    notes: list[str] = list(ARM_NOTES)
    if mock:
        notes.append("Every value is random. Candidate counts are real.")
    else:
        if metric == "coverage":
            if show_cands:
                notes.append("\"(n cand.)\" = distinct candidates that solved at least one "
                             "instance, of 20 generated. Pools are unequal because some "
                             "generated heuristics crash.")
            else:
                # The cells no longer carry the pool size, so the caveat has to: equal
                # coverage off unequal pools is not the same result as equal coverage
                # off equal pools.
                notes.append("Each arm generated 20 candidates per domain, of which "
                             "unequal numbers run at all -- coverage here is over the "
                             "candidates that worked, not over 20.")
            notes.append("Coverage saturates: most domains sit at or near |I| for nearly "
                         "every system, so it does not discriminate between the arms.")
        elif metric == "winrate":
            notes.append("Each arm is paired against ARM 1, per instance, over instances "
                         "both solve -- so unequal candidate pools do not distort it. "
                         "Arm 1's own column is blank, and so are the baselines: they are "
                         "not paired against an arm. Arm 2 and Arm 3 are NOT compared to "
                         "each other; they differ in two inputs at once.")
        else:
            notes.append("Medians over solved runs only, so a domain where one arm solves "
                         "fewer instances is measured on an easier subset. Read alongside "
                         "the coverage table.")
            notes.append("No Total: a median of medians is not meaningful.")
        if not have_baseline and metric not in PAIRED_METRICS:
            notes.append("Paper columns show \"--\": no --baseline CSV was given.")

    if extra_note:
        notes.append(extra_note)
    notes.append(instance_set_note())
    return notes


def render(rows, out_or_pdf, metric, mock, have_baseline, extra_note="",
           show_cands=False):
    label, sub, direction = METRICS[metric]
    header = ["Domain", "$|I|$"] + PAPER_COLS + ARM_COLS
    notes = build_notes(metric, mock, have_baseline, show_cands, extra_note)

    # matplotlib does not wrap fig.text, so a long note silently runs off both edges
    # of the page. Wrap here rather than hand-splitting each string.
    wrapped = "\n".join(textwrap.fill(n, NOTE_WRAP) for n in notes)

    # Size the page to the notes instead of fixing it: the block is now long enough
    # that a fixed height would run it into the table, silently, with no error.
    note_in = (wrapped.count("\n") + 1) * NOTE_PT * NOTE_LEAD / 72 + NOTE_PAD_IN
    fig_h = TABLE_IN + TITLE_IN + note_in

    # Wider than the two-arm version: the third arm column pushed the table past the
    # page edge.
    fig, ax = plt.subplots(figsize=(17.5, fig_h))
    ax.axis("off")
    ax.set_position([0.02, note_in / fig_h, 0.96, TABLE_IN / fig_h])

    tbl = ax.table(cellText=rows, colLabels=header, cellLoc="center", loc="upper center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.0)
    tbl.auto_set_column_width(col=list(range(len(header))))

    ncols = len(header)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#eeeeee"); cell.set_text_props(weight="bold")
            cell.set_height(0.20)
        if r == len(rows):
            cell.set_facecolor("#f6f6f6"); cell.set_text_props(weight="bold")
        if c >= ncols - len(ARM_COLS):
            cell.set_facecolor("#fff4e0" if r != 0 else "#ffe6bd")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")

    fig.suptitle(f"{label} by domain — ablation arms alongside the paper's columns",
                 y=0.985, fontsize=13, weight="bold")
    ax.text(0.5, 1.005,
            ("LAYOUT MOCKUP — ALL NUMBERS ARE FABRICATED, NOT RESULTS" if mock else
             f"Gemini 3 Flash, GBFS. {sub} — {direction} is better."),
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10.5 if mock else 9.5,
            color="#b00000" if mock else "#333333",
            weight="bold" if mock else "normal")

    fig.text(0.5, (NOTE_PAD_IN / 2) / fig_h, wrapped, ha="center", va="bottom",
             fontsize=NOTE_PT, color="#333333", linespacing=NOTE_LEAD)

    if mock:
        fig.text(0.5, 0.5, "FABRICATED SAMPLE", fontsize=54, color="#d00000",
                 alpha=0.10, ha="center", va="center", rotation=22, weight="bold")

    if isinstance(out_or_pdf, PdfPages):
        out_or_pdf.savefig(fig)
    else:
        fig.savefig(out_or_pdf)
    plt.close(fig)


def markdown(rows, metric, have_baseline, present, extra_note="", show_cands=False):
    """One metric as a GitHub-flavoured markdown table, planner-labelled."""
    label, sub, direction = METRICS[metric]
    # $...$ is LaTeX for the PDF; markdown wants plain text.
    def plain(c):
        return c.replace("$^P$", "^P").replace("$|I|$", "|I|")

    head = ["Domain", "\\|I\\|"]
    head += [f"{PLANNER_OF.get(c, '')} / {plain(c)}".lstrip(" /") for c in PAPER_COLS]
    head += [f"Ablation / {a.replace(chr(10), ' ')}" for a in ARM_COLS]

    out = [f"### {label} — {sub} ({direction} is better)", ""]
    out.append("Gemini 3 Flash, GBFS. Arm columns are the prompt ablation; "
               "the rest are the paper's systems.")
    out.append("")
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] * len(head)) + "|")
    for r in rows:
        cells = [plain(str(c)) if c != "" else "" for c in r]
        # escape the pipe that appears inside a literal cell value, not the borders
        cells = [c.replace("|", "\\|") for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    notes: list[str] = list(ARM_NOTES)
    if metric == "coverage":
        if show_cands:
            notes.append("`(n cand.)` = candidates that solved at least one instance, of 20 "
                         "generated. Pools are unequal because some generated heuristics crash.")
        else:
            notes.append("Each arm generated 20 candidates per domain and unequal numbers of "
                         "them run at all, so coverage is over the candidates that worked.")
        notes.append("Coverage saturates — most domains sit at or near |I| for nearly every "
                     "system, so it cannot separate the arms. The signal is in node counts.")
    elif metric == "winrate":
        notes.append("Paired per instance against **Arm 1**, over instances both solve, so "
                     "unequal candidate pools do not distort it. Arm 1's own column is blank.")
        notes.append("Arms 2 and 3 are **not** compared to each other — they differ in two "
                     "prompt inputs at once.")
    else:
        notes.append("Medians over solved runs only, so a domain where one arm solves fewer "
                     "instances is measured on an easier subset. Read with the coverage table.")
        notes.append("No Total row value: a median of medians is not meaningful.")
    if not have_baseline and metric != "winrate":
        notes.append("Paper columns are `--`: no baseline CSV was supplied.")
    for a in ARM_ORDER:
        if a not in present:
            notes.append(f"`{a}` was not evaluated; its column reads *not run*.")

    notes.append(instance_set_note())
    if extra_note:
        notes.append(extra_note)
    out += [f"- {n}" for n in notes]
    return "\n".join(out)


def virtual_best(by_algo):
    """{algo: {arm: rows}} -> {arm: rows}, best row per (arm, domain, instance, cand).

    Mirrors how every column of the paper is computed: a virtual best over GBFS, A* and
    WA*. Without it an arm number is not comparable to a paper column. Keyed on the
    candidate too, so this is per-candidate-best-algorithm, not an oracle over
    candidates -- that would silently reward whichever arm had a luckier draw.
    """
    merged = {}
    for algo, arms in by_algo.items():
        if algo == VB_ALGO:
            continue
        for arm, rows in arms.items():
            best = merged.setdefault(arm, {})
            for r in rows:
                v = num(r, "expanded_nodes")
                if v is None:
                    continue
                k = (r["domain_name"], r["problem_name"], r["heuristic_name"])
                prev = best.get(k)
                # Store the value alongside the row: re-deriving it invites a None
                # comparison, since num() is Optional by contract.
                if prev is None or v < prev[0]:
                    best[k] = (v, r)
    return {arm: [r for _, r in d.values()] for arm, d in merged.items()}


def gather(args):
    """Read the CSVs once.

    Returns ({algo: {arm: rows}}, {algo: {arm: {domain: n_candidates}}}, baseline rows).

    Splitting by algorithm is not optional: base_name() strips the _GBFS/_StandardA/_WA
    suffix to find the arm, so pooling a multi-algorithm CSV would silently average
    three search algorithms into one series and every number would be meaningless.
    """
    by_algo, unknown, no_algo = {}, set(), 0
    for path in args.input:
        for r in load(path):
            name = r["experiment_name"]
            b, algo = base_name(name), algo_of(name)
            if algo is None:
                no_algo += 1
                continue
            for arm, suffix in ARM_SUFFIXES:
                if b.endswith(suffix):
                    by_algo.setdefault(algo, {}).setdefault(arm, []).append(r)
                    break
            else:
                unknown.add(b)
    if unknown:
        print(f"WARNING ignoring unrecognised experiment_name(s): {sorted(unknown)}")
    if no_algo:
        print(f"WARNING ignoring {no_algo} row(s) whose experiment_name carries no "
              f"_GBFS/_StandardA/_WA suffix -- the algorithm cannot be determined.")
    if not by_algo:
        raise SystemExit("no arm rows found; expected experiment_name like "
                         "<Model>-arm1-no-example_GBFS")

    if args.virtual_best:
        vb = virtual_best(by_algo)
        if vb:
            by_algo[VB_ALGO] = vb

    cands = {algo: {a: {d: len({r["heuristic_name"] for r in rs
                                if r["domain_name"] == d}) for d in DOMS}
                    for a, rs in arms.items()}
             for algo, arms in by_algo.items()}
    brows = load(args.baseline) if args.baseline else []
    return by_algo, cands, brows


def baseline_agg(brows, metric):
    if not brows or metric in PAIRED_METRICS:
        return {}
    out = {}
    for col, prefix in BASELINE_SYSTEMS.items():
        out[col] = agg([r for r in brows if base_name(r["experiment_name"]) == prefix], metric)
    out["LLM-VB"] = agg([r for r in brows if base_name(r["experiment_name"]) in LLM_VB_MODELS],
                        metric)
    return out


def winrate(arm_rows, arm):
    """domain -> 'w/n (p%)' for `arm` beating the reference arm, paired per instance.

    Paired on instance and restricted to instances both arms solve, so it is not
    distorted by the arms having different numbers of usable candidates.
    """
    ref = best_per_instance(arm_rows.get(REFERENCE_ARM, []))
    cur = best_per_instance(arm_rows.get(arm, []))
    shared = set(ref) & set(cur)
    per, tw, tn = {}, 0, 0
    for dom in DOMS:
        ks = [k for k in shared if k[0] == dom]
        if not ks:
            continue
        w = sum(1 for k in ks if cur[k] < ref[k])
        per[dom] = f"{w}/{len(ks)}  ({100*w/len(ks):.0f}%)"
        tw += w; tn += len(ks)
    per["__overall__"] = f"{tw}/{tn}  ({100*tw/tn:.0f}%)" if tn else "--"
    return per


def noderatio(arm_rows, arm):
    """domain -> median of (arm / arm1) expanded nodes, paired per instance.

    The companion to winrate, and the one that answers "how much does it matter".
    A win rate counts instances; this measures magnitude. They can disagree sharply:
    a 90% win rate next to a 0.97x ratio means the arm wins most instances by almost
    nothing, which is a coin flip near parity, not an effect. Report both or neither.
    """
    ref = best_per_instance(arm_rows.get(REFERENCE_ARM, []))
    cur = best_per_instance(arm_rows.get(arm, []))
    shared = [k for k in set(ref) & set(cur) if ref[k] > 0]
    per, allr = {}, []
    for dom in DOMS:
        rs = [cur[k] / ref[k] for k in shared if k[0] == dom]
        if not rs:
            continue
        per[dom] = f"{statistics.median(rs):.2f}x  (n={len(rs)})"
        allr += rs
    per["__overall__"] = (f"{statistics.median(allr):.2f}x  (n={len(allr)})"
                          if allr else "--")
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", nargs="?", help="output PDF (single-metric mode)")
    # No default: "not given" has to be distinguishable from "coverage", so that
    # --markdown/--all-metrics can emit every metric unless one was actually asked for.
    ap.add_argument("--metric", choices=sorted(METRICS), default=None,
                    help="one metric only. Single-PDF mode defaults to coverage; with "
                         "--markdown or --all-metrics, omitting this emits every metric "
                         "and giving it narrows the output to that one.")
    ap.add_argument("--all-metrics", metavar="OUTDIR",
                    help="write one PDF per metric into OUTDIR, plus combined.pdf")
    ap.add_argument("--markdown", metavar="FILE", nargs="?", const="-",
                    help="emit markdown tables instead of PDFs (default stdout). "
                         "Column headers name the planner. Covers every metric.")
    ap.add_argument("-i", "--input", action="append", default=[], metavar="CSV",
                    help="ablation results CSV; repeatable")
    ap.add_argument("--baseline", metavar="CSV", help="results CSV for the paper's columns")
    ap.add_argument("--algorithm", metavar="LIST",
                    help="restrict to these search algorithms, comma-separated "
                         f"({'/'.join(ALGOS)}, or {VB_ALGO}). Default: every one found. "
                         "Each gets its own table set -- they are never pooled.")
    ap.add_argument("--candidate-counts", action="store_true",
                    help="print each arm's usable-candidate pool size beside its "
                         "coverage number. Off by default: the coverage table reports "
                         "coverage. The counts are always in the stdout summary.")
    ap.add_argument("--virtual-best", action="store_true",
                    help="also emit a virtual-best-over-algorithms set, matching how "
                         "the paper's columns are computed. Needs all three algorithms "
                         "for an arm to be meaningful for that arm.")
    ap.add_argument("--mock", action="store_true",
                    help="fabricate a coverage table; layout check only")
    args = ap.parse_args()

    if args.mock:
        if args.input:
            ap.error("--mock fabricates everything; do not combine it with real input.")
        rows = []
        for dom in DOMS:
            n = DOMAIN_SIZES[dom]
            vals = [random.randint(int(n * .35), n) for _ in PAPER_COLS]
            arm_cells = [str(random.randint(int(n * .4), n)) for _ in USABLE[dom]]
            if args.candidate_counts:
                arm_cells = [f"{s}  ({c} cand.)"
                             for s, c in zip(arm_cells, USABLE[dom])]
            rows.append([dom, str(n)] + [str(v) for v in vals] + arm_cells)
        rows.append(["Total", str(sum(DOMAIN_SIZES.values()))]
                    + ["--"] * (len(PAPER_COLS) + len(ARM_COLS)))
        render(rows, args.output or "mock.pdf", "coverage", mock=True,
               have_baseline=False, show_cands=args.candidate_counts)
        print(f"wrote {args.output or 'mock.pdf'}")
        return 0

    if not args.input:
        ap.error("give -i with at least one ablation CSV, or --mock for the layout.")
    if not args.all_metrics and not args.output and not args.markdown:
        ap.error("give an output PDF path, or use --all-metrics OUTDIR, or --markdown.")

    by_algo, cands_by_algo, brows = gather(args)
    have_baseline = bool(brows)

    # Drop baseline columns the CSV has no rows for, rather than printing 0 for a
    # system that was never run.
    if have_baseline:
        present_names = {base_name(r["experiment_name"]) for r in brows}
        gone = [c for c, prefix in BASELINE_SYSTEMS.items() if prefix not in present_names]
        if gone:
            for c in gone:
                PAPER_COLS.remove(c)
            print("dropping baseline column(s) with no rows in the baseline CSV: "
                  + ", ".join(c.replace("$^P$", "^P") for c in gone))

    order = [a for a in ALGOS if a in by_algo]
    if VB_ALGO in by_algo:
        order.append(VB_ALGO)
    if args.algorithm:
        want = {a.strip() for a in args.algorithm.split(",")}
        missing = want - set(by_algo)
        if missing:
            ap.error(f"no rows for algorithm(s): {', '.join(sorted(missing))}. "
                     f"Present: {', '.join(sorted(by_algo))}")
        order = [a for a in order if a in want]

    metrics_order = ["coverage", "winrate", "noderatio", "expanded_nodes",
                     "search_elapsed_time", "nodes_per_second", "solution_size"]
    single = args.metric or "coverage"
    wanted = ([m for m in metrics_order if m in METRICS]
              if (args.all_metrics or args.markdown) else [single])
    if args.metric:
        wanted = [args.metric]

    md_parts = []
    for algo in order:
        arm_rows = by_algo[algo]
        cands = cands_by_algo[algo]
        present = tuple(a for a in ARM_ORDER if a in arm_rows)
        label = ("virtual best over GBFS/A*/WA*" if algo == VB_ALGO else algo)
        note = f"Search algorithm: **{label}**."
        absent = [a for a in ARM_ORDER if a not in present]
        if absent:
            note += (f" Not evaluated under this algorithm: {', '.join(absent)} — "
                     "those columns read *not run*, and no cross-arm claim can use them.")
        if algo == VB_ALGO:
            # A "virtual best" over one algorithm is not a virtual best. If the arms
            # were not all run under all three, the VB silently favours whichever arm
            # had more algorithms to choose from -- so name the composition per arm.
            comp = {a: [al for al in ALGOS if a in by_algo.get(al, {})]
                    for a in present}
            note += "  Composed per arm from: " + "; ".join(
                f"{a} = {'+'.join(v) if v else 'nothing'}" for a, v in comp.items()) + "."
            uneven = {a for a, v in comp.items() if len(v) < len(ALGOS)}
            if uneven:
                note += (f"  **Not a fair comparison: {', '.join(sorted(uneven))} had "
                         "fewer algorithms to draw from, which biases the virtual best "
                         "against them.** Use the per-algorithm tables instead.")
        else:
            note += ("  The paper's columns are a virtual best over GBFS, A* and WA*, "
                     "so they are not comparable to a single-algorithm arm column; "
                     "compare arms to each other here.")
        if REFERENCE_ARM not in present:
            print(f"WARNING [{algo}] {REFERENCE_ARM} absent; paired metrics will be "
                  "empty for this algorithm.")

        def rows_for(metric, arm_rows=arm_rows, cands=cands, present=present):
            if metric in ("winrate", "noderatio"):
                fn = winrate if metric == "winrate" else noderatio
                # The reference arm has no ratio against itself; leave it blank rather
                # than printing a meaningless 1.00x / 0-of-n.
                arms = {REFERENCE_ARM: {d: "" for d in DOMS}}
                arms[REFERENCE_ARM]["__overall__"] = ""
                for a in present:
                    if a != REFERENCE_ARM:
                        arms[a] = fn(arm_rows, a)
                return build_rows(metric, arms, cands, {}, have_baseline, present,
                                  args.candidate_counts)
            arms = {a: agg(rs, metric) for a, rs in arm_rows.items()}
            return build_rows(metric, arms, cands, baseline_agg(brows, metric),
                              have_baseline, present, args.candidate_counts)

        if args.markdown:
            md_parts.append(f"# {label}\n\n{note}\n\n" + "\n\n".join(
                markdown(rows_for(m), m, have_baseline, present,
                         show_cands=args.candidate_counts) for m in wanted))
        elif args.all_metrics:
            out = os.path.join(args.all_metrics, algo)
            os.makedirs(out, exist_ok=True)
            combined = os.path.join(out, "ablation_all_tables.pdf")
            with PdfPages(combined) as pdf:
                for m in wanted:
                    rows = rows_for(m)
                    single = os.path.join(out, f"ablation_{m}.pdf")
                    for target in (single, pdf):
                        render(rows, target, m, mock=False,
                               have_baseline=have_baseline, extra_note=note,
                               show_cands=args.candidate_counts)
            print(f"  {algo}: {len(wanted)} PDFs + {os.path.basename(combined)} in {out}")
        else:
            render(rows_for(single), args.output, single, mock=False,
                   have_baseline=have_baseline, extra_note=note,
                   show_cands=args.candidate_counts)
            print(f"wrote {args.output} ({algo})")
            break

    if args.markdown:
        doc = "\n\n---\n\n".join(md_parts)
        if args.markdown == "-":
            print(doc)
        else:
            with open(args.markdown, "w", encoding="utf-8") as f:
                f.write(doc + "\n")
            print(f"wrote {args.markdown}")

    print()
    for algo in order:
        for a in ARM_ORDER:
            rs = by_algo[algo].get(a)
            if rs:
                n = sum(cands_by_algo[algo][a].values())
                print(f"  {algo:12} {a}: {len(rs)} solved runs, {n} candidates")
            else:
                print(f"  {algo:12} {a}: not present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
