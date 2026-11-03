#!/usr/bin/env python3
"""
Code-validity rates for generated heuristics: does it compile, and does it run?

Three separate numbers, because collapsing them misstates how often a model emits
broken code:

  compiles              ast.parse succeeds. Failure = the model produced text that is
                        not Python.
  pipeline artifact     does NOT parse as written, but parses once trailing JSON
                        envelope debris is stripped. Failure = OUR generator, not the
                        model. Counted separately and never as a model failure.
  runs                  the planner reaches GOAL on one small instance of the domain.
                        Failure = code that imports fine and breaks at run time -- the
                        common case, since validate_heuristic never calls initialize().

The pipeline-artifact column matters most for heuristics generated before the repair
landed in generate_heuristic.py: 27 of the 31 unparseable files in the corpus were
complete, valid code plus two stray characters. Those files are deliberately never
modified on disk (they are the record of what the model produced), so the repair has to
happen at measurement time instead.

    # the three ablation arms (default)
    python3 Experiments/python_scripts/measure_heuristic_validity.py

    # every model directory found on disk
    python3 Experiments/python_scripts/measure_heuristic_validity.py --all

    # named directories only
    python3 Experiments/python_scripts/measure_heuristic_validity.py \
        --models Gemini3Pro,GPT5,ClaudeOpus46

    # skip the (slow) execution pass; compile + artifact only
    ... --compile-only

    # rates only, into a file, no per-file failure list
    ... --summary-only -o outputs/tables/validity_arms.md

    # measure once, then re-render as often as you like without re-running anything
    ... --json validity.json -o validity.md
    ... --from-json validity.json --compact -o validity_compact.md

    # several dumps in one table (paper models above their ablation arms)
    ... --from-json paper.json --from-json arms.json --compact -o validity_all.md

Prints markdown tables to stdout, or to -o.
"""
import argparse
import ast
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(ROOT, "Pytrich/Heuristics/GeneratedHeuristics")
PY = sys.executable

ARM_DIRS = ["Gemini3Flash-arm1-no-example",
            "Gemini3Flash-arm2-no-example-hints",
            "Gemini3Flash-arm3-fixed-example"]

# One deliberately small instance per domain: this measures "does it run", not search
# quality, so a hard instance would only add timeouts.
DOMAIN_INSTANCE = {
    "Towers":            "Towers-pfile_03",
    "Barman_BDI":        "Barman-BDI-pfile01",
    "Blocksworld_GTOHP": "Blocksworld-GTOHP-p01",
    "Depots":            "Depots-p01",
    "Robot":             "Robot-pfile_01_001",
    "Rover_GTOHP":       "Rover-GTOHP-p01",
}

# Same tuple the generator strips, same conservative rule: only when the file does not
# already parse AND stripping makes it parse.
DEBRIS = ('"', '}', '"}', '"""', "'''", '```', '')


def parses(src):
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def api_era(src):
    """'current' | 'legacy' | 'unknown' -- which Pytrich Heuristic interface this targets.

    Pytrich changed the heuristic contract. The corpus spans both sides of it:

      legacy   __init__(self, model, ...)          model handed to the constructor
               __call__(self, state, task_network)
      current  __init__(self, param=None, ...)     no required positional args
               initialize(self, model, initial_node)
               __call__(self, parent_node, node)

    `--heuristic-file` constructs the class with NO arguments, so a legacy heuristic
    cannot be instantiated by it at all -- it dies with "__init__() missing 1 required
    positional argument: 'model'". That is a harness limitation, not the model emitting
    broken code, and counting it as a failure would report GPT5 at 83% when the truth is
    we cannot load it. Legacy files are reported in their own column instead.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "unknown"
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
        init = methods.get("__init__")
        required = []
        if init is not None:
            pos = [a.arg for a in init.args.args][1:]      # drop self
            required = pos[:len(pos) - len(init.args.defaults)] if pos else []
        if "initialize" in methods:
            # Arity is the era tell. The current contract is initialize(self, model,
            # initial_node) and __call__(self, parent_node, node); an older one took
            # initialize(self, model) and __call__(self, node). A file with the old
            # arity dies with "takes 2 positional arguments but 3 were given" -- again a
            # harness/era mismatch, not the model writing broken code.
            def nargs(name):
                fn = methods.get(name)
                return None if fn is None else len(fn.args.args) - 1
            if nargs("initialize") == 1 or nargs("__call__") == 1:
                return "legacy"
            # Current interface, but a required constructor argument still makes it
            # unloadable. That IS a defect in the generated code -- unlike the legacy
            # case -- so it is reported as a failure, just as its own kind: adding
            # "=None" would fix it, which is not the same as a logic bug.
            return "ctor" if required else "current"
        if required:
            return "legacy"
    return "unknown"


def is_pipeline_artifact(src):
    """True if trailing envelope debris alone is what stops this parsing."""
    lines = src.rstrip().split("\n")
    for cut in range(1, min(4, len(lines)) + 1):
        if lines[-cut].strip() not in DEBRIS:
            continue
        if parses("\n".join(lines[:-cut])):
            return True
    return False


def classify(path, sas, run):
    src = open(path, encoding="utf-8", errors="replace").read()
    if not parses(src):
        if is_pipeline_artifact(src):
            return "artifact", "envelope debris (our generator, not the model)"
        try:
            ast.parse(src)
        except SyntaxError as e:
            return "compile", f"SyntaxError line {e.lineno}: {e.msg}"
    era = api_era(src)
    if era == "legacy":
        return "legacy", "targets the pre-initialize() Heuristic API; --heuristic-file " \
                         "cannot construct it"
    if era == "ctor":
        return "ctor", "__init__ has a required positional argument, so it cannot be " \
                       "constructed with no args (a default would fix it)"
    if not run or sas is None:
        return "ok", ""
    try:
        p = subprocess.run(
            [PY, "__main__.py", "--sas_file", sas, "--heuristic-file", path,
             "-S", "Astar()", "-N", "AstarNode(G=0,H=1)"],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "runtime", "timeout (300 s) on a small instance"
    out = p.stdout + p.stderr
    if "Search Status : GOAL" in out:
        return "ok", ""
    for line in reversed(out.splitlines()):
        head = line.strip().split(":")[0]
        if head.endswith(("Error", "Exception")):
            return "runtime", line.strip()[:110]
    return "runtime", "ran, did not reach GOAL"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma-separated model directory names")
    ap.add_argument("--all", action="store_true", help="every model directory on disk")
    ap.add_argument("--compile-only", action="store_true",
                    help="skip the execution pass (fast)")
    ap.add_argument("--summary-only", action="store_true",
                    help="omit the per-file failure table, keeping the rates only")
    ap.add_argument("--compact", action="store_true",
                    help="three columns only: files, usable, model failure rate")
    ap.add_argument("-o", "--output", metavar="FILE",
                    help="write the markdown here instead of stdout")
    ap.add_argument("--json", metavar="FILE",
                    help="also dump the raw per-file verdicts here")
    ap.add_argument("--from-json", metavar="FILE", action="append",
                    help="re-render from a --json dump instead of re-running every "
                         "candidate. The measurement pass takes minutes; changing how "
                         "the table looks should not. Repeatable: several dumps merge "
                         "into one table, in the order given, first occurrence of a "
                         "model name winning.")
    args = ap.parse_args()

    out_lines = []
    def emit(s=""):
        out_lines.append(s)

    if args.from_json:
        rows, details, skipped, seen = [], [], set(), set()
        for path in args.from_json:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            for r in cached["rows"]:
                # Merging dumps that overlap would double-count a model into two rows
                # of the same table, so the first one wins and the rest are dropped.
                if r[0] in seen:
                    print(f"  ...{r[0]} already present; ignoring the copy in {path}",
                          file=sys.stderr)
                    continue
                seen.add(r[0])
                rows.append(tuple(r))
            details += [tuple(d) for d in cached["details"] if d[0] in seen]
            skipped |= set(cached["skipped"])
        return report(args, emit, out_lines, rows, details, skipped)

    if args.all:
        names = sorted({d for dom in os.listdir(GEN)
                        if os.path.isdir(os.path.join(GEN, dom))
                        for d in os.listdir(os.path.join(GEN, dom))
                        if os.path.isdir(os.path.join(GEN, dom, d))})
    elif args.models:
        names = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        names = ARM_DIRS

    rows, details, skipped = [], [], set()
    for model in names:
        tot = ok = art = bad_c = bad_r = leg = ctor = 0
        for dom in sorted(os.listdir(GEN)):
            d = os.path.join(GEN, dom, model)
            if not os.path.isdir(d):
                continue
            inst = DOMAIN_INSTANCE.get(dom)
            if inst is None:
                skipped.add(dom)
                continue
            sas = os.path.join(ROOT, "htn-benchmarks/sas-files", inst + ".psas")
            if not os.path.exists(sas):
                skipped.add(f"{dom} (missing {inst}.psas)")
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".py"):
                    continue
                tot += 1
                verdict, detail = classify(os.path.join(d, fn), sas,
                                           not args.compile_only)
                if verdict == "ok":
                    ok += 1
                elif verdict == "artifact":
                    art += 1
                elif verdict == "compile":
                    bad_c += 1
                elif verdict == "legacy":
                    leg += 1
                elif verdict == "ctor":
                    ctor += 1
                else:
                    bad_r += 1
                if verdict != "ok":
                    details.append((model, dom, fn, verdict, detail))
        if tot:
            rows.append((model, tot, ok, bad_c, art, bad_r, leg, ctor))
            print(f"  ...{model}: {tot} checked", file=sys.stderr)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "details": details,
                       "skipped": sorted(skipped)}, f, indent=1)
        print(f"wrote {args.json}", file=sys.stderr)

    return report(args, emit, out_lines, rows, details, skipped)


def report(args, emit, out_lines, rows, details, skipped):
    """Render the measured verdicts as markdown."""
    def pct(n, t):
        return f"{n} ({100 * n / t:.1f}%)" if t else "--"

    def failure_rate(tot, bc, art, br, leg, ctor):
        # Denominator is what we can actually judge: exclude our own artifact and the
        # files written against the old interface. A rate over `tot` would be a
        # statement about our harness dressed up as one about the model.
        meas = tot - art - leg
        return meas, (f"**{100 * (bc + br + ctor) / meas:.1f}%**" if meas
                      else "n/a (none measurable)")

    emit()
    narrowed = []
    if args.compact:
        emit("| model / arm | files | usable | model failure rate |")
        emit("|---|---|---|---|")
        for model, tot, ok, bc, art, br, leg, ctor in rows:
            meas, rate = failure_rate(tot, bc, art, br, leg, ctor)
            emit(f"| {model} | {tot} | {ok} | {rate} |")
            if meas != tot:
                # The compact table has no column for the denominator, and a rate over
                # a fraction of `files` must not be left looking like one over all of
                # them -- that is the difference between "GPT5 fails 20% of the time"
                # and "GPT5 fails 20% of the 16% of its files we can judge".
                narrowed.append(f"{model} {meas} of {tot}")
    else:
        emit("| model / arm | files | measurable | usable | model syntax error "
              "| our envelope artifact | legacy API (not measurable) | needs ctor arg "
              "| runtime crash | model failure rate |")
        emit("|---|---|---|---|---|---|---|---|---|---|")
        for model, tot, ok, bc, art, br, leg, ctor in rows:
            meas, rate = failure_rate(tot, bc, art, br, leg, ctor)
            emit(f"| {model} | {tot} | {meas} | {ok} | {pct(bc, tot)} | {pct(art, tot)} "
                  f"| {pct(leg, tot)} | {pct(ctor, tot)} | {pct(br, tot)} | {rate} |")

    # Explain a column only when it fired. On a corpus with no artifacts and no legacy
    # files -- which is every post-repair arm -- these two paragraphs are just noise
    # around three zeros.
    any_art = any(r[4] for r in rows)
    any_leg = any(r[6] for r in rows)

    emit()
    if args.compact and not (any_art or any_leg):
        # The compact table has no measurable column, so do not send the reader
        # looking for one: here the denominator really is `files`.
        emit("**Model failure rate = (syntax error + missing constructor default + "
             "runtime crash) / files.** Nothing is excluded: no file in this table was "
             "an artifact of our own generator, and none targets the pre-`initialize()` "
             "Heuristic API.")
    elif args.compact:
        emit("**Model failure rate = (syntax error + missing constructor default + "
             "runtime crash) / measurable**, where *measurable* excludes files that are "
             "our generator's artifact or target the pre-`initialize()` Heuristic API — "
             "neither is the model's doing. Run without `--compact` for the breakdown.")
        emit()
        emit("Where the denominator is NOT `files`, it is: " + "; ".join(narrowed) +
             ". A rate over a small denominator is a weak measurement, not a good "
             "score — read those alongside the count they came from.")
    else:
        emit("**Model failure rate = (syntax + ctor-arg + runtime) / measurable.** Two columns are "
              "excluded from the denominator because neither is the model's doing:")
        emit()
    if any_art and not args.compact:
        emit("* *our envelope artifact* — complete valid code plus a trailing `\"` or `}` "
              "from the JSON response envelope. A defect in our generator, fixed later; the "
              "files on disk are deliberately never rewritten, so it is repaired here at "
              "measurement time.")
    if any_leg and not args.compact:
        emit("* *legacy API* — written against the pre-`initialize()` Heuristic contract, "
              "where the model was passed to `__init__`. `--heuristic-file` constructs with "
              "no arguments and cannot load these at all. They may well have worked when "
              "generated; this harness cannot tell, so it does not guess.")
    if not (any_art or any_leg or args.compact):
        emit("* Both are zero throughout this table, so nothing is excluded here and the "
              "rate is over every file. The columns are kept because other corpora do "
              "populate them.")
    # Arm directories are named "<Model>-<arm>", so a merged table can put a model and
    # its own ablation arms side by side. That pair is the most misreadable thing here:
    # same model, rates far apart, and the difference is the population, not the model.
    names = [r[0] for r in rows]
    pairs = [(base, n) for base in names for n in names
             if n != base and n.startswith(base + "-")]
    if pairs:
        emit()
        emit("**" + ", ".join(sorted({b for b, _ in pairs})) + "** appears both on its "
             "own and as the base of an ablation-arm row (" +
             ", ".join(f"`{n}`" for _, n in sorted(pairs)) + "). Those are the same "
             "model over different file sets, so the gap between them says nothing "
             "about the model.")

    if len(rows) > 1:
        emit()
        # Rows sit in one table, so they read as comparable. Across model directories
        # they often are not: a raw batch of 20 per domain and a directory holding the
        # five heuristics that were kept are different populations, and the second will
        # score better for reasons that have nothing to do with the model.
        emit("Rows are comparable only where the directories hold comparable "
             "populations. A directory of raw batch generations and one holding "
             "heuristics that were kept, refined or hand-picked are not the same "
             "measurement, and the curated one will score better for reasons that are "
             "not about the model. Check the file counts before comparing rates.")

    emit()
    # Two different populations, two different numbers, one obvious misreading.
    emit("*usable* is not the candidate pool behind a coverage table. Here it means the "
         "heuristic reached GOAL on ONE designated small instance of its domain under "
         "`Astar()` with `AstarNode(G=0,H=1)`; there it means the candidate solved at "
         "least one instance of the evaluation set. A heuristic can pass one and fail "
         "the other, so the two counts differ by a candidate or two per arm.")
    if skipped:
        emit()
        emit(f"Skipped (no small instance mapped): {', '.join(sorted(skipped))}.")
    if args.summary_only:
        # The rates are only auditable against the per-file list, so say where it is
        # rather than letting a summary circulate as if nothing was dropped.
        emit()
        emit(f"Per-file failure list omitted (`--summary-only`); {len(details)} entries. "
             "Re-run without the flag to see which file failed how.")
    else:
        emit()
        emit("Failures:")
        emit()
        emit("| model / arm | domain | file | kind | detail |")
        emit("|---|---|---|---|---|")
        for model, dom, fn, verdict, detail in details:
            emit(f"| {model} | {dom} | `{fn}` | {verdict} | `{detail}` |")

    doc = "\n".join(out_lines)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(doc + "\n")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
