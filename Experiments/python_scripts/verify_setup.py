#!/usr/bin/env python3
"""
Does this checkout actually work? Run this after following README -> Setup.

Every check here corresponds to one documented setup step, and each failure
prints the command that fixes it.

    python3 Experiments/python_scripts/verify_setup.py           # everything
    python3 Experiments/python_scripts/verify_setup.py --quick   # skip the two subprocess runs
    python3 Experiments/python_scripts/verify_setup.py --planner-only   # skip LLM-generation checks

Uses only the standard library, deliberately: it has to run *before* setup is
complete in order to say what is missing. Exit status is 1 if any FAIL was
printed, so it can gate a CI step or an HPC submission script.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (import name, pip name, what breaks without it)
CORE_PACKAGES = [
    ("yaml", "pyyaml", "config/*.yaml parsing (planner, task lists, generator)"),
    ("psutil", "psutil", "imported by every search module at load time"),
]
GENERATION_PACKAGES = [
    ("requests", "requests", "LLM API calls"),
    ("dotenv", "python-dotenv", "reading .env for API keys -- hard import in generate_heuristic.py"),
]
OPTIONAL_PACKAGES = [
    ("matplotlib", "matplotlib", "PDF ablation tables; --markdown works without it"),
]

# panda_ground.py calls the parser and grounder by path from PandaBuilds/; the job
# scripts call the engine. All three are built by PandaBuilds/build_panda.sh.
PANDA_BINARIES = ["pandaPIparser", "pandaPIgrounder", "pandaPIengine"]

# provider -> env var, mirroring config/models.yaml:env_vars
API_KEY_ENVS = {"google": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY"}

SMOKE_DOMAIN = REPO / "htn-benchmarks/total-order/Blocksworld-GTOHP/domain.hddl"
SMOKE_PROBLEM = REPO / "htn-benchmarks/total-order/Blocksworld-GTOHP/p01.hddl"

failures = 0
warnings = 0


def report(ok, label, detail="", fix=None, warn_only=False):
    """One result line. `ok=False` with warn_only=True counts as a warning."""
    global failures, warnings
    if ok:
        mark = "PASS"
    elif warn_only:
        mark = "WARN"
        warnings += 1
    else:
        mark = "FAIL"
        failures += 1
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok and fix:
        print(f"         fix: {fix}")
    return ok


def executable_kind(path):
    """'elf', 'macho', 'script' or None, sniffed from magic bytes.

    Avoids shelling out to file(1), which is absent on some cluster images.
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
    except OSError:
        return None
    if magic[:4] == b"\x7fELF":
        return "elf"
    if magic[:4] in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe",
                     b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"):
        return "macho"
    if magic[:2] == b"#!":
        return "script"
    return None


def host_kind():
    return {"Linux": "elf", "Darwin": "macho"}.get(os.uname().sysname)


def check_python():
    print("Python")
    v = sys.version_info
    report(v >= (3, 8), "version >= 3.8", f"running {v.major}.{v.minor}.{v.micro}",
           fix="load a newer python module on HPC, or recreate the venv with `uv venv --python 3.12`")
    # 31 generated candidates (all GPT52, incl. 15 in Towers) annotate `Model | None`,
    # which is a SyntaxError before 3.10 -- they fail to import, not to run.
    report(v >= (3, 10), "version >= 3.10", "needed by 31 GPT52 candidates using `X | None`",
           fix="only matters if you evaluate GPT52 heuristics; 3.8 is fine otherwise",
           warn_only=True)


def check_packages(include_generation):
    print("\nPython packages")
    groups = list(CORE_PACKAGES)
    if include_generation:
        groups += GENERATION_PACKAGES
    for mod, pip_name, why in groups:
        found = importlib.util.find_spec(mod) is not None
        report(found, f"{pip_name}", why, fix=f"uv pip install {pip_name}")
    for mod, pip_name, why in OPTIONAL_PACKAGES:
        found = importlib.util.find_spec(mod) is not None
        report(found, f"{pip_name} (optional)", why,
               fix=f"uv pip install {pip_name}", warn_only=True)


def check_panda():
    print("\nPANDA tools (PandaBuilds/)")
    want = host_kind()
    for name in PANDA_BINARIES:
        path = REPO / "PandaBuilds" / name
        if not report(path.is_file(), name, "present",
                      fix="bash PandaBuilds/build_panda.sh"):
            continue
        report(os.access(path, os.X_OK), f"{name} executable", "",
               fix=f"chmod +x PandaBuilds/{name}")
        kind = executable_kind(path)
        if want is None:
            report(True, f"{name} format", f"{kind}; unknown host, not checked", warn_only=True)
        else:
            report(kind == want, f"{name} format", f"{kind}, host wants {want}",
                   fix="rebuild on this machine: bash PandaBuilds/build_panda.sh")


def check_benchmarks():
    print("\nBenchmarks")
    root = REPO / "htn-benchmarks/total-order"
    ok = report(SMOKE_DOMAIN.is_file(), "htn-benchmarks/total-order populated",
                f"{len(list(root.glob('*/domain.hddl')))} domains" if root.is_dir() else "missing",
                fix="these files come with the clone -- if they are missing, the checkout is "
                    "partial: run `git status` and `git checkout -- htn-benchmarks`")
    sas = REPO / "htn-benchmarks/sas-files"
    n_sas = len(list(sas.glob("*.psas"))) if sas.is_dir() else 0
    report(n_sas >= 180, "htn-benchmarks/sas-files/*.psas", f"{n_sas} of 180 grounded instances",
           fix="bash htn-benchmarks/ground_benchmarks.sh   (needed for the JobArrays runs)",
           warn_only=True)
    return ok


def check_api_keys():
    print("\nLLM API keys (generation only; never needed to evaluate)")
    env_file = REPO / ".env"
    report(env_file.is_file(), ".env present", "read automatically by generate_heuristic.py",
           fix="create .env with GEMINI_API_KEY=... (it is gitignored), or export the var",
           warn_only=True)
    # Load .env the same way the generator does, without depending on python-dotenv
    # being installed yet, and without ever printing a key.
    from_env_file = {}
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                from_env_file[k.strip()] = v.strip()
    any_key = False
    for provider, var in API_KEY_ENVS.items():
        where = "environment" if os.environ.get(var) else (".env" if from_env_file.get(var) else None)
        any_key = any_key or where is not None
        report(where is not None, f"{var} ({provider})", f"found in {where}" if where else "not set",
               fix=f"export {var}=... or add it to .env -- only needed to generate", warn_only=True)
    if not any_key:
        print("         note: no key found. Evaluating existing heuristics does not need one.")
    else:
        print("         note: this only checks a key is *present*, not that it works.")


def run(cmd, timeout):
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    except OSError as e:
        return None, str(e)


def check_planner_runs():
    print("\nEnd-to-end: planner")
    if not SMOKE_DOMAIN.is_file():
        report(False, "Blocksworld p01 solves", "benchmarks missing, cannot run",
               fix="fix the benchmark check above first")
        return
    # Grounding drops <Domain>-<problem>.psas in the working directory and the plan
    # goes to Experiments/outputs/plan-files/ by default. A verification run should
    # leave no trace, so send the plan to a temp path and remove the .psas afterwards
    # unless it was already there.
    sas = REPO / f"{SMOKE_DOMAIN.parent.name}-{SMOKE_PROBLEM.stem}.psas"
    sas_preexisting = sas.exists()
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run([sys.executable, "__main__.py", str(SMOKE_DOMAIN), str(SMOKE_PROBLEM),
                         "-S", "Astar()", "-H", "TDG()",
                         "-o", str(Path(tmp) / "verify.plan")], timeout=180)
    if sas.exists() and not sas_preexisting:
        sas.unlink()
    report(code == 0 and "Search Status : GOAL" in out, "Blocksworld p01 solves with TDG",
           "grounding + search + plan output" if code == 0 else f"exit {code}",
           fix="run the command by hand to see the error: "
               f"python3 __main__.py {SMOKE_DOMAIN.relative_to(REPO)} "
               f"{SMOKE_PROBLEM.relative_to(REPO)} -S \"Astar()\" -H \"TDG()\"")


def check_generator_runs():
    print("\nEnd-to-end: heuristic generator (no API call, no cost)")
    code, out = run([sys.executable, "generate_heuristic.py", "--domain", "towers",
                     "--model", "gemini3flash", "--dry-run"], timeout=120)
    ok = code == 0 and "DRY RUN" in out
    report(ok, "prompt assembles for towers/gemini3flash", "--dry-run builds the full prompt",
           fix="run it by hand: python3 generate_heuristic.py --domain towers "
               "--model gemini3flash --dry-run")
    # A silently truncated prompt is worse than a crash: it generates, it costs
    # money, and the arm it produces is not the arm you asked for.
    if ok:
        size = next((l for l in out.splitlines() if "Prompt length" in l), "")
        report("[ERROR:" not in out, "prompt contains no [ERROR: ...] placeholders", size.strip(),
               fix="a missing file is being pasted into the prompt as an error string; "
                   "check config/domains.yaml paths")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="skip the two subprocess runs (planner, generator)")
    ap.add_argument("--planner-only", action="store_true",
                    help="skip everything only needed to generate heuristics with an LLM")
    args = ap.parse_args()

    print(f"Verifying Pytrich setup in {REPO}\n")
    check_python()
    check_packages(include_generation=not args.planner_only)
    check_panda()
    check_benchmarks()
    if not args.planner_only:
        check_api_keys()
    if not args.quick:
        check_planner_runs()
        if not args.planner_only:
            check_generator_runs()

    print(f"\n{failures} failed, {warnings} warnings")
    if failures:
        print("Setup is incomplete -- fix the FAIL lines above, then re-run.")
    else:
        print("Setup looks usable." + (" Warnings are optional pieces." if warnings else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
