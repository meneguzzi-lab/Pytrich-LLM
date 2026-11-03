#!/usr/bin/env python3
"""
Generate task list for SLURM job arrays.
Each line represents one experiment to run.
"""

import argparse
import glob
import importlib.util
import inspect
import os
import re
import sys
import yaml


# Domain name mapping from script names to filesystem paths
DOMAIN_FILESYSTEM_MAP = {
    "Barman-BDI": "Barman_BDI",
    "Blocksworld-GTOHP": "Blocksworld_GTOHP",
    "Depots": "Depots",
    "Lamps": "Lamps",
    "Robot": "Robot",
    "Rover-GTOHP": "Rover_GTOHP",
    "Towers": "Towers",
}

# Node configurations for different algorithms
NODE_CONFIGS = {
    "StandardA": "AstarNode(G=1,H=1)",
    "GBFS": "AstarNode(G=0,H=1)",
    "WA": "AstarNode(G=1,H=5)",
}


def validate_heuristic(filepath):
    """
    Quick validation of a heuristic file.
    Returns (is_valid, reason) tuple.
    Checks: importable, has Heuristic subclass, __init__ takes no required args,
    __call__ has correct signature (self, parent_node, node).
    """
    # Ensure project root is on sys.path so heuristic imports (e.g., from Pytrich.Heuristics...)
    # can resolve. This is needed because generate_task_list.py runs from JobArrays/, not the root.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Step 1: Import the module
    try:
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            return False, "cannot create module spec"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        return False, f"import error: {type(e).__name__}: {e}"

    # Step 2: Find a Heuristic subclass
    heuristic_cls = None
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Check if it's a subclass of something named "Heuristic" but not "Heuristic" itself
        for base in inspect.getmro(obj):
            if base.__name__ == "Heuristic" and base is not obj:
                heuristic_cls = obj
                break
        if heuristic_cls:
            break

    if heuristic_cls is None:
        return False, "no Heuristic subclass found"

    # Step 3: Check __init__ signature — no required positional args beyond self
    try:
        init_sig = inspect.signature(heuristic_cls.__init__)
        for param_name, param in init_sig.parameters.items():
            if param_name == "self":
                continue
            if param.default is inspect.Parameter.empty and param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                return False, f"__init__ has required arg '{param_name}'"
    except (ValueError, TypeError) as e:
        return False, f"cannot inspect __init__: {e}"

    # Step 4: Check __call__ signature — should be (self, parent_node, node)
    try:
        call_sig = inspect.signature(heuristic_cls.__call__)
        call_params = list(call_sig.parameters.keys())
        if len(call_params) != 3 or call_params[0] != "self":
            return False, f"__call__ signature mismatch: expected (self, parent_node, node), got ({', '.join(call_params)})"
    except (ValueError, TypeError) as e:
        return False, f"cannot inspect __call__: {e}"

    return True, "ok"


def load_config(config_path):
    """Load planners configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_all_planners(config):
    """Get all planners (LLM + builtin) from config."""
    llm_planners = config.get('llm_planners', [])
    builtin_planners = config.get('builtin_planners', [])
    return llm_planners + builtin_planners


def get_llm_planners(config):
    """Get LLM planners from config."""
    return config.get('llm_planners', [])


def get_file_mode_planners(config):
    """LLM planners whose candidates are run by file path rather than registry key."""
    return config.get('file_mode_planners', [])


PROBLEM_EXTENSIONS = ('.hddl', '.pddl')


def strip_problem_extension(instance):
    """pfile01.hddl / pfile01.pddl -> pfile01; a name without either is returned as is."""
    for ext in PROBLEM_EXTENSIONS:
        if instance.endswith(ext):
            return instance[:-len(ext)]
    return instance


def get_instances(domain):
    """
    Auto-detect problem instances for a domain by scanning the benchmark folder.
    Returns sorted list of instance filenames (e.g., ['pfile01.hddl', 'pfile02.hddl', ...]).
    Problem files are .hddl in most domains and .pddl in some (Lamps, from IPC 2023).
    """
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Path to domain folder
    domain_path = os.path.join(project_root, "htn-benchmarks", "total-order", domain)

    if not os.path.exists(domain_path):
        print(f"Warning: Domain folder not found: {domain_path}", file=sys.stderr)
        return []

    # Find all problem files except domain.hddl
    instances = []
    for f in os.listdir(domain_path):
        if f.endswith(PROBLEM_EXTENSIONS) and f != 'domain.hddl':
            instances.append(f)

    # Sort naturally (pfile01, pfile02, ..., pfile10, pfile11, ...)
    instances.sort(key=lambda x: (len(x), x))

    return instances


def get_heuristics_llm(domain, planner, validate=True, file_mode=False):
    """
    Discover LLM heuristics for any planner.
    Returns list of heuristic function names (registry keys), or of file paths when
    file_mode is set or the planner is an ablation arm (<Model>-<arm>).
    When validate=True, each heuristic file is checked for importability and
    correct signatures before inclusion.
    """
    # Get project root for absolute path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Map domain name to filesystem name
    domain_fs = DOMAIN_FILESYSTEM_MAP.get(domain, domain)

    # Path to heuristics directory (absolute path for robustness)
    heuristics_dir = os.path.join(project_root, "Pytrich", "Heuristics", "GeneratedHeuristics", domain_fs, planner)

    if not os.path.exists(heuristics_dir):
        # Not an error - planner may not have heuristics for this domain
        return []

    # Domain prefix for function names (first part before hyphen or underscore, lowercase)
    domain_prefix = domain_fs.split("_")[0].split("-")[0].lower()

    # Try plural form first, then singular
    prefixes_to_try = [domain_prefix]
    if domain_prefix.endswith('s'):
        prefixes_to_try.append(domain_prefix[:-1])  # Remove trailing 's' (e.g., depots -> depot)

    heuristic_files = []
    used_prefix = domain_prefix

    for prefix in prefixes_to_try:
        pattern = os.path.join(heuristics_dir, f"{prefix}_heuristic_*.py")
        found_files = sorted(glob.glob(pattern))
        if found_files:
            heuristic_files = found_files
            used_prefix = prefix
            break

    if not heuristic_files:
        return []

    heuristics = []
    skipped = []
    planner_lower = planner.lower()

    # Ablation arm directories (`<Model>-<arm>`) and file-mode planners have no
    # constants.py entries: the registry key is derived from the model alone, so every
    # arm would collapse onto the same key. Emit the file path instead; the sbatch
    # template then runs it with --heuristic-file.
    if '-' in planner or file_mode:
        for filepath in heuristic_files:
            if validate:
                ok, reason = validate_heuristic(filepath)
                if not ok:
                    skipped.append((os.path.basename(filepath), reason))
                    continue
            heuristics.append(os.path.relpath(filepath, project_root))
        if skipped:
            print(f"  - {len(heuristic_files)} discovered, {len(skipped)} skipped (broken), "
                  f"{len(heuristics)} valid  [arm dir, --heuristic-file mode]")
            for name, reason in skipped:
                print(f"      skipped {name}: {reason}")
        else:
            print(f"  - {len(heuristics)} heuristics, all valid  [arm dir, --heuristic-file mode]")
        return heuristics

    for filepath in heuristic_files:
        filename = os.path.basename(filepath)

        # Validate heuristic if enabled
        if validate:
            is_valid, reason = validate_heuristic(filepath)
            if not is_valid:
                skipped.append((filename, reason))
                continue

        # Handle different naming patterns:
        # 1. Standard pattern: <prefix>_heuristic_<N>.py (Gemini, ChatGPT)
        # 2. Claude pattern: <prefix>_heuristic_claude_v<N>.py

        # Try standard pattern first: _heuristic_<number>.py
        match = re.search(r'_heuristic_(\d+)\.py$', filename)
        if match:
            heuristic_num = match.group(1)
            # Format: <domain_prefix>_<planner_lower>_heuristic_<N>()
            heuristic_name = f"{domain_prefix}_{planner_lower}_heuristic_{heuristic_num}()"
            heuristics.append(heuristic_name)
            continue

        # Try Claude versioned pattern: _heuristic_claude_v<N>.py
        match = re.search(r'_heuristic_claude_v(\d+)\.py$', filename)
        if match:
            heuristic_num = int(match.group(1)) - 1  # v1→0, v2→1, etc.
            heuristic_name = f"{domain_prefix}_{planner_lower}_heuristic_{heuristic_num}()"
            heuristics.append(heuristic_name)
            continue

        # Try generic versioned pattern: _heuristic_<name>_v<N>.py
        match = re.search(r'_heuristic_(\w+)_v(\d+)\.py$', filename)
        if match:
            heuristic_num = int(match.group(2)) - 1  # v1→0, v2→1, etc.
            heuristic_name = f"{domain_prefix}_{planner_lower}_heuristic_{heuristic_num}()"
            heuristics.append(heuristic_name)
            continue

    # Print validation summary
    if validate and skipped:
        total_discovered = len(heuristic_files)
        total_valid = total_discovered - len(skipped)
        print(f"  - {total_discovered} heuristics discovered, {len(skipped)} skipped (broken), {total_valid} valid")
        for filename, reason in skipped:
            print(f"    SKIP {filename}: {reason}")

    return heuristics


def get_heuristics_pytrich():
    """Get list of Pytrich built-in heuristics."""
    return ["TDG()", "NOVELTY()", "LMCOUNT(use_bid=True)"]


def get_heuristics_panda():
    """Get list of Panda built-in heuristics."""
    return ["rc2(ff)", "rc2(lmc)", "rc2(add)", "lmcount(lmtype=bid)"]


def get_sas_file_path(domain, instance, planner=""):
    """
    Get the path to the preprocessed SAS file.
    - Panda: htn-benchmarks/panda-sas-files/<Domain>-<instance>.sas
    - Others: htn-benchmarks/sas-files/<Domain>-<instance>.psas
    """
    instance_base = strip_problem_extension(instance)
    if planner == "Panda":
        return f"htn-benchmarks/panda-sas-files/{domain}-{instance_base}.sas"
    return f"htn-benchmarks/sas-files/{domain}-{instance_base}.psas"


def get_engine_name(planner, heuristic, algorithm):
    """Generate engine name for experiment identification."""
    if planner == "Panda":
        heuristic_clean = heuristic.replace("(", "").replace(")", "").replace("=", "").replace(",", "")
        return f"Panda-{heuristic_clean}_{algorithm}"
    elif planner == "Pytrich":
        heuristic_base = heuristic.split("(")[0]
        return f"Pytrich-{heuristic_base}_{algorithm}"
    else:
        # LLM planners (Gemini20Flash, Gemini25Pro, Claude, etc.)
        return f"{planner}_{algorithm}"


def generate_task_list(domain, planner, algorithm, output_file, llm_planners, validate=True,
                       file_mode_planners=()):
    """
    Generate task list for a specific (domain, planner, algorithm) combination.

    Args:
        domain: Domain name (e.g., "Barman-BDI")
        planner: Planner type (any LLM planner, Pytrich, Panda)
        algorithm: Search algorithm (StandardA, GBFS, WA)
        output_file: Path to output task list file
        llm_planners: List of LLM planner names from config
        validate: If True, validate LLM heuristic files before including them
    """
    # Get node configuration for algorithm
    node_config = NODE_CONFIGS.get(algorithm, "AstarNode(G=1,H=1)")

    # Get instances
    instances = get_instances(domain)

    # Get heuristics based on planner type
    if planner in llm_planners:
        heuristics = get_heuristics_llm(domain, planner, validate=validate,
                                        file_mode=planner in file_mode_planners)
    elif planner == "Pytrich":
        heuristics = get_heuristics_pytrich()
    elif planner == "Panda":
        heuristics = get_heuristics_panda()
    else:
        print(f"Error: Unknown planner type: {planner}", file=sys.stderr)
        return 0

    if not heuristics:
        print(f"Warning: No heuristics found for {domain}, {planner}", file=sys.stderr)
        return 0

    # Generate task list
    tasks = []
    for instance in instances:
        for heuristic in heuristics:
            # For Panda, instance name doesn't have the problem-file extension
            instance_name = instance if planner != "Panda" else strip_problem_extension(instance)

            sas_file = get_sas_file_path(domain, instance, planner)
            engine_name = get_engine_name(planner, heuristic, algorithm)

            # Task format: domain instance planner heuristic algorithm engine_name node_config sas_file
            task_line = f"{domain} {instance_name} {planner} {heuristic} {algorithm} {engine_name} {node_config} {sas_file}"
            tasks.append(task_line)

    # Write task list to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for task in tasks:
            f.write(task + '\n')

    print(f"Generated {len(tasks)} tasks for {domain} {planner} {algorithm}")
    print(f"  - {len(instances)} instances")
    print(f"  - {len(heuristics)} heuristics")
    print(f"  - Output: {output_file}")

    return len(tasks)


def main():
    # Determine default config path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_config = os.path.join(project_root, "config", "planners.yaml")

    parser = argparse.ArgumentParser(description="Generate task list for SLURM job arrays")
    parser.add_argument("--config", default=default_config,
                        help=f"Path to planners config file (default: {default_config})")
    parser.add_argument("--domain", help="Domain name (e.g., Barman-BDI)")
    parser.add_argument("--planner", help="Planner type (from config)")
    parser.add_argument("--algorithm", choices=["StandardA", "GBFS", "WA"],
                        help="Search algorithm")
    parser.add_argument("--output", help="Output task list file")
    parser.add_argument("--list-planners", action="store_true",
                        help="List available planners and exit")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip heuristic file validation (include all discovered files)")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    all_planners = get_all_planners(config)
    llm_planners = get_llm_planners(config)

    if args.list_planners:
        print("Available planners:")
        print(f"  LLM: {', '.join(llm_planners)}")
        print(f"  Built-in: {', '.join(config.get('builtin_planners', []))}")
        sys.exit(0)

    # Check required arguments for task generation
    missing = []
    if not args.domain:
        missing.append("--domain")
    if not args.planner:
        missing.append("--planner")
    if not args.algorithm:
        missing.append("--algorithm")
    if not args.output:
        missing.append("--output")
    if missing:
        print(f"Error: Missing required arguments: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    # Validate planner
    # An ablation arm is `<Model>-<arm>` -- accept it whenever the model half is a
    # known LLM planner, rather than requiring every arm to be listed in the config.
    # No model name contains a hyphen, so splitting on the first one is unambiguous.
    if args.planner not in all_planners:
        base = args.planner.split('-', 1)[0]
        if '-' in args.planner and base in llm_planners:
            llm_planners = llm_planners + [args.planner]
            all_planners = all_planners + [args.planner]
        else:
            print(f"Error: Unknown planner '{args.planner}'", file=sys.stderr)
            print(f"Available planners: {', '.join(all_planners)}", file=sys.stderr)
            print(f"Ablation arms are accepted as <Model>-<arm>, e.g. "
                  f"Gemini3Flash-arm1-no-example", file=sys.stderr)
            sys.exit(1)

    validate = not args.skip_validation
    num_tasks = generate_task_list(args.domain, args.planner, args.algorithm,
                                    args.output, llm_planners, validate=validate,
                                    file_mode_planners=get_file_mode_planners(config))

    if num_tasks == 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
