import argparse
import importlib.util
import inspect
import logging
import os
import sys

from Pytrich.DESCRIPTIONS import Descriptions
from Pytrich.planner import search_plan, SEARCHES, NODES
from Pytrich.constants import HEURISTICS
from Pytrich.Heuristics.heuristic import Heuristic
import Pytrich.FLAGS as FLAGS
from Pytrich.tools import parse_argument_string, parse_aggregation_function


def load_heuristic_from_file(file_path):
    """
    Dynamically import a heuristic class from a .py file path.

    Finds the first class in the module that is a subclass of Heuristic
    (but not Heuristic itself), instantiates it, and returns it.

    Args:
        file_path: Path to a .py file containing a Heuristic subclass.

    Returns:
        An instantiated heuristic object.

    Raises:
        ValueError: If no Heuristic subclass is found in the file.
        ImportError: If the file cannot be imported.
    """
    file_path = os.path.abspath(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot create module spec from '{file_path}'")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the first Heuristic subclass
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Heuristic) and obj is not Heuristic:
            return obj()

    raise ValueError(f"No Heuristic subclass found in '{file_path}'")

def main():
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument(
        "domain", nargs="?", 
        help="Path to the domain file (required if --sas_file is not provided)."
    )
    argparser.add_argument(
        "problem", nargs="?", 
        help="Path to the problem file (required if --sas_file is not provided)."
    )
    argparser.add_argument(
        "-en", "--experimentName",
        help="Indicate the experiment name (Optional)"
    )
    argparser.add_argument(
        "-dn", "--domainName",
        help="Indicate the domain name (Optional)"
    )
    argparser.add_argument(
        "-pn", "--problemName",
        help="Indicate the problem name (Optional)"
    )
    argparser.add_argument(
        "--sas_file", 
        help="Path to the SASplus file if the problem is already grounded."
    )
    argparser.add_argument(
        "-tor", "--totalorderreachability", 
        action="store_true",
        help="Use total-order reachability analysis during grounding post-processing"
    )
    argparser.add_argument(
        "-mg", "--monitorgrounder", 
        action="store_true",
        help="If set, enables monitoring grounding"
    )
    argparser.add_argument(
        "-ms", "--monitorsearch", 
        action="store_true",
        help="If set, enables monitoring of resources (time and memory) during searching"
    )

    argparser.add_argument(
        "-ml", "--monitorlandmarks", 
        action="store_true",
        help="If set, enables monitoring time during landmark generation"
    )

    argparser.add_argument(
        "-H", "--heuristic", default="TDG()",
        type=str,
        help='Specify heuristic in format "heuristic_name(param1=value1,param2=value2)"'
    )
    argparser.add_argument(
        "--heuristic-file",
        type=str,
        help="Path to a .py file containing a Heuristic subclass. "
             "Dynamically loads the heuristic, bypassing constants.py registration. "
             "Cannot be used together with -H or -A."
    )
    argparser.add_argument(
        "-A", "--aggregation",
        type=str,
        help='Specify aggregation function in format "<agg_function>([<heuristic>|<agg_function>])"'
    )
    argparser.add_argument(
        "-S", "--search", default="Astar()",
        type=str,
        help='Specify search algorithm in format "search_name(param1=value1,param2=value2) '
            '*heuristic is not a parameter of -S, use -H to pass a heuristic with their own parameters instead*"'
    )
    argparser.add_argument(
        "-N", "--node", default="AstarNode()",
        type=str,
        help='Specify node type in format "node_type(param1=value1,param2=value2)'
    )
    argparser.add_argument(
        "-o", "--output",
        type=str,
        help="Output path for the plan file. Defaults to Experiments/outputs/plan-files/<auto-name>.plan"
    )

    # Parse the arguments
    args = argparser.parse_args()
    desc = Descriptions()
    if args.experimentName:
        print(f"{desc('experiment_name', args.experimentName)}")
    
    # Print as a workaround since this does not show in the Configurations of the log file
    # Maybe it's a bug
    print(f"{desc('domain_name', args.domainName)}")
    print(f"{desc('problem_name', args.problemName)}")

    # Check for exclusive usage of sas_file or domain and problem
    if args.sas_file:
        if args.domain or args.problem:
            argparser.error("Cannot specify both --sas_file and domain/problem. Choose one.")
        if not os.path.isfile(args.sas_file):
            argparser.error(f"SAS file '{args.sas_file}' does not exist.")
    else:
        if not args.domain or not args.problem:
            argparser.error("Both domain and problem files must be specified if --sas_file is not provided.")
        if not os.path.isfile(args.domain):
            argparser.error(f"Domain file '{args.domain}' does not exist.")
        if not os.path.isfile(args.problem):
            argparser.error(f"Problem file '{args.problem}' does not exist.")

    # Assign flags
    FLAGS.MONITOR_SEARCH_RESOURCES = args.monitorsearch
    FLAGS.MONITOR_LM_TIME = args.monitorlandmarks
    FLAGS.USE_TO_REACHABILITY = args.totalorderreachability

    # Extract domain and problem names if provided
    domain_name = os.path.basename(os.path.dirname(args.domain)) if args.domain else None
    problem_name = os.path.splitext(os.path.basename(args.problem))[0] if args.problem else None
    # Validate mutual exclusivity of heuristic options
    if args.heuristic_file and (args.aggregation or args.heuristic != "TDG()"):
        argparser.error("--heuristic-file cannot be used together with -H or -A.")

    try:
        heuristic_function = None
        heuristic_name = None
        parameters = None
        if args.heuristic_file:
            if not os.path.isfile(args.heuristic_file):
                argparser.error(f"Heuristic file '{args.heuristic_file}' does not exist.")
            heuristic_function = load_heuristic_from_file(args.heuristic_file)
            heuristic_name = type(heuristic_function).__name__
            parameters = {}
        elif args.aggregation:
            heuristic_name, parameters = parse_argument_string(args.aggregation)
            heuristic_function = parse_aggregation_function(heuristic_name, parameters)
        else:
            heuristic_name, parameters = parse_argument_string(args.heuristic)
            heuristic_function = HEURISTICS[heuristic_name](**parameters)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    try:
        search_name, search_params = parse_argument_string(args.search)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        node_name, node_params = parse_argument_string(args.node)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Configuration:")
    print(f"  Domain: {domain_name}")
    print(f"  Problem: {problem_name}")
    print(f"  Search: {search_name}, Params: {search_params}")
    print(f"  Node: {node_name}, Params: {node_params}")
    print(f"  Heuristic: {heuristic_name}, Params: {parameters}")
    print()

    # Run the search plan
    result = search_plan(
        args.domain, args.problem, args.sas_file,
        heuristic_function,
        SEARCHES[search_name], NODES[node_name],
        search_params, node_params
    )
    if result:
        # Build filename from available identifiers
        parts = [v for v in [args.experimentName, args.domainName, args.problemName, heuristic_name] if v]
        plan_filename = "_".join(parts) + ".plan" if parts else "plan.plan"

        # Determine output path
        if args.output:
            plan_path = args.output
        else:
            plan_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Experiments", "outputs", "plan-files")
            os.makedirs(plan_dir, exist_ok=True)
            plan_path = os.path.join(plan_dir, plan_filename)

        # Write plan file
        os.makedirs(os.path.dirname(os.path.abspath(plan_path)), exist_ok=True)
        with open(plan_path, 'w') as f:
            for op in result:
                f.write(f"({op.name})\n")

        # Print plan to stdout + file path
        print("Plan:")
        for op in result:
            print(f"({op.name})")
        print(f"{desc('plan_file', plan_path)}")
    else:
        print("No plan found.")

if __name__ == "__main__":
    main()
