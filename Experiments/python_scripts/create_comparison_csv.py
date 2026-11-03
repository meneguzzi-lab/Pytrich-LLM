#!/usr/bin/env python3
"""
Create comparison CSV with best Panda and best LLM results per domain/problem.
Best = minimum expanded_nodes OR minimum solution_size (include both if different).
"""

import csv
import os
from collections import defaultdict

def load_csv(filepath):
    """Load CSV file and return list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def get_best_results(rows):
    """
    Find best results by expanded_nodes and solution_size.
    Returns list of 1-2 rows (deduplicated if same).
    """
    if not rows:
        return []

    # Filter rows with valid data
    valid_rows = []
    for r in rows:
        try:
            nodes = int(float(r.get('expanded_nodes', 0) or 0))
            size = int(float(r.get('solution_size', 0) or 0))
            if nodes > 0 or size > 0:  # Has some valid result
                valid_rows.append(r)
        except (ValueError, TypeError):
            continue

    if not valid_rows:
        return []

    # Find best by expanded_nodes
    best_nodes = min(valid_rows, key=lambda r: int(float(r.get('expanded_nodes', float('inf')) or float('inf'))))

    # Find best by solution_size
    best_solution = min(valid_rows, key=lambda r: int(float(r.get('solution_size', float('inf')) or float('inf'))))

    # Check if they're the same row
    if (best_nodes.get('expanded_nodes') == best_solution.get('expanded_nodes') and
        best_nodes.get('solution_size') == best_solution.get('solution_size') and
        best_nodes.get('heuristic_name') == best_solution.get('heuristic_name')):
        return [best_nodes]
    else:
        return [best_nodes, best_solution]

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create comparison CSV with best Panda and best LLM results')
    parser.add_argument('-i', '--input', default=None,
                       help='Single input CSV containing all results (Panda + LLM). '
                            'If not provided, reads separate panda_results.csv and llm_results.csv')
    parser.add_argument('-o', '--output-dir', default=None,
                       help='Output directory for comparison CSVs (default: same dir as input)')
    args = parser.parse_args()

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv_dir = os.path.join(script_dir, '..', 'outputs', 'csv-files')

    if args.input:
        # Single input file mode: split Panda vs LLM from one CSV
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
        print(f"Loading all results from: {args.input}")
        all_results = load_csv(args.input)
        print(f"  Found {len(all_results)} total results")

        panda_results = [r for r in all_results
                         if r.get('experiment_name', '').startswith('Panda')]
        llm_filtered = [r for r in all_results
                        if r.get('experiment_name', '').startswith(('Gemini', 'ChatGPT', 'Claude', 'GPT'))
                        and not r.get('experiment_name', '').startswith(('Panda', 'Pytrich'))]
    else:
        # Legacy mode: separate files
        out_dir = args.output_dir or default_csv_dir
        panda_csv = os.path.join(default_csv_dir, 'panda_results.csv')
        llm_csv = os.path.join(default_csv_dir, 'llm_results.csv')
        print(f"Loading Panda results from: {panda_csv}")
        panda_results = load_csv(panda_csv)
        print(f"Loading LLM results from: {llm_csv}")
        all_results = load_csv(llm_csv)
        llm_filtered = [r for r in all_results
                        if r.get('experiment_name', '').startswith(('Gemini', 'ChatGPT', 'Claude', 'GPT'))
                        and not r.get('experiment_name', '').startswith(('Panda', 'Pytrich'))]

    output_csv = os.path.join(out_dir, 'comparison_panda_llms.csv')
    llm_wins_csv = os.path.join(out_dir, 'llm_beats_panda.csv')

    print(f"  Found {len(panda_results)} Panda results")
    print(f"  Found {len(llm_filtered)} LLM results")

    # Identify distinct Panda variants (e.g., Panda-rc2ff, Panda-rc2lmc)
    def _panda_variant(experiment_name):
        """Extract Panda variant prefix (e.g., 'Panda-rc2ff') from experiment name."""
        return experiment_name.split('_')[0]

    panda_variants = sorted(set(_panda_variant(r['experiment_name']) for r in panda_results))
    if not panda_variants:
        panda_variants = ['Panda']
    print(f"  Panda variants: {panda_variants}")

    # Group Panda by variant AND domain/problem
    panda_by_variant_dp = {}
    for variant in panda_variants:
        panda_by_variant_dp[variant] = defaultdict(list)

    for r in panda_results:
        variant = _panda_variant(r['experiment_name'])
        key = (r['domain_name'], r['problem_name'])
        panda_by_variant_dp[variant][key].append(r)

    llm_by_dp = defaultdict(list)
    for r in llm_filtered:
        key = (r['domain_name'], r['problem_name'])
        llm_by_dp[key].append(r)

    # Get all domain/problem combinations
    all_panda_keys = set()
    for variant in panda_variants:
        all_panda_keys |= set(panda_by_variant_dp[variant].keys())
    all_keys = all_panda_keys | set(llm_by_dp.keys())
    print(f"\nProcessing {len(all_keys)} domain/problem combinations...")

    # Build comparison results
    comparison_rows = []
    llm_wins = []  # Track cases where LLM beats Panda

    for domain, problem in sorted(all_keys):
        # Get best results per Panda variant separately
        for variant in panda_variants:
            panda_rows = panda_by_variant_dp[variant].get((domain, problem), [])
            best_panda_variant = get_best_results(panda_rows)
            best_panda_variant = [dict(r) for r in best_panda_variant]
            comparison_rows.extend(best_panda_variant)

        # Get best LLM results
        llm_rows = llm_by_dp.get((domain, problem), [])
        best_llm = get_best_results(llm_rows)
        best_llm = [dict(r) for r in best_llm]
        comparison_rows.extend(best_llm)

        # Compare LLM vs each Panda variant for win/loss analysis
        if best_llm:
            for variant in panda_variants:
                panda_rows = panda_by_variant_dp[variant].get((domain, problem), [])
                best_panda = get_best_results(panda_rows)
                if not best_panda:
                    continue

                try:
                    panda_nodes = min(int(float(r.get('expanded_nodes', float('inf')) or float('inf')))
                                      for r in best_panda)
                    llm_nodes = min(int(float(r.get('expanded_nodes', float('inf')) or float('inf')))
                                    for r in best_llm)
                    panda_size = min(int(float(r.get('solution_size', float('inf')) or float('inf')))
                                     for r in best_panda)
                    llm_size = min(int(float(r.get('solution_size', float('inf')) or float('inf')))
                                   for r in best_llm)

                    # LLM wins on nodes
                    if llm_nodes < panda_nodes:
                        llm_wins.append({
                            'domain_name': domain,
                            'problem_name': problem,
                            'panda_variant': variant,
                            'metric': 'expanded_nodes',
                            'llm_value': llm_nodes,
                            'panda_value': panda_nodes,
                            'improvement': panda_nodes - llm_nodes,
                            'improvement_pct': round((panda_nodes - llm_nodes) / panda_nodes * 100, 2)
                        })

                    # LLM wins on solution size
                    if llm_size < panda_size:
                        llm_wins.append({
                            'domain_name': domain,
                            'problem_name': problem,
                            'panda_variant': variant,
                            'metric': 'solution_size',
                            'llm_value': llm_size,
                            'panda_value': panda_size,
                            'improvement': panda_size - llm_size,
                            'improvement_pct': round((panda_size - llm_size) / panda_size * 100, 2)
                        })
                except (ValueError, TypeError):
                    pass

    # Write comparison CSV
    headers = ['domain_name', 'problem_name', 'experiment_name', 'heuristic_name',
               'expanded_nodes', 'solution_size', 'search_elapsed_time']

    print(f"\nWriting {len(comparison_rows)} rows to: {output_csv}")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(comparison_rows)

    # Write LLM wins CSV
    if llm_wins:
        wins_headers = ['domain_name', 'problem_name', 'panda_variant', 'metric', 'llm_value',
                        'panda_value', 'improvement', 'improvement_pct']
        print(f"Writing {len(llm_wins)} LLM wins to: {llm_wins_csv}")

        with open(llm_wins_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=wins_headers)
            writer.writeheader()
            writer.writerows(llm_wins)

    # Summary
    panda_count = sum(1 for r in comparison_rows if r.get('experiment_name', '').startswith('Panda'))
    llm_count = len(comparison_rows) - panda_count
    print(f"\nSummary:")
    print(f"  Panda rows: {panda_count}")
    print(f"  LLM rows: {llm_count}")
    print(f"  Total: {len(comparison_rows)}")

    # Win/loss analysis per variant
    for variant in panda_variants:
        variant_wins = [w for w in llm_wins if w.get('panda_variant') == variant]
        nodes_wins = sum(1 for w in variant_wins if w['metric'] == 'expanded_nodes')
        size_wins = sum(1 for w in variant_wins if w['metric'] == 'solution_size')
        print(f"\nLLM vs {variant}:")
        print(f"  LLM wins on expanded_nodes: {nodes_wins}")
        print(f"  LLM wins on solution_size: {size_wins}")

    # Show domains
    domains = sorted(set(r['domain_name'] for r in comparison_rows))
    print(f"\nDomains: {domains}")

if __name__ == '__main__':
    main()
