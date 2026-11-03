#!/usr/bin/env python3
"""Generate comprehensive visualizations for Panda vs LLM comparison.

This script creates suites of plots comparing each baseline against LLM-generated
heuristics. It generates output directories for each baseline variant:
  - img-files/                    Combined (all Panda variants visible)
  - img-files/panda-rc2ff/        PANDA rc2ff vs LLMs
  - img-files/panda-rc2lmc/       PANDA rc2lmc vs LLMs
  - img-files/panda-lmcount/      PANDA lmcount-bid vs LLMs
  - img-files/pytrich-tdg/        Pytrich-TDG vs LLMs
  - img-files/pytrich-lmcount/    Pytrich-LMCOUNT vs LLMs

Usage:
    python generate_all_plots.py
    python generate_all_plots.py -i path/to/results.csv
    python generate_all_plots.py --output-dir custom/path
"""

import argparse
import os


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuration
DEFAULT_OUTPUT_DIR = 'Experiments/outputs/img-files'
DEFAULT_CSV_DIR = 'Experiments/outputs/csv-files'

# Consistent color palette
COLORS = {
    'panda': '#2ecc71',           # Green for Panda (generic/backward compat)
    'panda_rc2ff': '#2ecc71',     # Green for rc2ff
    'panda_rc2lmc': '#16a085',    # Teal for rc2lmc
    'panda_lmcount': '#8e44ad',   # Purple for PANDA lmcount
    'panda_rc2add': '#d35400',    # Orange for rc2add
    'pytrich_tdg': '#d35400',     # Orange for Pytrich-TDG
    'pytrich_lmcount': '#c0392b',  # Red for Pytrich-LMCOUNT
    'llm': '#3498db',             # Blue for LLM
    'win': '#27ae60',             # Green for LLM wins
    'tie': '#f39c12',             # Yellow for ties
    'loss': '#e74c3c',            # Red for LLM losses
}

# Mapping: suite key -> (experiment_name prefix, display label, color key)
BASELINE_CONFIG = {
    'rc2ff': (
        'Panda-rc2ff', 'Panda (rc2ff)', 'panda_rc2ff'),
    'rc2lmc': (
        'Panda-rc2lmc', 'Panda (rc2lmc)', 'panda_rc2lmc'),
    'rc2add': (
        'Panda-rc2add', 'Panda (rc2add)', 'panda_rc2add'),
    'panda-lmcount': (
        'Panda-lmcountlmtypebid', 'Panda (LMCount)', 'panda_lmcount'),
    'pytrich-tdg': (
        'Pytrich-TDG', 'Pytrich (TDG)', 'pytrich_tdg'),
    'pytrich-lmcount': (
        'Pytrich-LMCOUNT', 'Pytrich (LMCount)', 'pytrich_lmcount'),
}

# Font size constants
# TITLE_FONTSIZE = 14
TITLE_FONTSIZE = 20
# AXIS_LABEL_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 9
# ANNOTATION_FONTSIZE = 8
ANNOTATION_FONTSIZE = 16

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data(csv_dir, input_file=None, panda_variant=None):
    """Load CSV files and split into baseline/LLM rows.

    panda_variant controls which rows become the baseline:
      None / "both"        – all rows starting with "Panda"
      "rc2ff"              – Panda-rc2ff_* only
      "rc2lmc"             – Panda-rc2lmc_* only
      "panda-lmcount"      – Panda-lmcountlmtypebid_* only
      "pytrich-tdg"        – Pytrich-TDG_* only
      "pytrich-lmcount"    – Pytrich-LMCOUNT_* only

    LLM rows are always those not starting with "Panda" or "Pytrich".
    """
    if input_file:
        data_dir = os.path.dirname(os.path.abspath(input_file))
        all_data = pd.read_csv(input_file)
        baseline = all_data[
            all_data['experiment_name'].str.startswith('Panda', na=False)
        ]
        llm = all_data[
            ~all_data['experiment_name'].str.startswith(
                ('Panda', 'Pytrich'), na=False)
        ]
        comparison = pd.read_csv(
            os.path.join(data_dir, 'comparison_panda_llms.csv'))
        llm_wins = pd.read_csv(
            os.path.join(data_dir, 'llm_beats_panda.csv'))
    else:
        llm = pd.read_csv(os.path.join(csv_dir, 'llm_results.csv'))
        baseline = pd.read_csv(os.path.join(csv_dir, 'panda_results.csv'))
        comparison = pd.read_csv(
            os.path.join(csv_dir, 'comparison_panda_llms.csv'))
        llm_wins = pd.read_csv(
            os.path.join(csv_dir, 'llm_beats_panda.csv'))

    if not panda_variant or panda_variant == 'both':
        return llm, baseline, comparison, llm_wins

    # Resolve prefix from BASELINE_CONFIG
    if panda_variant in BASELINE_CONFIG:
        prefix = BASELINE_CONFIG[panda_variant][0]
    else:
        prefix = f'Panda-{panda_variant}'

    is_pytrich = panda_variant.startswith('pytrich-')

    if input_file:
        all_data = pd.read_csv(input_file)
        baseline = all_data[
            all_data['experiment_name'].str.startswith(prefix, na=False)
        ]
        if is_pytrich:
            # Rebuild comparison inline: baseline rows + LLM rows
            comparison = pd.concat([
                baseline,
                all_data[~all_data['experiment_name'].str.startswith(
                    ('Panda', 'Pytrich'), na=False)]
            ], ignore_index=True)
            # The win/loss CSV is computed against Panda, so it says nothing
            # about a Pytrich baseline. Rebuild it here with the same definition
            # create_comparison_csv.py uses: per problem, the LLM virtual best
            # (min over all LLM rows) against the baseline's virtual best (min
            # over its rows), one row per metric the LLM side wins.
            llm_wins = _wins_against_baseline(
                baseline,
                all_data[~all_data['experiment_name'].str.startswith(
                    ('Panda', 'Pytrich'), na=False)],
                prefix)
        else:
            comparison = comparison[
                ~comparison['experiment_name'].str.startswith(
                    'Panda', na=False) |
                comparison['experiment_name'].str.startswith(
                    prefix, na=False)
            ]
    else:
        baseline = baseline[
            baseline['experiment_name'].str.startswith(prefix, na=False)
        ]
        if not is_pytrich:
            comparison = comparison[
                ~comparison['experiment_name'].str.startswith(
                    'Panda', na=False) |
                comparison['experiment_name'].str.startswith(
                    prefix, na=False)
            ]

    return llm, baseline, comparison, llm_wins



def _wins_against_baseline(baseline, llm, variant):
    """Per-problem LLM-virtual-best wins against a baseline's virtual best.

    Mirrors create_comparison_csv.py: a win on a metric is a strictly smaller
    minimum over the LLM rows than over the baseline rows for that problem.
    Returns the llm_beats_panda.csv schema so the plots need no special case.
    """
    cols = ['domain_name', 'problem_name', 'panda_variant', 'metric',
            'llm_value', 'panda_value', 'improvement', 'improvement_pct']
    key = ['domain_name', 'problem_name']
    rows = []
    b = baseline.dropna(subset=['solution_size']).groupby(key).agg(
        nodes=('expanded_nodes', 'min'), size=('solution_size', 'min'))
    l = llm.dropna(subset=['solution_size']).groupby(key).agg(
        nodes=('expanded_nodes', 'min'), size=('solution_size', 'min'))
    m = l.join(b, lsuffix='_llm', rsuffix='_base', how='inner').reset_index()
    for _, r in m.iterrows():
        for metric, a, c in (('expanded_nodes', r['nodes_llm'], r['nodes_base']),
                             ('solution_size', r['size_llm'], r['size_base'])):
            if pd.notna(a) and pd.notna(c) and a < c and c > 0:
                rows.append({'domain_name': r['domain_name'], 'problem_name': r['problem_name'],
                             'panda_variant': variant, 'metric': metric,
                             'llm_value': a, 'panda_value': c, 'improvement': c - a,
                             'improvement_pct': round((c - a) / c * 100, 2)})
    return pd.DataFrame(rows, columns=cols)

def extract_model_name(experiment_name):
    """Extract the planner/model name from experiment_name."""
    if pd.isna(experiment_name):
        return 'Unknown'
    # Format: ModelName_Algorithm (e.g., Gemini20Flash_StandardA)
    parts = experiment_name.split('_')
    if len(parts) >= 1:
        name = parts[0]
        NAME_MAP = {
            'Panda-rc2ff': 'Panda (rc2ff)',
            'Panda-rc2lmc': 'Panda (rc2lmc)',
            'Panda-rc2add': 'Panda (rc2add)',
            'Panda-lmcountlmtypebid': 'Panda (LMCount)',
            'Pytrich-TDG': 'Pytrich (TDG)',
            'Pytrich-LMCOUNT': 'Pytrich (LMCount)',
        }
        return NAME_MAP.get(name, name)
    return experiment_name


def extract_algorithm(experiment_name):
    """Extract algorithm from experiment_name."""
    if pd.isna(experiment_name):
        return 'Unknown'
    parts = experiment_name.split('_')
    if len(parts) >= 2:
        algo = parts[1]
        # Display-friendly names
        ALGO_LABELS = {
            'WA': 'WA* (w=5)',
            'StandardA': 'A*',
        }
        return ALGO_LABELS.get(algo, algo)
    return 'Unknown'


def _panda_label(panda_variant):
    """Return display label for the baseline variant being plotted."""
    if panda_variant in BASELINE_CONFIG:
        return BASELINE_CONFIG[panda_variant][1]
    return 'Panda (best)'


def _panda_color(panda_variant):
    """Return color for the baseline variant being plotted."""
    if panda_variant in BASELINE_CONFIG:
        color_key = BASELINE_CONFIG[panda_variant][2]
        return COLORS.get(color_key, COLORS['panda_rc2ff'])
    return COLORS['panda_rc2ff']


def plot_coverage_by_domain(llm, panda, output_dir, panda_variant=None):
    """Bar chart: problems solved per domain by planner type."""
    print("Generating: coverage_by_domain.pdf")
    label = _panda_label(panda_variant)

    # Get unique problems solved by each planner type per domain
    llm_coverage = llm.groupby('domain_name')['problem_name'].nunique().reset_index()
    llm_coverage.columns = ['domain_name', 'LLM Best']

    panda_coverage = panda.groupby('domain_name')['problem_name'].nunique().reset_index()
    panda_coverage.columns = ['domain_name', label]

    # Merge
    coverage = pd.merge(llm_coverage, panda_coverage, on='domain_name', how='outer').fillna(0)
    coverage = coverage.sort_values('domain_name')

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(coverage))
    width = 0.35

    bars1 = ax.bar(x - width/2, coverage[label], width, label=label, color=_panda_color(panda_variant))
    bars2 = ax.bar(x + width/2, coverage['LLM Best'], width, label='LLM (any model)', color=COLORS['llm'])

    ax.set_xlabel('Domain', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Problems Solved', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title('Coverage: Problems Solved per Domain', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(coverage['domain_name'], rotation=45, ha='right', fontsize=TICK_LABEL_FONTSIZE)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coverage_by_domain.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_win_loss_stacked(comparison, output_dir, panda_variant=None):
    """Stacked bar: Win/Tie/Loss by domain for expanded nodes."""
    print("Generating: win_loss_by_domain.pdf")
    label = _panda_label(panda_variant)

    results = []

    for domain in comparison['domain_name'].unique():
        domain_data = comparison[comparison['domain_name'] == domain]
        problems = domain_data['problem_name'].unique()

        wins, ties, losses = 0, 0, 0
        for prob in problems:
            prob_data = domain_data[domain_data['problem_name'] == prob]

            panda_row = prob_data[prob_data['_is_baseline']]
            llm_rows = prob_data[~prob_data['_is_baseline']]

            if len(panda_row) == 0 or len(llm_rows) == 0:
                continue

            panda_nodes = panda_row['expanded_nodes'].min()
            llm_nodes = llm_rows['expanded_nodes'].min()

            if llm_nodes < panda_nodes:
                wins += 1
            elif llm_nodes > panda_nodes:
                losses += 1
            else:
                ties += 1

        if wins + ties + losses > 0:
            results.append({
                'domain': domain,
                'LLM Wins': wins,
                'Ties': ties,
                f'{label} Wins': losses
            })

    df_results = pd.DataFrame(results).sort_values('domain')

    if len(df_results) == 0:
        print("  Warning: No comparison data available for win/loss chart")
        return

    panda_col = f'{label} Wins'

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df_results))
    width = 0.6

    ax.bar(x, df_results['LLM Wins'], width, label='LLM Wins (fewer nodes)', color=COLORS['win'])
    ax.bar(x, df_results['Ties'], width, bottom=df_results['LLM Wins'], label='Tie', color=COLORS['tie'])
    ax.bar(x, df_results[panda_col], width,
           bottom=df_results['LLM Wins'] + df_results['Ties'],
           label=f'{label} Wins', color=COLORS['loss'])

    ax.set_xlabel('Domain', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Number of Problems', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f'Win/Tie/Loss Analysis: LLM vs {label} (Expanded Nodes)', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['domain'], rotation=45, ha='right', fontsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_loss_by_domain.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_win_loss_solution_size(comparison, output_dir, panda_variant=None):
    """Stacked bar: Win/Tie/Loss by domain for solution_size metric."""
    print("Generating: win_loss_by_domain_solution_size.pdf")
    label = _panda_label(panda_variant)

    results = []

    for domain in comparison['domain_name'].unique():
        domain_data = comparison[comparison['domain_name'] == domain]
        problems = domain_data['problem_name'].unique()

        wins, ties, losses = 0, 0, 0
        for prob in problems:
            prob_data = domain_data[domain_data['problem_name'] == prob]

            panda_row = prob_data[prob_data['_is_baseline']]
            llm_rows = prob_data[~prob_data['_is_baseline']]

            if len(panda_row) == 0 or len(llm_rows) == 0:
                continue

            panda_size = panda_row['solution_size'].min()
            llm_size = llm_rows['solution_size'].min()

            if llm_size < panda_size:
                wins += 1
            elif llm_size > panda_size:
                losses += 1
            else:
                ties += 1

        if wins + ties + losses > 0:
            results.append({
                'domain': domain,
                'LLM Wins': wins,
                'Ties': ties,
                f'{label} Wins': losses
            })

    df_results = pd.DataFrame(results).sort_values('domain')

    if len(df_results) == 0:
        print("  Warning: No comparison data available for solution size win/loss chart")
        return

    panda_col = f'{label} Wins'

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df_results))
    width = 0.6

    ax.bar(x, df_results['LLM Wins'], width, label='LLM Wins (shorter plans)', color=COLORS['win'])
    ax.bar(x, df_results['Ties'], width, bottom=df_results['LLM Wins'], label='Tie', color=COLORS['tie'])
    ax.bar(x, df_results[panda_col], width,
           bottom=df_results['LLM Wins'] + df_results['Ties'],
           label=f'{label} Wins', color=COLORS['loss'])

    ax.set_xlabel('Domain', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Number of Problems', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f'Win/Tie/Loss Analysis: LLM vs {label} (Solution Size)', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['domain'], rotation=45, ha='right', fontsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_loss_by_domain_solution_size.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_win_loss_both_metrics(comparison, output_dir, panda_variant=None):
    """Stacked bar: Win/Tie/Loss by domain when comparing BOTH metrics."""
    print("Generating: win_loss_by_domain_both.pdf")
    label = _panda_label(panda_variant)

    results = []

    for domain in comparison['domain_name'].unique():
        domain_data = comparison[comparison['domain_name'] == domain]
        problems = domain_data['problem_name'].unique()

        wins, ties, losses, mixed = 0, 0, 0, 0
        for prob in problems:
            prob_data = domain_data[domain_data['problem_name'] == prob]

            panda_row = prob_data[prob_data['_is_baseline']]
            llm_rows = prob_data[~prob_data['_is_baseline']]

            if len(panda_row) == 0 or len(llm_rows) == 0:
                continue

            panda_nodes = panda_row['expanded_nodes'].min()
            llm_nodes = llm_rows['expanded_nodes'].min()
            panda_size = panda_row['solution_size'].min()
            llm_size = llm_rows['solution_size'].min()

            llm_wins_nodes = llm_nodes < panda_nodes
            llm_wins_size = llm_size < panda_size
            panda_wins_nodes = llm_nodes > panda_nodes
            panda_wins_size = llm_size > panda_size

            if llm_wins_nodes and llm_wins_size:
                wins += 1
            elif panda_wins_nodes and panda_wins_size:
                losses += 1
            elif not llm_wins_nodes and not panda_wins_nodes and not llm_wins_size and not panda_wins_size:
                ties += 1
            else:
                mixed += 1

        if wins + ties + losses + mixed > 0:
            results.append({
                'domain': domain,
                'LLM Wins Both': wins,
                'Mixed': mixed,
                'Tie': ties,
                f'{label} Wins Both': losses
            })

    df_results = pd.DataFrame(results).sort_values('domain')

    if len(df_results) == 0:
        print("  Warning: No comparison data available for combined win/loss chart")
        return

    panda_col = f'{label} Wins Both'

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df_results))
    width = 0.6

    ax.bar(x, df_results['LLM Wins Both'], width, label='LLM Wins Both', color=COLORS['win'])
    ax.bar(x, df_results['Mixed'], width, bottom=df_results['LLM Wins Both'],
           label='Mixed (split wins)', color='#9b59b6')
    ax.bar(x, df_results['Tie'], width,
           bottom=df_results['LLM Wins Both'] + df_results['Mixed'],
           label='Tie', color=COLORS['tie'])
    ax.bar(x, df_results[panda_col], width,
           bottom=df_results['LLM Wins Both'] + df_results['Mixed'] + df_results['Tie'],
           label=f'{label} Wins Both', color=COLORS['loss'])

    ax.set_xlabel('Domain', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Number of Problems', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f'Win/Tie/Loss Analysis: LLM vs {label} (Both Nodes & Solution Size)', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['domain'], rotation=45, ha='right', fontsize=TICK_LABEL_FONTSIZE)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_loss_by_domain_both.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_scatter_expanded_nodes(comparison, output_dir, panda_variant=None):
    """Scatter plot: LLM vs Panda expanded_nodes."""
    print("Generating: scatter_expanded_nodes.pdf")
    label = _panda_label(panda_variant)

    scatter_data = []

    for domain in comparison['domain_name'].unique():
        domain_data = comparison[comparison['domain_name'] == domain]

        for prob in domain_data['problem_name'].unique():
            prob_data = domain_data[domain_data['problem_name'] == prob]

            panda_row = prob_data[prob_data['_is_baseline']]
            llm_rows = prob_data[~prob_data['_is_baseline']]

            if len(panda_row) == 0 or len(llm_rows) == 0:
                continue

            panda_nodes = panda_row['expanded_nodes'].min()
            llm_nodes = llm_rows['expanded_nodes'].min()

            if panda_nodes > 0 and llm_nodes > 0:
                scatter_data.append({
                    'domain': domain,
                    'problem': prob,
                    label: panda_nodes,
                    'LLM': llm_nodes
                })

    df_scatter = pd.DataFrame(scatter_data)

    if len(df_scatter) == 0:
        print("  Warning: No scatter data available")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    domains = df_scatter['domain'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(domains)))
    domain_colors = dict(zip(domains, colors))

    for domain in domains:
        domain_data = df_scatter[df_scatter['domain'] == domain]
        ax.scatter(domain_data[label], domain_data['LLM'],
                  label=domain, alpha=0.7, s=50, color=domain_colors[domain])

    min_val = min(df_scatter[label].min(), df_scatter['LLM'].min())
    max_val = max(df_scatter[label].max(), df_scatter['LLM'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x (equal)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f'{label} Expanded Nodes', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('LLM Expanded Nodes', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f'Expanded Nodes: LLM vs {label}\n(Below diagonal = LLM better)', fontsize=TITLE_FONTSIZE, fontweight='bold')

    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=ANNOTATION_FONTSIZE)
    ax.legend(loc='lower right', fontsize=ANNOTATION_FONTSIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_expanded_nodes.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_scatter_solution_size(comparison, output_dir, panda_variant=None):
    """Scatter plot: LLM vs Panda solution_size."""
    print("Generating: scatter_solution_size.pdf")
    label = _panda_label(panda_variant)

    scatter_data = []

    for domain in comparison['domain_name'].unique():
        domain_data = comparison[comparison['domain_name'] == domain]

        for prob in domain_data['problem_name'].unique():
            prob_data = domain_data[domain_data['problem_name'] == prob]

            panda_row = prob_data[prob_data['_is_baseline']]
            llm_rows = prob_data[~prob_data['_is_baseline']]

            if len(panda_row) == 0 or len(llm_rows) == 0:
                continue

            panda_sol = panda_row['solution_size'].min()
            llm_sol = llm_rows['solution_size'].min()

            if panda_sol > 0 and llm_sol > 0:
                scatter_data.append({
                    'domain': domain,
                    'problem': prob,
                    label: panda_sol,
                    'LLM': llm_sol
                })

    df_scatter = pd.DataFrame(scatter_data)

    if len(df_scatter) == 0:
        print("  Warning: No scatter data available")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    domains = df_scatter['domain'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(domains)))
    domain_colors = dict(zip(domains, colors))

    for domain in domains:
        domain_data = df_scatter[df_scatter['domain'] == domain]
        ax.scatter(domain_data[label], domain_data['LLM'],
                  label=domain, alpha=0.7, s=50, color=domain_colors[domain])

    min_val = min(df_scatter[label].min(), df_scatter['LLM'].min())
    max_val = max(df_scatter[label].max(), df_scatter['LLM'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x (equal)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f'{label} Solution Size', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('LLM Solution Size', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f'Solution Size: LLM vs {label}\n(Below diagonal = LLM better)', fontsize=TITLE_FONTSIZE, fontweight='bold')

    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=ANNOTATION_FONTSIZE)
    ax.legend(loc='lower right', fontsize=ANNOTATION_FONTSIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_solution_size.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_boxplot_expanded_nodes(llm, panda, output_dir, panda_variant=None):
    """Box plots: expanded_nodes distribution by planner."""
    print("Generating: boxplot_expanded_nodes.pdf")
    label = _panda_label(panda_variant)

    llm_data = llm[['domain_name', 'expanded_nodes']].copy()
    llm_data['Planner'] = 'LLM'
    llm_data['model'] = llm['experiment_name'].apply(extract_model_name)

    panda_data = panda[['domain_name', 'expanded_nodes']].copy()
    panda_data['Planner'] = label
    panda_data['model'] = label

    combined = pd.concat([llm_data, panda_data], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    domains = sorted(combined['domain_name'].unique())

    positions_panda = np.arange(len(domains)) * 3
    positions_llm = positions_panda + 1

    panda_by_domain = [combined[(combined['domain_name'] == d) & (combined['Planner'] == label)]['expanded_nodes'].dropna()
                       for d in domains]
    bp1 = ax.boxplot([x for x in panda_by_domain], positions=positions_panda, widths=0.6, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor(_panda_color(panda_variant))
        patch.set_alpha(0.7)

    llm_by_domain = [combined[(combined['domain_name'] == d) & (combined['Planner'] == 'LLM')]['expanded_nodes'].dropna()
                     for d in domains]
    bp2 = ax.boxplot([x for x in llm_by_domain], positions=positions_llm, widths=0.6, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor(COLORS['llm'])
        patch.set_alpha(0.7)

    ax.set_yscale('log')
    ax.set_xticks(positions_panda + 0.5)
    ax.set_xticklabels(domains, rotation=45, ha='right', fontsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel('Domain', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Expanded Nodes (log scale)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title('Expanded Nodes Distribution by Domain', fontsize=TITLE_FONTSIZE, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=_panda_color(panda_variant), alpha=0.7, label=label),
                      Patch(facecolor=COLORS['llm'], alpha=0.7, label='LLM')]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_expanded_nodes.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_violin_solution_size(llm, panda, output_dir, panda_variant=None):
    """Violin plots: solution_size distribution by planner."""
    print("Generating: violin_solution_size.pdf")
    label = _panda_label(panda_variant)

    llm_data = llm[['domain_name', 'solution_size']].copy()
    llm_data['Planner'] = 'LLM'

    panda_data = panda[['domain_name', 'solution_size']].copy()
    panda_data['Planner'] = label

    combined = pd.concat([llm_data, panda_data], ignore_index=True)
    combined = combined.dropna(subset=['solution_size'])

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.violinplot(data=combined, x='Planner', y='solution_size', hue='Planner',
                   palette={label: _panda_color(panda_variant), 'LLM': COLORS['llm']},
                   ax=ax, legend=False)

    ax.set_yscale('log')
    ax.set_xlabel('Planner Type', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Solution Size (log scale)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f'Solution Size Distribution: {label} vs LLM', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'violin_solution_size.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_cactus(llm, panda, output_dir, panda_variant=None):
    """Cactus plot: cumulative problems solved vs expanded_nodes (sorted)."""
    print("Generating: cactus_plot.pdf")
    label = _panda_label(panda_variant)

    llm_best = llm.groupby(['domain_name', 'problem_name']).agg({
        'expanded_nodes': 'min'
    }).reset_index()

    panda_best = panda.groupby(['domain_name', 'problem_name']).agg({
        'expanded_nodes': 'min'
    }).reset_index()

    llm_sorted = llm_best['expanded_nodes'].sort_values().reset_index(drop=True)
    panda_sorted = panda_best['expanded_nodes'].sort_values().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.step(range(1, len(panda_sorted) + 1), panda_sorted, where='post',
            label=f'{label} ({len(panda_sorted)} problems)', color=_panda_color(panda_variant), linewidth=2)
    ax.step(range(1, len(llm_sorted) + 1), llm_sorted, where='post',
            label=f'LLM Best ({len(llm_sorted)} problems)', color=COLORS['llm'], linewidth=2)

    ax.set_yscale('log')
    ax.set_xlabel('Number of Problems Solved', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Expanded Nodes (log scale)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title('Cactus Plot: Problems Solved vs Search Effort', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cactus_plot.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmap_coverage(llm, panda, output_dir, panda_variant=None):
    """Heatmap: Domains x Planners showing coverage percentage."""
    print("Generating: heatmap_coverage.pdf")

    llm = llm.copy()
    panda = panda.copy()
    llm['model'] = llm['experiment_name'].apply(extract_model_name)
    panda['model'] = panda['experiment_name'].apply(extract_model_name)

    # Get unique problems per domain (as denominator)
    all_problems = pd.concat([
        llm[['domain_name', 'problem_name']],
        panda[['domain_name', 'problem_name']]
    ]).drop_duplicates()
    total_per_domain = all_problems.groupby('domain_name').size()

    # Build list of Panda models present
    panda_models = sorted(panda['model'].unique().tolist())
    llm_models = sorted(llm['model'].unique().tolist())
    models = panda_models + llm_models
    domains = sorted(llm['domain_name'].unique())

    coverage_matrix = []

    for domain in domains:
        row = []
        total = total_per_domain.get(domain, 1)

        # Panda coverage (one or more variants)
        for model in panda_models:
            model_data = panda[(panda['domain_name'] == domain) & (panda['model'] == model)]
            solved = model_data['problem_name'].nunique()
            row.append(solved / total * 100)

        # LLM models coverage
        for model in llm_models:
            model_data = llm[(llm['domain_name'] == domain) & (llm['model'] == model)]
            solved = model_data['problem_name'].nunique()
            row.append(solved / total * 100)

        coverage_matrix.append(row)

    df_heatmap = pd.DataFrame(coverage_matrix, index=domains, columns=models)

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(df_heatmap, annot=True, fmt='.0f', cmap='RdYlGn',
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Coverage %'})

    ax.set_xlabel('Planner/Model', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Domain', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title('Coverage Heatmap: Domains x Planners (%)', fontsize=TITLE_FONTSIZE, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_coverage.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_improvement_by_domain(llm_wins, output_dir, panda_variant=None):
    """Grouped bar: improvement % by domain from llm_beats_panda.csv."""
    print("Generating: improvement_by_domain.pdf")
    label = _panda_label(panda_variant)

    if len(llm_wins) == 0:
        print("  Warning: No LLM wins data available")
        return

    improvement = llm_wins.groupby(['domain_name', 'metric']).agg({
        'improvement_pct': 'mean',
        'problem_name': 'count'
    }).reset_index()
    improvement.columns = ['domain', 'metric', 'avg_improvement', 'count']

    pivot = improvement.pivot(index='domain', columns='metric', values='avg_improvement').fillna(0)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(pivot))
    width = 0.35

    if 'expanded_nodes' in pivot.columns:
        ax.bar(x - width/2, pivot['expanded_nodes'], width, label='Expanded Nodes', color='#3498db')
    if 'solution_size' in pivot.columns:
        ax.bar(x + width/2, pivot['solution_size'], width, label='Solution Size', color='#e74c3c')

    ax.set_xlabel('Domain', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Average Improvement (%)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f'Average Improvement: LLM over {label} (when LLM wins)', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=TICK_LABEL_FONTSIZE)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_by_domain.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_algorithm_comparison(llm, output_dir, panda_variant=None):
    """Grouped bar: StandardA vs GBFS vs WA performance within each model."""
    print("Generating: algorithm_comparison.pdf")

    llm = llm.copy()
    llm['model'] = llm['experiment_name'].apply(extract_model_name)
    llm['algorithm'] = llm['experiment_name'].apply(extract_algorithm)

    algo_perf = llm.groupby(['model', 'algorithm']).agg({
        'expanded_nodes': 'median',
        'problem_name': 'count'
    }).reset_index()
    algo_perf.columns = ['model', 'algorithm', 'median_nodes', 'count']

    pivot = algo_perf.pivot(index='model', columns='algorithm', values='median_nodes').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))

    algorithms = [col for col in ['A*', 'GBFS', 'WA* (w=5)'] if col in pivot.columns]
    x = np.arange(len(pivot))
    width = 0.25

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, algo in enumerate(algorithms):
        offset = (i - len(algorithms)/2 + 0.5) * width
        ax.bar(x + offset, pivot[algo], width, label=algo, color=colors[i % len(colors)])

    ax.set_yscale('log')
    ax.set_xlabel('Model', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Median Expanded Nodes (log scale)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title('Algorithm Comparison Across LLM Models', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=TICK_LABEL_FONTSIZE)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(llm, panda, output_dir, panda_variant=None):
    """Bar chart comparing all models including Panda by median expanded nodes."""
    print("Generating: model_comparison.pdf")

    llm = llm.copy()
    panda = panda.copy()
    llm['model'] = llm['experiment_name'].apply(extract_model_name)
    panda['model'] = panda['experiment_name'].apply(extract_model_name)

    # Get median expanded nodes per LLM model
    llm_stats = llm.groupby('model').agg({
        'expanded_nodes': 'median',
        'problem_name': 'count'
    }).reset_index()
    llm_stats.columns = ['model', 'median_nodes', 'count']

    # Add each Panda variant as separate entries
    panda_stats = panda.groupby('model').agg({
        'expanded_nodes': 'median',
        'problem_name': 'count'
    }).reset_index()
    panda_stats.columns = ['model', 'median_nodes', 'count']

    all_stats = pd.concat([panda_stats, llm_stats], ignore_index=True)
    all_stats = all_stats.sort_values('median_nodes')

    fig, ax = plt.subplots(figsize=(10, 6))

    def _model_color(m):
        if 'rc2lmc' in m:
            return COLORS['panda_rc2lmc']
        elif 'rc2add' in m:
            return COLORS['panda_rc2add']
        elif 'lmcountlmtypebid' in m:
            return COLORS['panda_lmcount']
        elif 'Panda' in m:
            return COLORS['panda_rc2ff']
        return COLORS['llm']

    colors = [_model_color(m) for m in all_stats['model']]
    bars = ax.bar(all_stats['model'], all_stats['median_nodes'], color=colors)

    ax.set_yscale('log')
    ax.set_xlabel('Model', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Median Expanded Nodes (log scale)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title('Model Comparison: Median Search Effort', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    for bar, count in zip(bars, all_stats['count']):
        height = bar.get_height()
        ax.annotate(f'n={count}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_solution_quality_comparison(llm_wins, output_dir, panda_variant=None):
    """Histogram of improvement percentages for solution size."""
    print("Generating: solution_quality_histogram.pdf")
    label = _panda_label(panda_variant)

    sol_wins = llm_wins[llm_wins['metric'] == 'solution_size']

    if len(sol_wins) == 0:
        print("  Warning: No solution size improvements to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(sol_wins['improvement_pct'], bins=20, edgecolor='black', alpha=0.7, color=COLORS['win'])
    ax.axvline(x=sol_wins['improvement_pct'].median(), color='red', linestyle='--',
               label=f'Median: {sol_wins["improvement_pct"].median():.1f}%')

    ax.set_xlabel('Improvement Percentage', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Frequency', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f'Solution Quality Improvement: LLM vs {label}', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'solution_quality_histogram.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def run_plot_suite(csv_dir, input_file, output_dir, panda_variant=None):
    """Run all plot calls for a given baseline variant and output directory."""
    variant_desc = panda_variant or 'both'
    print(f"\n{'='*60}")
    print(f"Plot suite: variant={variant_desc}, output={output_dir}/")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    try:
        llm, panda, comparison, llm_wins = load_data(
            csv_dir, input_file=input_file, panda_variant=panda_variant)
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV file: {e}")
        return

    print(f"Loaded data ({variant_desc}):")
    print(f"  - LLM results: {len(llm)} records")
    print(f"  - Baseline results: {len(panda)} records")
    print(f"  - Comparison data: {len(comparison)} records")
    print(f"  - LLM wins: {len(llm_wins)} records")
    print()

    if len(panda) == 0:
        print(f"  Warning: No baseline data for '{variant_desc}', skipping")
        return

    # Tag baseline rows in comparison so plot functions work for any baseline.
    if panda_variant and panda_variant in BASELINE_CONFIG:
        baseline_pfx = BASELINE_CONFIG[panda_variant][0]
    else:
        baseline_pfx = 'Panda'
    comparison = comparison.copy()
    comparison['_is_baseline'] = (
        comparison['experiment_name'].str.startswith(baseline_pfx, na=False)
    )

    plot_coverage_by_domain(llm, panda, output_dir, panda_variant)
    plot_win_loss_stacked(comparison, output_dir, panda_variant)
    plot_win_loss_solution_size(comparison, output_dir, panda_variant)
    plot_win_loss_both_metrics(comparison, output_dir, panda_variant)
    plot_scatter_expanded_nodes(comparison, output_dir, panda_variant)
    plot_scatter_solution_size(comparison, output_dir, panda_variant)
    plot_boxplot_expanded_nodes(llm, panda, output_dir, panda_variant)
    plot_violin_solution_size(llm, panda, output_dir, panda_variant)
    plot_cactus(llm, panda, output_dir, panda_variant)
    plot_heatmap_coverage(llm, panda, output_dir, panda_variant)
    plot_improvement_by_domain(llm_wins, output_dir, panda_variant)
    plot_algorithm_comparison(llm, output_dir, panda_variant)
    plot_model_comparison(llm, panda, output_dir, panda_variant)
    plot_solution_quality_comparison(llm_wins, output_dir, panda_variant)

    print(f"\nPlot suite '{variant_desc}' complete -> {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive visualizations for Panda vs LLM comparison')
    parser.add_argument('--output-dir', '-o', default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory for plots (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--csv-dir', '-c', default=DEFAULT_CSV_DIR,
                       help=f'Directory containing CSV files (default: {DEFAULT_CSV_DIR})')
    parser.add_argument('-i', '--input', default=None,
                       help='Single input CSV containing all results (Panda + LLM). '
                            'Comparison/wins CSVs are read from the same directory.')
    args = parser.parse_args()

    suites = [
        (None,              args.output_dir),
        ('rc2ff',           os.path.join(args.output_dir, 'panda-rc2ff')),
        ('rc2lmc',          os.path.join(args.output_dir, 'panda-rc2lmc')),
        ('panda-lmcount',   os.path.join(args.output_dir, 'panda-lmcount')),
        ('pytrich-tdg',     os.path.join(args.output_dir, 'pytrich-tdg')),
        ('pytrich-lmcount', os.path.join(args.output_dir, 'pytrich-lmcount')),
    ]
    for variant, out_dir in suites:
        run_plot_suite(args.csv_dir, args.input, out_dir,
                       panda_variant=variant)

    print("\n" + "="*60)
    print("All plot suites generated successfully!")
    for variant, out_dir in suites:
        label = variant or 'combined'
        print(f"  {label:<20} {out_dir}/")


if __name__ == '__main__':
    main()
