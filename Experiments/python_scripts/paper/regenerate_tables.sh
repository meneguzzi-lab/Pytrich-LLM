#!/usr/bin/env bash
# Regenerate every table and figure of the paper from a results CSV and the ablation CSV.
#
#   bash Experiments/python_scripts/paper/regenerate_tables.sh [RESULTS.csv] [ABLATION.csv] [OUTDIR]
#
# Defaults: Experiments/results/results.csv, Experiments/results/ablation.csv, Experiments/outputs/paper.
# Needs pandas, numpy, matplotlib, seaborn and tabulate (requirements.txt).
set -euo pipefail
cd "$(dirname "$0")/../../.."
RESULTS="${1:-Experiments/results/results.csv}"
ABLATION="${2:-Experiments/results/ablation.csv}"
OUT="${3:-Experiments/outputs/paper}"
PY="${PY:-python3}"
P=Experiments/python_scripts/paper
mkdir -p "$OUT/figures" "$OUT/fixed_pair"

# Main text and appendix tables (LaTeX)
$PY $P/generate_coverage_table.py -i "$RESULTS" -o "$OUT/coverage_table.tex"
$PY $P/generate_planlength_table.py -i "$RESULTS" -o "$OUT/planlength_table.tex"
$PY $P/generate_planlength_table.py -i "$RESULTS" --intersection -o "$OUT/planlength_intersection_table.tex"
$PY $P/median_nodes_by_algo_domain.py "$RESULTS" --latex > "$OUT/median_nodes_by_algo.tex"

# Every number quoted in the results section, as markdown
$PY $P/results_numbers.py "$RESULTS" > "$OUT/results_numbers.md"

# Fixed (deployable) configurations and the prompt-context exclusion
$PY $P/fixed_pair_tables.py "$RESULTS" "$OUT/fixed_pair"
$PY $P/fixed_config.py "$RESULTS" > "$OUT/fixed_config.txt"
$PY $P/fixed_vs_pytrich.py "$RESULTS" > "$OUT/nodes_fixed_vs_pytrich.md"
$PY $P/prompt_context_excluded.py "$RESULTS" --markdown > "$OUT/prompt_context_excluded.md"

# Prompt ablation (appendix): one table set per search algorithm plus the virtual best
$PY Experiments/python_scripts/generate_ablation_table.py -i "$ABLATION" --baseline "$RESULTS" \
    --virtual-best --candidate-counts --markdown "$OUT/ablation_tables.md"

# Figures: comparison CSVs, then the plot suites (one per baseline)
TMP="$(mktemp -d)"; cp "$RESULTS" "$TMP/results.csv"
$PY Experiments/python_scripts/create_comparison_csv.py -i "$TMP/results.csv"
$PY Experiments/python_scripts/generate_all_plots.py -i "$TMP/results.csv" -o "$OUT/figures"
rm -rf "$TMP"
echo "tables and figures written under $OUT"
