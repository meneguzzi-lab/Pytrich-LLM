#!/bin/bash
#
# Master script to submit all HTN planning experiments as SLURM job arrays
# Usage: ./submit_all_experiments.sh [OPTIONS]
#
# Options:
#   --dry-run             Generate task lists but don't submit jobs
#   --domain <name>       Submit only specific domain(s) (can be repeated)
#   --planner <name>      Submit only specific planner(s) (can be repeated)
#   --algorithm <name>    Submit only specific algorithm(s) (can be repeated)
#   --ablation            Submit the prompt-ablation arms (Gemini 3 Flash, six domains)
#                         instead of the systems in config/planners.yaml
#   --skip-validation     Skip heuristic file validation
#   -h, --help            Show this help message

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration file
CONFIG_FILE="$PROJECT_ROOT/config/planners.yaml"

# Function to parse YAML arrays (simple parser for our format)
parse_yaml_array() {
    local key="$1"
    local file="$2"
    # Find the key, then extract items until next key or EOF
    # Also strip inline comments (anything after #)
    awk -v key="$key:" '
        $0 ~ "^" key { found=1; next }
        found && /^[a-z_]+:/ { found=0 }
        found && /^ *- / {
            gsub(/^ *- /, "")
            gsub(/ *#.*$/, "")  # Remove inline comments
            if (length($0) > 0) print
        }
    ' "$file"
}

# Load configuration from YAML
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse planners from config (compatible with bash 3.x)
IFS=$'\n' read -r -d '' -a LLM_PLANNERS < <(parse_yaml_array "llm_planners" "$CONFIG_FILE" && printf '\0') || true
IFS=$'\n' read -r -d '' -a BUILTIN_PLANNERS < <(parse_yaml_array "builtin_planners" "$CONFIG_FILE" && printf '\0') || true
IFS=$'\n' read -r -d '' -a CONFIG_DOMAINS < <(parse_yaml_array "domains" "$CONFIG_FILE" && printf '\0') || true
IFS=$'\n' read -r -d '' -a CONFIG_ALGORITHMS < <(parse_yaml_array "algorithms" "$CONFIG_FILE" && printf '\0') || true

# Combine LLM and builtin planners
PLANNERS=("${LLM_PLANNERS[@]}" "${BUILTIN_PLANNERS[@]}")
DOMAINS=("${CONFIG_DOMAINS[@]}")
ALGORITHMS=("${CONFIG_ALGORITHMS[@]}")

# Directories
TASK_LISTS_DIR="$SCRIPT_DIR/TaskLists"
LOGS_DIR="$PROJECT_ROOT/logs"

# Command-line options
DRY_RUN=false
ABLATION=false
SKIP_VALIDATION=false
FILTER_DOMAINS=()
FILTER_PLANNERS=()
FILTER_ALGORITHMS=()

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --domain)
            FILTER_DOMAINS+=("$2")
            shift 2
            ;;
        --planner)
            FILTER_PLANNERS+=("$2")
            shift 2
            ;;
        --algorithm)
            FILTER_ALGORITHMS+=("$2")
            shift 2
            ;;
        --ablation)
            ABLATION=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run             Generate task lists but don't submit jobs"
            echo "  --domain <name>       Submit only specific domain(s) (can be repeated)"
            echo "  --planner <name>      Submit only specific planner(s) (can be repeated)"
            echo "  --algorithm <name>    Submit only specific algorithm(s) (can be repeated)"
            echo "  --ablation            Submit the prompt-ablation arms instead of the main systems"
            echo "  --skip-validation     Skip heuristic file validation (include all files)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Configuration loaded from: $CONFIG_FILE"
            echo ""
            echo "Available domains: ${DOMAINS[*]}"
            echo "Available planners (LLM): ${LLM_PLANNERS[*]}"
            echo "Available planners (Built-in): ${BUILTIN_PLANNERS[*]}"
            echo "Available algorithms: ${ALGORITHMS[*]}"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# The prompt-ablation arms: three Gemini 3 Flash prompt conditions, run by file path,
# on the six IPC 2020 domains (Lamps has no arm candidates).
# Their logs go under JobArrays/outputs-ablation/ (sbatch exports the environment), so
# the two sweeps parse into separate CSVs.
if [ "$ABLATION" = true ]; then
    PLANNERS=(Gemini3Flash-arm1-no-example Gemini3Flash-arm2-no-example-hints Gemini3Flash-arm3-fixed-example)
    DOMAINS=(Barman-BDI Blocksworld-GTOHP Depots Robot Rover-GTOHP Towers)
    export OUTPUT_ROOT="${OUTPUT_ROOT:-JobArrays/outputs-ablation}"
fi

# Apply filters if specified
if [ ${#FILTER_DOMAINS[@]} -gt 0 ]; then
    DOMAINS=("${FILTER_DOMAINS[@]}")
fi

if [ ${#FILTER_PLANNERS[@]} -gt 0 ]; then
    PLANNERS=("${FILTER_PLANNERS[@]}")
fi

if [ ${#FILTER_ALGORITHMS[@]} -gt 0 ]; then
    ALGORITHMS=("${FILTER_ALGORITHMS[@]}")
fi

# Create directories if they don't exist
mkdir -p "$TASK_LISTS_DIR"

mkdir -p "$LOGS_DIR"

# Logs land in JobArrays/outputs/<engine>_<algorithm>/<domain>/<heuristic>/<instance>.out
# (see run_experiment.sbatch; OUTPUT_ROOT in the environment overrides the base folder).
# Re-submitting a combination overwrites only its own files.
mkdir -p "$PROJECT_ROOT/${OUTPUT_ROOT:-JobArrays/outputs}"

echo "==========================================="
echo "HTN Planning Experiments - Job Array Submission"
echo "==========================================="
echo "Config: $CONFIG_FILE"
echo "Domains: ${DOMAINS[*]}"
echo "Planners: ${PLANNERS[*]}"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Dry run: $DRY_RUN"
echo "Skip validation: $SKIP_VALIDATION"
echo "==========================================="
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

# Counters
total_arrays=0
total_tasks=0
successful_arrays=0
failed_arrays=0
skipped_arrays=0

# Job IDs for tracking
declare -a job_ids

# Iterate over all combinations
for domain in "${DOMAINS[@]}"; do
    echo "Processing domain: $domain"

    for planner in "${PLANNERS[@]}"; do
        for algorithm in "${ALGORITHMS[@]}"; do
            combination="${domain}_${planner}_${algorithm}"
            task_list_file="$TASK_LISTS_DIR/${combination}.txt"

            echo "  Combination: $planner + $algorithm"

            # Generate task list
            echo "    Generating task list..."
            VALIDATION_FLAG=""
            if [ "$SKIP_VALIDATION" = true ]; then
                VALIDATION_FLAG="--skip-validation"
            fi
            if python3 "$SCRIPT_DIR/generate_task_list.py" \
                --config "$CONFIG_FILE" \
                --domain "$domain" \
                --planner "$planner" \
                --algorithm "$algorithm" \
                --output "$task_list_file" \
                $VALIDATION_FLAG 2>&1; then

                # Count tasks
                if [ -f "$task_list_file" ]; then
                    array_size=$(wc -l < "$task_list_file")

                    if [ "$array_size" -gt 0 ]; then
                        total_arrays=$((total_arrays + 1))
                        total_tasks=$((total_tasks + array_size))

                        echo "    Task list: $array_size tasks"

                        # Submit job array unless dry-run
                        if [ "$DRY_RUN" = false ]; then
                            echo "    Submitting job array..."

                            if job_output=$(sbatch \
                                --array=1-${array_size}%5 \
                                --job-name="${combination}" \
                                "$SCRIPT_DIR/run_experiment.sbatch" \
                                "$task_list_file" 2>&1); then

                                # Extract job ID from sbatch output
                                job_id=$(echo "$job_output" | grep -oP 'Submitted batch job \K\d+')
                                job_ids+=("$job_id")
                                successful_arrays=$((successful_arrays + 1))

                                echo "    Submitted: Job ID $job_id"
                            else
                                echo "    Failed to submit job array"
                                echo "    Error: $job_output"
                                failed_arrays=$((failed_arrays + 1))
                            fi
                        else
                            echo "    [DRY RUN] Would submit array of size $array_size"
                            successful_arrays=$((successful_arrays + 1))
                        fi
                    else
                        echo "    Skipped: No tasks generated (empty task list)"
                        skipped_arrays=$((skipped_arrays + 1))
                    fi
                else
                    echo "    Error: Task list file not created"
                    failed_arrays=$((failed_arrays + 1))
                fi
            else
                echo "    Skipped: No heuristics found for this combination"
                skipped_arrays=$((skipped_arrays + 1))
            fi

            echo ""
        done
    done
done

echo "==========================================="
echo "Submission Summary"
echo "==========================================="
echo "Total job arrays: $total_arrays"
echo "Total tasks: $total_tasks"
echo "Successful submissions: $successful_arrays"
echo "Skipped (no heuristics): $skipped_arrays"
echo "Failed submissions: $failed_arrays"
echo ""

if [ "$DRY_RUN" = false ] && [ ${#job_ids[@]} -gt 0 ]; then
    echo "Submitted job IDs: ${job_ids[*]}"
    echo ""
    echo "Monitor jobs with:"
    echo "  squeue -u \$USER"
    echo "  squeue -j <job_id>"
    echo ""
    echo "Cancel all submitted jobs with:"
    echo "  scancel ${job_ids[*]}"
fi

echo "==========================================="

if [ "$failed_arrays" -gt 0 ]; then
    exit 1
fi

exit 0
