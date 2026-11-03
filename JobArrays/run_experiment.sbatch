#!/bin/bash
#SBATCH --job-name=pytrich_%A_%a
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=spot-compute
#SBATCH --exclude=compute016,hmem012,hmem013,hmem014,vhmem002
#SBATCH --time=01:00:00
#
# The partition and node list above are those of the Maxwell cluster (University of
# Aberdeen). spot-compute mixes CPU types; the excluded nodes are the ones that are not
# Intel Xeon Gold 6138, so that every run executes on the hardware the paper reports.
# Every log records the node it ran on (the "Node:" line). On another cluster, change
# these two lines.

# Generic SLURM job array template for running Pytrich experiments
# Usage: sbatch --array=1-N run_experiment.sbatch <task_list_file>
# Local: ./run_experiment.sbatch --local [--task-id N] <task_list_file>

# Ensure we're running from project root
# Determine Project Root
# On Slurm, SLURM_SUBMIT_DIR points to the directory where sbatch was called
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    # Fallback for local execution
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

cd "$PROJECT_ROOT"

# Create logs directory if it doesn't exist (after cd to ensure correct location)
mkdir -p logs

# Set Python path for module imports
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Pin Python's string-hash seed so set/dict iteration order, and with it node counts,
# is the same from run to run.
export PYTHONHASHSEED=0

# Parse arguments for local mode
LOCAL_MODE=false
TASK_ID=""
TASK_LIST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --task-id)
            TASK_ID="$2"
            shift 2
            ;;
        *)
            TASK_LIST="$1"
            shift
            ;;
    esac
done

# Set SLURM variables for local mode
if [ "$LOCAL_MODE" = true ]; then
    if [ -z "$TASK_ID" ]; then
        TASK_ID=1
    fi
    SLURM_ARRAY_TASK_ID="$TASK_ID"
    SLURM_JOB_ID="LOCAL"
    SLURMD_NODENAME="$(hostname)"
fi

# Check if task list file is provided
if [ -z "$TASK_LIST" ]; then
    echo "Error: Task list file not provided"
    echo "Usage: sbatch --array=1-N run_experiment.sbatch <task_list_file>"
    echo "Local: ./run_experiment.sbatch --local [--task-id N] <task_list_file>"
    exit 1
fi

# Check if task list file exists
if [ ! -f "$TASK_LIST" ]; then
    echo "Error: Task list file not found: $TASK_LIST"
    exit 1
fi

# Read the line corresponding to this array task ID
TASK_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST")

if [ -z "$TASK_LINE" ]; then
    echo "Error: No task found for array task ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse task parameters
# Format: domain instance planner heuristic algorithm engine_name node_config sas_file
read domain instance planner heuristic algorithm engine_name node_config sas_file <<< "$TASK_LINE"

# Clean names for organized output
instance_clean="${instance%.hddl}"
instance_clean="${instance_clean%.pddl}"   # Lamps (IPC 2023) ships .pddl problem files
heuristic_clean="${heuristic%%(*}"
heuristic_clean="${heuristic_clean:-NONE}"

# Ablation arms arrive as a .py path rather than a registry key, because arm files
# have no constants.py entry (see generate_task_list.get_heuristics_llm). Select the
# flag from the field's shape, and strip the path for the output directory name.
if [[ "$heuristic" == *.py ]]; then
    HEURISTIC_FLAG="--heuristic-file"
    heuristic_clean="$(basename "$heuristic" .py)"
else
    HEURISTIC_FLAG="-H"
fi

# Set up organized output directory
# OUTPUT_ROOT lets a run write beside earlier logs instead of over them, e.g.
# sbatch --export=ALL,OUTPUT_ROOT=JobArrays/outputs-timing ...
RESULT_DIR="${OUTPUT_ROOT:-JobArrays/outputs}/${engine_name}/${domain}/${heuristic_clean}"
mkdir -p "$RESULT_DIR" || { echo "Error: Cannot create output directory: $RESULT_DIR"; exit 1; }
RESULT_FILE="${RESULT_DIR}/${instance_clean}.out"

# Redirect all subsequent output to organized file
exec > "$RESULT_FILE" 2>&1

echo "=========================================="
echo "DEBUG: Current Working Directory: $(pwd)"
echo "DEBUG: Project Root: $PROJECT_ROOT"
echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SAS file requested: $sas_file"
echo "DEBUG: Checking if SAS file exists at $(pwd)/$sas_file"
ls -ld "htn-benchmarks" "htn-benchmarks/sas-files" "$sas_file" 2>/dev/null || echo "DEBUG: File or directory NOT FOUND"
echo "=========================================="

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "=========================================="
echo "Domain: $domain"
echo "Instance: $instance"
echo "Planner: $planner"
echo "Heuristic: $heuristic"
echo "Algorithm: $algorithm"
echo "Engine: $engine_name"
echo "Node Config: $node_config"
echo "SAS File: $sas_file"
echo "=========================================="

# Validate SAS file exists
if [ ! -f "$sas_file" ]; then
    echo "Error: SAS file not found: $sas_file"
    exit 1
fi

# Function to get weight parameter from algorithm
get_weight() {
    case "$1" in
        "StandardA"|"GBFS")
            echo "1"
            ;;
        "WA")
            echo "5"
            ;;
        *)
            echo "1"
            ;;
    esac
}

# Function to get g-value mode from algorithm
# Use action costs to match Pytrich (primitive operators only); GBFS ignores g-values
get_gvalue() {
    case "$1" in
        "StandardA"|"WA")
            echo "action"
            ;;
        "GBFS")
            echo "none"
            ;;
        *)
            echo "action"
            ;;
    esac
}

# Execute based on planner type
if [[ "$planner" == "Panda" ]]; then
    echo "Running Panda planner..."
    weight=$(get_weight "$algorithm")
    gvalue=$(get_gvalue "$algorithm")
    # Fixed seed for PANDA's per-node random tie-breaker (engine default is 42;
    # passed explicitly so the log records it). Record the binary so a sweep can
    # be tied to one build.
    PANDA_SEED=42
    PANDA_BIN=./PandaBuilds/pandaPIengine
    if command -v sha256sum >/dev/null 2>&1; then
        panda_hash=$(sha256sum "$PANDA_BIN" | cut -d' ' -f1)
    else
        panda_hash=$(shasum -a 256 "$PANDA_BIN" | cut -d' ' -f1)
    fi
    echo "Panda binary: $PANDA_BIN"
    echo "Panda binary sha256: $panda_hash"
    echo "Panda binary mtime: $(date -r "$PANDA_BIN" 2>/dev/null)"
    echo "Panda seed: $PANDA_SEED"
    echo "Panda gValue: $gvalue  weight: $weight"

    if [ "$LOCAL_MODE" = true ]; then
        "$PANDA_BIN" \
            "$sas_file" \
            -H "$heuristic" \
            -g "$gvalue" \
            -w "$weight" \
            -S "$PANDA_SEED" 2>&1
    else
        srun --time=00:30:00 "$PANDA_BIN" \
            "$sas_file" \
            -H "$heuristic" \
            -g "$gvalue" \
            -w "$weight" \
            -S "$PANDA_SEED" 2>&1
    fi

    exit_code=$?

else
    # Python-based planners (Pytrich built-ins and LLM heuristics)
    echo "Running Python-based planner ($planner)..."
    PLAN_FILE="${RESULT_DIR}/${instance_clean}.plan"

    if [ "$LOCAL_MODE" = true ]; then
        python3 __main__.py \
            -en "$engine_name" \
            -dn "$domain" \
            -pn "$instance" \
            --sas_file "$sas_file" \
            $HEURISTIC_FLAG "$heuristic" \
            -N "$node_config" \
            -o "$PLAN_FILE" 2>&1
    else
        srun --time=00:30:00 python3 __main__.py \
            -en "$engine_name" \
            -dn "$domain" \
            -pn "$instance" \
            --sas_file "$sas_file" \
            $HEURISTIC_FLAG "$heuristic" \
            -N "$node_config" \
            -o "$PLAN_FILE" 2>&1
    fi

    exit_code=$?
fi

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
