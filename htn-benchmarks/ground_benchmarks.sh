#!/usr/bin/env bash
# Ground every benchmark instance from the committed parser output (panda-htn-files/*.htn)
# into the two grounded formats the experiments read:
#
#   sas-files/<Domain>-<instance>.psas    pandaPIgrounder -q -D -e   (read by Pytrich)
#   panda-sas-files/<Domain>-<instance>.sas   pandaPIgrounder -q     (read by PANDA)
#
# The .htn files are the output of pandaPIparser with parameter splitting (its default)
# and are committed because the grounding is deterministic from them: the files this
# script writes are byte-identical to the ones used for the paper. Existing files are
# skipped, so the script can be re-run. The largest Rover instances take about a minute.
#
#   bash htn-benchmarks/ground_benchmarks.sh            # every instance (180)
#   bash htn-benchmarks/ground_benchmarks.sh Towers     # one domain
set -euo pipefail
cd "$(dirname "$0")"
GROUNDER=../PandaBuilds/pandaPIgrounder
[ -x "$GROUNDER" ] || { echo "grounder not found or not executable: $GROUNDER (see README, PANDA binaries)"; exit 1; }
mkdir -p sas-files panda-sas-files
pattern="${1:-}"
n=0; skipped=0
for htn in panda-htn-files/${pattern}*.htn; do
    base="$(basename "$htn" .htn)"
    psas="sas-files/$base.psas"; sas="panda-sas-files/$base.sas"
    if [ -s "$psas" ] && [ -s "$sas" ]; then skipped=$((skipped+1)); continue; fi
    echo "grounding $base"
    [ -s "$psas" ] || "$GROUNDER" -q -D -e "$htn" "$psas" > /dev/null
    [ -s "$sas" ]  || "$GROUNDER" -q "$htn" "$sas" > /dev/null
    n=$((n+1))
done
echo "grounded $n instance(s), $skipped already present"
