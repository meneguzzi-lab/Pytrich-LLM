#!/usr/bin/env bash
# Build the three PANDA tools the experiments call and place them in this folder:
#
#   PandaBuilds/pandaPIparser     parses HDDL (Pytrich/Grounder/panda_ground.py)
#   PandaBuilds/pandaPIgrounder   grounds the parsed problem (panda_ground.py, htn-benchmarks/ground_benchmarks.sh)
#   PandaBuilds/pandaPIengine     the PANDA planner, the baseline of the paper (JobArrays/run_experiment.sbatch)
#
# Sources are cloned under PandaBuilds/src/ at the pinned commits below. Requirements:
# git, g++ with C++17, make, cmake, flex (2.6 or newer), bison (3.5 or newer), gengetopt
# and zip. On macOS, Homebrew's bison and flex must be used (brew install bison flex
# gengetopt cmake) because the system ones are too old; the script picks them up
# automatically when present.
#
#   bash PandaBuilds/build_panda.sh            # everything
#   bash PandaBuilds/build_panda.sh engine     # one tool: parser, grounder or engine
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
mkdir -p src

PARSER_REPO=https://github.com/panda-planner-dev/pandaPIparser.git
PARSER_COMMIT=88c0995c72c5ce2685e9546ec153a3545977ed81
GROUNDER_REPO=https://github.com/panda-planner-dev/pandaPIgrounder.git
GROUNDER_COMMIT=4ff15b2
ENGINE_REPO=https://github.com/panda-planner-dev/pandaPIengine.git
ENGINE_COMMIT=810f04388667db5e3e4f114e960a4efbb43b1ac0

clone() {  # clone <repo> <commit> <dir>
    if [ ! -d "src/$3/.git" ]; then git clone "$1" "src/$3"; fi
    git -C "src/$3" fetch -q origin
    git -C "src/$3" checkout -q "$2"
}

build_parser() {
    clone "$PARSER_REPO" "$PARSER_COMMIT" pandaPIparser
    local flags=()
    [ -x /opt/homebrew/opt/bison/bin/bison ] && flags+=(BISON=/opt/homebrew/opt/bison/bin/bison)
    [ -x /opt/homebrew/opt/flex/bin/flex ] && flags+=(FLEX=/opt/homebrew/opt/flex/bin/flex)
    (cd src/pandaPIparser && make -j "${flags[@]}")
    cp src/pandaPIparser/pandaPIparser "$HERE/"
}

build_grounder() {
    clone "$GROUNDER_REPO" "$GROUNDER_COMMIT" pandaPIgrounder
    (cd src/pandaPIgrounder
     git submodule update --init
     cd cpddl
     git apply --check ../0002-makefile.patch 2>/dev/null && git apply ../0002-makefile.patch
     make boruvka opts bliss lpsolve
     make
     cd ../src
     make)
    cp src/pandaPIgrounder/pandaPIgrounder "$HERE/"
}

build_engine() {
    clone "$ENGINE_REPO" "$ENGINE_COMMIT" pandaPIengine
    (cd src/pandaPIengine && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release ../src && make -j)
    cp src/pandaPIengine/build/pandaPIengine "$HERE/"
}

case "${1:-all}" in
    parser)   build_parser ;;
    grounder) build_grounder ;;
    engine)   build_engine ;;
    all)      build_parser; build_grounder; build_engine ;;
    *) echo "usage: $0 [parser|grounder|engine|all]"; exit 2 ;;
esac
echo "built:"; ls -l "$HERE"/pandaPI* 2>/dev/null | awk '{print "  " $NF}'
