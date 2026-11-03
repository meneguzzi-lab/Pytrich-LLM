# PANDA tools

The planner and the job scripts call three PANDA binaries at the root of this folder:
`pandaPIparser` and `pandaPIgrounder` for grounding (`Pytrich/Grounder/panda_ground.py`,
`htn-benchmarks/ground_benchmarks.sh`) and `pandaPIengine`, the PANDA planner that is the
baseline of the paper (`JobArrays/run_experiment.sbatch`). Binaries are not committed;
build them from source with

```bash
bash PandaBuilds/build_panda.sh
```

which clones the three repositories under `PandaBuilds/src/` at the commits below and
copies the binaries here.

| Tool | Repository | Commit |
| ---- | ---------- | ------ |
| pandaPIparser | https://github.com/panda-planner-dev/pandaPIparser | `88c0995c72c5ce2685e9546ec153a3545977ed81` |
| pandaPIgrounder | https://github.com/panda-planner-dev/pandaPIgrounder | `4ff15b2` (with its `cpddl` and `h2-fd-preprocessor` submodules) |
| pandaPIengine | https://github.com/panda-planner-dev/pandaPIengine | `810f04388667db5e3e4f114e960a4efbb43b1ac0` |

Requirements: git, a C++17 compiler, make, cmake, flex 2.6 or newer, bison 3.5 or newer,
gengetopt and zip. On a cluster without these in the default environment, load a recent
GCC module before building and before running (the engine links `libstdc++` dynamically,
so the same module must be loaded in the job script's environment; `sbatch` passes the
submitting shell's environment to the job). The grounder is built with GCC on Linux; its
bundled `cpddl` library does not compile with Apple clang as pinned. On macOS the parser
and the engine build after `brew install bison flex gengetopt cmake` (the script uses
Homebrew's bison and flex when present), which is enough to run the planner on
instances grounded elsewhere.

The grounder output is deterministic across builds, so the grounded benchmarks are
byte-identical whichever machine produces them. The engine's search is not guaranteed to
be identical across compilers and revisions: the PANDA numbers of the paper come from an
engine compiled on the Maxwell cluster (SHA-256 `24103830b15ccd1b76baedef05e099085b3b889f
ff09b56226841338d34e5922`, recorded in every PANDA log), whose source revision was not
recorded. Expect a rebuilt engine to match the paper's PANDA coverage within an instance
or two per domain, and its node counts closely but not necessarily exactly; the Pytrich
and LLM results do not depend on the engine.
