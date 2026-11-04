# <img src="Icon/pytrich-icon.png" alt="Pytrich" width="36" style="border-radius: 50%;"/> Pytrich-LLM: HTN Planning with LLM-Generated Heuristics

[![DOI](https://zenodo.org/badge/1444167963.svg)](https://doi.org/10.5281/zenodo.23931424)

Code, generated heuristics, benchmarks and results for the paper

> **Hierarchical Task Network Planning with LLM-Generated Heuristics**
> Felipe Meneguzzi, Alexandre Buchweitz, Augusto B. Corrêa, Victor Scherer Putrich and André Grahl Pereira.
> *Advances in Neural Information Processing Systems 39 (NeurIPS 2026).*

Hierarchical Task Network (HTN) planning decomposes high-level tasks through a method library until only executable actions remain.
Domain-independent HTN heuristics are still far less informative than their classical-planning counterparts.
Following the methodology of Corrêa, Pereira and Seipp ([Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code](https://arxiv.org/abs/2503.18809), NeurIPS 2025), we ask large language models to write domain-specific heuristics as Python classes for the [Pytrich](https://github.com/schererl/Pytrich) total-order HTN planner, generate twenty candidates per model and domain, select on the smallest instance only, and evaluate the result on seven standard total-order benchmark domains (the six IPC 2020 domains Barman, Blocksworld, Depots, Robot, Rover and Towers, plus Lamps from IPC 2023; 180 instances).
We compare against the TDG and LMCount domain-independent heuristics in the same planner and against the [PANDA](https://panda-planner-dev.github.io/) planner with four heuristics, under A\*, greedy best-first search and weighted A\* (w = 5), with 8 GB of memory and 30 minutes per run on one core.

The headline numbers of the paper, on all 180 instances (coverage is the number of instances solved by any of the three search algorithms):

| System | Barman | Blocksworld | Depots | Robot | Rover | Towers | Lamps | Total |
| ------ | -----: | ----------: | -----: | ----: | ----: | -----: | ----: | ----: |
| PANDA rc2(ff) | 18 | 29 | 27 | 20 | 26 | 14 | 19 | 153 |
| PANDA rc2(lmc) | 15 | 26 | 23 | 20 | 19 | 13 | 18 | 134 |
| PANDA rc2(add) | 16 | 30 | 22 | 20 | 27 | 14 | 19 | 148 |
| PANDA LMCount | 18 | 30 | 29 | 20 | 20 | 14 | 15 | 146 |
| Pytrich TDG | 18 | 20 | 23 | 11 | 30 | 15 | 15 | 132 |
| Pytrich LMCount | 19 | 30 | 27 | 11 | 16 | 15 | 12 | 130 |
| Claude Opus 4.6 (best single heuristic) | 20 | 28 | 30 | 11 | 30 | 15 | 14 | 148 |
| LLM, one model and algorithm fixed per domain | 18 | 28 | 26 | 11 | 30 | 15 | 15 | 143 |
| LLM virtual best (five models) | 20 | 29 | 30 | 11 | 30 | 15 | 20 | 155 |

On the instances both sides solve, the LLM virtual best expands fewer nodes than TDG on 107 of 132 instances (median node ratio 0.74) and than LMCount on 126 of 129 (0.46).
Every table and figure of the paper is built from `Experiments/results/results.csv` and `Experiments/results/ablation.csv` by the scripts in this repository; Section 4 below regenerates them, and Section 3 re-runs the experiments that produced the two files.

## 1. What is in the repository

| Path | Content |
| ---- | ------- |
| `__main__.py`, `Pytrich/` | The Pytrich planner: grounding through PANDA, A\*-family search, the TDG, LMCount and Novelty heuristics |
| `Pytrich/Heuristics/GeneratedHeuristics/<Domain>/<Model>/` | Every generated heuristic evaluated in the paper, exactly as the model returned it (including candidates that fail to import or crash): nine models on the six IPC 2020 domains, three on Lamps, and the three prompt-ablation arms (`Gemini3Flash-arm*`) |
| `Pytrich/Heuristics/GeneratedExamples/` | The worked examples the prompt shows (one per domain, plus Snake for ablation arm 3 on Towers) |
| `Pytrich/constants.py` | Registry of the generated heuristics that import, keyed `<domain>_<model>_heuristic_<n>` |
| `generate_heuristic.py`, `templates/`, `config/` | The heuristic generator: prompt template, domain hints and worked examples (`config/domains.yaml`), model and API settings (`config/models.yaml`), systems and domains of the sweep (`config/planners.yaml`) |
| `htn-benchmarks/` | The seven domains in HDDL and the parser output of every instance, plus the script that grounds them |
| `PandaBuilds/` | Build script and pinned sources for the PANDA parser, grounder and engine (binaries are built, not committed) |
| `JobArrays/` | SLURM job-array scripts: task-list generator, the sbatch template, the submission script and the timing-run task list |
| `Experiments/python_scripts/` | Log parser, plan validator, comparison against the record, ablation tables, code-validity measurement, setup check; `paper/` holds the scripts for every table and figure of the paper |
| `Experiments/results/` | The dataset of record (Section 5) |

## 2. Setup

Python 3.10 or newer (some generated candidates use `X | None` annotations), `pip install -r requirements.txt`.
Only `pyyaml` and `psutil` are needed to run the planner and the job scripts; `requests` and `python-dotenv` are for generating heuristics, and `pandas`, `numpy`, `matplotlib`, `seaborn` and `tabulate` for the tables and figures.

**PANDA tools.**
The planner grounds through PANDA's parser and grounder, and the PANDA baselines run its engine.
Build the three from source (pinned commits, a few minutes; needs a C++17 compiler, cmake, flex, bison, gengetopt and zip):

```bash
bash PandaBuilds/build_panda.sh
```

See `PandaBuilds/README.md` for the sources, the requirements per platform, and what a rebuilt engine does and does not reproduce.

**Grounded benchmarks.**
The experiments read pre-grounded instances, which are large and not committed.
Generate them once from the committed parser output; the output is byte-identical to the files used for the paper:

```bash
bash htn-benchmarks/ground_benchmarks.sh      # 180 instances, a few minutes
```

**Check everything.**

```bash
python3 Experiments/python_scripts/verify_setup.py --planner-only
```

It checks the Python version and packages, the PANDA binaries for this platform, the benchmarks, and then solves Blocksworld p01 with TDG.
A non-zero exit status means something above is missing, and each failure line prints the fix.

**Quick start.** Solve an instance with TDG, with a registered generated heuristic, and with a heuristic file:

```bash
python3 __main__.py htn-benchmarks/total-order/Blocksworld-GTOHP/domain.hddl \
    htn-benchmarks/total-order/Blocksworld-GTOHP/p01.hddl -S "Astar()" -H "TDG()"

python3 __main__.py --sas_file htn-benchmarks/sas-files/Towers-pfile_01.psas \
    -S "Astar()" -N "AstarNode(G=0,H=1)" -H "towers_gemini3flash_heuristic_0()"

python3 __main__.py --sas_file htn-benchmarks/sas-files/Towers-pfile_01.psas \
    -S "Astar()" -N "AstarNode(G=0,H=1)" \
    --heuristic-file Pytrich/Heuristics/GeneratedHeuristics/Towers/ClaudeOpus46/towers_heuristic_0.py
```

`-N "AstarNode(G=g,H=h)"` selects the search: `G=1,H=1` is A\*, `G=0,H=1` is greedy best-first search and `G=1,H=5` is weighted A\*.
The run ends with `Search Status : GOAL`, the expanded-node count and the plan.

## 3. Reproducing the experiments

Everything runs as SLURM job arrays, one benchmark instance per task, 1 core, 8 GB, 30 minutes of wall-clock enforced by `srun`, with `PYTHONHASHSEED=0` so that Pytrich's node counts are deterministic and PANDA seeded with 42.
The SLURM header of `JobArrays/run_experiment.sbatch` names the Maxwell partition and excludes the nodes that are not Intel Xeon Gold 6138, the processor every run of the record executed on; on another cluster, edit those two lines.
Every log records its node.

**Step 1. Main sweep.**
Every system of `config/planners.yaml` on every domain under the three search algorithms: 46,080 tasks in 153 job arrays (every generated candidate that imports, times every instance, plus TDG, LMCount, Novelty and the four PANDA heuristics).

```bash
cd JobArrays
./submit_all_experiments.sh --dry-run        # writes the task lists and prints the counts
./submit_all_experiments.sh                  # submits; --domain, --planner and --algorithm filter
```

Logs land in `JobArrays/outputs/<System>_<Algorithm>/<Domain>/<heuristic>/<instance>.out`, one file per task.
An empty `.out` means the task was killed for memory; a log without `Search Status` means it hit the time limit.

**Step 2. Prompt ablation** (appendix of the paper): the three Gemini 3 Flash prompt arms on the six IPC 2020 domains, 26,100 tasks.

```bash
./submit_all_experiments.sh --ablation       # logs go under JobArrays/outputs-ablation/
```

**Step 3. Parse the logs into one CSV each.**

```bash
cd ..
python3 Experiments/python_scripts/parse_all_logs_all_domains.py -i JobArrays/outputs -o results_new.csv --pattern "*.out"
```

For the ablation, parse `JobArrays/outputs-ablation` into `ablation_new.csv` the same way.
The CSV has one row per solved run: `domain_name, problem_name, experiment_name, search_elapsed_time, solution_size, expanded_nodes, nodes_per_second, heuristic_name, heuristic_elapsed_time, ...`, where `experiment_name` is `<System>_<Algorithm>` with algorithm `GBFS`, `StandardA` (A\*) or `WA` (weighted A\*).

**Step 4. Validate every plan** (Pytrich and LLM runs; PANDA verifies its own plans):

```bash
python3 Experiments/python_scripts/validate_plans.py --sas-dir htn-benchmarks/sas-files \
    --out-dir JobArrays/outputs --csv plan_validation_new.csv
```

Every plan is replayed against the grounded model: each action must be applicable in turn and the final state must contain every goal fact.
The `verdict` column reads `valid` for every solved run (`not solved` for a log without a plan, `INVALID` otherwise).

**Step 5. Compare with the record.**

```bash
python3 Experiments/python_scripts/compare_with_record.py results_new.csv
python3 Experiments/python_scripts/compare_with_record.py results_new.csv --per-run --details
python3 Experiments/python_scripts/compare_with_record.py ablation_new.csv --record Experiments/results/ablation.csv
```

The first form prints, per system, the coverage in the record and in the new file and how many of the instances both solve have the same per-instance minimum node count; `--per-run` does the same per (system, heuristic, algorithm), which is the right view for a partial sweep.
Every Pytrich and LLM node count reproduces exactly; coverage may differ by one or two instances per system for runs that finish within a few percent of the 30-minute or 8 GB limit. PANDA's node counts depend on the engine build (see `PandaBuilds/README.md`), so compare its coverage rather than its nodes.
The record itself lists the instances each system solves, so any difference can be traced to the log of that run.

**Step 6. Regenerate the paper's tables and figures** from the new CSVs (Section 4) and compare with the shipped ones.

**Timing run (optional).**
`Experiments/results/timing_gbfs.csv` holds wall-clock measurements of TDG, LMCount and each model's best greedy best-first candidate per domain on the six IPC 2020 domains.
`python3 JobArrays/timing_task_list.py --output JobArrays/TaskLists/timing_GBFS.txt` writes the 1,020 task lines, which the docstring shows how to submit under `OUTPUT_ROOT=JobArrays/outputs-timing`.

**Running one task without SLURM.**
`bash JobArrays/run_experiment.sbatch --local --task-id N <task list>` runs line N of a task list on the current machine, writing the same log layout under `JobArrays/outputs/` (or `OUTPUT_ROOT`).
Use it to smoke-test a task list before submitting it.

## 4. Regenerating the tables and figures

```bash
bash Experiments/python_scripts/paper/regenerate_tables.sh \
    [Experiments/results/results.csv] [Experiments/results/ablation.csv] [Experiments/outputs/paper]
```

With the shipped CSVs this writes, under the output directory: the coverage table, the two plan-length tables and the per-algorithm median-node table as LaTeX (`coverage_table.tex`, `planlength_table.tex`, `planlength_intersection_table.tex`, `median_nodes_by_algo.tex`, identical to the paper's), every number quoted in the results section (`results_numbers.md`), the fixed-configuration tables (`fixed_pair/`, `fixed_config.txt`, `nodes_fixed_vs_pytrich.md`), the coverage with the prompt-context instances removed (`prompt_context_excluded.md`), the prompt-ablation tables of the appendix (`ablation_tables.md`), and the fourteen comparison figures against each baseline (`figures/`).
Each script can also be run on its own; `python3 <script> --help` or its docstring gives the options.

The code-validity table of the appendix (how many candidates per ablation arm fail to compile or crash at run time) is measured by running every candidate on one small instance of its domain:

```bash
python3 Experiments/python_scripts/measure_heuristic_validity.py --summary-only
```

## 5. The dataset of record (`Experiments/results/`)

| File | Rows | Content |
| ---- | ---: | ------- |
| `results.csv` | 23,497 | Every solved run of every system of the main comparison on the 180 instances. Only solved runs appear; an instance a system has no row for is unsolved. The Pytrich baselines (`Pytrich-TDG`, `Pytrich-LMCOUNT`, `Pytrich-NOVELTY`) carry no algorithm suffix, up to three rows per instance. Claude Opus 4.6 rows use the key `ClaudeOpus46`; the `heuristic_name` of a candidate run by file is its class name. |
| `ablation.csv` | 14,064 | The three Gemini 3 Flash prompt arms (`Gemini3Flash-arm1-no-example`: no hints and a stub example; `arm2-no-example-hints`: the hints restored; `arm3-fixed-example`: a cross-domain worked example restored) under the three algorithms on the six IPC 2020 domains. |
| `plan_validation_asrun.csv`, `plan_validation_hw6138.csv` | 72,932 and 5,090 | The validator's verdict for every Pytrich and LLM log behind the two files above, one row per log (`batch` names the sweep: `main`, `lamps`, `rover-large` for the three largest Rover instances, `rerun` for the runs repeated under the fixed goal test, `ablation`, and `hw6138` for the runs repeated on the uniform hardware). Every run in the two files is `valid`; the `INVALID` rows are runs whose plan failed the check under an earlier goal test of the planner, which were dropped and re-run after the fix. |
| `timing_gbfs.csv` | 758 | Wall-clock timing of TDG, LMCount and each model's best greedy best-first candidate per domain. |
| `api_costs.csv` | 557 | Tokens and cost of the generation calls of the Claude Opus 4.6, Gemini 3 Flash (including the ablation arms) and Gemini 2.5 Pro Lamps batches. |

Every run in `results.csv` and `ablation.csv` executed on one core of an Intel Xeon Gold 6138 under the limits above.
Node counts of Pytrich runs are deterministic under `PYTHONHASHSEED=0`; PANDA runs use seed 42 and an engine compiled on the cluster (SHA-256 starting `24103830`, recorded in every PANDA log).
Plan length is counted in primitive actions on every side.
The plan validator checks action applicability and the goal; it does not check that a plan refines the initial task network.

Model keys in the CSVs: `ClaudeOpus46` (Claude Opus 4.6, `claude-opus-4-6`), `ClaudeSonnet45`, `Gemini20Flash`, `Gemini25Pro`, `Gemini3Flash`, `Gemini3Pro`, `ChatGPT4o`, `GPT5`, `GPT52`; baselines `Pytrich-TDG`, `Pytrich-LMCOUNT` (bidirectional landmarks, `LMCOUNT(use_bid=True)`), `Panda-rc2ff`, `Panda-rc2lmc`, `Panda-rc2add`, `Panda-lmcountlmtypebid`.
Claude Sonnet 4.5 and GPT-4o were run on Depots only; GPT-5 and GPT-5.2 produced heuristics that run on Towers only; Gemini 3 Pro's Depots candidates all fail at run time; Lamps was run with Claude Opus 4.6, Gemini 3 Flash and Gemini 2.5 Pro.

## 6. Generating heuristics

Generation needs an API key, exported or placed in a `.env` file at the repository root (`GEMINI_API_KEY`, `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`); evaluating the shipped heuristics needs none.
Twenty independent one-shot candidates for one domain and model:

```bash
python3 generate_heuristic.py --domain towers --model gemini3flash --batch 20
```

This writes `Pytrich/Heuristics/GeneratedHeuristics/Towers/Gemini3Flash/towers_heuristic_<n>.py`, appends the prompt that produced each file as a comment block at its end, registers every candidate that imports in `Pytrich/constants.py`, and logs tokens and cost to `Experiments/results/api_costs.csv`.
The prompt (`templates/prompt_template.py`) contains the domain HDDL, the smallest and largest instance, the domain's hint block (`config/domains.yaml`), the Pytrich heuristic interface, a worked example heuristic, and design guidance.
`--dry-run` previews the prompt and `--output-prompt FILE` writes it out, both without an API call; `--list-models` and `--list-domains` show what is configured.
`--thinking {off,low,medium,high}` and `--temperature` map onto each provider's controls (`config/models.yaml`); the paper's batches used the provider defaults, and the ablation arms `--thinking medium --temperature 1.0`.
`--count N` runs an iterative loop instead of independent samples: generate, test on the smallest instance against TDG, feed the result back, repeat.

The ablation arms vary one prompt input each relative to arm 1:

```bash
python3 generate_heuristic.py -d barman -m gemini3flash --batch 20 --thinking medium --temperature 1.0 \
    --example stub --hints none --run-tag arm1-no-example
python3 generate_heuristic.py -d barman -m gemini3flash --batch 20 --thinking medium --temperature 1.0 \
    --example stub --hints full --run-tag arm2-no-example-hints
python3 generate_heuristic.py -d barman -m gemini3flash --batch 20 --thinking medium --temperature 1.0 \
    --example fixed:Pytrich/Heuristics/GeneratedExamples/towers_heuristic_example.py --hints none \
    --run-tag arm3-fixed-example        # Towers itself takes snake_heuristic.py
```

Arm folders (`<Model>-<tag>`) are not registered in `constants.py`; the job scripts run them by file path.
Note that `gemini-2.0-flash` and `gemini-3-pro-preview` have since been retired from the Gemini API, so those two models' heuristics cannot be regenerated; the shipped files are the ones evaluated.

## Citation

```bibtex
@inproceedings{MeneguzziBuchweitzCorreaEtAl2026,
  title     = {Hierarchical Task Network Planning with {LLM}-Generated Heuristics},
  author    = {Meneguzzi, Felipe and Buchweitz, Alexandre and Corr{\^e}a, Augusto B. and Putrich, Victor Scherer and Pereira, Andr{\'e} Grahl},
  booktitle = {Advances in Neural Information Processing Systems 39 (NeurIPS 2026)},
  year      = {2026}
}
```

The repository itself is archived on Zenodo under the concept DOI [10.5281/zenodo.23931424](https://doi.org/10.5281/zenodo.23931424), which always resolves to the latest release; the entry below cites the version the paper reports:

```bibtex
@misc{MeneguzziBuchweitzCorreaEtAl2026Software,
  title     = {{Pytrich-LLM}: Code, Generated Heuristics, Benchmarks and Results of ``Hierarchical Task Network Planning with {LLM}-Generated Heuristics''},
  author    = {Meneguzzi, Felipe and Buchweitz, Alexandre and Corr{\^e}a, Augusto B. and Putrich, Victor Scherer and Pereira, Andr{\'e} Grahl},
  publisher = {Zenodo},
  year      = {2026},
  version   = {1.0.3},
  doi       = {10.5281/zenodo.23957133}
}
```

## Licence and acknowledgements

MIT licence (see `LICENSE`).
Pytrich is the planner of Victor Scherer Putrich ([schererl/Pytrich](https://github.com/schererl/Pytrich)); the benchmark domains come from the IPC 2020 and 2023 HTN tracks via [schererl/htn-benchmarks](https://github.com/schererl/htn-benchmarks); PANDA is by the [panda-planner-dev](https://github.com/panda-planner-dev) group.
