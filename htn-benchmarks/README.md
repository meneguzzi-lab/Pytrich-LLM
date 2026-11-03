# Benchmarks

The seven total-order HTN domains of the paper, from the IPC 2020 track (Barman-BDI,
Blocksworld-GTOHP, Depots, Robot, Rover-GTOHP, Towers) and the IPC 2023 track (Lamps),
as distributed in [schererl/htn-benchmarks](https://github.com/schererl/htn-benchmarks).

| Folder | Content |
| ------ | ------- |
| `total-order/<Domain>/` | `domain.hddl` and the problem files (`.hddl`; `.pddl` for Lamps), 180 instances in all |
| `panda-htn-files/` | `<Domain>-<instance>.htn`, the output of `pandaPIparser` for every instance (with parameter splitting, the parser's default) |
| `sas-files/` | `<Domain>-<instance>.psas`, grounded for Pytrich (`pandaPIgrounder -q -D -e`); generated, not committed |
| `panda-sas-files/` | `<Domain>-<instance>.sas`, grounded for PANDA (`pandaPIgrounder -q`); generated, not committed |

```bash
bash htn-benchmarks/ground_benchmarks.sh          # all 180 instances
bash htn-benchmarks/ground_benchmarks.sh Rover    # one domain
```

Grounding is deterministic from the `.htn` files, so the generated files are byte-identical
to the ones the experiments ran on. The `.htn` files are committed rather than regenerated
because not every parser build applies parameter splitting; the grounder has no such
variation.
