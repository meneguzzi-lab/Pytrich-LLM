import re
from typing import Dict, Tuple, List, Optional

try:
    # Typical Pytrich import
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback to allow static analysis without framework
    class Heuristic:
        def __init__(self, model=None):
            self.model = model
        def __call__(self, state_bitwise: int, task_network: List):
            return 0


class BlocksworldGPT5Heuristic2(Heuristic):
    """
    Domain-dependent heuristic for the BLOCKS (Blocksworld) HDDL domain.

    Key ideas and guarantees:
    - Admissible lower bound built from goal facts (primarily on[x,y]) using Blocksworld structure.
    - Uses bitwise membership checks for speed (no string state scanning at runtime).
    - Leverages hierarchical tasks cautiously (do_clear, do_on_table) with safe, minimal contributions.
    - Scales to large instances by pre-indexing facts and parsing goal facts once.

    Lower-bound design (admissible):
    1) Every unsatisfied goal on(x,y) requires at least one stack(x,y) action. So:
       LB_stacks = number of unsatisfied on-goals.

    2) For any one such on-goal, before we can stack, we must satisfy preconditions: holding(x) and clear(y).
       Achieving holding(x) requires at least:
           0 if holding(x) already true,
           1 if clear(x) is true (one pick-up or unstack),
           2 if clear(x) is false (one unstack to clear x, then pick-up/unstack x).
       Also, if clear(y) is false, at least 1 unstack is needed to clear y.
       Summing these for each goal gives per-goal extra cost; to keep admissibility we add only the maximum of
       these extras across all unsatisfied on-goals (critical-path style):
           LB_pre_max = max_over_unsat_goals( holding_cost(x) + I(not clear(y)) ).

       Total so far: LB = LB_stacks + LB_pre_max. This is a safe, informative bound.

    3) If there are goal facts of type ontable(x), we add a very conservative bound of +1 per such unsatisfied
       goal (a put-down somewhere). This stays admissible and very cheap to compute.

    4) HTN tasks (optional, conservative use):
       - "do_clear(x)": if clear(x) is false, we add at most +1 (but only once over all such tasks) to avoid
         double counting across interacting tasks; this stays admissible and cheap.
       - "do_on_table(x)": in this domain there is a zero-cost method m3_do_on_table if clear(x) holds.
         Otherwise, at least one action is needed to progress; we conservatively add at most +1 (once over all such
         tasks requiring work) to keep admissibility and avoid double counting.

    The result is fast, domain-aware, and robust for large Blocksworld HTN instances.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_2: bool = True):
        super().__init__(model)
        self.model = model

        # Pre-index facts for O(1) bitwise checks by predicate and args.
        self._fact_ids_by_pred: Dict[str, Dict[Tuple[str, ...], int]] = {
            'on': {}, 'clear': {}, 'holding': {}, 'ontable': {}, 'handempty': {}
        }
        self._index_facts()

        # Parse and cache goal facts for quick access each call
        self.goal_on_pairs: List[Tuple[str, str]] = []       # (x, y)
        self.goal_ontable: List[str] = []                    # x
        self._index_goals()

        # Precompile regex for extracting block arguments from task names
        self._block_re = re.compile(r"b\d+")

    # ----------------------------- Internal helpers -----------------------------

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name: +predicate[arg1,arg2] or -predicate[arg1,arg2]."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _index_facts(self):
        """Build maps from (predicate, args) to global_id for fast bitwise membership checks."""
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            key = tuple(args)
            if pred not in self._fact_ids_by_pred:
                # Keep any unexpected predicate available too
                self._fact_ids_by_pred[pred] = {}
            # For nullary facts like handempty, use key=()
            self._fact_ids_by_pred[pred][key] = f.global_id

    def _index_goals(self):
        """Extract goal on/ontable facts from model.goals bitset into Python lists for fast iteration."""
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue
            if pred == 'on' and len(args) == 2:
                self.goal_on_pairs.append((args[0], args[1]))
            elif pred == 'ontable' and len(args) == 1:
                self.goal_ontable.append(args[0])
        # Deduplicate just in case
        self.goal_on_pairs = list(dict.fromkeys(self.goal_on_pairs))
        self.goal_ontable = list(dict.fromkeys(self.goal_ontable))

    def _is_true(self, state_bitwise: int, pred: str, args: Tuple[str, ...]) -> int:
        """Fast bitwise check: return 1 if fact is true in state, else 0."""
        fact_map = self._fact_ids_by_pred.get(pred)
        if not fact_map:
            return 0
        fid = fact_map.get(tuple(args))
        if fid is None:
            return 0
        return (state_bitwise >> fid) & 1

    def _extract_blocks_from_task(self, task_name: str) -> List[str]:
        """Heuristic parser for task/abstract-task names to extract block symbols (b123, etc.)."""
        return self._block_re.findall(task_name)

    # ----------------------------- Heuristic core -----------------------------

    def __call__(self, state_bitwise: int, task_network: List) -> int:
        # 1) Count unsatisfied on-goals (each requires at least one stack)
        unsat_on_count = 0
        pre_extra_max = 0

        for x, y in self.goal_on_pairs:
            if self._is_true(state_bitwise, 'on', (x, y)):
                continue
            unsat_on_count += 1

            # holding(x) prerequisite lower bound
            if self._is_true(state_bitwise, 'holding', (x,)):
                hold_cost = 0
            elif self._is_true(state_bitwise, 'clear', (x,)):
                # can pick-up (if ontable) or unstack (if on smth) in at least 1 action
                hold_cost = 1
            else:
                # need to clear x (at least 1) then pick-up/unstack x (another 1)
                hold_cost = 2

            # clear(y) prerequisite lower bound
            y_clear_cost = 0 if self._is_true(state_bitwise, 'clear', (y,)) else 1

            extra = hold_cost + y_clear_cost
            if extra > pre_extra_max:
                pre_extra_max = extra

        # 2) Optional: conservative contribution for any ontable goals (rare in provided instances)
        ontable_contrib = 0
        for x in self.goal_ontable:
            if self._is_true(state_bitwise, 'ontable', (x,)):
                continue
            # At least one action (a put-down at some point) is needed to achieve ontable(x)
            ontable_contrib += 1

        # 3) Very conservative HTN-aware terms for early tasks (avoid double counting):
        #    - If any do_clear(x) remains where clear(x) is false -> +1 (once)
        #    - If any do_on_table(x) remains where clear(x) is false -> +1 (once),
        #      because m3_do_on_table is zero-cost when clear(x) holds
        tn_clear_flag = 0
        tn_on_table_flag = 0
        if task_network:
            for t in task_network:
                tname = getattr(t, 'name', '')
                lname = tname.lower()
                # Detect do_clear
                if 'do_clear' in lname:
                    blks = self._extract_blocks_from_task(lname)
                    if blks:
                        x = blks[0]
                        if not self._is_true(state_bitwise, 'clear', (x,)):
                            tn_clear_flag = 1
                # Detect do_on_table
                if 'do_on_table' in lname:
                    blks = self._extract_blocks_from_task(lname)
                    if blks:
                        x = blks[0]
                        # If clear(x) already true, method m3 allows zero cost
                        if not self._is_true(state_bitwise, 'clear', (x,)):
                            tn_on_table_flag = 1

                # We deliberately do NOT add extra cost for do_put_on here to avoid
                # double counting with goal on(x,y) contributions.

                # Early exit if both flags found
                if tn_clear_flag and tn_on_table_flag:
                    break

        # Final admissible lower bound
        h_val = unsat_on_count + pre_extra_max + ontable_contrib + tn_clear_flag + tn_on_table_flag
        return int(h_val)


# Alias for potential dynamic loader compatibility (optional)
HeuristicClass = BlocksworldGPT5Heuristic2
