from typing import Dict, List, Tuple, Optional

# Try both import styles to be robust in different runner layouts
from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic3(Heuristic):
    """
    Domain-dependent HTN heuristic for the Depots domain.

    Design overview (admissible lower bound, fast bitwise checks):
    - Goals in these instances are On(c, s) facts. In the domain, the only action that adds on(c, s)
      is Drop(h, c, s, p). Therefore, if on(c, s) is not yet true, at least one Drop is needed.
    - Preconditions of Drop require lifting(h, c) and clear(s). Reaching lifting(h, c) requires
      either Lift(h, c, s', p) or Unload(h, c, t, p). Clearing s requires lifting the crate on s.
    - We compute a per-goal local lower bound using only facts that can be checked bitwise in the
      current state:
         cost(on(c, s)) >= 1 (for the final Drop)
                           + [0 if already lifting(c) else min actions to be lifting(c)]
                           + [0 if clear(s) else 1]
      where the min actions to be lifting(c) are estimated as:
         - 0 if some lifting(h, c) is already true
         - 1 if c is in some truck (one Unload makes lifting true)
         - 1 + [1 if not clear(c) else 0] otherwise (one Lift needed, but if c is not clear,
           we need at least one extra Lift to clear whatever is on top of c first)
    - To preserve admissibility in the presence of many goals with possible shared helpful actions,
      we use the max-aggregation over all unsatisfied goal on-facts (h_max style). This is a safe
      lower bound: achieving all goals requires at least the actions required by the hardest one.
    - For the hierarchical part, the planner must also reduce the current task network (TN) to empty.
      A trivial admissible lower bound for TN is the number of primitive operators currently in the
      TN (each must be executed). We combine both requirements via max: the final plan must both
      achieve all goals and empty the TN, thus cost >= max(goal-lb, TN-lb). Using max keeps the
      heuristic admissible.

    Efficiency:
    - Pre-index facts by predicate/arguments to avoid string scans at heuristic time.
    - Only bitwise membership checks are used for current-state queries.

    Notes:
    - This heuristic is simple and scalable. It ignores travel/relocation costs (Drive, moving
      hoists), which underestimates cost but keeps the heuristic admissible and fast.
    - The parameter name is kept as requested: depots_gpt5_heuristic_3.
    """

    def __init__(self, model, depots_gpt5_heuristic_3: bool = True):
        super().__init__(model)
        self.depots_gpt5_heuristic_3 = depots_gpt5_heuristic_3

        # Indices for fast bitwise checks
        self.clear_id_by_surface: Dict[str, int] = {}
        self.lifting_ids_by_crate: Dict[str, List[int]] = {}
        self.in_ids_by_crate: Dict[str, List[int]] = {}
        self.on_id_by_pair: Dict[Tuple[str, str], int] = {}
        self.on_any_ids_by_crate: Dict[str, List[int]] = {}

        # Pre-parse all facts into indices
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate:
                continue

            if predicate == 'clear' and len(args) == 1:
                s = args[0]
                self.clear_id_by_surface[s] = fact.global_id

            elif predicate == 'lifting' and len(args) == 2:
                h, c = args
                self.lifting_ids_by_crate.setdefault(c, []).append(fact.global_id)

            elif predicate == 'in' and len(args) == 2:
                c, t = args
                self.in_ids_by_crate.setdefault(c, []).append(fact.global_id)

            elif predicate == 'on' and len(args) == 2:
                c, s = args
                self.on_id_by_pair[(c, s)] = fact.global_id
                self.on_any_ids_by_crate.setdefault(c, []).append(fact.global_id)

        # Cache goals: we only care about on-goals (domain-specific)
        self.goal_on_pairs: List[Tuple[str, str]] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, gargs = self._parse_fact_name(gname)
            if pred == 'on' and len(gargs) == 2:
                self.goal_on_pairs.append((gargs[0], gargs[1]))

        # Try to cache a way to recognize Operator instances robustly
        self._operator_sample_type = type(self.model.operators[0]) if getattr(self.model, 'operators', None) and len(self.model.operators) > 0 else None

    # ----------------------------- Core heuristic API -----------------------------
    def h(self, state_bitwise: int, task_network: List) -> int:
        # Lower bound from goals (on-facts)
        goal_lb = 0
        for (c, s2) in self.goal_on_pairs:
            cost_cs = self._cost_for_goal_on(state_bitwise, c, s2)
            if cost_cs > goal_lb:
                goal_lb = cost_cs

        # Lower bound from current task network: count primitive operators present
        tn_lb = 0
        if task_network:
            for task in task_network:
                if self._is_operator(task):
                    tn_lb += 1

        # Admissible combination: both goals must be true and TN must be empty
        # => cost >= max(goal_lb, tn_lb)
        return max(goal_lb, tn_lb)

    # ----------------------------- Helpers -----------------------------
    def _is_operator(self, task) -> bool:
        # Prefer direct isinstance check if we have a sample operator type
        if self._operator_sample_type is not None:
            try:
                return isinstance(task, self._operator_sample_type)
            except Exception:
                pass
        # Fallback: heuristic structural check
        return hasattr(task, 'apply') and hasattr(task, 'add_effects') and hasattr(task, 'del_effects') and hasattr(task, 'cost')

    def _is_true(self, state_bitwise: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    def _any_true(self, state_bitwise: int, ids: Optional[List[int]]) -> bool:
        if not ids:
            return False
        for fid in ids:
            if ((state_bitwise >> fid) & 1) == 1:
                return True
        return False

    def _cost_for_goal_on(self, state_bitwise: int, c: str, s2: str) -> int:
        # If already satisfied, cost is 0
        gid = self.on_id_by_pair.get((c, s2))
        if gid is not None and self._is_true(state_bitwise, gid):
            return 0

        # Final Drop is necessary to achieve on(c, s2)
        need_drop = 1

        # Need to be holding c (lifting(h, c)) before Drop
        if self._any_true(state_bitwise, self.lifting_ids_by_crate.get(c)):
            need_lift_c = 0
        else:
            # If c is in some truck, one Unload is enough to make lifting true
            if self._any_true(state_bitwise, self.in_ids_by_crate.get(c)):
                need_lift_c = 1
            else:
                # Otherwise, at least one Lift is needed; but if c is not clear, we need
                # at least one extra Lift to clear its top first.
                need_lift_c = 1
                clear_c_id = self.clear_id_by_surface.get(c)
                if clear_c_id is not None and not self._is_true(state_bitwise, clear_c_id):
                    need_lift_c += 1

        # Drop requires clear(s2). If s2 is not clear, at least one Lift (of its top crate) is needed
        need_clear_s2 = 0
        clear_s2_id = self.clear_id_by_surface.get(s2)
        if clear_s2_id is not None and not self._is_true(state_bitwise, clear_s2_id):
            need_clear_s2 = 1

        # Local admissible lower bound for this on-goal
        return need_drop + need_lift_c + need_clear_s2

    @staticmethod
    def _parse_fact_name(name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []


# Alias requested parameter name remains in __init__: depots_gpt5_heuristic_3
# Class name is DepotsGPT5Heuristic3, inheriting from Heuristic.
