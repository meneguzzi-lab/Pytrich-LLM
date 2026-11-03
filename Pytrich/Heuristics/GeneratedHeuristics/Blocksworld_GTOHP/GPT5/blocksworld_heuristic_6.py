import re
from typing import Dict, Tuple, List, Optional, Iterable, Union

try:
    # Expected location in Pytrich
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:  # pragma: no cover - fallback stub for standalone tests
    class Heuristic:
        def __init__(self, model):
            self.model = model
        def __call__(self, state, task_network):
            return 0


class BlocksworldGPT5Heuristic6(Heuristic):
    """
    Domain-dependent heuristic for BLOCKS (Blocksworld HDDL) tailored for Pytrich.

    High-level idea (admissible lower bound):
    - Goals are conjunctive on(x,y) facts. Any unsatisfied on(x,y) requires at least
      one stack action (lower bound +1). Moreover, if x or y is not clear in the
      current state, at least one unstack action is required beforehand to make
      them clear (each adds +1 to the lower bound). Summing over all goal pairs
      gives a safe, informative lower bound based on domain structure (stacking
      requires clear preconditions that can only be achieved by removing blockers).
    - Additionally, the hierarchical setting requires finishing the current task
      network (TN). Any primitive actions already present in the TN must still be
      executed. We add the exact count of primitive operators in the current TN as
      another lower bound. To remain admissible while considering both the goal and
      the HTN refinement, we return max(goal_lower_bound, tn_lower_bound).

    Rationale for admissibility and guidance:
    - For each unsatisfied on(x,y), at least one stack is required: +1.
    - If x is not clear, at least one unstack is required to clear x before we can
      move/stack it: +1.
    - If y is not clear, at least one unstack is required to clear y before x can
      be stacked onto y: +1.
      These unstack costs are not double-counted with the stacks, and they capture
      inevitable precondition work in Blocksworld, yet remain conservative because
      we only count one unblock per endpoint per goal link.
    - Counting pending primitive operators in the TN is an exact lower bound for the
      remaining plan cost. Using max with the goal lower bound keeps the heuristic
      admissible while leveraging hierarchical progress.

    Implementation details:
    - Uses bitwise checks for state membership (fast on large instances).
    - Precomputes mappings from grounded facts to global ids for quick lookup.
    - Extracts and caches goal on(x,y) facts at initialization to avoid recomputing.
    - Robustly iterates over the task network: supports both ids and component
      objects using model.get_component when needed.

    Notes:
    - We intentionally avoid assigning fixed costs to abstract tasks (do_put_on,
      do_clear, do_on_table, do_move) because some have zero-cost decompositions
      (NOP methods) when already satisfied in the state. Without safely parsing
      their grounded arguments, any positive constant could make the heuristic
      inadmissible. Instead, we rely on pending primitive operators in the TN plus
      the goal-based bound above.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_6: Optional[dict] = None):
        super().__init__(model)
        self.cfg = blocksworld_gpt5_heuristic_6 or {}

        # Cache: predicate -> arg tuple(s) -> global_id
        self.on_gid_map: Dict[Tuple[str, str], int] = {}
        self.clear_gid_map: Dict[str, int] = {}
        self.ontable_gid_map: Dict[str, int] = {}

        # Build fact dictionaries
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            if pred == 'on' and len(args) == 2:
                self.on_gid_map[(args[0], args[1])] = f.global_id
            elif pred == 'clear' and len(args) == 1:
                self.clear_gid_map[args[0]] = f.global_id
            elif pred == 'ontable' and len(args) == 1:
                self.ontable_gid_map[args[0]] = f.global_id

        # Extract goal on(x,y) facts and pre-resolve the relevant GIDs for quick checks
        self.goal_items: List[Tuple[int, int, int]] = []  # list of (on_gid, clear_x_gid_or_-1, clear_y_gid_or_-1)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'on' and len(args) == 2:
                x, y = args
                on_gid = self.on_gid_map.get((x, y), -1)
                cx_gid = self.clear_gid_map.get(x, -1)
                cy_gid = self.clear_gid_map.get(y, -1)
                if on_gid != -1:
                    self.goal_items.append((on_gid, cx_gid, cy_gid))
        # Done. All future checks are O(1) bit tests.

    # ------------------------ Public interface ------------------------
    def __call__(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        # Lower bound from the goal (unsatisfied on + required precondition clears)
        goal_lb = self._goal_lower_bound(state_bitwise)
        # Lower bound from the current task network (count of pending primitive ops)
        tn_lb = self._tn_lower_bound(task_network)
        # Admissible combination: both must be achieved, so take max
        return max(goal_lb, tn_lb)

    # ------------------------ Core computations ------------------------
    def _goal_lower_bound(self, state_bitwise: int) -> int:
        lb = 0
        for on_gid, cx_gid, cy_gid in self.goal_items:
            if not self._is_true(state_bitwise, on_gid):
                # At least one stack to make on(x,y)
                cost = 1
                # If x is not clear, at least one unstack to clear it first
                if cx_gid != -1 and not self._is_true(state_bitwise, cx_gid):
                    cost += 1
                # If y is not clear, at least one unstack to clear it first
                if cy_gid != -1 and not self._is_true(state_bitwise, cy_gid):
                    cost += 1
                lb += cost
        return lb

    def _tn_lower_bound(self, task_network: List[Union[int, object]]) -> int:
        # Count only primitive operators (each will cost at least 1 to execute).
        # This is an exact LB for executing remaining primitive tasks already present in the TN.
        if not task_network:
            return 0
        count = 0
        for comp in self._iter_tn_components(task_network):
            # Strictly count Operators; AbstractTasks are ignored unless expanded
            from_component = getattr(comp, '__class__', type(comp)).__name__
            if from_component == 'Operator':
                count += 1
        return count

    # ------------------------ Utilities ------------------------
    def _iter_tn_components(self, task_network: Iterable[Union[int, object]]) -> Iterable[Union[object, int]]:
        for item in task_network:
            if isinstance(item, int):
                try:
                    yield self.model.get_component(item)
                except Exception:
                    # If the TN already holds objects, just yield the id as-is (won't be counted)
                    yield item
            else:
                yield item

    @staticmethod
    def _is_true(state_bitwise: int, gid: int) -> bool:
        return ((state_bitwise >> gid) & 1) == 1

    @staticmethod
    def _parse_fact_name(name: str):
        """
        Parse grounded fact name in Pytrich format: +predicate[arg1,arg2] or -predicate[arg1,arg2].
        Returns (predicate, [args]) or (None, []).
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []


# Optional convenience alias (if the framework expects a factory function)
def blocksworld_gpt5_heuristic_6(model):
    """Factory to instantiate the heuristic with default settings."""
    return BlocksworldGPT5Heuristic6(model)
