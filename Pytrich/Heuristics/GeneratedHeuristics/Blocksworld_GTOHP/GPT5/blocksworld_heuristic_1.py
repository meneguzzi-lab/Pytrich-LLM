import re
from typing import Dict, Tuple, List, Optional

try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback minimal base class if framework import path differs in testing.
    class Heuristic:
        def __init__(self, model, *args, **kwargs):
            self.model = model
        def evaluate(self, state_bitwise: int, task_network: List = None) -> int:
            return 0


class BlocksworldGPT5Heuristic1(Heuristic):
    """
    Blocksworld domain-dependent heuristic for Pytrich Planner.

    Design highlights (admissible lower bound):
    - Goal-driven lower bound: For every goal +on[x,y] not yet true, at least one stack(x,y) is needed.
      Additionally, to perform stack(x,y), we must hold x; if not currently holding x, we need at least one
      action to acquire it (either pick-up or unstack). Therefore:
        cost(on(x,y) not true) >= 1 (stack) + [0 if holding(x) else 1]
      This remains a valid, fast, and admissible lower bound across all search states.

    - Task-network lower bound: Count remaining primitive operators in the current task network TN.
      Each remaining primitive operator contributes at least 1 step to any plan that finishes TN.
      We ignore the 'nop' operator if present (cost 0). For abstract tasks, we do not add a cost here to avoid
      potential double counting effects; the goal-based bound already provides strong guidance.

    - Final heuristic value is the max of both lower bounds (admissible combination of admissible bounds).

    Implementation details:
    - Uses model.goals and model.state_explicit_repr to enumerate goal facts.
    - Parses facts in the grounded "+predicate[arg1,arg2]" format and builds fast dictionaries for lookups.
    - Uses bitwise checks (state >> fact.global_id) & 1 for O(1) truth tests.

    Practical behavior:
    - Scales well to very large Blocksworld instances (e.g., 1000 blocks) because it relies on simple
      precomputed mappings and constant-time state tests.
    - Provides decisive guidance on how many stack/get-holding actions remain for goal on-pairs and
      leverages primitive-action counts as the task network is refined.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_1: bool = True, debug: bool = False):
        super().__init__(model)
        self.debug = debug
        self.enabled = bool(blocksworld_gpt5_heuristic_1)

        # Internal maps for fast access
        # Fact dictionaries keyed by arguments
        self.on_fact_by_xy: Dict[Tuple[str, str], int] = {}       # (x,y) -> global_id of +on[x,y]
        self.holding_fact_by_x: Dict[str, int] = {}               # x -> global_id of +holding[x]
        self.ontable_fact_by_x: Dict[str, int] = {}               # x -> global_id of +ontable[x]
        self.clear_fact_by_x: Dict[str, int] = {}                 # x -> global_id of +clear[x]
        self.handempty_fact_id: Optional[int] = None              # global_id of +handempty

        # Name -> global_id lookup for all facts (for goal name to id resolution)
        self.fact_name_to_id: Dict[str, int] = {}

        # Pre-parse all facts in the model for fast predicates lookup
        for f in self.model.facts:
            name = f.name
            self.fact_name_to_id[name] = f.global_id
            predicate, args = self._parse_fact_name(name)
            if predicate is None:
                continue
            # Only consider positive facts for truth-checking in the heuristic
            if not name.startswith('+'):
                continue
            if predicate == 'on' and len(args) == 2:
                self.on_fact_by_xy[(args[0], args[1])] = f.global_id
            elif predicate == 'holding' and len(args) == 1:
                self.holding_fact_by_x[args[0]] = f.global_id
            elif predicate == 'ontable' and len(args) == 1:
                self.ontable_fact_by_x[args[0]] = f.global_id
            elif predicate == 'clear' and len(args) == 1:
                self.clear_fact_by_x[args[0]] = f.global_id
            elif predicate == 'handempty':
                self.handempty_fact_id = f.global_id

        # Parse goal to extract +on[x,y] facts and their IDs
        self.goal_on_ids: List[Tuple[int, str, str]] = []  # list of (fact_id, x, y)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            predicate, args = self._parse_fact_name(gname)
            if predicate == 'on' and len(args) == 2:
                # Prefer direct name lookup (exact goal fact), fallback to on_fact_by_xy by args
                fid = self.fact_name_to_id.get(gname)
                if fid is None:
                    fid = self.on_fact_by_xy.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_ids.append((fid, args[0], args[1]))

    def _parse_fact_name(self, name: str):
        """
        Parse grounded fact name of the form +predicate[arg1,arg2] or -predicate[arg1,arg2].
        Returns (predicate, [args]) or (None, []).
        """
        name = name.strip()
        if name.startswith('+') or name.startswith('-'):
            core = name[1:]
            bracket_idx = core.find('[')
            if bracket_idx > 0 and core.endswith(']'):
                predicate = core[:bracket_idx]
                args_str = core[bracket_idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _bit_is_true(state_bitwise: int, fact_global_id: int) -> bool:
        return ((state_bitwise >> fact_global_id) & 1) == 1

    def _is_holding(self, state_bitwise: int, x: str) -> bool:
        fid = self.holding_fact_by_x.get(x)
        return self._bit_is_true(state_bitwise, fid) if fid is not None else False

    def _is_goal_on_true(self, state_bitwise: int, on_fact_id: int) -> bool:
        return self._bit_is_true(state_bitwise, on_fact_id)

    def _goal_lower_bound(self, state_bitwise: int) -> int:
        """
        Lower bound from goal +on[x,y] facts:
        For each goal +on[x,y] not currently true:
          - At least 1 stack action is needed.
          - Additionally, if not currently holding x, at least 1 action is needed to acquire x (pick-up/unstack).
        Summed across all goal on-pairs.
        """
        if not self.goal_on_ids:
            return 0
        h = 0
        for on_fid, x, _y in self.goal_on_ids:
            if self._is_goal_on_true(state_bitwise, on_fid):
                continue  # already satisfied
            # +1 for the necessary stack
            h += 1
            # +1 if we still need to acquire x (not holding x now)
            if not self._is_holding(state_bitwise, x):
                h += 1
        return h

    def _primitive_tn_lower_bound(self, task_network: Optional[List]) -> int:
        """
        Lower bound from the current task network (TN): number of primitive operators pending.
        We do not add costs for abstract tasks here to avoid double counting. 'nop' operators are ignored.
        """
        if not task_network:
            return 0
        count = 0
        for t in task_network:
            # Primitive operators are instances of Operator; abstract tasks are AbstractTask.
            # We import type names from the model module namespace.
            # "isinstance" checks are safe due to shared class definitions in Pytrich.
            if t.__class__.__name__ == 'Operator':
                name = getattr(t, 'name', '') or ''
                # Ignore 'nop' operators (cost 0)
                if name.startswith('nop'):
                    continue
                count += 1
        return count

    def evaluate(self, state_bitwise: int, task_network: Optional[List] = None) -> int:
        if not self.enabled:
            return 0
        # Admissible combination via max of independent lower bounds
        h_goal = self._goal_lower_bound(state_bitwise)
        h_tn_prim = self._primitive_tn_lower_bound(task_network)
        h_val = max(h_goal, h_tn_prim)

        if self.debug:
            # Light-weight debug info (optional):
            pass
        return h_val


__all__ = [
    'BlocksworldGPT5Heuristic1'
]
