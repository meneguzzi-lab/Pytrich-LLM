from typing import Dict, Tuple, List, Optional, Union

# Attempt to import the base Heuristic class from Pytrich
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Minimal fallback to allow static analysis; the planner will provide the real base class.
    class Heuristic:
        def __init__(self, model, *args, **kwargs):
            self.model = model
        def h(self, state_bitwise: int, task_network: List[Union[int, object]]):
            return 0
        def tie_breaker(self, state_bitwise: int, task_network: List[Union[int, object]]):
            return 0


class BlocksworldGPT5Heuristic18(Heuristic):
    """
    Domain-dependent heuristic for the BLOCKS world HDDL domain designed for the Pytrich Planner.

    Design goals:
    - Safe and fast bitwise checks for large instances (e.g., 1000 blocks)
    - Admissible lower bound using domain knowledge
    - Considers both the goal facts and the remaining (primitive) tasks in the task network

    Heuristic outline:
    1) Goal guidance (admissible lower bound, h_max over on-goal pairs):
       For each goal on(x,y), if not true, we compute a relaxed, optimistic action count to make it true.
       We use the following lower bound per goal pair:
         cost(on(x,y)) = 0 if on(x,y) already true
                          else 1 + max(cost_hold_x, cost_clear_y)
         where
           - cost_hold_x = 0 if holding(x) else 1 (at least one action to get to holding x under relaxation)
           - cost_clear_y = 0 if clear(y) else 1 (at least one action to clear y under relaxation)
         The final +1 is for the final stack(x,y).
       Taking the maximum over all on-goal pairs yields an admissible lower bound (h_max style),
       avoiding overestimation that would arise from summation.

    2) Task network contribution (primitive lower bound):
       Count remaining primitive operators in the current task network (excluding 'nop') as a mandatory
       number of steps still to execute. This yields another admissible lower bound (they must be applied).

    3) Final heuristic value:
       h(state, tn) = max(h_goals_max, primitive_ops_remaining)

    Tie-breaking:
       Prefer states that already satisfy more goal on-facts: tie_breaker = -satisfied_goal_count.

    Implementation details:
    - All state membership checks are done through bitwise operations on fact.global_id.
    - We pre-index facts by predicate and arguments for O(1) access per check.
    - We parse model.goals once at initialization to identify relevant '+on[x,y]' pairs.

    Notes on admissibility:
    - h_goals_max is a standard h_max-like estimate over individual on-goal lower bounds -> admissible.
    - primitive_ops_remaining is a hard lower bound: all non-nop primitive operators in the task network must execute.
    - Combining by max keeps the overall heuristic admissible.

    This heuristic is intentionally simple to keep per-node computation fast while still providing
    meaningful guidance on both small and very large instances.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_18: bool = True):
        super().__init__(model)
        self.model = model

        # Indexing structures for fast lookups
        self.fact_by_name: Dict[str, object] = {}
        self.on_index: Dict[Tuple[str, str], object] = {}
        self.clear_index: Dict[str, object] = {}
        self.ontable_index: Dict[str, object] = {}
        self.holding_index: Dict[str, object] = {}
        self.handempty_fact: Optional[object] = None

        # Build indices over all facts
        for f in getattr(self.model, 'facts', []):
            self.fact_by_name[f.name] = f
            pred, args = self._parse_fact_name(f.name)
            if not pred or not f.name.startswith('+'):
                continue
            if pred == 'on' and len(args) == 2:
                self.on_index[(args[0], args[1])] = f
            elif pred == 'clear' and len(args) == 1:
                self.clear_index[args[0]] = f
            elif pred == 'ontable' and len(args) == 1:
                self.ontable_index[args[0]] = f
            elif pred == 'holding' and len(args) == 1:
                self.holding_index[args[0]] = f
            elif pred == 'handempty':
                self.handempty_fact = f

        # Parse and cache goal on-facts once
        self.goal_on_items: List[Tuple[Optional[object], Optional[object], Optional[object]]] = []
        # Each tuple: (on_fact(x,y), holding_fact(x), clear_fact(y))
        try:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        except Exception:
            goal_fact_names = []
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'on' and len(args) == 2 and name.startswith('+'):
                x, y = args[0], args[1]
                on_f = self.on_index.get((x, y))
                hold_f = self.holding_index.get(x)
                clr_y = self.clear_index.get(y)
                self.goal_on_items.append((on_f, hold_f, clr_y))

    # -----------------------------
    # Public Heuristic API
    # -----------------------------
    def h(self, state_bitwise: int, task_network: Optional[List[Union[int, object]]] = None) -> int:
        # Component 1: h_goals_max over goal on-facts
        h_goals = 0
        for on_fact, hold_fact_x, clear_fact_y in self.goal_on_items:
            if on_fact is None:
                # If we cannot find the fact in the model (shouldn't happen), skip safely
                continue
            if self._is_true(state_bitwise, on_fact):
                continue
            # Compute lower bound cost to achieve on(x,y)
            # cost = 1 (final stack) + max(cost_hold_x, cost_clear_y)
            cost_hold_x = 0
            if hold_fact_x is None or not self._is_true(state_bitwise, hold_fact_x):
                cost_hold_x = 1
            cost_clear_y = 0
            if clear_fact_y is None or not self._is_true(state_bitwise, clear_fact_y):
                cost_clear_y = 1
            cost_here = 1 + max(cost_hold_x, cost_clear_y)
            if cost_here > h_goals:
                h_goals = cost_here

        # Component 2: hard lower bound on remaining primitive operators in the task network
        primitive_ops_remaining = 0
        if task_network:
            for comp in self._iter_task_network_components(task_network):
                if self._is_operator(comp):
                    # Count as 1 step unless this is the 'nop' action
                    name_l = str(getattr(comp, 'name', '')).lower()
                    if not name_l.startswith('nop'):
                        primitive_ops_remaining += 1

        # Final admissible heuristic: max of the two lower bounds
        return max(h_goals, primitive_ops_remaining)

    def tie_breaker(self, state_bitwise: int, task_network: Optional[List[Union[int, object]]] = None) -> int:
        # Prefer states that satisfy more goal on-facts (more negative is better)
        satisfied = 0
        for on_fact, _, _ in self.goal_on_items:
            if on_fact is not None and self._is_true(state_bitwise, on_fact):
                satisfied += 1
        return -satisfied

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        name = name.strip()
        if name.startswith('+') or name.startswith('-'):
            name_ = name[1:]
            bracket_idx = name_.find('[')
            if bracket_idx > 0 and name_.endswith(']'):
                predicate = name_[:bracket_idx]
                args_str = name_[bracket_idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _is_operator(comp: object) -> bool:
        # Operators in model expose 'apply' and 'add_effects' attributes; decompositions/abstract tasks do not.
        return hasattr(comp, 'apply') and hasattr(comp, 'add_effects')

    def _get_component(self, cid_or_obj: Union[int, object]) -> object:
        if isinstance(cid_or_obj, int):
            return self.model.get_component(cid_or_obj)
        return cid_or_obj

    def _iter_task_network_components(self, task_network: List[Union[int, object]]):
        for item in task_network:
            yield self._get_component(item)

    @staticmethod
    def _is_true(state_bitwise: int, fact_obj: Optional[object]) -> bool:
        if fact_obj is None:
            return False
        gid = getattr(fact_obj, 'global_id', None)
        if gid is None:
            return False
        return ((state_bitwise >> gid) & 1) == 1
