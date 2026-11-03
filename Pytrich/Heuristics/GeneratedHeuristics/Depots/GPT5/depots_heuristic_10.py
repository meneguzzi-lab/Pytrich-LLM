from typing import Dict, List, Tuple, Optional, Any

from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic10(Heuristic):
    """
    Domain-dependent heuristic for the Depots HDDL domain.

    High-level idea
    ----------------
    We combine two admissible lower bounds and take the max of them:
      - Goal-based LB: estimates remaining primitive actions required to achieve all remaining goal On facts.
      - Task-network LB: estimates a lower bound on the actions still required to complete the current task network
        (both remaining primitive operators and abstract tasks under optimistic assumptions).

    We design the LB using domain knowledge:
      - To establish on(c, s):
          * If already true: 0
          * If c is currently being lifted by some hoist: at least 1 (Drop)
          * Else if c is in some truck: at least 2 (Unload + Drop)
          * Else (c is on some other support): at least 2 (Lift + Drop)
        We ignore truck driving and hoist positioning for a safe (admissible) lower bound.
      - For abstract tasks in the task network (TN):
          * Each remaining primitive operator (except 'nop') contributes at least 1.
          * do_put_on(c, s): same base as above for goals (1 if c is lifting, else 2 if goal not yet true);
            we avoid adding additional surface-clearing cost to remain conservative and efficient.
          * do_clear(s, p): at least 1 if s is not clear, else 0.
          * do_get_truck(t, p): at least 1 if truck t is not at p, else 0. (Drive is 1 step in this domain)
          * do_lift_crate(c, p, h): 0 if already lifting(h, c), else at least 1.
          * do_load_truck(c, s, p, t): at least 2 (Lift + Load); if c is already lifting by any hoist: at least 1 (Load)
          * do_unload_truck(c, s, p, t): at least 2 (Unload + Drop); if c is already lifting: at least 1 (Drop)

    We return: max(goal_LB, tasknet_LB). This is admissible because both goals and an empty TN are jointly required
    to terminate; hence the true remaining cost is >= the maximum of their independent lower bounds.

    Implementation notes
    --------------------
    - We precompute dictionaries to enable O(1) checks by bitwise membership (global_id) without reconstructing states.
      For facts, we only rely on bitwise checks; we never enumerate explicit states during evaluation (per the framework guidance).
    - We avoid scanning all candidate 'on[*, s]' facts at runtime for performance. The goal LB only needs to know if
      a crate is currently being lifted or is in a truck; otherwise we count it as "on some surface" and charge 2.
      This keeps evaluation fast even on very large instances.
    - We do not add extra "clear(s)" costs for target surfaces in the goal LB to maintain strong admissibility and
      avoid potential double counting across multiple tasks that may clear the same surface anyway.

    This heuristic is tuned for scalability (large Depots instances) and reduces node expansions by providing
    informative, yet safe, estimates that exploit key domain structure.
    """

    def __init__(self, model, depots_gpt5_heuristic_10: bool = True):
        super().__init__(model)
        self.name = "DepotsGPT5Heuristic10"

        # Internal caches / indices
        self.fact_name_to_id: Dict[str, int] = {}
        self.in_by_crate: Dict[str, List[int]] = {}
        self.lifting_by_crate: Dict[str, List[int]] = {}
        self.lifting_pair: Dict[Tuple[str, str], int] = {}  # (hoist, crate) -> fact_id
        self.clear_fact: Dict[str, int] = {}               # obj -> fact_id for clear[obj]
        self.at_fact: Dict[Tuple[str, str], int] = {}      # (locatable, place) -> fact_id

        # Extract and index facts
        self._index_facts()

        # Precompute goal on-facts (ids and parsed args)
        self.goal_on_entries: List[Tuple[int, str, str]] = []  # list of (goal_fact_id, crate, surface)
        self.goal_target_by_crate: Dict[str, str] = {}
        self._index_goal_on_facts()

    # ------------------------------
    # Public heuristic API
    # ------------------------------
    def evaluate(self, state_bitwise: int, task_network: List[Any]) -> int:
        goal_lb = self._goal_lower_bound(state_bitwise)
        tasks_lb = self._tasknet_lower_bound(state_bitwise, task_network)
        h_val = max(goal_lb, tasks_lb)
        if h_val < 0:
            return 0
        return h_val

    __call__ = evaluate

    # ------------------------------
    # Index building
    # ------------------------------
    def _index_facts(self):
        for f in self.model.facts:
            self.fact_name_to_id[f.name] = f.global_id
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            if pred == "in" and len(args) == 2:
                c, t = args
                self.in_by_crate.setdefault(c, []).append(f.global_id)
            elif pred == "lifting" and len(args) == 2:
                h, c = args
                self.lifting_by_crate.setdefault(c, []).append(f.global_id)
                self.lifting_pair[(h, c)] = f.global_id
            elif pred == "clear" and len(args) == 1:
                x = args[0]
                self.clear_fact[x] = f.global_id
            elif pred == "at" and len(args) == 2:
                x, p = args
                self.at_fact[(x, p)] = f.global_id

    def _index_goal_on_facts(self):
        # Enumerate goal bits directly (fast) and parse only those that correspond to on[c, s]
        goals_bits = self.model.goals
        bit_pos = 0
        while goals_bits:
            if goals_bits & 1:
                fact = self.model.facts[bit_pos]
                pred, args = self._parse_fact_name(fact.name)
                if pred == "on" and len(args) == 2:
                    c, s = args
                    self.goal_on_entries.append((fact.global_id, c, s))
                    # If multiple goals mention the same crate, the last one wins; this is fine for LB purposes.
                    self.goal_target_by_crate[c] = s
            goals_bits >>= 1
            bit_pos += 1

    # ------------------------------
    # Lower-bound components
    # ------------------------------
    def _goal_lower_bound(self, state_bitwise: int) -> int:
        cost = 0
        for fid, crate, surface in self.goal_on_entries:
            if self._is_true(state_bitwise, fid):
                continue  # goal already satisfied

            # If the crate is currently being lifted by any hoist, a single Drop can achieve on(c, s) (in best case)
            lifting_list = self.lifting_by_crate.get(crate, [])
            if any(self._is_true(state_bitwise, lf) for lf in lifting_list):
                cost += 1
                continue

            # If the crate is in any truck, we need Unload + Drop at minimum
            in_list = self.in_by_crate.get(crate, [])
            if any(self._is_true(state_bitwise, inf) for inf in in_list):
                cost += 2
                continue

            # Otherwise, crate must be on some other support (wrong place) -> need at least Lift + Drop
            cost += 2
        return cost

    def _tasknet_lower_bound(self, state_bitwise: int, task_network: List[Any]) -> int:
        if not task_network:
            return 0

        primitive_ops_lb = 0
        abstract_tasks_lb = 0

        for comp in task_network:
            # Detect whether it's an Operator (primitive) or AbstractTask (decomposable) using attribute presence
            if hasattr(comp, 'pos_precons') and hasattr(comp, 'add_effects') and hasattr(comp, 'del_effects'):
                # Likely an Operator
                name, args = self._parse_component_name(comp.name)
                if name.lower() == 'nop':
                    continue
                primitive_ops_lb += 1  # every remaining primitive operator (except nop) must execute once
            else:
                # Likely an AbstractTask
                tname, targs = self._parse_component_name(getattr(comp, 'name', ''))
                abstract_tasks_lb += self._abstract_task_lb(state_bitwise, tname, targs)

        return primitive_ops_lb + abstract_tasks_lb

    def _abstract_task_lb(self, state_bitwise: int, tname: str, targs: List[str]) -> int:
        tname = tname or ""
        if not tname:
            return 0

        # Map known abstract tasks to optimistic lower bounds using domain knowledge
        if tname == 'do_put_on':
            # do_put_on(c, s2)
            if len(targs) >= 2:
                c, s2 = targs[0], targs[1]
                # If goal on[c, s2] is already true, LB = 0; otherwise same base rule as in goal LB
                goal_fid = self._goal_on_fact_id(c, s2)
                if goal_fid is not None and self._is_true(state_bitwise, goal_fid):
                    return 0
                # If lifting any hoist has c -> Drop is enough
                lifting_list = self.lifting_by_crate.get(c, [])
                if any(self._is_true(state_bitwise, lf) for lf in lifting_list):
                    return 1
                # If in truck -> Unload + Drop
                in_list = self.in_by_crate.get(c, [])
                if any(self._is_true(state_bitwise, inf) for inf in in_list):
                    return 2
                # Else assume on some other support -> Lift + Drop
                return 2
            return 1  # default safe LB

        if tname == 'do_clear':
            # do_clear(s1, p1)
            if len(targs) >= 1:
                s1 = targs[0]
                fid = self.clear_fact.get(s1)
                if fid is not None and self._is_true(state_bitwise, fid):
                    return 0
                return 1  # need at least one Lift to clear a surface
            return 1

        if tname == 'do_get_truck':
            # do_get_truck(t, p1)
            if len(targs) >= 2:
                t, p1 = targs[0], targs[1]
                fid = self.at_fact.get((t, p1))
                if fid is not None and self._is_true(state_bitwise, fid):
                    return 0
                return 1  # at least one Drive operator
            return 1

        if tname == 'do_lift_crate':
            # do_lift_crate(c, p, h)
            if len(targs) >= 3:
                c, _p, h = targs[0], targs[1], targs[2]
                fid = self.lifting_pair.get((h, c))
                if fid is not None and self._is_true(state_bitwise, fid):
                    return 0
                return 1  # at least one Lift or Unload
            return 1

        if tname == 'do_load_truck':
            # do_load_truck(c, s, p, t)
            if len(targs) >= 4:
                c = targs[0]
                lifting_list = self.lifting_by_crate.get(c, [])
                if any(self._is_true(state_bitwise, lf) for lf in lifting_list):
                    return 1  # can Load directly in best case
                return 2      # Lift + Load
            return 2

        if tname == 'do_unload_truck':
            # do_unload_truck(c, s, p, t)
            if len(targs) >= 4:
                c = targs[0]
                lifting_list = self.lifting_by_crate.get(c, [])
                if any(self._is_true(state_bitwise, lf) for lf in lifting_list):
                    return 1  # already lifting -> just Drop in best case
                return 2      # Unload + Drop
            return 2

        # Unknown abstract task: return a very conservative 0 (admissible but not informative)
        return 0

    # ------------------------------
    # Utilities
    # ------------------------------
    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact names in the model: format "+predicate[arg1,arg2]" or "-predicate[arg1,arg2]".
        Returns (predicate, [args...]) or (None, []).
        """
        if not name:
            return None, []
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
        bracket_idx = s.find('[')
        if bracket_idx > 0 and s.endswith(']'):
            predicate = s[:bracket_idx]
            args_str = s[bracket_idx + 1:-1]
            args = args_str.split(',') if args_str else []
            return predicate, args
        return None, []

    @staticmethod
    def _parse_component_name(name: str) -> Tuple[str, List[str]]:
        """
        Parse operator/abstract task instance names: expected format like "do_put_on(crate1,pallet2)".
        Returns (head_name, [args...]); if parsing fails, returns (name, []).
        """
        if not name:
            return "", []
        s = name.strip()
        l = s.find('(')
        if l < 0 or not s.endswith(')'):
            return s, []
        head = s[:l]
        inside = s[l + 1:-1]
        # Split by commas while trimming whitespace
        args = [a.strip() for a in inside.split(',')] if inside else []
        return head, args

    def _goal_on_fact_id(self, crate: str, surface: str) -> Optional[int]:
        # Build name "+on[c,s]" and look it up; faster to derive by name than scanning.
        key = f"+on[{crate},{surface}]"
        return self.fact_name_to_id.get(key)

    @staticmethod
    def _is_true(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1


# Factory alias for consistency if the framework expects a callable to construct this heuristic
def build(model):
    return DepotsGPT5Heuristic10(model)
