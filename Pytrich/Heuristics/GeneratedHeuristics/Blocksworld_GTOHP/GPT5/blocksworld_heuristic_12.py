from typing import Dict, Tuple, List, Optional, Union, Set

try:
    # Common path in Pytrich
    from Heuristics.heuristic import Heuristic
except Exception:
    # Alternate capitalization seen in some forks
    from Pytrich.Heuristics.heuristic import Heuristic


class BlocksworldGPT5Heuristic12(Heuristic):
    """
    Blocksworld domain-dependent heuristic for HTN planning in Pytrich.

    Design goals
    ------------
    - Admissible and lightweight for very large instances (1k blocks).
    - Considers both classical goal facts and the hierarchical task network (HTN).
    - Uses domain structure of Blocksworld and the provided HTN methods.

    Heuristic definition (admissible lower bound)
    --------------------------------------------
    h(state, tn) = max(goal_lb(state), tn_lb(state, tn))

    Where:
      - goal_lb(state): A lower bound on the number of primitive actions needed to
        achieve the ON goals from the current state. We use:
            goal_lb = max(
                unsatisfied_goal_on_count,                       # at least one STACK per unsatisfied on(x,y)
                blocked_supports_count,                           # unique y in unsatisfied goals where clear(y) is false => at least one UNSTACK somewhere
                blocked_movers_count                              # unique x in unsatisfied goals where not clear(x) and not holding(x)
            )
        This is safe because: each of these counts represents disjoint necessary
        action requirements; taking their max preserves admissibility.

      - tn_lb(state, tn): A lower bound on the number of primitive actions needed
        to complete the current task network. We conservatively count:
          * For each primitive operator in tn (except 'nop'): +1 (it must be executed).
          * For each abstract task in tn:
              - do_put_on(x,y): 0 if on(x,y) holds now (method m0 with nop), else 1
              - do_on_table(x): 0 if clear(x) holds now (method m3 with nop), else 1
              - do_move(x,y): 1 (no zero-cost method)
              - do_clear(x): 0 if clear(x) holds now (method m6 with nop), else 1
            For any other abstract task: +1 as a conservative bound.
        Summing these minima yields a lower bound for finishing the HTN from this
        node. We then take max with goal_lb to avoid double counting between HTN and goals.

    Implementation notes
    --------------------
    - All state membership checks are done with bit tests using global_id.
    - Grounded fact names follow the '+predicate[arg1,arg2]' convention provided.
    - We preindex important predicate facts: on, clear, ontable, holding, handempty.
    - Goals are parsed once from model.goals (bitset) using model.state_explicit_repr.
    - Robust handling of task_network elements whether they are objects or ids.

    This heuristic scales well and provides effective guidance while remaining
    admissible and simple.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_12: Optional[dict] = None):
        super().__init__(model)
        self.model = model

        # Fact indices for O(1) bit-tests
        self._on_idx: Dict[Tuple[str, str], int] = {}
        self._clear_idx: Dict[str, int] = {}
        self._ontable_idx: Dict[str, int] = {}
        self._holding_idx: Dict[str, int] = {}
        self._handempty_idx: Optional[int] = None

        # Parse and index all facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if pred == 'on' and len(args) == 2:
                self._on_idx[(args[0], args[1])] = fact.global_id
            elif pred == 'clear' and len(args) == 1:
                self._clear_idx[args[0]] = fact.global_id
            elif pred == 'ontable' and len(args) == 1:
                self._ontable_idx[args[0]] = fact.global_id
            elif pred == 'holding' and len(args) == 1:
                self._holding_idx[args[0]] = fact.global_id
            elif pred == 'handempty':
                self._handempty_idx = fact.global_id

        # Parse goal ON-facts (bitset -> explicit list of "+on[x,y]")
        self.goal_on_pairs: List[Tuple[str, str]] = []
        self.goal_on_fact_ids: List[int] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                self.goal_on_pairs.append((args[0], args[1]))
                fid = self._on_idx.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_fact_ids.append(fid)

    # ---------------------------- Public API ----------------------------
    def evaluate(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        """
        Compute h(n) for the given state and task network.
        """
        goal_lb = self._goal_lower_bound(state_bitwise)
        tn_lb = self._tn_lower_bound(state_bitwise, task_network)
        return max(goal_lb, tn_lb)

    # Support common alternative method names some planners expect
    def h(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        return self.evaluate(state_bitwise, task_network)

    def heuristic(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        return self.evaluate(state_bitwise, task_network)

    # -------------------------- Lower bounds ---------------------------
    def _goal_lower_bound(self, state_bitwise: int) -> int:
        """
        Lower bound from ON goals only.
        - unsat_stack: number of ON goals not yet true (each needs at least a STACK)
        - blocked_supports: unique y among unsatisfied goals where clear(y) is false (each needs at least an UNSTACK)
        - blocked_movers: unique x among unsatisfied goals where not clear(x) and not holding(x)
        Return max of these counts (admissible).
        """
        unsat_pairs: List[Tuple[str, str]] = []
        unsat_stack = 0
        for fid, pair in zip(self.goal_on_fact_ids, self.goal_on_pairs):
            if not self._is_true(state_bitwise, fid):
                unsat_stack += 1
                unsat_pairs.append(pair)

        if unsat_stack == 0:
            return 0

        # Count unique y whose clear(y) is false
        blocked_supports_set: Set[str] = set()
        for (x, y) in unsat_pairs:
            cy = self._clear_idx.get(y)
            if cy is not None and not self._is_true(state_bitwise, cy):
                blocked_supports_set.add(y)
        blocked_supports = len(blocked_supports_set)

        # Count unique x where not clear(x) and not holding(x)
        blocked_movers_set: Set[str] = set()
        for (x, _y) in unsat_pairs:
            cx = self._clear_idx.get(x)
            hx = self._holding_idx.get(x)
            cx_true = self._is_true(state_bitwise, cx) if cx is not None else False
            hx_true = self._is_true(state_bitwise, hx) if hx is not None else False
            if (not cx_true) and (not hx_true):
                blocked_movers_set.add(x)
        blocked_movers = len(blocked_movers_set)

        return max(unsat_stack, blocked_supports, blocked_movers)

    def _tn_lower_bound(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        """
        Lower bound from the (partial) task network.
        - For primitive operators explicitly in the TN: +1 each (except nop).
        - For abstract tasks, use zero-cost method applicability (NOP) from domain methods:
            * do_put_on(x,y): 0 if on(x,y) holds now, else 1
            * do_on_table(x): 0 if clear(x) holds now, else 1
            * do_move(x,y): 1  (no zero-cost method available)
            * do_clear(x):    0 if clear(x) holds now, else 1
          Any other abstract tasks: +1
        """
        if not task_network:
            return 0

        lb = 0
        for t in task_network:
            comp = self._ensure_component(t)
            if comp is None:
                # Be conservative but safe
                lb += 1
                continue

            name = getattr(comp, 'name', '')
            # Heuristically detect operator vs abstract by attribute presence
            is_operator = hasattr(comp, 'get_precons') and hasattr(comp, 'apply')
            if is_operator:
                # Primitive operator - pay 1 unless it's a nop
                base, _args = self._parse_name_and_args(name)
                if base != 'nop':
                    lb += 1
                continue

            # Abstract task: name contains task type and args
            base, args = self._parse_name_and_args(name)
            base = base or ''
            if base == 'do_put_on' and len(args) >= 2:
                x, y = args[0], args[1]
                fid = self._on_idx.get((x, y))
                if fid is not None and self._is_true(state_bitwise, fid):
                    lb += 0
                else:
                    lb += 1
            elif base == 'do_on_table' and len(args) >= 1:
                x = args[0]
                cx = self._clear_idx.get(x)
                if cx is not None and self._is_true(state_bitwise, cx):
                    lb += 0
                else:
                    lb += 1
            elif base == 'do_move':
                # No zero-cost method exists in domain
                lb += 1
            elif base == 'do_clear' and len(args) >= 1:
                x = args[0]
                cx = self._clear_idx.get(x)
                if cx is not None and self._is_true(state_bitwise, cx):
                    lb += 0
                else:
                    lb += 1
            else:
                # Unknown abstract task type: be conservative but admissible
                lb += 1

        return lb

    # ---------------------------- Utilities ----------------------------
    @staticmethod
    def _parse_fact_name(name: str):
        """
        Parse grounded fact names of the form "+predicate[arg1,arg2]" or "-predicate[arg1,arg2]".
        Returns (predicate, [args]). If parsing fails, returns (None, []).
        """
        if not isinstance(name, str):
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
    def _parse_name_and_args(name: str) -> Tuple[str, List[str]]:
        """
        Robustly parse operator/abstract-task names. Supports formats like:
          - "do_put_on[b1,b2]"
          - "pick-up[b1]"
          - Fallback to bare names -> (name, [])
        """
        if not isinstance(name, str):
            return '', []
        s = name.strip()
        # Prefer blocksworld-like [args]
        b = s.find('[')
        if b != -1 and s.endswith(']'):
            head = s[:b]
            inside = s[b + 1:-1]
            args = [a.strip() for a in inside.split(',')] if inside else []
            return head, args
        # Secondary: (args)
        p = s.find('(')
        if p != -1 and s.endswith(')'):
            head = s[:p]
            inside = s[p + 1:-1]
            # args might be comma or space separated depending on generator
            if ',' in inside:
                args = [a.strip() for a in inside.split(',') if a.strip()]
            else:
                args = [a.strip() for a in inside.split() if a.strip()]
            return head, args
        return s, []

    def _is_true(self, state_bitwise: int, fact_global_id: Optional[int]) -> bool:
        if fact_global_id is None:
            return False
        return (state_bitwise >> fact_global_id) & 1 == 1

    def _ensure_component(self, comp_or_id: Union[int, object]):
        """
        Accept either a component object or its global_id. Return the object or None if invalid.
        """
        if isinstance(comp_or_id, int):
            try:
                return self.model.get_component(comp_or_id)
            except Exception:
                return None
        return comp_or_id
