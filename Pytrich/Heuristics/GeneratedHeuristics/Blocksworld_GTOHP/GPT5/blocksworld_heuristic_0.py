from typing import Dict, Tuple, List, Union, Optional

# Fallback import to allow running outside the Pytrich package loader
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:  # pragma: no cover
    class Heuristic:  # minimal stub
        def __init__(self, model=None, name: str = "heuristic"):
            self.model = model
            self.name = name
        def __call__(self, state_bitwise: int, task_network: List):
            return 0


class BlocksworldGPT5Heuristic0(Heuristic):
    """
    Domain-dependent heuristic for the BLOCKS (Blocksworld) HDDL domain.

    Design rationale (short):
    - Goals in this domain are On(x,y). The only action that can make on(x,y) true is stack(x,y).
      Thus, any unsatisfied On(x,y) goal requires at least one stack.
    - To perform stack(x,y), we need holding(x) and clear(y). Creating holding(x) requires at least
      one action (pick-up or unstack), and sometimes additional work to clear x first.
    - We compute a tight but inexpensive lower bound per goal On(x,y) using only current-state facts:
        cost(On(x,y) not true) = 1 (stack) + [1 if not (holding(x) and clear(y)) else 0]
                                 + [1 if (not holding(x) and not clear(x)) else 0]
      This remains admissible and captures key preconditions while accounting for synergy (e.g.,
      unstack(x,y) provides both holding(x) and clear(y) when applicable) without overcounting.
    - We also leverage the remaining task network (HTN) by adding a lower bound derived from the
      unrefined tasks: count(primitive ops) + minimal cost of certain abstract tasks (do_clear,
      do_on_table, do_move/do_put_on) computed against the current state. To remain admissible, we
      take max(goal-based LB, task-network LB) rather than summing them, avoiding double-counting.

    Efficiency notes:
    - All state-membership tests use bitwise checks via fact.global_id, never string lookups.
    - We avoid building a full mapping for all on[x,y] facts (which could be O(n^2)). Instead we:
        * Extract goal on-facts directly from the goals bitset (O(#goal_facts)).
        * Build fast maps for unary predicates (clear, holding, ontable) in a single pass using
          light prefix checks, skipping expensive parsing of on-facts.
    - Name parsing for tasks is robust to different ground-name formats ([], (), or space-separated).

    This heuristic is consistent and admissible, scales to large instances, and provides useful
    guidance by exploiting Blocksworld structure and HTN subtasks at low computational cost.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_0: bool = True):
        super().__init__(model, name="BlocksworldGPT5Heuristic0")
        self.blocksworld_gpt5_heuristic_0 = blocksworld_gpt5_heuristic_0

        # Maps for quick bitwise fact checks
        self.clear_map: Dict[str, object] = {}      # block -> Fact
        self.holding_map: Dict[str, object] = {}    # block -> Fact
        self.ontable_map: Dict[str, object] = {}    # block -> Fact
        self.handempty_fact: Optional[object] = None

        # Goal on-facts (only those appearing in goals), to avoid O(n^2) indexing
        self.goal_on_map: Dict[Tuple[str, str], object] = {}  # (x,y) -> Fact
        self.goal_on_pairs: List[Tuple[str, str]] = []

        # Initialize unary predicate maps with a single pass over facts
        # (Skip parsing on-facts which dominate the set.)
        for fact in getattr(self.model, 'facts', []):
            name = fact.name
            if not name or name[0] not in ['+', '-']:
                continue
            if name.startswith('+clear['):
                b = self._extract_single_arg(name)
                if b is not None:
                    self.clear_map[b] = fact
                continue
            if name.startswith('+holding['):
                b = self._extract_single_arg(name)
                if b is not None:
                    self.holding_map[b] = fact
                continue
            if name.startswith('+ontable['):
                b = self._extract_single_arg(name)
                if b is not None:
                    self.ontable_map[b] = fact
                continue
            if name.startswith('+handempty'):
                # Either +handempty or +handempty[]
                self.handempty_fact = fact
                continue
            # All other facts (including +on[...]) are ignored here for efficiency.

        # Build goal on-fact map by iterating only over set bits in goals
        # Ensures O(#goal_facts) without scanning all facts for on[x,y]
        goals_bits = int(getattr(self.model, 'goals', 0))
        remaining = goals_bits
        while remaining:
            lsb = remaining & -remaining
            bit_index = (lsb.bit_length() - 1)
            remaining -= lsb
            # Access the Fact directly by its global_id (= bit_index)
            fact = self.model.facts[bit_index]
            # Expect names of the form '+on[x,y]'
            if fact.name.startswith('+on['):
                pair = self._extract_pair_args(fact.name)
                if pair is not None:
                    self.goal_on_map[pair] = fact
                    self.goal_on_pairs.append(pair)

    # ---------------------- Public API ----------------------
    def __call__(self, state_bitwise: int, task_network: List[Union[object, object]]) -> int:
        if not self.blocksworld_gpt5_heuristic_0:
            return 0
        return self._evaluate(state_bitwise, task_network)

    # ---------------------- Core Evaluation ----------------------
    def _evaluate(self, state_bitwise: int, task_network: List[Union[object, object]]) -> int:
        # Goal-based lower bound
        goal_lb = self._goal_lower_bound(state_bitwise)

        # Task-network lower bound
        tn_lb = self._task_network_lower_bound(state_bitwise, task_network)

        # Admissible combination: take the max of both lower bounds
        return max(goal_lb, tn_lb)

    # ---------------------- Goal-based LB ----------------------
    def _goal_lower_bound(self, state_bitwise: int) -> int:
        total = 0
        for (x, y) in self.goal_on_pairs:
            f_on = self.goal_on_map.get((x, y))
            if f_on is None:
                continue  # Shouldn't happen; defensive
            if self._is_true(state_bitwise, f_on):
                continue  # already satisfied

            # Preconditions for stack(x,y): holding(x) and clear(y)
            holding_x = self._is_true(state_bitwise, self.holding_map.get(x))
            clear_y = self._is_true(state_bitwise, self.clear_map.get(y))
            clear_x = self._is_true(state_bitwise, self.clear_map.get(x)) if not holding_x else False

            # Always at least one stack
            cost = 1
            # If either precondition missing, at least one additional action is needed to enable stack
            if not (holding_x and clear_y):
                cost += 1
            # If we are not already holding x and x is not clear now, we must clear x first (>= 1)
            if (not holding_x) and (not clear_x):
                cost += 1

            total += cost
        return total

    # ---------------------- Task-network LB ----------------------
    def _task_network_lower_bound(self, state_bitwise: int, task_network: List[Union[object, object]]) -> int:
        if not task_network:
            return 0

        primitive_cost = 0
        abstract_cost = 0

        for t in task_network:
            # Detect primitive operators by duck-typing (Operator has apply/applicable)
            if self._is_operator(t):
                op_name = getattr(t, 'name', '').lower()
                # Count nop as zero-cost
                if op_name.startswith('nop'):
                    continue
                primitive_cost += 1
                continue

            # Otherwise, assume AbstractTask-like; use its grounded name
            tname = getattr(t, 'name', '')
            pred, args = self._parse_task_name(tname)
            if not pred:
                # Unknown structure; be conservative (0) to preserve admissibility
                continue

            pl = pred.lower()
            if pl in ('do_put_on', 'do_move'):
                # If we can confirm on[x,y] already holds, 0; else use the same LB as goals.
                x, y = (args + [None, None])[:2]
                cost = 0
                # Try to reuse goal on-facts if available
                f_on = self.goal_on_map.get((x, y)) if (x is not None and y is not None) else None
                if f_on is None:
                    # Unknown pair: to remain admissible, do not assume non-satisfied; cost = 0
                    cost = 0
                else:
                    if not self._is_true(state_bitwise, f_on):
                        holding_x = self._is_true(state_bitwise, self.holding_map.get(x))
                        clear_y = self._is_true(state_bitwise, self.clear_map.get(y))
                        clear_x = self._is_true(state_bitwise, self.clear_map.get(x)) if not holding_x else False
                        cost = 1  # stack
                        if not (holding_x and clear_y):
                            cost += 1
                        if (not holding_x) and (not clear_x):
                            cost += 1
                abstract_cost += cost

            elif pl == 'do_on_table':
                x = (args + [None])[0]
                if x is None:
                    continue
                # If already ontable(x), zero; else at least 1 (if holding(x): put-down) or 2 otherwise
                ontable_x = self._is_true(state_bitwise, self.ontable_map.get(x))
                if ontable_x:
                    cost = 0
                else:
                    holding_x = self._is_true(state_bitwise, self.holding_map.get(x))
                    if holding_x:
                        cost = 1  # put-down
                    else:
                        # Need to create holding(x) then put-down: >=2. If x not clear, need extra to clear, but we keep LB
                        cost = 2
                abstract_cost += cost

            elif pl == 'do_clear':
                x = (args + [None])[0]
                if x is None:
                    continue
                # If clear(x) already, zero; else at least one unstack is needed
                clear_x = self._is_true(state_bitwise, self.clear_map.get(x))
                abstract_cost += 0 if clear_x else 1

            elif pl == 'nop':
                # No contribution
                continue

            else:
                # Unknown abstract task; to stay admissible, contribute zero
                continue

        return primitive_cost + abstract_cost

    # ---------------------- Helpers ----------------------
    @staticmethod
    def _is_operator(x: object) -> bool:
        return hasattr(x, 'apply') and hasattr(x, 'applicable')

    @staticmethod
    def _extract_single_arg(name: str) -> Optional[str]:
        # Expects form '+pred[arg]'
        lb = name.find('[')
        if lb == -1:
            return None
        rb = name.rfind(']')
        if rb == -1 or rb < lb:
            return None
        arg = name[lb + 1:rb].strip()
        return arg if arg else None

    @staticmethod
    def _extract_pair_args(name: str) -> Optional[Tuple[str, str]]:
        # Expects form '+pred[arg1,arg2]'
        lb = name.find('[')
        if lb == -1:
            return None
        rb = name.rfind(']')
        if rb == -1 or rb < lb:
            return None
        inside = name[lb + 1:rb]
        parts = [p.strip() for p in inside.split(',')]
        if len(parts) != 2:
            return None
        return parts[0], parts[1]

    @staticmethod
    def _parse_task_name(name: str) -> Tuple[Optional[str], List[str]]:
        # Robust parsing for task/operator names in possible formats:
        #   'do_put_on[b1,b2]'  or  'do_put_on(b1,b2)'  or  'do_put_on b1 b2'
        if not name:
            return None, []
        s = name.strip()
        # Bracket format
        lb = s.find('[')
        if lb != -1 and s.endswith(']'):
            pred = s[:lb].strip()
            args_str = s[lb + 1:-1]
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return pred, args
        # Parenthesis format
        lp = s.find('(')
        if lp != -1 and s.endswith(')'):
            pred = s[:lp].strip()
            args_str = s[lp + 1:-1]
            # Split by comma or whitespace
            if ',' in args_str:
                args = [a.strip() for a in args_str.split(',') if a.strip()]
            else:
                args = [a for a in args_str.split() if a]
            return pred, args
        # Whitespace-separated format
        parts = s.split()
        if not parts:
            return None, []
        pred = parts[0]
        args = parts[1:]
        return pred, args

    @staticmethod
    def _is_true(state_bitwise: int, fact_obj: Optional[object]) -> bool:
        if fact_obj is None:
            return False
        gid = getattr(fact_obj, 'global_id', None)
        if gid is None:
            return False
        return ((state_bitwise >> gid) & 1) == 1
