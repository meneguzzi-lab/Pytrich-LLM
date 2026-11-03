from typing import List, Tuple, Set, Dict, Optional, Union

try:
    # Expected location in Pytrich
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback stub to keep this module importable in isolation/tests
    class Heuristic:
        def __init__(self, *args, **kwargs):
            pass


class BlocksworldGPT5Heuristic4(Heuristic):
    """
    Domain-dependent heuristic for BLOCKSWORLD (HDDL) in Pytrich.

    Key ideas (all used as admissible lower bounds; final value = max of them):
    - Goal-on lower bound: Each goal on(x,y) that is false requires at least one 'stack' action. Count them.
    - Goal-top clearance: Any block that is a top-of-goal-stack (i.e., no block should be on it in the goal)
      must be clear in the final state. If it is not clear now, at least one unstack is required. Count them.
    - HTN do_put_on lower bound: For each unique (x,y) appearing in remaining do_put_on tasks that is currently
      false, at least one 'stack' is required at some point.
    - HTN do_move lower bound: Similar to do_put_on; if on(x,y) is false, at least one 'stack'.
    - HTN do_on_table lower bound: For each unique x with a do_on_table task that is currently not on-table,
      at least one action (unstack or put-down) is required at some point.
    - HTN do_clear lower bound: For each unique x with a do_clear task that is currently not clear,
      at least one action (an unstack of the blocker) is required at some point.

    The final heuristic is the maximum of these counts, ensuring admissibility while capturing both
    the goal structure and the current HTN task network.

    Design rationale:
    - Counting unsatisfied goal "on" facts is a classic admissible lower bound in Blocksworld because
      only 'stack' adds on(x,y), and each such goal needs at least one stack unless already true now.
    - Counting top-of-goal blocks that are currently not clear is also admissible: at least one unstack
      is required to make them clear at some point.
    - Considering remaining abstract tasks makes the heuristic sensitive to the hierarchy and remaining
      refinement effort, but to avoid double-counting across interacting tasks, we use separate lower bounds
      per task type and take their maximum with the goal bounds.

    This heuristic focuses on scalability: all computations are O(|goals| + |task_network|), and
    state membership tests use bitwise checks via Fact.global_id.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_4: bool = True):
        super().__init__()
        self.model = model
        self.blocksworld_gpt5_heuristic_4 = blocksworld_gpt5_heuristic_4
        self.name = "BlocksworldGPT5Heuristic4"

        # Fact indexing for O(1) bitwise checks
        self._fact_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        self._index_relevant_facts()

        # Parse goal facts once
        self.goal_on_pairs: Set[Tuple[str, str]] = set()
        self.goal_blocks_all: Set[str] = set()
        self.goal_support_y_set: Set[str] = set()
        self.goal_top_blocks: Set[str] = set()
        self._index_goal_structure()

    # ------------------------
    # Public interface expected by Pytrich
    # ------------------------
    def evaluate(self, state_bitwise: int, task_network: List[Union[object, int]]) -> int:
        """
        Compute the heuristic value for a given state and (remaining) task network.

        Args:
            state_bitwise: int bitvector of current state (use bitwise tests with Fact.global_id)
            task_network: list of remaining tasks/operators; we parse only abstract task names
                          like 'do_put_on[b1,b2]'.
        Returns:
            Non-negative integer heuristic value.
        """
        # 1) Goal-on lower bound: count unsatisfied on(x,y)
        lb_goal_on = 0
        for (x, y) in self.goal_on_pairs:
            if not self._is_true(state_bitwise, 'on', (x, y)):
                lb_goal_on += 1

        # 2) Goal-top clearance lower bound: blocks that must be clear in goal but are not clear now
        lb_goal_tops_clear = 0
        for b in self.goal_top_blocks:
            if not self._is_true(state_bitwise, 'clear', (b,)):
                lb_goal_tops_clear += 1

        # 3) HTN task-based lower bounds (unique targets to avoid accidental duplicate pairs)
        tn_put_on_pairs: Set[Tuple[str, str]] = set()
        tn_move_pairs: Set[Tuple[str, str]] = set()
        tn_on_table_blocks: Set[str] = set()
        tn_clear_blocks: Set[str] = set()

        for comp in (task_network or []):
            name = getattr(comp, 'name', None)
            if not name or not isinstance(name, str):
                # If it's an Operator or different kind, we simply skip; heuristic focuses on abstract tasks
                continue
            pred, args = self._parse_component_name(name)
            if not pred:
                continue

            # Normalize based on domain abstract task names
            if pred == 'do_put_on' and len(args) == 2:
                tn_put_on_pairs.add((args[0], args[1]))
            elif pred == 'do_move' and len(args) >= 2:
                tn_move_pairs.add((args[0], args[1]))
            elif pred == 'do_on_table' and len(args) == 1:
                tn_on_table_blocks.add(args[0])
            elif pred == 'do_clear' and len(args) == 1:
                tn_clear_blocks.add(args[0])

        # 3a) For each unique do_put_on pair not already true, one 'stack' is minimally required
        lb_tn_put_on = 0
        for (x, y) in tn_put_on_pairs:
            if not self._is_true(state_bitwise, 'on', (x, y)):
                lb_tn_put_on += 1

        # 3b) For each unique do_move pair not already satisfied, one 'stack' is minimally required
        lb_tn_move = 0
        for (x, y) in tn_move_pairs:
            if not self._is_true(state_bitwise, 'on', (x, y)):
                lb_tn_move += 1

        # 3c) For each unique do_on_table block not yet on-table, at least one action is required
        lb_tn_on_table = 0
        for x in tn_on_table_blocks:
            if not self._is_true(state_bitwise, 'ontable', (x,)):
                lb_tn_on_table += 1

        # 3d) For each unique do_clear block not yet clear, at least one action is required
        lb_tn_clear = 0
        for x in tn_clear_blocks:
            if not self._is_true(state_bitwise, 'clear', (x,)):
                lb_tn_clear += 1

        # Final heuristic = max of admissible lower bounds
        h_val = max(lb_goal_on, lb_goal_tops_clear, lb_tn_put_on, lb_tn_move, lb_tn_on_table, lb_tn_clear)
        return int(h_val)

    # Optional convenience: allow calling the instance as a function
    __call__ = evaluate

    # ------------------------
    # Internal helpers
    # ------------------------
    def _index_relevant_facts(self):
        """
        Build index mapping (predicate, args_tuple) -> global_id for fast state membership checks.
        We index only the needed predicates for this heuristic: on, clear, ontable, holding, handempty.
        """
        relevant = {"on", "clear", "ontable", "holding", "handempty"}
        for fact in getattr(self.model, 'facts', []):
            try:
                name = fact.name
            except Exception:
                continue
            pred, args = self._parse_fact_name_any(name)
            if pred in relevant:
                self._fact_by_key[(pred, args)] = fact.global_id

    def _index_goal_structure(self):
        """
        Extract and cache goal relations and the set of blocks that must be clear at the goal (tops of stacks).
        """
        self.goal_on_pairs.clear()
        self.goal_blocks_all.clear()
        self.goal_support_y_set.clear()
        self.goal_top_blocks.clear()

        try:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        except Exception:
            goal_fact_names = []

        for gname in goal_fact_names:
            pred, args = self._parse_fact_name_sign(gname)
            if pred == 'on' and len(args) == 2:
                x, y = args
                self.goal_on_pairs.add((x, y))
                self.goal_blocks_all.add(x)
                self.goal_blocks_all.add(y)
                self.goal_support_y_set.add(y)

        # Top-of-goal blocks are those that should not have any block on them in the goal.
        # Among all blocks that appear in the goal graph, those not in support_y must be clear in the goal.
        self.goal_top_blocks = {b for b in self.goal_blocks_all if b not in self.goal_support_y_set}

    def _is_true(self, state_bitwise: int, predicate: str, args: Tuple[str, ...]) -> bool:
        """
        Fast bitwise membership test for predicate(args) in state.
        Returns False if the fact is not grounded/indexed or not true in the state.
        """
        gid = self._fact_by_key.get((predicate, args))
        if gid is None:
            return False
        return ((state_bitwise >> gid) & 1) == 1

    @staticmethod
    def _parse_fact_name_sign(name: str) -> Tuple[Optional[str], Tuple[str, ...]]:
        """
        Parse a grounded fact name that may include a leading + or - sign.
        Expected formats:
        - '+on[b1,b2]'
        - '-clear[b3]'
        - 'on[b1,b2]' (fallback if no sign exists)
        Returns: (predicate, tuple(args)) or (None, ()) if unparsable.
        """
        if not name or not isinstance(name, str):
            return None, tuple()
        s = name.strip()
        if s and (s[0] == '+' or s[0] == '-'):
            s = s[1:]
        lb = s.find('[')
        if lb < 0:
            # No args case, like 'handempty' or malformed
            pred = s
            return pred, tuple()
        pred = s[:lb]
        args_str = s[lb + 1: -1] if s.endswith(']') else s[lb + 1:]
        if not args_str:
            return pred, tuple()
        args = tuple(a.strip() for a in args_str.split(',') if a.strip())
        return pred, args

    @staticmethod
    def _parse_fact_name_any(name: str) -> Tuple[Optional[str], Tuple[str, ...]]:
        """
        General parser for fact-like names; tolerant to presence/absence of +/-.
        """
        return BlocksworldGPT5Heuristic4._parse_fact_name_sign(name)

    @staticmethod
    def _parse_component_name(name: str) -> Tuple[Optional[str], Tuple[str, ...]]:
        """
        Parse component (task/operator) names expected in the model, typically like 'do_put_on[b1,b2]'.
        Returns (predicate, tuple(args)) if parseable; otherwise (None, ()).
        """
        if not name or not isinstance(name, str):
            return None, tuple()
        s = name.strip()
        # No +/- expected for tasks/operators
        lb = s.find('[')
        if lb < 0:
            # Might be 0-arity name like 'handempty' operator but not used here
            return s, tuple()
        pred = s[:lb]
        args_str = s[lb + 1: -1] if s.endswith(']') else s[lb + 1:]
        args = tuple(a.strip() for a in args_str.split(',') if a.strip()) if args_str else tuple()
        return pred, args

    # Optional helper if the planner expects a string name
    def get_name(self) -> str:
        return self.name
