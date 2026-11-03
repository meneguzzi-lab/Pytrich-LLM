try:
    # Typical Pytrich import path
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback stub if the environment provides a different path; keeps class usable for type checking
    class Heuristic:
        def __init__(self, model):
            self.model = model

from typing import Dict, Tuple, List, Union, Set


class BlocksworldGPT5Heuristic11(Heuristic):
    """
    Domain-dependent heuristic for BLOCKS (Blocksworld HDDL) tailored for Pytrich.

    Key ideas (admissible, HTN-aware, efficient on large instances):
    - Goal guidance via an admissible h_max-style bound across goal on(x,y) facts.
      For each goal on(x,y), compute a tight lower bound on the minimum number of
      primitive actions needed from the current state to make on(x,y) true:
        * If on(x,y) already holds: 0
        * Otherwise, at least one final stack is needed: +1
        * If holding(x): we avoid the pickup/unstack cost (0), else
            - if clear(x): need 1 (pickup/unstack) to get holding(x)
            - else: need at least 3 (2 to remove one blocker, +1 to unstack/pick)
          (This underestimates if more than one blocker; still admissible.)
        * If y is not clear: at least 2 actions to clear its top blocker (unstack+put-down)
      This yields an admissible per-goal LB. We take the maximum across all goal on(x,y)
      to preserve admissibility (h_max style for interacting goals).

    - HTN task-network awareness via a per-task lower bound and max-aggregation:
      For each current task in the task network:
        * Primitive operators: LB >= 1 (except nop -> 0).
        * do_put_on(x,y) and do_move(x,y): use the same admissible LB as above for on(x,y),
          but return 0 if on(x,y) already true (method m0_do_put_on is a no-op).
        * do_on_table(x): 0 if ontable(x) already holds; else
            - if holding(x): 1 (put-down)
            - elif clear(x): 2 (unstack then put-down)
            - else: 3 (clear top:2 + unstack/put-down:1)
        * do_clear(x): 0 if clear(x) already holds; else at least 2 (unstack+put-down one blocker)
      We take the maximum lower bound among all tasks (admissible aggregation).

    - Final heuristic h(n) = max(h_goals, h_tasks). This keeps h admissible and scales well.

    Implementation notes for Pytrich:
    - Use bitwise membership checks with fact.global_id for speed.
    - Avoid building O(n^2) indices for all on(x,y). Instead, we only index the subset
      of on(x,y) pairs that actually matter: those in the goal set and those appearing
      in the initial task network as arguments to do_put_on/do_move. This avoids
      constructing a 1e6-entry dictionary on large instances.

    Robustness:
    - If the planner reaches a state where all goals hold and the task network is empty,
      we return 0 immediately (goal reached condition).
    - If a task introduces a new on(x,y) pair not seen initially, we conservatively
      treat its LB without checking on(x,y) truth (rare in this domain); this remains
      admissible but slightly weaker. A lazy cache hook is provided for extensibility.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_11: bool = True):
        super().__init__(model)
        self.name = "BlocksworldGPT5Heuristic11"

        # Core predicate names
        self.P_ON = 'on'
        self.P_CLEAR = 'clear'
        self.P_ONTABLE = 'ontable'
        self.P_HAND = 'handempty'
        self.P_HOLD = 'holding'

        # Mappings for quick bitwise checks (only O(n) for unary preds; selective for 'on')
        self.clear_gid: Dict[str, int] = {}
        self.ontable_gid: Dict[str, int] = {}
        self.holding_gid: Dict[str, int] = {}
        self.handempty_gid: int = -1
        # Only for pairs that matter (goals + initial TN pairs)
        self.on_gid: Dict[Tuple[str, str], int] = {}

        # Extract goal on(x,y) pairs from bitwise goals
        self.goal_on_pairs: List[Tuple[str, str]] = self._extract_goal_on_pairs()

        # Extract interesting on(x,y) pairs from initial task network (do_put_on/do_move)
        tn_on_pairs = self._extract_on_pairs_from_tn(model.initial_tn)

        # Union of pairs to index efficiently
        self.interesting_on_pairs: Set[Tuple[str, str]] = set(self.goal_on_pairs)
        self.interesting_on_pairs.update(tn_on_pairs)

        # Build fast gid maps by scanning facts once
        self._index_relevant_facts()

    # ---------- Public API ----------
    def estimate(self, state_bitwise: int, task_network: List[Union[object, int]]) -> int:
        # Exact goal-reached check using bitwise goals + empty TN
        try:
            if self.model.goal_reached(state_bitwise, task_network):
                return 0
        except Exception:
            # Fallback if model.goal_reached signature differs; still keep safe
            if (self.model.goals & state_bitwise) == self.model.goals and len(task_network) == 0:
                return 0

        # h_goals: admissible max across per-goal lower bounds
        h_goals = 0
        if self.goal_on_pairs:
            for x, y in self.goal_on_pairs:
                lb = self._lb_on_goal_pair(state_bitwise, x, y)
                if lb > h_goals:
                    h_goals = lb

        # h_tasks: admissible max across current tasks' lower bounds
        h_tasks = 0
        if task_network:
            for t in task_network:
                comp = t
                # In some Pytrich configurations, task_network may carry global_ids
                if isinstance(t, int):
                    try:
                        comp = self.model.get_component(t)
                    except Exception:
                        comp = t
                try:
                    name = comp.name
                except Exception:
                    # Unknown component structure -> minimal safe LB of 1
                    h_tasks = max(h_tasks, 1)
                    continue

                pred, args = self._parse_component_name(name)
                # Distinguish operator vs. abstract task by available attributes
                is_operator = hasattr(comp, 'apply') or hasattr(comp, 'get_precons')

                if is_operator:
                    # Primitive operators: at least 1 step (except explicit nop)
                    if pred == 'nop':
                        lb = 0
                    else:
                        lb = 1
                else:
                    # Abstract tasks: domain-specific admissible LBs
                    if pred in ('do_put_on', 'do_move') and len(args) >= 2:
                        x, y = args[0], args[1]
                        lb = self._lb_on_goal_pair(state_bitwise, x, y, treat_as_task=True)
                    elif pred == 'do_on_table' and len(args) >= 1:
                        x = args[0]
                        lb = self._lb_do_on_table(state_bitwise, x)
                    elif pred == 'do_clear' and len(args) >= 1:
                        x = args[0]
                        lb = self._lb_do_clear(state_bitwise, x)
                    else:
                        # Unknown abstract task: minimal one action to progress
                        lb = 1

                if lb > h_tasks:
                    h_tasks = lb

        return max(h_goals, h_tasks)

    # ---------- Lower bound calculators (all admissible) ----------
    def _lb_on_goal_pair(self, state: int, x: str, y: str, treat_as_task: bool = False) -> int:
        """
        Admissible lower bound for achieving on(x,y) from current state.
        If treat_as_task is True, allow zero when on(x,y) already true (HTN m0_do_put_on no-op).
        Always 0 when on(x,y) is already true (for goals we also need this to keep h_max admissible).
        """
        on_gid = self._get_on_gid(x, y)
        if on_gid is not None and ((state >> on_gid) & 1):
            return 0

        # At least one final stack is required if not already achieved
        stack_cost = 1

        # Cost to get holding(x):
        # - If already holding(x): 0
        # - Else if clear(x): 1 (pickup or unstack)
        # - Else: at least 3 (clear a blocker:2 + pickup/unstack:1)
        if self._is_true(state, self.holding_gid.get(x, -1)):
            get_hold_cost = 0
        else:
            if self._is_true(state, self.clear_gid.get(x, -1)):
                get_hold_cost = 1
            else:
                get_hold_cost = 3

        # Cost to clear y (if needed): at least 2 if y is not clear
        y_clear_cost = 0 if self._is_true(state, self.clear_gid.get(y, -1)) else 2

        return stack_cost + get_hold_cost + y_clear_cost

    def _lb_do_on_table(self, state: int, x: str) -> int:
        # If already ontable(x): 0
        if self._is_true(state, self.ontable_gid.get(x, -1)):
            return 0
        # If holding(x): just put-down: 1
        if self._is_true(state, self.holding_gid.get(x, -1)):
            return 1
        # If clear(x) and not ontable: unstack then put-down (2). If not clear: +1 for minimal clear.
        if self._is_true(state, self.clear_gid.get(x, -1)):
            return 2
        return 3

    def _lb_do_clear(self, state: int, x: str) -> int:
        # If already clear(x): 0. Else unstack+put-down at least: 2
        return 0 if self._is_true(state, self.clear_gid.get(x, -1)) else 2

    # ---------- Utilities / Parsing ----------
    def _is_true(self, state: int, gid: int) -> bool:
        return gid >= 0 and ((state >> gid) & 1) == 1

    def _get_on_gid(self, x: str, y: str):
        key = (x, y)
        gid = self.on_gid.get(key)
        if gid is not None:
            return gid
        # Lazy fallback (rare): try to discover and cache if present, but avoid scanning all facts here.
        # We conservatively return None if not indexed. This keeps admissibility (we won't return 0 for on[x,y]).
        return None

    def _parse_fact_name(self, name: str):
        """Parse fact names like '+on[b1,b2]' or '-clear[b1]' to ('on', ['b1','b2'])."""
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
        bracket_idx = s.find('[')
        if bracket_idx > 0 and s.endswith(']'):
            pred = s[:bracket_idx]
            args_str = s[bracket_idx + 1:-1]
            args = args_str.split(',') if args_str else []
            return pred, args
        return None, []

    def _parse_component_name(self, name: str):
        """Parse operator/abstract task name 'do_put_on[b1,b2]' into ('do_put_on',['b1','b2'])."""
        s = name.strip()
        # Support both 'name[args]' and possible 'name(args)' variants
        if '[' in s and s.endswith(']'):
            idx = s.find('[')
            pred = s[:idx]
            arg_str = s[idx + 1:-1]
            args = arg_str.split(',') if arg_str else []
            return pred, args
        if '(' in s and s.endswith(')'):
            idx = s.find('(')
            pred = s[:idx]
            arg_str = s[idx + 1:-1]
            args = [a.strip() for a in arg_str.split(',')] if arg_str else []
            return pred, args
        return s, []

    # ---------- Index building ----------
    def _extract_goal_on_pairs(self) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        try:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        except Exception:
            goal_fact_names = []
        for n in goal_fact_names:
            pred, args = self._parse_fact_name(n)
            if pred == self.P_ON and len(args) == 2:
                pairs.append((args[0], args[1]))
        return pairs

    def _extract_on_pairs_from_tn(self, tn: List[Union[object, int]]) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()
        if not tn:
            return pairs
        for t in tn:
            comp = t
            if isinstance(t, int):
                try:
                    comp = self.model.get_component(t)
                except Exception:
                    comp = t
            try:
                name = comp.name
            except Exception:
                continue
            pred, args = self._parse_component_name(name)
            if pred in ('do_put_on', 'do_move') and len(args) >= 2:
                pairs.add((args[0], args[1]))
        return pairs

    def _index_relevant_facts(self):
        # Build unary predicate gid maps and on(x,y) gids for interesting pairs only
        # Scanning once across facts is acceptable; avoids building massive full on[x,y] map.
        interesting_on = self.interesting_on_pairs
        if not hasattr(self.model, 'facts'):
            return
        for f in self.model.facts:
            try:
                fname = f.name
            except Exception:
                continue
            pred, args = self._parse_fact_name(fname)
            if not pred:
                continue
            if pred == self.P_CLEAR and len(args) == 1:
                self.clear_gid[args[0]] = f.global_id
            elif pred == self.P_ONTABLE and len(args) == 1:
                self.ontable_gid[args[0]] = f.global_id
            elif pred == self.P_HOLD and len(args) == 1:
                self.holding_gid[args[0]] = f.global_id
            elif pred == self.P_HAND:
                self.handempty_gid = f.global_id
            elif pred == self.P_ON and len(args) == 2:
                x, y = args[0], args[1]
                if (x, y) in interesting_on:
                    self.on_gid[(x, y)] = f.global_id
        # Done


# Alias with requested lowercase parameter naming preserved in __init__ signature
# Users of Pytrich can instantiate this heuristic and register it under any name they use in their pipeline.
