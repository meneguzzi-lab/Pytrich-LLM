from typing import Dict, Tuple, List, Optional, Set

# Robust import of the Heuristic base class across typical Pytrich layouts
try:
    from .Heuristic import Heuristic  # type: ignore
except Exception:
    try:
        from .heuristic import Heuristic  # type: ignore
    except Exception:
        try:
            from Pytrich.Heuristics.heuristic import Heuristic  # type: ignore
        except Exception:
            from Heuristics.heuristic import Heuristic  # type: ignore


class BlocksworldGPT5Heuristic19(Heuristic):
    """
    Blocksworld domain-dependent heuristic for Pytrich (HDDL/HTN planning).

    Design summary (admissible lower bound):
    - Goal-based bound:
      For each goal on(x,y) not yet true, at least one stack is needed, and at least
      one prior acquire-holding (pick-up or unstack) is needed. Thus a safe lower bound
      is 2 per unsatisfied on-goal. If the agent is already holding one of the needed x,
      we give a 1-action credit (stack directly), i.e., 2*U - 1 where U is the count of
      unsatisfied on-goals and the hold credit applies if holding matches any such x.
      If we are holding some unrelated block while we still need to achieve goals, we
      also need at least one extra put-down before any acquire, so we add +1 in that case.

      h_goals = 2 * (#unsatisfied on-goals) - (1 if holding matches some goal-x else 0)
                 + (1 if holding is not None and holding not in goal-Xs and unsatisfied>0 else 0)

      This is admissible and exploits the single-arm/one-held-block constraint.

    - Task-network bound:
      We additionally compute a lower bound from the current task network (HTN):
       * do_put_on(x,y) or do_move(x,y): 0 if on(x,y) already holds; else 1 if holding(x), else 2.
       * do_on_table(x): 0 if ontable(x) already holds; else 1 if holding(x), else 2.
       * do_clear(x): 0 if clear(x) already holds; else 1.
       * Primitive operators remaining in the TN: count 1 each except nop (0).
       * We add the same initial-put-down bound (if holding an unrelated block and there
         exists at least one task requiring an acquire-holding step).

      This leverages HTN structure without overcounting interactions. It is domain-specific
      yet keeps evaluation O(|goals| + |TN|).

    - Final heuristic value is max(h_goals, h_tasks), not a sum, to avoid double-counting
      when tasks and goals overlap (e.g., do_put_on tasks whose effects are also in the goals).

    Implementation details:
    - Facts are looked up via bitwise membership using precomputed Fact.global_id.
      We build fast maps for +on, +ontable, +clear, +holding, +handempty.
    - Parsing is robust to names like "do_put_on(b1,b2)", "do_put_on[b1,b2]", or spaced.
    - The heuristic is designed to scale to the largest provided instance.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_19: Optional[dict] = None):
        self.model = model
        self.config = blocksworld_gpt5_heuristic_19 or {}

        # Maps for quick fact lookup: (predicate, (args...)) -> Fact
        self.fact_pos_map: Dict[Tuple[str, Tuple[str, ...]], object] = {}
        # Specialized fast indices
        self.on_facts: Dict[Tuple[str, str], object] = {}
        self.ontable_facts: Dict[str, object] = {}
        self.clear_facts: Dict[str, object] = {}
        self.holding_facts: Dict[str, object] = {}
        self.handempty_fact: Optional[object] = None

        self._preprocess_facts()
        self.goal_on_list: List[Tuple[str, str, int]] = self._extract_goal_on_list()

    # ------------------------- Core Heuristic API -------------------------
    def evaluate(self, state_bitwise: int, task_network: List[object]) -> int:
        # Compute goal-based bound
        h_goals = self._goal_based_bound(state_bitwise)
        # Compute task-network based bound
        h_tasks = self._task_network_bound(state_bitwise, task_network)
        # Use max to avoid double counting
        return max(h_goals, h_tasks)

    # ------------------------- Preprocessing ------------------------------
    def _preprocess_facts(self) -> None:
        # Build positive-fact indices for fast bitwise checks
        for fact in getattr(self.model, 'facts', []):
            pred, args = self._parse_fact_name(getattr(fact, 'name', ''))
            if pred is None:
                continue
            if not getattr(fact, 'name', '').startswith('+'):
                # Only index positive facts; goals and state-true checks are over +facts
                continue
            key = (pred, tuple(args))
            self.fact_pos_map[key] = fact
            if pred == 'on' and len(args) == 2:
                self.on_facts[(args[0], args[1])] = fact
            elif pred == 'ontable' and len(args) == 1:
                self.ontable_facts[args[0]] = fact
            elif pred == 'clear' and len(args) == 1:
                self.clear_facts[args[0]] = fact
            elif pred == 'holding' and len(args) == 1:
                self.holding_facts[args[0]] = fact
            elif pred == 'handempty' and len(args) == 0:
                self.handempty_fact = fact

    def _extract_goal_on_list(self) -> List[Tuple[str, str, int]]:
        goal_names = self.model.state_explicit_repr(self.model.goals)
        goal_on: List[Tuple[str, str, int]] = []
        for name in goal_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'on' and len(args) == 2:
                key = (pred, tuple(args))
                fact = self.fact_pos_map.get(key)
                if fact is not None:
                    goal_on.append((args[0], args[1], fact.global_id))
        return goal_on

    # ------------------------- Goal-based bound ---------------------------
    def _goal_based_bound(self, state_bitwise: int) -> int:
        # Count unsatisfied on-goals and whether any required x is already held
        unsatisfied = 0
        unsat_xs: Set[str] = set()
        has_hold_on_goal_x = False

        held_block = self._get_held_block(state_bitwise)

        for x, y, fid in self.goal_on_list:
            if ((state_bitwise >> fid) & 1) == 1:
                continue  # already satisfied
            unsatisfied += 1
            unsat_xs.add(x)
            if held_block is not None and held_block == x:
                has_hold_on_goal_x = True

        if unsatisfied == 0:
            return 0

        hold_credit = 1 if has_hold_on_goal_x else 0
        # Base: one stack + one acquire per unsatisfied on-goal
        h = 2 * unsatisfied - hold_credit

        # If holding an unrelated block and still have goals, need at least one put-down
        if held_block is not None and held_block not in unsat_xs:
            h += 1

        return max(0, h)

    # ------------------------- Task-network bound -------------------------
    def _task_network_bound(self, state_bitwise: int, task_network: List[object]) -> int:
        h = 0
        held_block = self._get_held_block(state_bitwise)
        acquire_task_xs: Set[str] = set()  # x's from tasks that would require an acquire step

        for node in task_network:
            # Primitive operator: count at least 1 (except nop)
            if self._is_operator(node):
                op_name = getattr(node, 'name', '')
                pred, _ = self._parse_component_name(op_name)
                if pred == 'nop':
                    # nop has no effect; counting 0 keeps admissibility
                    continue
                h += 1
                continue

            # Abstract task: domain-specific minimal decompositions
            if self._is_abstract_task(node):
                name = getattr(node, 'name', '')
                pred, args = self._parse_component_name(name)
                if pred in ('do_put_on', 'do_move') and len(args) >= 2:
                    x, y = args[0], args[1]
                    if self._is_true(state_bitwise, 'on', (x, y)):
                        cost = 0
                    else:
                        if self._is_true(state_bitwise, 'holding', (x,)):
                            cost = 1  # can stack directly
                        else:
                            cost = 2  # need acquire + stack
                            acquire_task_xs.add(x)
                    h += cost
                elif pred == 'do_on_table' and len(args) >= 1:
                    x = args[0]
                    if self._is_true(state_bitwise, 'ontable', (x,)):
                        cost = 0
                    else:
                        if self._is_true(state_bitwise, 'holding', (x,)):
                            cost = 1  # can put-down directly
                        else:
                            cost = 2  # need unstack/pick-up + put-down
                            acquire_task_xs.add(x)
                    h += cost
                elif pred == 'do_clear' and len(args) >= 1:
                    x = args[0]
                    if self._is_true(state_bitwise, 'clear', (x,)):
                        h += 0
                    else:
                        h += 1  # at least one unstack to make x clear
                else:
                    # Unknown/other abstract tasks in this domain -> 0 (admissible fallback)
                    h += 0
            else:
                # Unrecognized node type -> be safe and assume 0
                h += 0

        # If holding some block that is not immediately useful for any acquire-needing task,
        # then at least one put-down is needed before we can pick-up/unstack for those tasks.
        if held_block is not None and len(acquire_task_xs) > 0 and held_block not in acquire_task_xs:
            h += 1

        return max(0, h)

    # ------------------------- Helpers ------------------------------------
    def _get_held_block(self, state_bitwise: int) -> Optional[str]:
        # Find which block (if any) is currently held
        for x, fact in self.holding_facts.items():
            if ((state_bitwise >> fact.global_id) & 1) == 1:
                return x
        return None

    def _is_true(self, state_bitwise: int, predicate: str, args: Tuple[str, ...]) -> bool:
        fact = self.fact_pos_map.get((predicate, tuple(args)))
        if fact is None:
            return False
        return ((state_bitwise >> fact.global_id) & 1) == 1

    @staticmethod
    def _parse_fact_name(name: str):
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name_ = name[1:]
            bracket_idx = name_.find("[")
            if bracket_idx > 0 and name_.endswith("]"):
                predicate = name_[:bracket_idx]
                args_str = name_[bracket_idx + 1 : -1]
                args = [a.strip() for a in args_str.split(",")] if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _parse_component_name(name: str) -> Tuple[str, List[str]]:
        # Accept names like 'do_put_on(b1,b2)', 'do_put_on[b1,b2]', or spaced 'do_put_on b1 b2'
        s = (name or '').strip()
        if not s:
            return '', []
        # Parentheses case
        for oc, cc in (("(", ")"), ("[", "]")):
            idx = s.find(oc)
            if idx != -1 and s.endswith(cc):
                pred = s[:idx].strip()
                inside = s[idx + 1 : -1].strip()
                if inside:
                    return pred, [a.strip() for a in inside.split(",")]
                return pred, []
        # Space-separated fallback
        toks = s.split()
        if len(toks) >= 2:
            return toks[0], toks[1:]
        return s, []

    @staticmethod
    def _is_operator(node: object) -> bool:
        # Duck-typing: operators usually have 'apply' and 'cost'
        return hasattr(node, 'apply') and hasattr(node, 'cost') and hasattr(node, 'pos_precons')

    @staticmethod
    def _is_abstract_task(node: object) -> bool:
        # Duck-typing: abstract tasks have 'decompositions' and 'name'; no 'apply'
        return hasattr(node, 'decompositions') and hasattr(node, 'name') and not hasattr(node, 'apply')
