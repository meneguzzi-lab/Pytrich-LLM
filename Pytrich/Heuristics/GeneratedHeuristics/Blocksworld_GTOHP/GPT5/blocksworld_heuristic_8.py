import re
from typing import Dict, List, Optional, Tuple, Union

# Try to import the framework's Heuristic base class. Provide a small fallback for type checking.
try:
    from Heuristics.heuristic import Heuristic as BaseHeuristic
except Exception:
    try:
        from Pytrich.Heuristics.heuristic import Heuristic as BaseHeuristic  # alternative path in some setups
    except Exception:
        class BaseHeuristic:
            def __init__(self, model, name: str = "heuristic"):
                self.model = model
                self.name = name
            def evaluate(self, state, task_network=None) -> int:
                return 0


class BlocksworldGPT5Heuristic8(BaseHeuristic):
    """
    Domain-dependent heuristic for the BLOCKS world HDDL domain.

    Design rationale (concise):
    - In Blocksworld, achieving on(x,y) requires exactly one stack(x,y) action; no other action adds on(x,y).
      Therefore a safe and very cheap lower bound is the count of goal on-pairs that are not yet true.
      This yields an admissible h_g (goal-based lower bound).
    - In HTN settings, we must also estimate the minimum remaining effort to refine the current task network.
      We provide h_tn (task-network lower bound) by:
        * Adding the exact costs of any primitive operators already in the task network (since they must be executed).
        * For each remaining abstract task, counting a cheap, type-specific lower bound:
            - do_put_on(x,y) or do_move(x,y): at least one stack is eventually needed if on(x,y) not already true.
            - do_on_table(x): ontable(x) comes from put-down(x); at least one action if it's not already on table.
            - do_clear(x): if not clear(x), at least one action (e.g., unstack something from x or put/stack x) is needed.
          Unknown/other abstract tasks are conservatively assigned a lower bound of 1.
      Summing these gives a safe lower bound on the cost still required to complete the task network, h_tn.
    - The search must end with both the goals satisfied AND the task network empty; hence the overall lower bound is
      the maximum of the two independent lower bounds: h(n) = max(h_g, h_tn). This keeps the heuristic admissible.

    Efficiency:
    - All state membership checks use bitwise tests via global_id (no string membership on states).
    - Goal facts are precompiled to a list of fact IDs for O(1) bit checks during evaluate().
    - Fact lookup is indexed by (predicate, tuple(args)).

    Notes:
    - This heuristic is intentionally simple (fast to compute) yet informative. On large instances, counting unsatisfied
      on-goals provides strong guidance; refining abstract tasks with minimal per-type costs helps focus decomposition.
    - The heuristic is domain-dependent but robust to variations in task name formatting by extracting block symbols via regex.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_8: bool = True):
        super().__init__(model, name="BlocksworldGPT5Heuristic8")

        # Index facts for O(1) bit checks by (predicate, tuple(args))
        self._fact_index: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        for f in getattr(self.model, 'facts', []):
            if f is None:
                continue
            pred, args = self._parse_fact_name(f.name)
            if pred:
                key = (pred, tuple(args))
                self._fact_index[key] = f.global_id

        # Precompute goal on-fact IDs (one stack needed per unsatisfied on)
        self._goal_on_fact_ids: List[int] = []
        try:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        except Exception:
            goal_fact_names = []
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                fid = self._fact_index.get((pred, tuple(args)))
                if fid is not None:
                    self._goal_on_fact_ids.append(fid)

    # -------------------------- Core Heuristic API --------------------------
    def evaluate(self, state_bitwise: int, task_network: Optional[List[object]] = None) -> int:
        if state_bitwise is None:
            return 0

        # Lower bound due to goals: count how many goal on(x,y) are not yet true
        h_goal = 0
        for fid in self._goal_on_fact_ids:
            if ((state_bitwise >> fid) & 1) == 0:
                h_goal += 1

        # Lower bound to refine the remaining task network (HTN part)
        h_primitive = 0
        h_abstract = 0
        if task_network:
            for t in task_network:
                # Primitive operator: sum its exact cost
                if self._is_operator(t):
                    try:
                        h_primitive += int(getattr(t, 'cost', 1))
                    except Exception:
                        h_primitive += 1
                    continue

                # Abstract task: add a cheap, type-specific lower bound
                if self._is_abstract_task(t):
                    tname = getattr(t, 'name', '')
                    ttype = self._get_task_type(tname)
                    blocks = self._extract_blocks(tname)

                    if ttype in ('do_put_on', 'do_move'):
                        # Need on(x,y); if not already true, at least 1 stack
                        if len(blocks) >= 2:
                            x, y = blocks[0], blocks[1]
                            if not self._is_true(state_bitwise, 'on', (x, y)):
                                h_abstract += 1
                        else:
                            # Unknown arguments -> still assume at least 1 action
                            h_abstract += 1

                    elif ttype == 'do_on_table':
                        # ontable(x) must be true; otherwise at least one put-down
                        if len(blocks) >= 1:
                            x = blocks[0]
                            if not self._is_true(state_bitwise, 'ontable', (x,)):
                                h_abstract += 1
                        else:
                            h_abstract += 1

                    elif ttype == 'do_clear':
                        # If clear(x) not true, at least one action is required
                        if len(blocks) >= 1:
                            x = blocks[0]
                            if not self._is_true(state_bitwise, 'clear', (x,)):
                                h_abstract += 1
                        else:
                            h_abstract += 1

                    else:
                        # Unknown abstract task: conservatively assume at least one action
                        h_abstract += 1

                # If it's neither clearly an operator nor an abstract task, be conservative: no addition.
                # This avoids any risk of overestimation from misclassification.

        h_tn = h_primitive + h_abstract

        # Overall admissible lower bound
        return max(h_goal, h_tn)

    # ------------------------------ Utilities ------------------------------
    def _is_operator(self, obj) -> bool:
        # Duck-typing: Operators have 'apply' and 'cost' in the provided framework
        return hasattr(obj, 'apply') and hasattr(obj, 'cost') and hasattr(obj, 'add_effects')

    def _is_abstract_task(self, obj) -> bool:
        # Duck-typing: AbstractTask has 'decompositions'
        return hasattr(obj, 'decompositions') and hasattr(obj, 'name')

    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact names of the form '+predicate[arg1,arg2]' or '-predicate[arg1,arg2]'.
        Returns (predicate, [args]) or (None, []).
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

    def _get_fact_id(self, predicate: str, args: Tuple[str, ...]) -> Optional[int]:
        return self._fact_index.get((predicate, args))

    def _is_true(self, state_bitwise: int, predicate: str, args: Tuple[str, ...]) -> bool:
        fid = self._get_fact_id(predicate, args)
        if fid is None:
            return False
        return ((state_bitwise >> fid) & 1) == 1

    def _get_task_type(self, name: str) -> str:
        lname = (name or '').lower()
        for key in ('do_put_on', 'do_move', 'do_on_table', 'do_clear'):
            if key in lname:
                return key
        return ''

    def _extract_blocks(self, name: str) -> List[str]:
        """
        Extract block symbols robustly from a task name. Works with formats such as:
          - 'do_put_on b4 b2'
          - 'do_put_on(b4,b2)'
          - 'do_put_on__b4__b2'
          - Any other string containing tokens like 'b123'
        Returns a list like ['b4', 'b2'] in the order they appear.
        """
        if not isinstance(name, str):
            return []
        return re.findall(r"\b[bB]\d+\b", name)

    # Optional: provide a string representation for debugging
    def __repr__(self):
        return f"<BlocksworldGPT5Heuristic8 name={self.name}>"
