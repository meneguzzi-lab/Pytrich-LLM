from typing import Dict, Tuple, Set, List, Optional, Union

# Pytrich base class (provided by the framework)
from Pytrich.Heuristics.heuristic import Heuristic


class BlocksworldGPT5Heuristic3(Heuristic):
    """
    Domain-dependent heuristic for HDDL Blocksworld in the Pytrich Planner.

    Key ideas (admissible lower bounds):
    - Any target on(x,y) requires exactly one 'stack' action to add on(x,y).
      Lower bound L_stack = number of required on(x,y) that are currently false.
    - Any target ontable(x) (from tasks or goals) requires exactly one 'put-down' to add ontable(x).
      Lower bound L_putdown = number of required ontable(x) that are currently false.
    - To clear a block b that is not clear, at least one 'unstack' is required (remove the block on top of b).
      We union all blocks that are required to be clear explicitly (do_clear tasks, clear goals),
      and blocks that must be clear to enable a stack (both x and y in do_put_on/do_move targets),
      then count how many of those are currently not clear. This yields L_unstack.

    These three lower bounds refer to disjoint action types (stack, put-down, unstack),
    thus they safely add without double counting effects across action types:
      LB_domain = L_stack + L_putdown + L_unstack

    Additionally, we consider the current task network's pending primitive operators:
      LB_ops = sum(cost(op) for op in TN if op is a primitive operator)
    This is a trivial admissible bound: you must execute at least these operators.

    Final heuristic value is:  h = max(LB_domain, LB_ops)

    This balances guidance between (a) domain-level invariants and (b) the concrete remainder
    of the current task network, while remaining admissible and efficient.

    Notes on parsing and performance:
    - We rely on the model's bitwise state. Every Fact has a global_id. We check truth via bitwise shifts.
    - Goal facts are retrieved via model.state_explicit_repr(self.model.goals) which returns ['+on[b1,b2]', ...].
    - We pre-index facts of interest: on[x,y], ontable[x], clear[x], holding[x], handempty.
    - We parse AbstractTask and Operator names robustly (accepting both [...] and (...)).

    Robustness across instances:
    - Scales to very large instances by O(1) bitwise checks and lightweight set logic.
    - Works both when the task network contains only abstract tasks or also pending primitives.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_3: bool = True):
        super().__init__(model)
        self.model = model
        self.blocksworld_gpt5_heuristic_3 = blocksworld_gpt5_heuristic_3

        # Index of facts by predicate and arguments for O(1) bitwise checks
        self.on_index: Dict[Tuple[str, str], int] = {}
        self.ontable_index: Dict[str, int] = {}
        self.clear_index: Dict[str, int] = {}
        self.holding_index: Dict[str, int] = {}
        self.handempty_id: Optional[int] = None

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            if pred == 'on' and len(args) == 2:
                self.on_index[(args[0], args[1])] = fact.global_id
            elif pred == 'ontable' and len(args) == 1:
                self.ontable_index[args[0]] = fact.global_id
            elif pred == 'clear' and len(args) == 1:
                self.clear_index[args[0]] = fact.global_id
            elif pred == 'holding' and len(args) == 1:
                self.holding_index[args[0]] = fact.global_id
            elif pred == 'handempty':
                self.handempty_id = fact.global_id

        # Pre-extract goal-derived requirements (static over the search space)
        self.goal_required_on: Set[Tuple[str, str]] = set()
        self.goal_required_ontable: Set[str] = set()
        self.goal_required_clear: Set[str] = set()

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred is None:
                continue
            if pred == 'on' and len(args) == 2:
                self.goal_required_on.add((args[0], args[1]))
            elif pred == 'ontable' and len(args) == 1:
                self.goal_required_ontable.add(args[0])
            elif pred == 'clear' and len(args) == 1:
                self.goal_required_clear.add(args[0])
        # Other goal predicates (holding/handempty) are ignored in this LB; could be added safely if desired.

    # -----------------------------
    # Public Heuristic API
    # -----------------------------
    def __call__(self, state_bitwise: int, task_network: List[Union[object, object]]) -> int:
        return self._estimate(state_bitwise, task_network)

    # -----------------------------
    # Core Heuristic Computation
    # -----------------------------
    def _estimate(self, state_bitwise: int, task_network: List[Union[object, object]]) -> int:
        # Extract TN-driven requirements and pending primitive op costs
        tn_req_on, tn_req_ontable, tn_req_clear, pending_ops_cost = self._collect_requirements_from_task_network(task_network)

        # Merge goal requirements with TN requirements
        req_on = set(self.goal_required_on)
        req_on.update(tn_req_on)

        req_ontable = set(self.goal_required_ontable)
        req_ontable.update(tn_req_ontable)

        req_clear = set(self.goal_required_clear)
        req_clear.update(tn_req_clear)

        # Domain action-type lower bounds
        lb_domain = self._domain_lower_bound(state_bitwise, req_on, req_ontable, req_clear)

        # Current pending primitive operators lower bound (exact for those operators)
        lb_ops = pending_ops_cost

        # Final admissible heuristic
        return max(lb_domain, lb_ops)

    # -----------------------------
    # Lower bound components
    # -----------------------------
    def _domain_lower_bound(self,
                            state_bitwise: int,
                            req_on: Set[Tuple[str, str]],
                            req_ontable: Set[str],
                            req_clear: Set[str]) -> int:
        # L_stack: one stack per required on(x,y) that is currently false
        l_stack = 0
        for (x, y) in req_on:
            fid = self.on_index.get((x, y))
            if fid is None:
                # If the fact is not in the model's universe, skip
                continue
            if not self._is_true(state_bitwise, fid):
                l_stack += 1

        # L_putdown: one put-down per required ontable(x) that is currently false
        l_putdown = 0
        for x in req_ontable:
            fid = self.ontable_index.get(x)
            if fid is None:
                continue
            if not self._is_true(state_bitwise, fid):
                l_putdown += 1

        # L_unstack: count distinct blocks that must be cleared and are not clear now
        # Blocks that must be clear include:
        # - explicit do_clear targets from TN/Goals
        # - x and y for any required on(x,y) (x must be clear to be held, y must be clear to receive x)
        blocks_to_clear: Set[str] = set()
        # From explicit clear requirements
        for b in req_clear:
            if not self._is_clear(state_bitwise, b):
                blocks_to_clear.add(b)
        # From on requirements
        for (x, y) in req_on:
            if not self._is_clear(state_bitwise, x):
                blocks_to_clear.add(x)
            if not self._is_clear(state_bitwise, y):
                blocks_to_clear.add(y)
        l_unstack = len(blocks_to_clear)

        return l_stack + l_putdown + l_unstack

    # -----------------------------
    # Task-network parsing
    # -----------------------------
    def _collect_requirements_from_task_network(self, task_network: List[Union[object, object]]):
        req_on: Set[Tuple[str, str]] = set()
        req_ontable: Set[str] = set()
        req_clear: Set[str] = set()

        pending_ops_cost: int = 0

        for t in task_network or []:
            # Identify Operators vs AbstractTasks by available attributes
            # Operator has: cost, precons, etc. AbstractTask has: decompositions. We use duck-typing on attributes.
            if hasattr(t, 'cost') and hasattr(t, 'add_effects'):
                # Primitive operator
                # Safely sum the operator's actual cost as admissible lower bound
                try:
                    op_cost = int(getattr(t, 'cost', 1))
                except Exception:
                    op_cost = 1
                pending_ops_cost += max(0, op_cost)
                # We do not add their implicit requirements here to avoid double counting with LB_domain
                continue

            # Abstract task: parse its name to detect block arguments
            if hasattr(t, 'name'):
                pred, args = self._parse_component_name(getattr(t, 'name'))
                if pred is None:
                    continue
                if pred == 'do_put_on' and len(args) >= 2:
                    req_on.add((args[0], args[1]))
                elif pred == 'do_move' and len(args) >= 2:
                    req_on.add((args[0], args[1]))
                elif pred == 'do_on_table' and len(args) >= 1:
                    req_ontable.add(args[0])
                elif pred == 'do_clear' and len(args) >= 1:
                    req_clear.add(args[0])

        return req_on, req_ontable, req_clear, pending_ops_cost

    # -----------------------------
    # Utility methods
    # -----------------------------
    def _is_true(self, state_bitwise: int, fact_id: int) -> bool:
        if fact_id is None:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    def _is_clear(self, state_bitwise: int, block: str) -> bool:
        fid = self.clear_index.get(block)
        if fid is None:
            return False
        return self._is_true(state_bitwise, fid)

    @staticmethod
    def _parse_fact_name(name: str):
        """
        Parse grounded fact name given in format "+pred[arg1,arg2]" or "-pred[arg1,arg2]".
        Returns (predicate, [args...]) or (None, []).
        """
        if not isinstance(name, str):
            return None, []
        s = name.strip()
        if s.startswith("+") or s.startswith("-"):
            s = s[1:]
        # Expect [...]
        lb = s.find('[')
        rb = s.rfind(']')
        if lb > 0 and rb > lb:
            pred = s[:lb].strip()
            args_str = s[lb + 1:rb].strip()
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return pred, args
        return None, []

    @staticmethod
    def _parse_component_name(name: str):
        """
        Robust parser for operator/abstract task names. Supports formats like:
          - do_put_on[b1,b2]
          - do_put_on(b1,b2)
          - stack[b1,b2]
          - stack(b1,b2)
        Returns (predicate, [args...]) or (None, []).
        """
        if not isinstance(name, str):
            return None, []
        s = name.strip()
        # Prefer [ ] first
        lb = s.find('[')
        rb = s.rfind(']')
        if lb > 0 and rb > lb:
            pred = s[:lb].strip()
            args_str = s[lb + 1:rb].strip()
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return pred, args
        # Fallback to ( )
        lb = s.find('(')
        rb = s.rfind(')')
        if lb > 0 and rb > lb:
            pred = s[:lb].strip()
            inner = s[lb + 1:rb].strip()
            # Accept either comma or whitespace separation
            if ',' in inner:
                args = [a.strip() for a in inner.split(',') if a.strip()]
            else:
                args = [a for a in inner.split() if a]
            return pred, args
        # If no bracket content, maybe it's a zero-arity symbol
        return s, []


# Factory function (optional, depending on framework usage)
def get_heuristic(model):
    return BlocksworldGPT5Heuristic3(model)
