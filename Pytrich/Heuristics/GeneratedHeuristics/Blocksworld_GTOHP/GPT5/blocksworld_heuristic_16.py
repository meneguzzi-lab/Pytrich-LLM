# Domain-dependent heuristic for BLOCKSWORLD (HDDL / HTN) in Pytrich
# Class name: BlocksworldGPT5Heuristic16
# Inherits from Heuristic
# __init__ parameter name: blocksworld_gpt5_heuristic_16 (all lowercase)
#
# Design rationale (short):
# - We compute an admissible (lower-bound) estimate of the remaining effort to achieve the goal
#   relations in Blocksworld, tailored to the provided hierarchical domain.
# - We use only bitwise checks for current state (fast) and parse the goal facts once at init.
# - Lower bound components per unsatisfied goal on(x,y):
#     1) At least one stack action is necessary to put x on y (s1)
#     2) If y is blocked (not clear), at least one unstack is necessary to clear y (s2)
#     3) If x has some block on top (i.e., x is not clear and not being held), at least one unstack is
#        necessary to clear x before acquiring it (s3)
#   The sum h = s1 + s2 + s3 is admissible because these represent disjoint necessary action types:
#   stacks (s1) are distinct from unstack actions; clearing the support y (s2) acts on y; clearing the
#   target x (s3) acts on x; a single unstack cannot simultaneously clear x and y.
# - We avoid over-counting acquisition steps that could overlap with support-clearing by not adding a
#   separate global "not holding x" term (which could double count when x is the blocker on some y).
# - Complexity: O(|goal_on|) per evaluation; suitable for large instances.
#
# Practical notes:
# - Goals are obtained from model.goals via model.state_explicit_repr(self.model.goals) as required.
# - We pre-index facts (on, clear, holding) by parsing Fact.name once during initialization.
# - We use only bitwise state checks ((state >> fact.global_id) & 1) during heuristic evaluation.

from typing import Dict, Tuple, List, Set, Optional

try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback for alternative import structures
    from .Heuristic import Heuristic  # type: ignore


class BlocksworldGPT5Heuristic16(Heuristic):
    """
    Domain-dependent admissible heuristic for the BLOCKSWORLD HDDL domain.

    Heuristic value is a sum of three disjoint lower bounds over unsatisfied goal on(x,y) facts:
      - s1: number of unsatisfied on(x,y) goals (>= 1 stack each)
      - s2: number of such goals whose support y is currently blocked (not clear[y]) (>= 1 unstack)
      - s3: number of such goals whose target x currently has a block on top (not clear[x] and not holding[x]) (>= 1 unstack)

    Final heuristic: h = s1 + s2 + s3

    This leverages Blocksworld structure and HTN methods (which always eventually realize stack/unstack
    patterns) while remaining efficient and scalable. We avoid counting acquisition actions separately
    to prevent double-counting with support-clearing unstack actions.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_16: bool = True):
        super().__init__(model)
        self.model = model
        self._enabled = blocksworld_gpt5_heuristic_16

        # Parsed fact indices
        # on[(x,y)] -> Fact
        self.on_facts: Dict[Tuple[str, str], object] = {}
        # clear[x] -> Fact
        self.clear_facts: Dict[str, object] = {}
        # holding[x] -> Fact
        self.holding_facts: Dict[str, object] = {}
        # handempty -> Fact (optional, not used directly in this heuristic)
        self.handempty_fact: Optional[object] = None

        # Goal on(x,y) pairs (from model.goals bitwise integer)
        self.goal_on_pairs: List[Tuple[str, str]] = []
        self.goal_on_fact_ids: Set[int] = set()

        # Small LRU cache for state heuristic values
        self._cache: Dict[int, int] = {}
        self._cache_order: List[int] = []
        self._cache_capacity: int = 4096

        # Build indices over facts
        self._index_facts()
        # Parse the goal bitset into concrete goal on(x,y) facts
        self._init_goals()

    def name(self) -> str:
        return "BlocksworldGPT5Heuristic16"

    # --------------- Utility parsing and indexing ---------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _index_facts(self):
        # model.facts is an indexable collection: facts[global_id] -> Fact
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None:
                continue
            if predicate == "on" and len(args) == 2:
                x, y = args[0].strip(), args[1].strip()
                self.on_facts[(x, y)] = fact
            elif predicate == "clear" and len(args) == 1:
                x = args[0].strip()
                self.clear_facts[x] = fact
            elif predicate == "holding" and len(args) == 1:
                x = args[0].strip()
                self.holding_facts[x] = fact
            elif predicate == "handempty":
                self.handempty_fact = fact

    def _init_goals(self):
        # model.goals is a bitwise int; need explicit repr to collect goal facts (strings)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for fname in goal_fact_names:
            predicate, args = self._parse_fact_name(fname)
            if predicate == "on" and len(args) == 2:
                x, y = args[0].strip(), args[1].strip()
                self.goal_on_pairs.append((x, y))
                # If this concrete on(x,y) fact exists in our fact table, record its id for fast checks
                f = self.on_facts.get((x, y))
                if f is not None:
                    self.goal_on_fact_ids.add(f.global_id)

    # --------------- Bitwise state helpers ---------------
    @staticmethod
    def _is_true(state_bitwise: int, fact_obj: Optional[object]) -> bool:
        if fact_obj is None:
            return False
        return (state_bitwise >> fact_obj.global_id) & 1 == 1

    # --------------- Heuristic evaluation ---------------
    def evaluate(self, state_bitwise: int, task_network: List[object] = None) -> int:
        if not self._enabled:
            return 0
        # Check cache
        cached = self._cache.get(state_bitwise, None)
        if cached is not None:
            return cached

        s1 = 0  # unsatisfied goal on(x,y) -> at least one stack per goal
        s2 = 0  # support y blocked -> at least one unstack per such y
        s3 = 0  # target x has some block on top -> at least one unstack per such x

        # For each goal on(x,y), compute components if not currently satisfied
        for (x, y) in self.goal_on_pairs:
            on_fact = self.on_facts.get((x, y))
            if on_fact is None:
                # If fact is missing in the model, conservatively treat as unsatisfied (still admissible)
                unsatisfied = True
            else:
                unsatisfied = ((state_bitwise >> on_fact.global_id) & 1) == 0

            if not unsatisfied:
                continue

            s1 += 1  # need at least one stack to set on(x,y)

            # If y is not clear, then y has a blocker that must be unstacked at least once
            y_clear_fact = self.clear_facts.get(y)
            if not self._is_true(state_bitwise, y_clear_fact):
                s2 += 1

            # If x has a block on top, we need at least one unstack to clear x (but don't count if holding[x])
            x_holding_fact = self.holding_facts.get(x)
            x_is_holding = self._is_true(state_bitwise, x_holding_fact)
            x_clear_fact = self.clear_facts.get(x)
            if (not x_is_holding) and (not self._is_true(state_bitwise, x_clear_fact)):
                s3 += 1

        h_val = s1 + s2 + s3

        # Update cache (simple LRU)
        if len(self._cache) >= self._cache_capacity:
            # Evict oldest
            oldest_key = self._cache_order.pop(0)
            self._cache.pop(oldest_key, None)
        self._cache[state_bitwise] = h_val
        self._cache_order.append(state_bitwise)

        return h_val

    # Optional: alias for engines expecting __call__
    def __call__(self, state_bitwise: int, task_network: List[object] = None) -> int:
        return self.evaluate(state_bitwise, task_network)
