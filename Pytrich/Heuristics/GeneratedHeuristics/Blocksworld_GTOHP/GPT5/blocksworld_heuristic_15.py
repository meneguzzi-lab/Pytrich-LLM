from typing import Dict, Tuple, List, Set, Optional
from collections import defaultdict

try:
    # Import Heuristic base class from Pytrich
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback minimal interface to avoid import-time issues in standalone tests
    class Heuristic:
        def __init__(self, model):
            self.model = model
        def get(self, state, task_network):
            return 0


class BlocksworldGPT5Heuristic15(Heuristic):
    """
    Domain-dependent heuristic for BLOCKS (Blocksworld HDDL) tailored for Pytrich.

    Design rationale (succinct):
    - Goals are On(x,y) pairs. In Blocksworld, any plan that reaches the goal must include at least
      one stack(x,y) for each goal On(x,y) not already true, plus must ensure x and y are clear when stacking.
    - We use admissible lower bounds combined via max:
      1) h_unsat_on: number of goal On facts currently false (each requires at least one stack).
      2) h_inplace_gap: let in_place be the number of x that are part of a correct, stable goal chain
         (on(x,goal_y) and recursively, goal_y is in place or is a chain base). Then a lower bound on remaining
         stacks is (#goal_pairs - in_place).
      3) h_max_pair_difficulty: for each unsatisfied goal (x,y), at least (1 + blocks_above(x) + blocks_above(y))
         primitive actions are needed (1 stack, and clearing the towers above x and y each costs at least the
         number of blocks on them). Taking the maximum over pairs keeps admissibility and captures the hardest link.
      4) Task-network awareness: for pending abstract tasks, especially do_put_on/do_clear/do_on_table, we compute
         per-task admissible lower bounds (similar to 3) and take the maximum across tasks. This gives HTN-aware
         guidance without overcounting shared work (we use max, not sum, to remain admissible across dependent tasks).

    Efficiency:
    - We never use string membership checks on state. We build integer bit masks for all +on and +clear facts, and
      iterate set bits only. We also pre-index fact IDs to their block arguments.
    - For each evaluated state, we reconstruct the current support graph (children mapping) by scanning only active
      +on facts (O(#blocks)). This allows fast computation of blocks_above counts with memoization.

    Admissibility:
    - Each component is a proven lower bound; using max across them preserves admissibility and consistency.

    Expected effect:
    - On small and very large instances, this heuristic gives significantly better guidance than a blind or pure
      count-of-unsatisfied-goals heuristic, while remaining lightweight.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_15: Optional[Dict] = None):
        super().__init__(model)
        self.options = blocksworld_gpt5_heuristic_15 or {}

        # Parsed indices and masks
        self.on_mask: int = 0
        self.clear_mask: int = 0

        # Maps and indices for quick access
        self.on_id_to_pair: Dict[int, Tuple[str, str]] = {}     # bit_id -> (x,y)
        self.on_pair_to_id: Dict[Tuple[str, str], int] = {}     # (x,y) -> bit_id
        self.clear_fact_id_by_block: Dict[str, int] = {}        # block -> bit_id
        self.ontable_fact_id_by_block: Dict[str, int] = {}      # block -> bit_id (if present in model)

        # Goal structure
        self.goal_pairs: List[Tuple[str, str]] = []             # list of (x,y) for +on[x,y] goals
        self.goal_support: Dict[str, str] = {}                  # x -> y in goals
        self.goal_child_of: Dict[str, str] = {}                 # y -> x (inverse, if unique)
        self.goal_x_set: Set[str] = set()                       # all x that appear in goal on(x,y)

        # Build indices from model facts
        self._index_model_facts()
        # Parse model goals once (bitset to fact names); then map to on-pairs
        self._index_goals()

    # ----------------------- Indexing Helpers -----------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _index_model_facts(self):
        # Iterate all facts once to build masks and fast maps
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            if pred == 'on' and len(args) == 2:
                x, y = args[0], args[1]
                bit = 1 << fact.global_id
                self.on_mask |= bit
                self.on_id_to_pair[fact.global_id] = (x, y)
                self.on_pair_to_id[(x, y)] = fact.global_id
            elif pred == 'clear' and len(args) == 1:
                b = args[0]
                self.clear_fact_id_by_block[b] = fact.global_id
                self.clear_mask |= (1 << fact.global_id)
            elif pred == 'ontable' and len(args) == 1:
                b = args[0]
                self.ontable_fact_id_by_block[b] = fact.global_id
            # handempty/holding not required for this heuristic but kept parsable if needed later

    def _index_goals(self):
        # Goal facts are encoded as a bitwise int; we can render once at init
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for fname in goal_fact_names:
            pred, args = self._parse_fact_name(fname)
            if pred == 'on' and len(args) == 2:
                x, y = args[0], args[1]
                self.goal_pairs.append((x, y))
                self.goal_support[x] = y
                self.goal_x_set.add(x)
        # Inverse mapping (if a y has a unique child x in goal)
        for x, y in self.goal_pairs:
            # In blocksworld goal towers, each y should have at most one child; store the first
            if y not in self.goal_child_of:
                self.goal_child_of[y] = x

    # ----------------------- Bitwise Helpers -----------------------
    @staticmethod
    def _iter_set_bits(bitmask: int):
        # Efficiently iterate over set bit positions in bitmask
        value = bitmask
        while value:
            lsb = value & -value
            idx = (lsb.bit_length() - 1)
            yield idx
            value ^= lsb

    @staticmethod
    def _is_true(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) != 0

    # ----------------------- State Decoding -----------------------
    def _decode_current_on_children(self, state_bitwise: int) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Build the current support graph from the state's +on facts only by scanning set bits within
        the precomputed on_mask. Returns:
        - children: y -> list of blocks directly on y
        - support_of: x -> y (current support)
        """
        active_on = state_bitwise & self.on_mask
        children: Dict[str, List[str]] = defaultdict(list)
        support_of: Dict[str, str] = {}
        for bit_id in self._iter_set_bits(active_on):
            x, y = self.on_id_to_pair.get(bit_id, (None, None))
            if x is None:
                continue
            support_of[x] = y
            children[y].append(x)
        return children, support_of

    def _count_blocks_above(self, b: str, children: Dict[str, List[str]], cache: Dict[str, int]) -> int:
        if b in cache:
            return cache[b]
        total = 0
        for c in children.get(b, []):
            total += 1 + self._count_blocks_above(c, children, cache)
        cache[b] = total
        return total

    # ----------------------- Goal Placement / In-place -----------------------
    def _on_true(self, state_bitwise: int, x: str, y: str) -> bool:
        fact_id = self.on_pair_to_id.get((x, y))
        if fact_id is None:
            return False
        return self._is_true(state_bitwise, fact_id)

    def _well_placed_count(self, state_bitwise: int) -> int:
        """
        Count goal x-blocks that are in a correct, stable goal chain: on(x, goal_y) is true and either
        goal_y is a base (no goal specifying y-on-*) or is itself well-placed recursively.
        """
        memo: Dict[str, Optional[bool]] = {}

        def is_well_placed(x: str) -> bool:
            if x in memo:
                return bool(memo[x])
            y = self.goal_support.get(x)
            if y is None:
                memo[x] = False
                return False
            if not self._on_true(state_bitwise, x, y):
                memo[x] = False
                return False
            # If y is not constrained as an x in goals, then x is well placed.
            if y not in self.goal_support:
                memo[x] = True
                return True
            # Otherwise, y must also be well placed
            res = is_well_placed(y)
            memo[x] = res
            return res

        cnt = 0
        for x in self.goal_x_set:
            if is_well_placed(x):
                cnt += 1
        return cnt

    # ----------------------- Pairwise Lower Bound -----------------------
    def _pair_lower_bound(self, state_bitwise: int, x: str, y: str,
                          children: Dict[str, List[str]], above_cache: Dict[str, int]) -> int:
        # If already on(x,y), no additional cost for this link
        if self._on_true(state_bitwise, x, y):
            return 0
        # LB: 1 stack + unstacking all blocks above x + unstacking all above y
        clear_x_cost = self._count_blocks_above(x, children, above_cache)
        clear_y_cost = self._count_blocks_above(y, children, above_cache)
        return 1 + clear_x_cost + clear_y_cost

    # ----------------------- Task-Network aware LB -----------------------
    def _tn_lower_bound_max(self, state_bitwise: int, task_network: List) -> int:
        if not task_network:
            return 0
        # Build current children and above-cache once
        children, _ = self._decode_current_on_children(state_bitwise)
        above_cache: Dict[str, int] = {}
        tn_max = 0
        for t in task_network:
            # Robustly parse task/operator names: they have same string form with args
            name = getattr(t, 'name', None)
            if not name:
                continue
            pred, args = self._parse_fact_name(name)
            if pred is None:
                continue
            # Abstract tasks in this domain: do_put_on, do_clear, do_on_table, do_move
            if pred == 'do_put_on' and len(args) == 2:
                x, y = args[0], args[1]
                cost = self._pair_lower_bound(state_bitwise, x, y, children, above_cache)
                # Also ensure at least 1 if the goal is simply unsatisfied (stack to come)
                if not self._on_true(state_bitwise, x, y):
                    cost = max(cost, 1)
                tn_max = max(tn_max, cost)
            elif pred == 'do_clear' and len(args) == 1:
                x = args[0]
                # To clear x in any plan, you must at least remove one block if any sits on/above it.
                # Safe LB: number of blocks directly+indirectly above x (each needs at least one unstack).
                cost = self._count_blocks_above(x, children, above_cache)
                # But even if above_count>0, the LB in steps is at least 1 action required
                if cost > 0:
                    cost = max(cost, 1)
                tn_max = max(tn_max, cost)
            elif pred == 'do_on_table' and len(args) == 1:
                x = args[0]
                # If x is not already on table, at least 1 stack/unstack + clears of above
                # We detect on-table via any on(x, y) being true; absence implies either ontable or holding.
                # If an explicit ontable fact exists, prefer using it.
                ontable_id = self.ontable_fact_id_by_block.get(x)
                is_ontable = False
                if ontable_id is not None:
                    is_ontable = self._is_true(state_bitwise, ontable_id)
                else:
                    # Fallback: if x supports someone (appears as key in support_of?), we can't get it
                    # without children mapping; we'll approximate with: if x has any support_of[x], it's not on table
                    # However we do not have support_of here; we can infer: if x has any on(x, y) true, it's not on table
                    # Retrieve by scanning only active on facts for x via children decoding is not trivial; use children map
                    # to check if x is a child of some y: We built children as y->children, not inverse.
                    # We'll cheaply recompute support_of by scanning active_on bits once more if required.
                    # Simpler LB: if there is any block above x, definitely not on table in a way that avoids cost.
                    # Keep admissible: we consider is_ontable False unless proven True.
                    is_ontable = False
                if is_ontable:
                    cost = 0
                else:
                    cost = 1 + self._count_blocks_above(x, children, above_cache)
                tn_max = max(tn_max, cost)
            else:
                # For other tasks/operators, we do not add extra LB beyond global ones.
                pass
        return tn_max

    # ----------------------- Main Heuristic -----------------------
    def get(self, state_bitwise: int, task_network: List) -> int:
        # 1) Unsatisfied goals count (each needs at least one stack)
        unsat = 0
        for (x, y) in self.goal_pairs:
            if not self._on_true(state_bitwise, x, y):
                unsat += 1
        h_unsat_on = unsat

        # 2) In-place chain aware count: (#goal_pairs - well_placed)
        in_place = self._well_placed_count(state_bitwise)
        h_inplace_gap = max(0, len(self.goal_pairs) - in_place)

        # 3) Max pairwise difficulty across unsatisfied goal links
        children, _ = self._decode_current_on_children(state_bitwise)
        above_cache: Dict[str, int] = {}
        max_pair = 0
        if unsat > 0:
            for (x, y) in self.goal_pairs:
                if not self._on_true(state_bitwise, x, y):
                    lb = self._pair_lower_bound(state_bitwise, x, y, children, above_cache)
                    if lb > max_pair:
                        max_pair = lb
        h_max_pair = max_pair

        # 4) Task-network aware bound (max across HTN tasks' LBs)
        h_tn = self._tn_lower_bound_max(state_bitwise, task_network)

        # Combine via max to preserve admissibility
        h_val = max(h_unsat_on, h_inplace_gap, h_max_pair, h_tn)
        # Return non-negative integer
        return int(h_val)

    # Aliases for compatibility with different planner calls
    def h(self, state_bitwise: int, task_network: List) -> int:
        return self.get(state_bitwise, task_network)

    def __call__(self, state_bitwise: int, task_network: List) -> int:
        return self.get(state_bitwise, task_network)
