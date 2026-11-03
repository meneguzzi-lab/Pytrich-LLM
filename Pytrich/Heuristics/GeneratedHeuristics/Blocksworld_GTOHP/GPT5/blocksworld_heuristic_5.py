from typing import Dict, Set, List, Tuple, Optional

try:
    from .Heuristic import Heuristic
except Exception:
    # Fallback for environments where relative import is not available
    from Pytrich.Heuristics.heuristic import Heuristic


class BlocksworldGPT5Heuristic5(Heuristic):
    """
    Domain-dependent heuristic for the BLOCKS (Blocksworld) HDDL domain.

    Design rationale (admissible, domain-specific, hierarchy-aware):
    - Each unsatisfied goal on(x,y) must be achieved by exactly one 'stack x y' action -> lower bound +1 each.
    - To execute a stack, we must be holding x (either via unstack or pick-up). If x is not currently held,
      at least one additional action is required somewhere in the plan to obtain x in hand. However, this action can
      sometimes coincide with a required 'clearance' unstack (e.g., unstacking a block that also happens to be a goal-top).
      To avoid double-counting while still being informative, we:
        * Add +1 per unsatisfied goal-top x if x is NOT currently held (prep cost), and
        * Count "clearance moves" as the number of unique non-goal-top blocks currently above any x or any y
          (i.e., blocks that must be moved at least once) using a union over relevant towers, explicitly excluding
          all goal-top blocks to avoid overlap with their per-goal prep cost.

    This yields an admissible estimate:
      h = (#unsatisfied on-goals)               [one stack per on-goal]
          + (#unsatisfied goal-tops not held)   [one pickup/unstack per x, if not already holding x]
          + (unique count of non-goal-top blocks that are initially above any goal x or y) [clearance unstack moves]

    Notes:
    - We parse the current state once (using model.state_explicit_repr(state)) to build immediate on/holding maps.
      We never do repeated membership tests over state lists (to avoid timeouts on large instances).
    - We pre-parse the goal facts once in __init__, and we only consider on-goal pairs (x,y).
    - We build the immediate "above" relation (who is on top of whom right now) and then collect blocks above any
      x or any y (with careful exclusions to maintain admissibility and avoid double counting across goals).
    - This heuristic is efficient enough for the largest provided instance (1,000 blocks) since it iterates only over
      true facts in the current state (O(n) with small constants) and goal literals (O(n)). The above-chain collection
      traverses tower chains that, in practice, are near-linear and visited minimally due to set-union aggregation.

    Implementation details:
    - Grounded Fact Format: "+predicate[arg1,arg2]" (see _parse_fact_name below).
    - Goals are accessed via model.state_explicit_repr(self.model.goals). We parse and cache goal on-pairs.
    - All state membership tests for facts use a single pass over state_explicit_repr(state) to build maps/sets.

    Admissibility: By design, each component of the sum is a necessary action count that is carefully de-overlapped.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_5: bool = True):
        super().__init__(model)
        self.enabled = blocksworld_gpt5_heuristic_5
        self.model = model

        # Parse and cache all goal on(x,y) pairs
        self.goal_on_pairs: List[Tuple[str, str]] = []
        self.goal_tops: Set[str] = set()  # set of all x that appear in on(x,y) goal literals

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                x, y = args[0], args[1]
                self.goal_on_pairs.append((x, y))
                self.goal_tops.add(x)
        # Precompute a set of all goal supports Y (not strictly required, but handy if needed later)
        self.goal_supports: Set[str] = {y for _, y in self.goal_on_pairs}

    # ------------------------------------------------------------------
    # Core heuristic API
    # ------------------------------------------------------------------
    def h(self, state_bitwise: int, task_network: List = None) -> int:
        if not self.enabled:
            return 0

        # Build current state mappings by scanning only true facts once
        # - support_of[x] = y if +on[x,y]
        # - above_of[y] = x if +on[x,y] (immediate above)
        # - holding_set = {x | +holding[x]}
        support_of: Dict[str, str] = {}
        above_of: Dict[str, str] = {}
        holding_set: Set[str] = set()

        fact_names = self.model.state_explicit_repr(state_bitwise)
        for fname in fact_names:
            pred, args = self._parse_fact_name(fname)
            if pred == 'on' and len(args) == 2:
                x, y = args[0], args[1]
                support_of[x] = y
                # There is at most one block immediately on any y in legal blocksworld states
                above_of[y] = x
            elif pred == 'holding' and len(args) == 1:
                holding_set.add(args[0])
            # We do not need clear/ontable/handempty explicitly for this heuristic

        # Helper: collect all blocks strictly above a given base block (following immediate "above_of")
        # Excludes any blocks in exclude_set (and continues traversal past them) to avoid double counting.
        def collect_above_excluding(base: str, exclude_set: Set[str]) -> Set[str]:
            res: Set[str] = set()
            cur = above_of.get(base)
            while cur is not None:
                if cur not in exclude_set:
                    res.add(cur)
                # climb further regardless, but always exclude nodes in exclude_set from the result
                cur = above_of.get(cur)
            return res

        # Compute unsatisfied goals, their prep-cost, and union of clearance-required moves
        unsat_goal_count = 0
        prep_cost = 0  # +1 for each unsatisfied goal-top x that is not currently held
        clearance_blocks: Set[str] = set()  # union of blocks (non-goal-tops) strictly above any goal x or y

        # Exclude all goal-tops from clearance to avoid overlap with their own prep-cost
        exclude_for_clearance: Set[str] = set(self.goal_tops)

        for (x, y) in self.goal_on_pairs:
            # If already satisfied, no cost from this goal
            if support_of.get(x) == y:
                continue

            unsat_goal_count += 1  # one stack to achieve on(x,y)

            # Prep cost: must get x into hand unless we already hold it
            if x not in holding_set:
                prep_cost += 1

            # Clearance: all non-goal-top blocks above x and above y must be moved at least once
            # - Above x: everything strictly above x (excluding all goal-tops)
            clearance_blocks |= collect_above_excluding(x, exclude_for_clearance)

            # - Above y: everything above y, excluding x and all goal-tops (avoid overlap with x's prep and other tops)
            exclude_xy = set(exclude_for_clearance)
            exclude_xy.add(x)
            clearance_blocks |= collect_above_excluding(y, exclude_xy)

        # Total heuristic value
        h_val = unsat_goal_count + prep_cost + len(clearance_blocks)

        # Optional tiny admissible tie-breaker: penalize remaining tasks very lightly without risking inadmissibility.
        # We keep it zero (commented) to remain strictly conservative across unknown task semantics.
        # if task_network:
        #     h_val += 0  # change to tiny weights if you want tie-breaking that remains admissible

        return int(h_val)

    # Provide __call__ for planner integrations that invoke heuristics as callables
    def __call__(self, state_bitwise: int, task_network: List = None) -> int:
        return self.h(state_bitwise, task_network)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact names of the form "+predicate[arg1,arg2]" or "-predicate[arg1,arg2]" into (predicate, [args]).
        Returns (None, []) if the format is unexpected.
        """
        if not name:
            return None, []
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name_ = name[1:]
            bracket_idx = name_.find("[")
            if bracket_idx > 0 and name_.endswith("]"):
                predicate = name_[:bracket_idx]
                args_str = name_[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []
