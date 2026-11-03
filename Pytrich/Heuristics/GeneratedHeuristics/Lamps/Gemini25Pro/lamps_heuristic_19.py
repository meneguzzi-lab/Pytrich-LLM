from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict
import time

"""
Domain-dependent heuristic for the Lamps domain.

This heuristic is based on the number of misplaced lamps, which provides an 
admissible lower bound on the number of actions required to solve the puzzle.

**1. Bottleneck Analysis**

The primary bottleneck in the Lamps domain is the combinatorial complexity of the 
`propagate` tasks. A single `flip` action chosen by a `turn` method can trigger a 
cascade of subsequent `flip`s, making the outcome of any single choice difficult to 
predict. The challenge is to find a sequence of initial `turn`s whose cumulative effects 
(including all propagations) result in the goal state.

**2. Admissible Lower Bound**

This heuristic uses a single, effective admissible lower bound:

*   **Term 1: Number of lamps in the wrong state (`N_wrong`).**
    *   **Admissibility Sketch:** The cost of a plan is the total number of primitive
        `flip` actions executed. Each `flip` action changes the status of exactly
        one lamp, for example, from `(on x y Y)` to `(on x y N)`. To fix a lamp
        that is in the wrong state, at least one `flip` action must be applied
        to it. Therefore, to fix `N_wrong` misplaced lamps, a minimum of `N_wrong`
        `flip` actions are required. This value is a consistent and admissible 
        lower bound `h(n) <= h*(n)` on the remaining cost.
"""


class LampsGemini25ProHeuristic19(Heuristic):
    """
    A heuristic for the Lamps domain that calculates the number of lamps
    that are not in their goal state.
    """

    def __init__(self, lamps_gemini25pro_heuristic_19=None, name="LampsGemini25ProHeuristic19"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Preprocessed domain information
        # Maps (x, y) coordinate tuple to its goal status ('Y' or 'N')
        self.goal_on_status = {}
        # Maps (x, y) to a dict {'Y': fact, 'N': fact} for quick lookup
        self.on_facts = defaultdict(dict)
        # A sorted list of all (x, y) coordinate tuples
        self.coords = []

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search starts. Preprocesses the model to build
        fast lookup tables for goal states and lamp facts.
        """
        start_time = time.time()
        self.model = model

        # Extract static domain information for fast lookups in __call__
        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Calculate the initial heuristic value
        h_value = self._compute_heuristic(initial_node.state)

        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name like '+predicate[arg1,arg2]' into a 
        (predicate, [args]) tuple.
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _preprocess(self):
        """
        Analyzes the model's facts and goals to build lookup dictionaries.
        - Populates `self.goal_on_status` with the target state of each lamp.
        - Populates `self.on_facts` to map each lamp coordinate to its corresponding
          '(on x y Y)' and '(on x y N)' fact objects.
        """
        # 1. Extract goal positions from the bitwise goal state
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, status = args
                    self.goal_on_status[(x, y)] = status

        # 2. Index all 'on' facts by coordinate for fast state checking
        coords_set = set()
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args
                coord = (x, y)
                coords_set.add(coord)
                self.on_facts[coord][status] = fact
        
        # Store a sorted list for deterministic iteration
        self.coords = sorted(list(coords_set))

    def _compute_heuristic(self, state_bitwise: int) -> int:
        """
        Calculates the heuristic value by counting how many lamps are not in
        their goal state.
        """
        h_value = 0
        for coord in self.coords:
            goal_status = self.goal_on_status.get(coord)
            if goal_status is None:
                # This coordinate is not part of the goal, so we don't care about its state.
                continue

            # Determine the current status ('Y' or 'N') of the lamp at `coord`
            # We assume 'Y' and 'N' are the only statuses and are mutually exclusive.
            fact_for_Y = self.on_facts[coord]['Y']
            is_on_Y = (state_bitwise >> fact_for_Y.global_id) & 1
            current_status = 'Y' if is_on_Y else 'N'

            if current_status != goal_status:
                h_value += 1
        
        return h_value

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        This method must be very fast as it's called for every node expansion.
        """
        h_value = self._compute_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"{self.name}(lamps={len(self.coords)})"

    def __str__(self):
        return f"{self.name} for a {len(self.coords)}-lamp grid"

    def __output__(self):
        return (
            f"{self.name} Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size (Num Lamps): {len(self.coords)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
