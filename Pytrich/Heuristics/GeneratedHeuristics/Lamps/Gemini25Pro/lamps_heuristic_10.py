import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Lamps domain.

This heuristic calculates the Hamming distance between the current state and the goal state.

**Bottleneck Analysis**
The primary bottleneck in the Lamps domain is the complex, non-local effect of a single 'flip' action.
A flip at one coordinate can trigger state-dependent propagation chains, making it difficult to predict
the full consequences of an action and find a short path to the goal.

**Admissible Lower Bounds**
This heuristic uses a single, powerful lower bound:
1.  **Hamming Distance:** The number of lamps currently in a different state than their goal state.
    -   **Admissibility Sketch:** The plan cost is the total number of primitive 'flip' actions performed.
        Each 'flip' action, whether it's the start of a 'turn' or part of a propagation, changes the state
        of exactly one lamp. If 'k' lamps are in the wrong state, at least 'k' individual state changes
        are required to reach the goal. Therefore, 'k' is a strict lower bound on the total number of
        'flip' actions, and the heuristic is admissible.
"""


class LampsGemini25ProHeuristic10(Heuristic):
    """
    An admissible heuristic for the Lamps domain based on Hamming distance.

    This heuristic estimates the remaining cost by counting the number of lamps
    that are not in their required goal state. This is a lower bound on the total
    number of 'flip' actions required, as each misplaced lamp must have its state
    changed at least once.
    """

    def __init__(self, lamps_gemini25pro_heuristic_10=None, name="LampsGemini25ProHeuristic10"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # self.goal_state: maps (x, y) coord tuples to goal status 'Y' or 'N'
        self.goal_state = {}
        # self.on_fact_map: maps (x, y, status) tuples to their corresponding Fact object
        self.on_fact_map = {}
        # self.coords: a set of all (x, y) coordinate tuples in the grid
        self.coords = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Preprocesses the domain model to build fast lookup tables for goals and states.

        This method is called once before the search begins. It parses all 'on' facts
        to understand the grid structure and caches the goal state for efficient comparison.
        """
        start_time = time.time()
        self.model = model

        # 1. Preprocess all facts to build the coordinate set and the on_fact_map.
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.coords.add((x, y))
                self.on_fact_map[(x, y, s)] = fact

        # 2. Preprocess the goal state from model.goals bitmask.
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for goal_name in goal_fact_names:
            predicate, args = self._parse_fact_name(goal_name)
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.goal_state[(x, y)] = s

        self.preprocessing_time = time.time() - start_time

        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.

        This method is called for every expanded node and must be very fast.
        It relies on the precomputed tables from initialize().
        """
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return float(h_value)

    def _compute(self, state_bitwise: int) -> int:
        """
        Core heuristic calculation: counts the number of misplaced lamps.
        """
        misplaced_count = 0
        if not self.goal_state:
            return 0

        # Iterate through each lamp on the grid
        for x, y in self.coords:
            goal_s = self.goal_state.get((x, y))

            # If this lamp is not in the goal, we don't care about its state.
            if goal_s is None:
                continue

            current_s = None
            # Find the current state of the lamp at (x, y) using fast bitwise checks
            fact_y = self.on_fact_map.get((x, y, 'Y'))
            if fact_y and ((state_bitwise >> fact_y.global_id) & 1):
                current_s = 'Y'
            else:
                # If it's not 'Y', it must be 'N' (assuming a valid state)
                fact_n = self.on_fact_map.get((x, y, 'N'))
                if fact_n and ((state_bitwise >> fact_n.global_id) & 1):
                    current_s = 'N'

            if current_s is not None and current_s != goal_s:
                misplaced_count += 1

        return misplaced_count

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name into a (predicate, [args]) tuple.
        Example: '+on[l0,l0,Y]' -> ('on', ['l0', 'l0', 'Y'])
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

    def __repr__(self):
        return f"LampsGemini25ProHeuristic10(grid_size={len(self.coords)})"

    def __str__(self):
        return f"LampsGemini25ProHeuristic10 for a grid of size {len(self.coords)}"

    def __output__(self):
        return (
            f"Lamps Heuristic Info (LampsGemini25ProHeuristic10):\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {len(self.coords)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
            f"\tInitial H-Value: {self.initial_h}\n"
        )
