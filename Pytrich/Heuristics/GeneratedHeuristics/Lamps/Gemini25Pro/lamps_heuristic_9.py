from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

"""
Domain-dependent heuristic for the Lamps domain.

This heuristic is designed to be admissible for the Lamps domain, which models a puzzle similar to "Lights Out".

Heuristic Analysis:

1.  **Domain Bottleneck:** The primary bottleneck is the cascading effect of the `flip` action. A single `turn` (which initiates a `flip` at a chosen coordinate) propagates to neighbors with the same state, causing a chain reaction. This makes it difficult to predict the full effect of an action and requires careful selection of flips to avoid undoing previous progress. The cost of a `turn` is not 1, but the total number of primitive `flip` actions it triggers.

2.  **Admissible Lower Bound (h_mismatch):** The heuristic calculates the number of lamps that are not in their goal state. 
    *   **Admissibility Sketch:** To solve the puzzle, every lamp currently in an incorrect state must have its state changed at least once. The only primitive action that changes a lamp's state is `flip`. Each `flip` action changes the state of exactly one lamp. Therefore, if `k` lamps are in the wrong state, a minimum of `k` `flip` actions are required to reach the goal. Since the planner's cost is the number of primitive actions, this count is an admissible lower bound on the remaining cost.

Implementation Details:
-   **Preprocessing (`initialize`):** The heuristic pre-processes the model to build efficient look-up tables. It maps each coordinate `(x, y)` and status `s` to the global ID of the corresponding `(on x y s)` fact. It also parses the goal state to create a map from each coordinate to its required goal status.
-   **Heuristic Calculation (`__call__`):** For a given state, the heuristic iterates through all lamp coordinates. For each lamp, it uses the precomputed maps and a fast bitwise check to see if the lamp's current state matches its goal state. It sums the number of these mismatches.

This heuristic follows the 'goal-count' pattern and is state-aware, providing a much better estimate than a blind or simple task-counting heuristic.
"""

class LampsGemini25ProHeuristic9(Heuristic):
    """
    An admissible heuristic for the Lamps domain based on counting mismatched lamps.
    """

    def __init__(self, lamps_gemini25pro_heuristic_9=None, name="LampsGemini25ProHeuristic9"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Precomputed data structures
        self.coords = []              # List of all (x, y) coordinates
        self.on_fact_ids = {}         # Map: (x, y, status) -> global_id
        self.goal_on_status = {}      # Map: (x, y) -> goal_status ("Y" or "N")

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called ONCE before search starts. Preprocesses the model to build fast lookup tables.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._compute(initial_node.state, initial_node.task_network)

        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name like '+predicate[arg1,arg2]' into (predicate, [arg1, arg2]).
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
        Extracts domain structure from the model's facts and goals.
        - Populates `self.on_fact_ids` for fast state checking.
        - Populates `self.goal_on_status` to know the target state for each lamp.
        """
        coords_set = set()

        # 1. Parse all facts to find coordinates and fact IDs
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                coords_set.add((x, y))
                self.on_fact_ids[(x, y, status)] = fact.global_id
        
        self.coords = sorted(list(coords_set))

        # 2. Parse goal state to get the target status for each lamp
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, status = args[0], args[1], args[2]
                    self.goal_on_status[(x, y)] = status

    def _compute(self, state_bitwise: int, task_network) -> float:
        """
        Calculates the number of lamps not in their goal state.
        """
        # If goal is met and task network is empty, cost is 0
        if self.model.goal_reached(state_bitwise, task_network):
            return 0.0

        mismatch_count = 0
        for x, y in self.coords:
            goal_status = self.goal_on_status.get((x, y))

            # If a lamp's goal state isn't specified, we don't count it.
            if goal_status is None:
                continue
            
            # Get the global_id for the fact representing the lamp being in its goal state
            fact_id_for_goal_state = self.on_fact_ids.get((x, y, goal_status))

            if fact_id_for_goal_state is not None:
                # Use a fast bitwise check to see if the lamp is in its goal state
                is_in_goal_state = (state_bitwise >> fact_id_for_goal_state) & 1
                if not is_in_goal_state:
                    mismatch_count += 1
        
        return float(mismatch_count)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculate heuristic value during search. This must be fast.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsGemini25ProHeuristic9(lamps={len(self.coords)})"

    def __str__(self):
        return f"LampsGemini25ProHeuristic9 with {len(self.coords)} lamps"

    def __output__(self):
        return (
            f"Lamps Mismatch Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Lamps: {len(self.coords)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
            f"\tInitial H-Value: {self.initial_h}\n"
        )
