from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

"""
Domain-dependent heuristic for the Lamps domain.

This heuristic calculates the number of lamps that are not in their correct goal state.
This is equivalent to the Hamming distance between the current grid state and the goal grid state.

**Bottleneck Analysis:**
The primary bottleneck is flipping each of the misplaced lamps into its correct state. Each
such flip requires at least one 'flip' action, which is the primitive action and unit of cost.

**Admissibility Sketch:**
- The heuristic h(n) is the count of lamps (x, y) where the current state `on(x, y, s_current)` does not match the goal state `on(x, y, s_goal)`.
- To change the state of a single lamp at (x, y), the primitive action `flip(x, y, ...)` must be executed, either as the start of a 'turn' or as part of a propagation.
- Each `flip` action has a cost of 1 and can change the state of at most one lamp.
- Therefore, the number of mismatched lamps is a lower bound on the number of `flip` actions required to reach the goal. Since the total cost includes `flip` actions and non-negative method precondition costs, this heuristic is admissible.
"""

class LampsGemini25ProHeuristic1(Heuristic):
    """
    A heuristic for the Lamps domain that counts the number of lamps in an incorrect state.
    """

    def __init__(self, lamps_gemini25pro_heuristic_1=None, name="LampsGemini25ProHeuristic1"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Maps (x, y) coordinate tuple to the required goal status ('Y' or 'N')
        self.goal_states = {}

        # Maps (x, y) coordinate tuple to a dict of statuses and their fact objects
        # e.g., {(x,y): {"Y": fact_on_y, "N": fact_on_n}}
        self.coord_facts = {}

        # A set of all (x, y) coordinates that are part of the goal condition
        self.goal_coords = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before the search starts. Preprocesses the domain to build
        data structures for fast heuristic computation.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name like '+predicate[arg1,arg2]' into a tuple.
        Returns: (predicate, [arg1, arg2, ...]) or (None, []) if parsing fails.
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
        Extracts goal states and maps coordinates to 'on' facts from the model.
        This populates self.goal_states, self.coord_facts, and self.goal_coords.
        """
        # 1. Get goal state configuration from the bitwise goal representation
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, status = args[0], args[1], args[2]
                    self.goal_states[(x, y)] = status

        self.goal_coords = set(self.goal_states.keys())

        # 2. Map all coordinates to their corresponding "on" facts for fast lookups
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                coord = (x, y)
                if coord not in self.coord_facts:
                    self.coord_facts[coord] = {}
                self.coord_facts[coord][status] = fact

    def _compute(self, state_bitwise: int) -> int:
        """
        Calculates the heuristic value for a given state by counting mismatched lamps.
        """
        mismatch_count = 0
        if not self.goal_coords:
            return 0

        for coord in self.goal_coords:
            goal_status = self.goal_states.get(coord)
            # This should always be found, but defensive check is good.
            if goal_status is None:
                continue

            facts_for_coord = self.coord_facts.get(coord)
            # This check ensures that the lamp exists in the domain definition.
            if not facts_for_coord or "Y" not in facts_for_coord:
                continue

            fact_on_Y = facts_for_coord["Y"]

            # Check if the lamp is currently "Y" (on) using a fast bitwise operation
            is_on = (state_bitwise >> fact_on_Y.global_id) & 1
            current_status = "Y" if is_on else "N"

            if current_status != goal_status:
                mismatch_count += 1

        return mismatch_count

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a search node.
        This is called for every expanded node, so it must be very fast.
        """
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsGemini25ProHeuristic1(goals={len(self.goal_coords)})"

    def __str__(self):
        return f"LampsGemini25ProHeuristic1 with {len(self.goal_coords)} goal positions"

    def __output__(self):
        return (
            f"Lamps Gemini Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGoal Coords Tracked: {len(self.goal_coords)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
            f"\tInitial H-Value: {self.initial_h}\n"
        )
