import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic1(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic calculates the number of rings that are not in their final goal positions.
    It is a goal-count heuristic, which is admissible because each 'move' action can
    at most place one ring in its correct final position, and the cost of each action is 1.
    Therefore, the heuristic value is always less than or equal to the true cost to reach the goal.

    Rationale:
    1.  **Admissibility:** The primary design goal is to create an admissible heuristic (h(n) <= h*(n))
        to guarantee optimal solutions with algorithms like A*. A simple count of misplaced rings
        satisfies this, as each move corrects the position of at most one ring.

    2.  **Simplicity and Efficiency:** The Towers of Hanoi problem has an exponential search space.
        A computationally inexpensive heuristic is crucial to avoid adding significant overhead
        to each node evaluation. This heuristic requires a single pass over the rings for each state.

    3.  **Domain Insight:** The core challenge in Towers of Hanoi is getting every ring to its designated
        spot. Counting how many are not there provides a direct, albeit simple, measure of progress
        towards the final goal configuration.

    4.  **Preprocessing:** The goal configuration is static and defined by `goal_on` predicates.
        This information is extracted once during the `initialize` phase and stored for quick
        access, making the heuristic calculation at each node faster.
    """

    def __init__(self, towers_gemini25pro_heuristic_1=None):
        """
        Initializes the TowersGemini25ProHeuristic1.

        :param towers_gemini25pro_heuristic_1: A placeholder parameter to match the specified naming convention.
        """
        super().__init__(name="towers_gemini25pro_heuristic_1")
        self.model = None
        self.goal_positions = {}
        self.rings = set()
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by preprocessing the planning model.
        It identifies all rings and their goal positions from the static predicates.

        :param model: The planning model containing facts, operators, etc.
        :param initial_node: The initial node of the search.
        :return: The result of the parent class's initialize method.
        """
        start_time = time.time()
        self.model = model

        # Precompute the set of rings and their final goal positions.
        # This information is static throughout the search.
        for fact in self.model.facts:
            fact_name = fact.name.strip("()").lower()
            parts = fact_name.split()
            if not parts:
                continue

            # Identify rings. A ring is any object that can be 'on' something,
            # 'smallerThan' something, or has a 'goal_on' state.
            if parts[0] in ["on", "goal_on", "smallerthan"] and len(parts) > 1:
                self.rings.add(parts[1])

            # Store the goal configuration from 'goal_on' predicates.
            if parts[0] == "goal_on" and len(parts) == 3:
                ring, on_object = parts[1], parts[2]
                self.goal_positions[ring] = on_object

        self.preprocessing_time = time.time() - start_time

        # Calculate the heuristic for the initial state.
        h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_heuristic(self, node: HTNNode) -> int:
        """
        Calculates the heuristic value for a given search node.
        The heuristic value is the number of rings not in their goal position.

        :param node: The HTNNode to evaluate.
        :return: The integer heuristic value.
        """
        # The heuristic value for a goal state is 0.
        if self.model.goal_reached(node.state, node.task_network):
            return 0

        current_facts = self.model.state_explicit_repr(node.state)
        current_positions = {}
        for fact_name in current_facts:
            if fact_name.lower().startswith("(on "):
                parts = fact_name.strip("()").lower().split()
                if len(parts) == 3:
                    ring, on_object = parts[1], parts[2]
                    current_positions[ring] = on_object

        misplaced_rings_count = 0
        for ring in self.rings:
            goal_pos = self.goal_positions.get(ring)
            current_pos = current_positions.get(ring)

            # A ring is misplaced if its current position is known but doesn't match the goal.
            if goal_pos is not None and current_pos != goal_pos:
                misplaced_rings_count += 1

        return misplaced_rings_count

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Callable method to compute the heuristic value for a node.

        :param parent_node: The parent node of the node being evaluated.
        :param node: The HTNNode to evaluate.
        :return: The integer heuristic value.
        """
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic1(rings={len(self.rings)}, goals={len(self.goal_positions)})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro Heuristic 1 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of rings tracked: {len(self.rings)}\n"
            f"\tNumber of goal positions: {len(self.goal_positions)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
