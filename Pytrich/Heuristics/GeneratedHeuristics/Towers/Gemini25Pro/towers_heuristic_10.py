import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic10(Heuristic):
    """
    Domain-Dependent Heuristic for the Towers of Hanoi domain.

    This heuristic is a goal-counting heuristic. It calculates the number of rings
    that are not in their final, specified goal positions.

    Rationale:
    1.  **Admissibility**: The number of misplaced rings is a lower bound on the
        number of 'move' actions required to solve the puzzle. Each misplaced ring
        must be moved at least once. Therefore, h(n) <= h*(n).
    2.  **Efficiency**: The heuristic performs a one-time preprocessing step to
        identify the set of goal facts. For each state evaluation, it performs a
        fast check of the state's bitmask against the precomputed set of goal
        fact IDs. This is computationally very cheap.
    3.  **Informedness**: While simple, this heuristic is more informed than a
        zero heuristic. It effectively guides the search towards states that
        incrementally build the final goal tower. For a problem like Towers of
        Hanoi where the goal is a specific configuration, counting satisfied
        goal conditions is a direct measure of progress.
    4.  **Simplicity & Robustness**: By relying on the `model.goals` attribute
        provided by the Pytrich framework, the heuristic avoids complex parsing
        of the domain or problem files. It directly measures progress towards
        the planner's defined goal conditions.
    """
    def __init__(self, towers_gemini25pro_heuristic_10):
        """
        Initializes the TowersGemini25ProHeuristic10.

        :param towers_gemini25pro_heuristic_10: A dictionary of parameters for the heuristic (currently unused).
        """
        super().__init__(name="towers_gemini25pro_heuristic_10")
        self.params = towers_gemini25pro_heuristic_10
        self.model = None
        self.goal_fact_ids = set()
        self.num_goals = 0
        self.preprocessing_time = 0.0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by preprocessing the planning model.
        It identifies and stores the global IDs of the goal facts.

        :param model: The planning model.
        :param initial_node: The initial node of the search.
        :return: The result of the parent class's initialize method.
        """
        start_time = time.time()
        self.model = model

        # The model.goals attribute contains the set of Fact objects that represent the goal state.
        # We extract their global_ids for fast checking against the state bitmask.
        if model.goals:
            self.goal_fact_ids = {fact.global_id for fact in model.goals}

        self.num_goals = len(self.goal_fact_ids)
        self.preprocessing_time = time.time() - start_time

        # Calculate the heuristic value for the initial state.
        initial_h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, initial_h_value)

    def _calculate_heuristic(self, state: int) -> int:
        """
        Calculates the heuristic value for a given state.
        The value is the number of unsatisfied goal conditions (misplaced rings).

        :param state: The state bitmask to evaluate.
        :return: The integer heuristic value.
        """
        if self.num_goals == 0:
            return 0

        # Count how many of the goal facts are present in the current state.
        achieved_count = 0
        for fact_id in self.goal_fact_ids:
            # Check if the bit at the position of fact_id is set in the state.
            if (state >> fact_id) & 1:
                achieved_count += 1
        
        # The heuristic value is the number of goals not yet achieved.
        return self.num_goals - achieved_count

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        The main heuristic evaluation function called by the planner.

        :param parent_node: The parent node in the search tree.
        :param node: The current node to be evaluated.
        :return: The heuristic value for the current node's state.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic10(tracked_goals={self.num_goals})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        """
        Returns a formatted string with information about the heuristic.
        """
        return (
            f"Towers Gemini Heuristic 10 Info:\n"
            f"\tName: {self.name}\n"
            f"\tType: Goal-Counting (Misplaced Rings)\n"
            f"\tNumber of Goal Facts Tracked: {self.num_goals}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f} s\n"
        )
