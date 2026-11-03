import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini25ProHeuristic11(Heuristic):
    """
    A domain-dependent heuristic for the Barman domain.

    This heuristic evaluates the quality of a decomposition method based on the
    subtasks it introduces. The core idea is to guide the search towards more
    efficient plans by penalizing methods that lead to common bottlenecks and
    rewarding methods that represent significant progress towards making a cocktail.

    Rationale:
    1.  **Bottleneck Penalties**: The Barman domain is constrained by a limited number
        of hands and the need to clean containers (shots, shakers) between uses.
        -   **Cleaning Penalty**: Methods introducing cleaning subtasks (`AchieveClean...`)
            are penalized because cleaning is non-productive overhead.
        -   **Manipulation Penalty**: Methods involving frequent grasping and releasing
            (`AchieveHolding`, `AchieveHandEmpty`) are penalized as they indicate
            complex hand management, a primary source of inefficiency.

    2.  **Progress Rewards**: Certain actions are key milestones in the cocktail-making
        process.
        -   **Progress Bonus**: Methods that include `shake` or `DoPourShakerToShot` are
            rewarded (given a negative penalty) because they represent irreversible
            progress towards the final product.

    3.  **Complexity Estimation**: The number of subtasks in a decomposition serves
        as a baseline estimate of the plan segment's complexity. A small penalty is
        applied per subtask to favor simpler, more direct methods.

    4.  **Admissibility**: The heuristic values are kept small to act primarily as a
        guiding force and tie-breaker in search algorithms like A*, minimizing the
        risk of making the heuristic inadmissible.
    """
    def __init__(self, barman_gemini25pro_heuristic_11=None):
        super().__init__(name="barman_gemini25pro_heuristic_11")
        if barman_gemini25pro_heuristic_11 is None:
            barman_gemini25pro_heuristic_11 = {}

        # Heuristic parameters taken from input dict, with defaults for robustness.
        self.subtask_penalty = barman_gemini25pro_heuristic_11.get("subtask_penalty", 0.05)
        self.cleaning_penalty = barman_gemini25pro_heuristic_11.get("cleaning_penalty", 0.2)
        self.manipulation_penalty = barman_gemini25pro_heuristic_11.get("manipulation_penalty", 0.1)
        self.progress_bonus = barman_gemini25pro_heuristic_11.get("progress_bonus", -0.1)
        self.model = None

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by storing a reference to the planning model.
        """
        self.model = model
        return super().initialize(model, 0)

    def __call__(self, parent_node, node):
        """
        Calculates the heuristic value for the given search node.
        This is the main entry point called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def _calculate_node_heuristic(self, node):
        """
        Computes the heuristic value. The heuristic is based on the decomposition
        method chosen at this node, as this represents a key decision point in the search.
        """
        # The heuristic estimates the cost of the *choice* (the method).
        # If no method is applied at this node, it's a primitive task, so no heuristic cost.
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Scores a decomposition method based on the complexity and type of its subtasks.
        A lower score is better, indicating a more promising path.
        """
        method_name = method.name.lower()

        # Null methods mean the goal is already satisfied. They are the best choice, cost = 0.
        if "null" in method_name:
            return 0

        # If a method has no subtasks (and isn't a null method), it's a terminal.
        if not hasattr(method, 'task_network') or not method.task_network:
            return 0

        score = 0.0

        # Add a base penalty for each subtask to prefer simpler decompositions.
        score += len(method.task_network) * self.subtask_penalty

        # Analyze subtasks for specific patterns to adjust the score.
        for subtask in method.task_network:
            subtask_name = subtask.name.lower()

            # Penalize cleaning operations.
            if "clean" in subtask_name:
                score += self.cleaning_penalty

            # Penalize low-level hand manipulations.
            if "achieveholding" in subtask_name or "achievehandempty" in subtask_name:
                score += self.manipulation_penalty

            # Reward key progress steps: shaking or pouring the final cocktail.
            if "shake" in subtask_name or "dopourshakertoshot" in subtask_name:
                score += self.progress_bonus

        # The heuristic value must be non-negative.
        return max(0, score)

    def __repr__(self):
        return (f"BarmanGemini25ProHeuristic11(subtask={self.subtask_penalty}, "
                f"cleaning={self.cleaning_penalty}, manip={self.manipulation_penalty}, "
                f"progress={self.progress_bonus})")

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Barman Domain Heuristic ('{self.name}') Info:\n"
            f"\tSubtask Penalty: {self.subtask_penalty}\n"
            f"\tCleaning Penalty: {self.cleaning_penalty}\n"
            f"\tManipulation Penalty: {self.manipulation_penalty}\n"
            f"\tProgress Bonus: {self.progress_bonus}\n"
        )
