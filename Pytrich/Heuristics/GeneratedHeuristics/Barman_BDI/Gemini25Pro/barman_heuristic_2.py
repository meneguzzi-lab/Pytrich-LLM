import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini25ProHeuristic2(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Barman domain.

    Rationale:
    The Barman domain involves preparing cocktails, a process with significant overhead
    related to resource management. The main bottlenecks are:
    1.  **Container State:** Shakers and shots must be clean before use, requiring costly
        cleaning cycles if they are dirty.
    2.  **Hand Availability:** With only two hands, many actions are serialized. Operations
        requiring two hands (like shaking or cleaning) are particularly constrained.

    This heuristic guides the search by assigning a complexity score to each decomposition
    method. The score is a small, positive value intended to represent the "cost" or
    "overhead" introduced by a method, ensuring the heuristic remains admissible.

    Heuristic Strategy:
    - **Precomputation:** All methods are analyzed once during initialization. A score is
      calculated based on the types of subtasks in each method's decomposition and stored.
    - **Scoring Logic:**
      - **Null Methods:** Methods that achieve an already-satisfied goal (e.g., `CleanShotNull`)
        have zero cost and are always preferred.
      - **Penalize Overhead:** Methods that introduce cleaning subtasks (`AchieveClean...`) receive
        the highest penalty, as cleaning is pure overhead.
      - **Penalize Complexity:** Methods for core tasks like preparing ingredients (`AchieveContains...Ingredient`)
        or shaking receive moderate penalties, reflecting their multi-step nature.
      - **Reward Progress:** Methods involving pouring are given a small penalty, as they represent
        progress towards the final goal.
      - **Factor in Constraints:** Hand management tasks (`AchieveHolding`, `AchieveHandEmpty`)
        receive a minor penalty to account for resource contention.

    By using small, floating-point penalties, the heuristic provides a fine-grained gradient
    to guide the search towards more efficient plans (e.g., those that minimize cleaning)
    without violating the admissibility requirement for optimal planners.
    """
    def __init__(self, barman_gemini25pro_heuristic_2: dict = {}):
        """
        Initializes the heuristic with configurable penalties.

        :param barman_gemini25pro_heuristic_2: A dictionary of penalty values to tune the heuristic's behavior.
        """
        self.cleaning_penalty = barman_gemini25pro_heuristic_2.get('cleaning_penalty', 0.3)
        self.ingredient_penalty = barman_gemini25pro_heuristic_2.get('ingredient_penalty', 0.2)
        self.shake_penalty = barman_gemini25pro_heuristic_2.get('shake_penalty', 0.15)
        self.pour_penalty = barman_gemini25pro_heuristic_2.get('pour_penalty', 0.05)
        self.hand_management_penalty = barman_gemini25pro_heuristic_2.get('hand_management_penalty', 0.02)
        self.subtask_count_penalty = barman_gemini25pro_heuristic_2.get('subtask_count_penalty', 0.01)

        super().__init__(name="barman_gemini25pro_heuristic_2")
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-analyzing the domain's methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all methods in the domain to pre-calculate their heuristic scores.
        This avoids recalculating scores during the search, making evaluation very fast.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            self.method_scores[method_name] = self._score_method_structure(method)

    def _score_method_structure(self, method) -> float:
        """
        Calculates a heuristic score for a method based on its structural complexity.
        Lower scores are better. The score represents the method's inherent complexity and overhead.
        """
        method_name_lower = method.name.lower()

        # Null methods are perfect (cost 0), as they indicate a goal is already met.
        if 'null' in method_name_lower or not method.task_network:
            return 0.0

        # Start with a base penalty proportional to the number of subtasks.
        score = len(method.task_network) * self.subtask_count_penalty

        for subtask in method.task_network:
            subtask_name = subtask.name.lower()

            # Penalize cleaning tasks heavily as they are pure overhead.
            if 'achieveclean' in subtask_name:
                score += self.cleaning_penalty

            # Penalize ingredient preparation steps, which are multi-action tasks.
            elif 'achievecontainsshakeringredient' in subtask_name or 'achievecontainsshotingredient' in subtask_name:
                score += self.ingredient_penalty

            # Penalize shaking, a key but resource-intensive step (requires two hands).
            elif 'shake' in subtask_name:
                score += self.shake_penalty

            # Lightly penalize pouring, as it signifies progress towards the goal.
            elif 'pour' in subtask_name or 'dopour' in subtask_name:
                score += self.pour_penalty

            # Small penalty for hand management (picking up, putting down), which consumes resources.
            elif 'achieveholding' in subtask_name or 'achievehandempty' in subtask_name:
                score += self.hand_management_penalty

        return score

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        It retrieves the pre-calculated score for the method that generated the node.
        """
        # For primitive actions or initial nodes, the estimated future cost is 0.
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        # Retrieve the pre-calculated score for the method.
        method_name = node.method.name
        return self.method_scores.get(method_name, 1.0)  # Default to 1.0 if not found

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main heuristic evaluation function called by the planner.
        Returns the score of the method that produced the current node, guiding the search
        towards simpler and more direct decompositions.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic2(CleaningPenalty={self.cleaning_penalty}, IngredientPenalty={self.ingredient_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Barman Gemini 2.5 Pro Heuristic 2 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tCleaning Penalty: {self.cleaning_penalty}\n"
            f"\tIngredient Penalty: {self.ingredient_penalty}\n"
            f"\tShake Penalty: {self.shake_penalty}\n"
            f"\tPour Penalty: {self.pour_penalty}\n"
            f"\tHand Management Penalty: {self.hand_management_penalty}\n"
            f"\tSubtask Count Penalty: {self.subtask_count_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
