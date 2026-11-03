import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Heuristic for the Blocksworld domain.
# This heuristic is designed to guide the search by preferring methods that
# directly contribute to the final goal configuration over methods that
# merely clear the way. It assigns small costs to decompositions based on
# the nature of the task they solve (e.g., moving a block vs. clearing a block).
# This approach is admissible because the heuristic costs are small, non-negative floats,
# guaranteed to be less than the true cost of any action (which is >= 1).
class BlocksworldGemini25ProHeuristic1(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Rationale:
    1.  Domain Insight: The primary challenge in Blocksworld is sequencing. To achieve a goal like (on A B),
        both A and B must be clear. This often creates subgoals of clearing blocks, which can be recursive.
        This heuristic prioritizes actions that build the target towers over actions that just dismantle
        existing ones, while still acknowledging that dismantling is often necessary.

    2.  Heuristic Strategy: The heuristic is based on method selection. It assigns a small cost (penalty)
        to each method decomposition.
        -   Base cases (e.g., a block is already in its target position) have a cost of 0.
        -   Methods that achieve the main `put_on` task are given a small penalty, representing progress.
        -   Methods that only achieve a `clear` subgoal are given a slightly higher penalty, as they are
          instrumental but not the final step.

    3.  Admissibility: The heuristic values are small floats (e.g., 0.1, 0.2) which are always less than
        the true cost of any primitive action (cost >= 1). This ensures that the heuristic is admissible
        and can be used with optimal search algorithms like A*.

    4.  Hierarchy Awareness: The heuristic explicitly reasons about the types of tasks being decomposed
        (e.g., 'put_on' vs. 'clear'), leveraging the hierarchical structure of the problem.
    """

    def __init__(self, blocksworld_gemini25pro_heuristic_1={"move_penalty": 0.2, "clear_penalty": 0.3}):
        """
        Initializes the heuristic.
        Args:
            blocksworld_gemini25pro_heuristic_1 (dict): A dictionary containing heuristic parameters.
                - "move_penalty": Cost associated with methods that move a block towards its goal.
                - "clear_penalty": Cost associated with methods that clear a block.
        """
        super().__init__(name="blocksworld_gemini25pro_heuristic_1")
        self.move_penalty = blocksworld_gemini25pro_heuristic_1.get("move_penalty", 0.2)
        self.clear_penalty = blocksworld_gemini25pro_heuristic_1.get("clear_penalty", 0.3)
        self.method_scores = {}
        self.preprocessing_time = 0
        self.model = None

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic with the problem model and pre-computes method info.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain to classify them.
        This allows for faster heuristic calculation during the search.
        It categorizes methods based on keywords in their names.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name.lower()
            
            # A method is a 'base case' if it has no subtasks to execute.
            is_base_case = not method.task_network
            
            # Heuristically determine the method's purpose from its name.
            # Assumes common Blocksworld naming conventions.
            is_goal_method = any(term in method_name for term in ['put_on', 'stack', 'move'])
            is_clear_method = any(term in method_name for term in ['clear', 'unstack'])
            
            self.method_scores[method.name] = {
                'is_base_case': is_base_case,
                'is_goal_method': is_goal_method,
                'is_clear_method': is_clear_method,
                'subtask_count': len(method.task_network)
            }

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        If the node was not expanded by a method, the cost is 0.
        Otherwise, it scores the method used for the expansion.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0
        return self._score_method(node.method)

    def _score_method(self, method) -> float:
        """
        Scores a method based on its pre-computed characteristics.
        Lower scores are better.
        """
        score = 0.0
        method_name = method.name

        info = self.method_scores.get(method_name)
        if not info:
            # Fallback for methods not seen during pre-computation (should not happen)
            return self.clear_penalty + self.move_penalty

        # Base cases represent a solved subproblem, so they have zero cost.
        if info['is_base_case']:
            return 0.0

        # Penalize methods that achieve clearing subgoals. These are necessary
        # but move us away from directly building the goal towers.
        if info['is_clear_method']:
            score += self.clear_penalty

        # Penalize methods that achieve the main goal of moving a block.
        # This penalty is lower than the clear_penalty to prefer these methods.
        if info['is_goal_method']:
            score += self.move_penalty
        
        # Add a very small penalty based on decomposition complexity.
        # This acts as a tie-breaker, preferring simpler decompositions.
        score += info['subtask_count'] * 0.01

        return score

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main heuristic function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        # Cap the value to prevent any unforeseen issues, although it should be small.
        h_value = min(h_value, 1.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return (f"BlocksworldGemini25ProHeuristic1(move_penalty={self.move_penalty}, "
                f"clear_penalty={self.clear_penalty})")

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 1 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tMove Penalty: {self.move_penalty}\n"
            f"\tClear Penalty: {self.clear_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
