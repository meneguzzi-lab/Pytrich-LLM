import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic17(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic guides the planner by assigning costs to different decomposition
    methods based on their role in the recursive Towers of Hanoi algorithm. The core
    idea is to penalize methods that lead to deep recursion (moving large stacks of disks)
    more than those that represent base cases or primitive actions.

    The heuristic operates by analyzing method names, as direct state inspection is
    not assumed to be available or efficient within the heuristic function.

    - Base cases (e.g., 'exchangeClear') are given a cost of 0, as they terminate a
      planning sub-problem.
    - Primitive actions (wrapped in methods like 'newMethod21') are given a small,
      fixed cost.
    - Recursive methods (e.g., 'm-rotateTower', 'exchangeLR') that are essential for
      moving sub-towers are given the highest penalty, as they generate the most sub-tasks.
    """
    def __init__(self, towers_gemini25pro_heuristic_17=0.5, primitive_penalty=0.1, name="towers_gemini25pro_heuristic_17"):
        """
        Initializes the heuristic.
        :param towers_gemini25pro_heuristic_17: The penalty for methods involving deep recursion (e.g., rotating a tower).
        :param primitive_penalty: The penalty for methods that decompose into a single primitive action.
        :param name: The name of the heuristic.
        """
        super().__init__(name=name)
        self.recursive_penalty = towers_gemini25pro_heuristic_17
        self.primitive_penalty = primitive_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic with the model and precomputes method scores.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precomputes scores for all methods in the domain model for faster evaluation during search.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            score = self._score_method_by_name(method.name)
            self.method_scores[method.name] = score

    def _score_method_by_name(self, method_name):
        """
        Scores a method based on its name, which indicates its function in the Hanoi algorithm.
        Lower scores are better (lower estimated cost).
        """
        mn_lower = method_name.lower()
        if 'exchangeclear' in mn_lower:
            # Base case for the exchange task, meaning a tower is clear.
            # This terminates a recursive branch, so it has zero cost.
            return 0.0
        elif 'm-rotatetower' in mn_lower or 'exchangelr' in mn_lower or 'exchangerl' in mn_lower:
            # These methods implement or call the main recursive step of moving n-1 disks.
            # This is the most expensive part of the algorithm, so it gets the highest penalty.
            return self.recursive_penalty
        elif 'newmethod21' in mn_lower:
            # This method is a direct wrapper for the primitive 'move' action.
            # It represents a single, tangible step in the plan.
            return self.primitive_penalty
        elif 'm-shifttower' in mn_lower or 'selecteddirection' in mn_lower or 'm-selectdirection' in mn_lower:
            # These methods handle the logic for selecting which sub-tower to move next.
            # They are part of the recursive structure but less complex than moving the whole tower.
            # Assigning an intermediate penalty.
            return (self.recursive_penalty + self.primitive_penalty) / 2.0
        else:
            # Default case for any unrecognized methods.
            return self.primitive_penalty

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method = node.method
        # Use precomputed score if available, otherwise compute it on the fly.
        return self.method_scores.get(method.name, self._score_method_by_name(method.name))

    def __call__(self, parent_node, node):
        """
        Calculates the heuristic value for the node.
        The values are kept small and cost-based to guide the search effectively.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic(recursive_penalty={self.recursive_penalty}, primitive_penalty={self.primitive_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 17 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tRecursive Penalty: {self.recursive_penalty}\n"
            f"\tPrimitive Penalty: {self.primitive_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
