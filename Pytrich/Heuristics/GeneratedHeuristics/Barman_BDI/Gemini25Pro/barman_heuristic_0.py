import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Barman domain.
class BarmanGemini25ProHeuristic0(Heuristic):
    """
    This heuristic for the Barman domain estimates the cost to reach a goal
    by analyzing the complexity of the methods used in the plan. It assigns
    a small, admissible cost (penalty) to each method based on the actions
    it likely entails, such as cocktail preparation, ingredient handling,
    cleaning, and basic manipulation.

    Rationale:
    1.  Hierarchy-Aware Costs: The cost penalties are designed to reflect the
        hierarchical nature of the tasks. High-level tasks like making a full
        cocktail receive a higher penalty than simpler, subordinate tasks like
        cleaning a glass or picking up an object. This guides the search
        towards completing necessary sub-goals in a logical order.
    2.  Focus on Key Challenges: The domain's main challenges are resource
        management (two hands) and state management (clean/dirty containers).
        Methods involving more complex state changes and resource coordination
        (e.g., `MakeCocktail`, `CleanFullShot`) are penalized more heavily
        than simple state changes (e.g., `PickUp`).
    3.  Admissibility: All heuristic values are small, positive costs,
        representing a fraction of the full plan cost. "Null" methods, which
        apply when a task is already complete, have a cost of zero. This
        ensures that the heuristic is admissible (h(n) <= h*(n)) and does not
        overestimate the true cost to the goal, which is crucial for
        optimal planners like A*.
    4.  Efficiency: The heuristic pre-computes the cost for each possible
        method during an initialization phase. During the search, it performs
        a quick lookup to retrieve the method's score, ensuring minimal
        computational overhead. This balances accuracy with performance,
        which is vital for large problem instances.
    """

    def __init__(self, barman_gemini25pro_heuristic_0):
        super().__init__(name="barman_gemini25pro_heuristic_0")
        self.method_penalties = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing the cost for each method
        in the domain model.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_penalties()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_penalties(self):
        """
        Analyzes all decomposition methods in the domain and assigns a
        pre-defined penalty based on their name, which is indicative of their
        complexity. Lower penalties are better.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name.lower() if hasattr(method, 'name') else str(method).lower()
            penalty = 0.0

            # Assign penalties based on method complexity.
            # Higher-level, more complex tasks get higher penalties.
            if "makeandpourcocktail" in method_name:
                penalty = 0.8
            elif "makecocktail" in method_name:
                penalty = 0.6
            elif "addingredienttousedshaker" in method_name or "addingredienttoemptyshaker" in method_name:
                penalty = 0.4
            elif "addingredienttoshot" in method_name:
                penalty = 0.3
            elif "cleanfull" in method_name: # Cleaning a full container is more work
                penalty = 0.2
            elif "cleanempty" in method_name:
                penalty = 0.15
            elif "dopourshakertoshot" in method_name:
                penalty = 0.1
            elif "pickup" in method_name or "emptyhand" in method_name or "putdown" in method_name:
                penalty = 0.05
            elif "null" in method_name: # Null methods mean the goal is already achieved
                penalty = 0.0
            else: # Default for any other method
                penalty = 0.1

            if hasattr(method, 'name'):
                self.method_penalties[method.name] = penalty

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        The value is based on the method that was used to generate this node.
        """
        # The heuristic value is the penalty of the method used to decompose the parent's task
        if not hasattr(node, 'method') or not node.method:
            return 0.0  # No method applied, e.g., the initial node

        method_name = node.method.name if hasattr(node.method, 'name') else None
        if method_name and method_name in self.method_penalties:
            return self.method_penalties[method_name]
        
        return 0.1 # Return a default small penalty if method not found

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic0(MethodsAnalyzed={len(self.method_penalties)})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Barman Gemini 2.5 Pro Heuristic 0 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_penalties)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
        )
