import time
from Pytrich.Heuristics.heuristic import Heuristic

class BarmanGemini25ProHeuristic10(Heuristic):
    """
    A domain-dependent heuristic for the Barman HDDL domain.

    This heuristic estimates the cost to reach a goal by analyzing the complexity of the
    methods chosen during planning. It assigns a small, pre-calculated cost to each method
    based on the type of operation it represents (e.g., making a cocktail, cleaning a glass,
    manipulating an object). The core idea is that methods decomposing into more complex or
    numerous primitive actions should have a higher heuristic cost.

    Key Principles:
    1.  **Admissibility**: Costs are kept very small and are designed to be a lower bound on the
        actual plan cost. "Null" methods, which apply when a subgoal is already satisfied,
        are assigned a cost of zero.
    2.  **Hierarchy-Aware**: The heuristic values reflect the planning hierarchy. High-level tasks
        like making a full cocktail have a higher cost than sub-tasks like cleaning a shot glass.
    3.  **Efficiency**: Method costs are pre-computed during initialization to ensure that
        heuristic evaluation during the search is a fast dictionary lookup.
    4.  **Domain-Specific Insight**: The costs are based on the typical workflow in the Barman
        domain: making a cocktail is the most complex task, followed by preparing ingredients,
        cleaning, and finally basic object manipulation (picking up/putting down).
    """

    def __init__(self, barman_gemini25pro_heuristic_10):
        """
        Initializes the heuristic and defines the cost penalties for different categories of actions.
        """
        super().__init__(name="barman_gemini25pro_heuristic_10")
        # Define penalties for different categories of operations.
        # These values are chosen to reflect the relative complexity and ensure admissibility.
        self.penalties = {
            'cocktail_make': 0.5,    # Cost for the entire process of making a cocktail in a shaker
            'ingredient_add': 0.2,   # Cost for preparing and adding a single ingredient
            'clean': 0.1,            # Cost for cleaning a container (shot or shaker)
            'pour': 0.1,             # Cost for pouring liquid from one container to another
            'shake': 0.1,            # Cost for the shaking action
            'manipulate': 0.05,      # Cost for basic hand actions like grasping or dropping
        }
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing scores for all methods in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain model and assigns a heuristic score
        to each one based on its name and implied complexity.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            method_name_lower = method_name.lower()
            score = 0

            # Null methods correspond to already-satisfied goals, so their cost is 0.
            if "null" in method_name_lower:
                score = 0
            # Highest-level task: make a cocktail and pour it into a shot.
            elif "makeandpourcocktail" in method_name_lower:
                score = self.penalties['cocktail_make'] + self.penalties['clean'] + self.penalties['pour']
            # Making a cocktail in the shaker (requires 2 ingredients + shaking).
            elif "makecocktail" in method_name_lower:
                score = 2 * self.penalties['ingredient_add'] + self.penalties['shake']
            # Adding an ingredient to the shaker (requires prepping it in a shot first).
            elif "addingredienttoshaker" in method_name_lower:
                score = self.penalties['ingredient_add'] + self.penalties['pour']
            # Getting an ingredient from a dispenser into a shot glass.
            elif "addingredienttoshot" in method_name_lower:
                score = self.penalties['ingredient_add']
            # Cleaning any container.
            elif "clean" in method_name_lower:
                score = self.penalties['clean']
            # Basic object/hand manipulation.
            elif any(s in method_name_lower for s in ["pickup", "emptyhand", "putdown", "holding", "ontable"]):
                score = self.penalties['manipulate']
            # Specific task for pouring.
            elif "dopourshakertoshot" in method_name_lower:
                 score = self.penalties['pour']

            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        The value is based on the method being used for decomposition in that node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0
        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Retrieves the pre-computed score for a given method.
        """
        method_name = method.name
        return self.method_scores.get(method_name, 0) # Default to 0 if not found

    def __call__(self, parent_node, node):
        """
        The main entry point for the heuristic function.
        It calculates and returns the heuristic value for the current node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic10()"

    def __str__(self):
        return f"BarmanGemini25ProHeuristic10 with penalties: {self.penalties}"

    def __output__(self):
        """
        Provides a formatted string with summary information about the heuristic.
        """
        return (
            f"Barman Gemini 2.5 Pro Heuristic 10 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Scored: {len(self.method_scores)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
        )
