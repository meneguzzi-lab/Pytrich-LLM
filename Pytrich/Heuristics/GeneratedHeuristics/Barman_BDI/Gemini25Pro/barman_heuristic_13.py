import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini25ProHeuristic13(Heuristic):
    """
    A domain-dependent heuristic for the Barman domain.

    This heuristic estimates the cost of a decomposition based on the method chosen.
    It assigns static penalties to methods based on the complexity and resource-intensiveness
    of the sub-plan they are expected to generate. The primary insight is that cleaning
    the single shaker is the main bottleneck in problems with multiple cocktail goals.
    Therefore, methods that lead to cleaning the shaker receive the highest penalty.

    The heuristic values are small floats, designed to guide the search by prioritizing
    simpler or more direct decomposition paths, while aiming to remain admissible.
    """
    def __init__(self, barman_gemini25pro_heuristic_13):
        super().__init__(name="barman_gemini25pro_heuristic_13")
        self.config = barman_gemini25pro_heuristic_13
        self.method_scores = {}
        self.preprocessing_time = 0

        # Static penalties associated with method types. Higher values indicate more
        # costly or complex operations that should be deprioritized if possible.
        self.penalties = {
            'clean_full_shaker': 1.0,
            'make_cocktail': 0.9,
            'clean_empty_shaker': 0.8,
            'make_and_pour': 0.7,
            'add_ingredient_to_shaker': 0.6,
            'clean_full_shot': 0.5,
            'add_ingredient_to_shot': 0.4,
            'pour_cocktail': 0.3,
            'clean_empty_shot': 0.2,
            'manipulation': 0.1,
            'base_case': 0.0
        }

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
        Analyzes all decomposition methods from the model and caches their heuristic scores.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions
        for method in methods:
            self.method_scores[method.name] = self._score_method_by_name(method.name)

    def _score_method_by_name(self, method_name):
        """
        Calculates a heuristic score for a method based on its name.
        The name often indicates the method's purpose and complexity.
        """
        method_lower = method_name.lower()

        # Null/base case methods represent an already achieved goal and have zero cost.
        if 'null' in method_lower:
            return self.penalties['base_case']

        # Cleaning methods are penalized heavily, especially for the shaker.
        if 'cleanfullshaker' in method_lower:
            return self.penalties['clean_full_shaker']
        if 'cleanemptyshaker' in method_lower:
            return self.penalties['clean_empty_shaker']
        if 'cleanfullshot' in method_lower:
            return self.penalties['clean_full_shot']
        if 'cleanemptyshot' in method_lower:
            return self.penalties['clean_empty_shot']

        # Cocktail preparation methods are complex and get high penalties.
        if 'makecocktail' in method_lower:
            return self.penalties['make_cocktail']
        if 'makeandpourcocktail' in method_lower:
            return self.penalties['make_and_pour']

        # Ingredient handling methods are intermediate steps.
        if 'addingredienttoemptyshaker' in method_lower or 'addingredienttousedshaker' in method_lower:
            return self.penalties['add_ingredient_to_shaker']
        if 'addingredienttoshot' in method_lower:
            return self.penalties['add_ingredient_to_shot']

        # Pouring is a key step but less complex than making the whole cocktail.
        if 'pour_shaker_to_shot' in method_lower:
             return self.penalties['pour_cocktail']

        # Basic manipulations have the lowest non-zero cost.
        if 'pickup' in method_lower or 'emptyhand' in method_lower or 'putdown' in method_lower:
            return self.penalties['manipulation']
            
        # Default to zero cost for any unclassified method.
        return self.penalties['base_case']

    def _calculate_node_heuristic(self, node):
        """
        Retrieves or computes the heuristic value for a node based on its method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        if method_name in self.method_scores:
            return self.method_scores[method_name]
        else:
            # Fallback for methods not seen during pre-computation.
            score = self._score_method_by_name(method_name)
            self.method_scores[method_name] = score
            return score

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic13(penalties={len(self.penalties)})"

    def __str__(self):
        return "BarmanGemini25ProHeuristic13"

    def __output__(self):
        return (
            f"Barman Gemini 2.5 Pro Heuristic 13 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
