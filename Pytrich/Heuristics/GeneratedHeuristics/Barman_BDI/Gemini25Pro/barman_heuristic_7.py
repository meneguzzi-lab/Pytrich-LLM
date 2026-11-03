import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini25ProHeuristic7(Heuristic):
    """
    A domain-dependent heuristic for the Barman domain.

    This heuristic estimates the remaining cost to achieve the planning goals by
    analyzing the decomposition method being applied at each search node. The core
    idea is to assign a small, admissible cost to each method based on the complexity
    of the sub-problem it solves.

    Rationale:
    The Barman domain involves a highly sequential process for creating cocktails.
    A single cocktail requires:
    1. Preparing ingredients (cleaning and filling shots).
    2. Combining ingredients in a shaker.
    3. Shaking the mixture.
    4. Preparing the serving glass (cleaning).
    5. Pouring the final cocktail.

    The heuristic assigns costs that reflect this hierarchy of tasks. Methods
    that decompose high-level tasks (like making a whole cocktail) receive a higher
    heuristic value than methods for low-level tasks (like picking up a glass).
    "Null" methods, which apply when a sub-goal is already met, are given a cost of
    zero, strongly prioritizing them to prune the search space.

    The costs are kept small (well below 1.0, the cost of a primitive action) to
    ensure the heuristic is admissible and can be used with A* search to find
    optimal plans.
    """
    def __init__(self, barman_gemini25pro_heuristic_7):
        """
        Initializes the heuristic and sets the penalty values for different
        categories of methods.
        
        The barman_gemini25pro_heuristic_7 parameter is for configuration, but default values
        are used here for simplicity.
        """
        super().__init__(name="barman_gemini25pro_heuristic_7")
        # Heuristic costs for different method types, reflecting task complexity.
        # Higher cost for more complex operations.
        self.penalties = {
            'cocktail_full': 0.5,     # Method to make and pour a full cocktail.
            'cocktail_make': 0.4,     # Method to just make the cocktail in the shaker.
            'add_ingredient': 0.2,    # Method to add one ingredient to the shaker (a multi-step process).
            'fill_shot': 0.1,         # Method to fill a shot from a dispenser.
            'cleaning': 0.05,         # Method for cleaning a container (frequent utility task).
            'manipulation': 0.01,     # Method for a basic action like grasping or dropping.
            'base_case': 0.0          # Method for a goal that is already satisfied (no cost).
        }
        
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing scores for all methods in the domain.
        This avoids expensive string comparisons during the search.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods from the model and assigns a heuristic
        score to each one based on its name, which indicates its role in the plan.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            score = self.penalties['manipulation']  # Default to a small cost.

            # Assign scores based on method names, from most complex to least.
            if 'Null' in method_name:
                score = self.penalties['base_case']
            elif 'MakeAndPourCocktail' in method_name:
                score = self.penalties['cocktail_full']
            elif 'MakeCocktail' in method_name:
                score = self.penalties['cocktail_make']
            elif 'AddIngredientTo' in method_name and 'Shaker' in method_name:
                score = self.penalties['add_ingredient']
            elif 'AddIngredientTo' in method_name and 'Shot' in method_name:
                score = self.penalties['fill_shot']
            elif 'Clean' in method_name:
                score = self.penalties['cleaning']
            elif 'DoPourShakerToShot' in method_name:
                score = self.penalties['fill_shot']
            elif any(s in method_name for s in ['PickUp', 'Holding', 'EmptyHand', 'HandEmpty', 'PutDown', 'OnTable']):
                score = self.penalties['manipulation']
            
            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        It retrieves the pre-computed score for the method applied in the node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        # Look up the precomputed score. Default to a small cost if not found.
        return self.method_scores.get(method_name, self.penalties['manipulation'])

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value
        
    def __repr__(self):
        return f"BarmanGemini25ProHeuristic7()"

    def __str__(self):
        return f"BarmanGemini25ProHeuristic7()"

    def __output__(self):
        """
        Returns a string with summary information about the heuristic.
        """
        return (
            f"Barman Gemini 2.5 Pro Heuristic 7 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
            f"\tPenalties: {self.penalties}\n"
        )
