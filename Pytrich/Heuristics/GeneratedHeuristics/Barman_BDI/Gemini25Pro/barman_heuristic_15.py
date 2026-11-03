import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini25ProHeuristic15(Heuristic):
    """
    A domain-dependent heuristic for the Barman HDDL domain.

    This heuristic estimates the cost of a decomposition method based on its
    semantic role in the cocktail-making process. It assigns higher costs to
    more complex, high-level tasks and lower costs to simpler, low-level
    manipulation or base-case methods. The goal is to guide the planner towards
    simpler and more direct solutions, prioritizing decompositions that are closer
    to primitive actions or that satisfy goals that are already met.

    The costs are precomputed for each method during initialization to ensure
    efficiency during the search. The heuristic values are small floats, designed
    to be admissible and provide gentle guidance without overriding the path cost.
    """
    def __init__(self, 
                 barman_gemini25pro_heuristic_15="barman_gemini25pro_heuristic_15",
                 manipulation_penalty=0.1,
                 clean_empty_penalty=0.15,
                 clean_full_penalty=0.2,
                 add_ingredient_shot_penalty=0.25,
                 add_ingredient_shaker_penalty=0.3,
                 make_cocktail_penalty=0.4,
                 serve_cocktail_penalty=0.5,
                 default_penalty=0.6):
        
        super().__init__(name=barman_gemini25pro_heuristic_15)
        self.manipulation_penalty = manipulation_penalty
        self.clean_empty_penalty = clean_empty_penalty
        self.clean_full_penalty = clean_full_penalty
        self.add_ingredient_shot_penalty = add_ingredient_shot_penalty
        self.add_ingredient_shaker_penalty = add_ingredient_shaker_penalty
        self.make_cocktail_penalty = make_cocktail_penalty
        self.serve_cocktail_penalty = serve_cocktail_penalty
        self.default_penalty = default_penalty
        
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute method costs.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precompute a cost for each decomposition method based on its name,
        which reflects its complexity and purpose in the Barman domain.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name
            score = self.default_penalty  # Default score for unrecognized methods

            # Lower score is better (lower cost)
            if "Null" in method_name:
                # Base case: The goal is already satisfied. No cost.
                score = 0.0
            elif "MakeAndPourCocktail" in method_name:
                # Top-level task: make and serve a cocktail. Highest complexity.
                score = self.serve_cocktail_penalty
            elif "MakeCocktail" in method_name:
                # Sub-goal: make a cocktail in the shaker. Involves multiple ingredients and shaking.
                score = self.make_cocktail_penalty
            elif "AddIngredientToUsedShaker" in method_name or "AddIngredientToEmptyShaker" in method_name:
                # Sub-goal: add an ingredient to the shaker. Requires preparing the ingredient first.
                score = self.add_ingredient_shaker_penalty
            elif "AddIngredientToShot" in method_name:
                # Sub-goal: prepare an ingredient in a shot glass. Requires cleaning and filling.
                score = self.add_ingredient_shot_penalty
            elif "CleanFull" in method_name:
                # Task: Clean a container that is full. Requires emptying first, so it's more expensive.
                score = self.clean_full_penalty
            elif "CleanEmpty" in method_name:
                # Task: Clean an empty container. Simpler than cleaning a full one.
                score = self.clean_empty_penalty
            elif any(task in method_name for task in ["CleanShot", "CleanShaker"]):
                # Generic cleaning task, assume intermediate complexity.
                score = self.clean_empty_penalty
            elif any(m in method_name for m in ["PickUp", "EmptyHand", "PutDown", "AchieveHolding", "AchieveHandEmpty", "AchieveOnTable"]):
                # Basic manipulation tasks involving hands and objects. Low cost.
                score = self.manipulation_penalty
            elif "DoPour" in method_name or "pour_shaker_to_shot_action" in method_name:
                # Pouring actions are also fundamental manipulations.
                score = self.manipulation_penalty
            
            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a node by looking up the precomputed
        score of the method used to generate this node's task network.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0
        
        method_name = node.method.name
        return self.method_scores.get(method_name, self.default_penalty)

    def __call__(self, parent_node, node):
        """
        Calculate and return the heuristic value for the given node.
        The value is kept small and non-negative to ensure admissibility.
        """
        h_value = self._calculate_node_heuristic(node)
        h_value = max(0.0, h_value)  # Ensure non-negative
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic15(name={self.name})"

    def __str__(self):
        return f"BarmanGemini25ProHeuristic15 - A cost-based heuristic for the Barman domain."

    def __output__(self):
        return (
            f"Barman Gemini 2.5 Pro Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
            f"\tPenalty Levels:\n"
            f"\t  Manipulation: {self.manipulation_penalty}\n"
            f"\t  Clean (Empty): {self.clean_empty_penalty}\n"
            f"\t  Clean (Full): {self.clean_full_penalty}\n"
            f"\t  Add Ingredient (Shot): {self.add_ingredient_shot_penalty}\n"
            f"\t  Add Ingredient (Shaker): {self.add_ingredient_shaker_penalty}\n"
            f"\t  Make Cocktail: {self.make_cocktail_penalty}\n"
            f"\t  Serve Cocktail: {self.serve_cocktail_penalty}\n"
        )
