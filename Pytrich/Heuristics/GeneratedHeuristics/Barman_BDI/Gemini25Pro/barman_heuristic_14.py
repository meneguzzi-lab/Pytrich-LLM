import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini25ProHeuristic14(Heuristic):
    """
    A domain-dependent heuristic for the Barman domain.

    This heuristic evaluates the promise of a decomposition method based on its role
    in the cocktail-making process. It assigns a small, admissible cost to each method,
    guiding the search towards more direct and efficient paths to the goal.

    The core idea is to penalize actions that represent overhead (cleaning, manipulation)
    more than actions that directly contribute to creating and serving a cocktail.
    The heuristic values are precomputed for all methods in the domain during initialization
    to ensure fast evaluation during the search.

    Heuristic Design Rationale:
    1.  **Goal-Oriented Scoring**: Methods are scored based on their proximity to the final
        goal of serving a cocktail. Making a cocktail is preferred over preparing an
        ingredient, which is preferred over cleaning a container.
    2.  **Cost of Preparation**: Cleaning is a significant source of actions in the Barman
        domain, as containers must be clean before use. Therefore, methods involving
        cleaning are assigned the highest penalties.
    3.  **Manipulation Overhead**: Actions like picking up and putting down containers are
        necessary but represent planning overhead. They are given a small penalty.
    4.  **Base Cases**: Methods with 'Null' in their name represent an already-achieved
        subgoal. These are the most desirable and are assigned a cost of 0.
    5.  **Efficiency**: By precomputing scores for each method, the heuristic lookup is a
        constant-time operation during the search, minimizing computational overhead.
    """

    def __init__(self, barman_gemini25pro_heuristic_14=None):
        """
        Initializes the heuristic with predefined penalties for different types of methods.
        """
        super().__init__(name="barman_gemini25pro_heuristic_14")

        # Penalties are abstract costs assigned to method categories.
        # Lower values are better.
        self.penalties = {
            "cleaning": 0.6,        # High cost for cleaning actions
            "mixing": 0.5,          # Cost for shaking/making the cocktail
            "ingredient_add": 0.4,  # Cost for adding an ingredient to the shaker
            "ingredient_prep": 0.3, # Cost for getting an ingredient into a shot
            "serving": 0.2,         # Cost for the final pour
            "manipulation": 0.1,    # Low cost for pick/place actions
            "high_level": 0.8       # Cost for very abstract methods
        }

        self.method_scores = {}
        self.preprocessing_time = 0.0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-calculating the scores for all methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain and assigns a heuristic score
        to each one based on its name and function. The scores are stored for quick retrieval.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            name = method.name
            name_lower = name.lower()
            score = 0.0

            # Base cases (goal already satisfied) are the best, cost is 0.
            if "null" in name_lower:
                score = 0.0
            # High-level method that encapsulates the entire process.
            elif "makeandpourcocktail" in name_lower:
                score = self.penalties["high_level"]
            # Method for mixing ingredients in the shaker.
            elif "makecocktail" in name_lower:
                score = self.penalties["mixing"]
            # Cleaning methods are costly due to extra steps.
            elif "clean" in name_lower:
                score = self.penalties["cleaning"]
                if "full" in name_lower:
                    score += 0.1  # Add extra cost for emptying the container first.
            # Adding a prepared ingredient to the shaker.
            elif "addingredientto" in name_lower and "shaker" in name_lower:
                score = self.penalties["ingredient_add"]
            # Preparing an ingredient by filling a shot glass.
            elif "addingredientto" in name_lower and "shot" in name_lower:
                score = self.penalties["ingredient_prep"]
            # Serving the final cocktail.
            elif "pour_shaker_to_shot" in name_lower:
                score = self.penalties["serving"]
            # Basic manipulation of containers and hands.
            elif any(k in name_lower for k in ["pickup", "emptyhand", "putdown"]):
                score = self.penalties["manipulation"]
            else:
                # A small default cost for any uncategorized methods.
                score = self.penalties["manipulation"]

            self.method_scores[name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node by looking up the score
        of the method used for decomposition.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        method_name = node.method.name
        return self.method_scores.get(method_name, 0.1)  # Use default if not found

    def __call__(self, parent_node, node):
        """
        The main entry point for the heuristic evaluation.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic14()"

    def __str__(self):
        return f"BarmanGemini25ProHeuristic14 with penalties: {self.penalties}"

    def __output__(self):
        """
        Returns a formatted string with summary information about the heuristic.
        """
        info_str = (
            f"Barman Gemini 2.5 Pro Heuristic (BarmanGemini25ProHeuristic14) Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tPenalties Used:\n"
        )
        for penalty_name, value in self.penalties.items():
            info_str += f"\t\t- {penalty_name}: {value}\n"
        return info_str
