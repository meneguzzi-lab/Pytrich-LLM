import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini25ProHeuristic8(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Barman domain.

    This heuristic provides a cost estimate for search nodes based on the complexity
    of the decomposition method being applied. It analyzes the hierarchical structure
    of the Barman domain and assigns higher heuristic values (costs) to methods that
    tackle more complex, high-level tasks, and lower costs to those handling simpler,
    low-level sub-tasks.

    The core idea is to guide the planner to solve necessary prerequisites (like
    cleaning and manipulation) before attempting complex sequences like making and
    serving a full cocktail. The costs are pre-computed for efficiency, making the
    heuristic evaluation at search time a fast dictionary lookup.

    The cost hierarchy is as follows (from lowest to highest):
    - 0: Base cases (e.g., `...Null` methods where the goal is already met).
    - 1: Basic manipulations (picking up, putting down).
    - 2-3: Cleaning tasks (differentiating between cleaning empty vs. full containers).
    - 4: Preparing a single ingredient in a shot glass.
    - 5: Adding a prepared ingredient to the main shaker.
    - 8: Making a complete cocktail in the shaker (combining ingredients).
    - 10: The entire process of making and serving a cocktail in a shot glass.
    """
    def __init__(self, barman_gemini25pro_heuristic_8):
        """
        Initializes the BarmanGemini25ProHeuristic8.

        Args:
            barman_gemini25pro_heuristic_8: This parameter is unused but included to match the required signature.
        """
        super().__init__(name="barman_gemini25pro_heuristic_8")
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-computing method scores.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # The initial node has no method applied, so its heuristic contribution is 0.
        initial_h_value = 0
        return super().initialize(model, initial_h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain model and assigns a
        pre-computed heuristic score to each.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            score = self._score_method_by_name(method_name.lower())
            self.method_scores[method_name] = score

    def _score_method_by_name(self, method_name_lower: str) -> int:
        """
        Assigns a heuristic cost to a method based on its name, reflecting its
        complexity within the task hierarchy.
        """
        # Category 0: Base Cases (Cost 0) - Goal is already satisfied.
        if 'null' in method_name_lower:
            return 0

        # Category 7: Top-Level Goal (Cost 10) - Make and pour a cocktail.
        if 'makeandpourcocktail' in method_name_lower:
            return 10

        # Category 6: Make Cocktail in Shaker (Cost 8) - A major sub-goal.
        if 'makecocktail' in method_name_lower:
            return 8

        # Category 5: Add Ingredient to Shaker (Cost 5) - Requires ingredient prep.
        if 'addingredienttoemptyshaker' in method_name_lower or \
           'addingredienttousedshaker' in method_name_lower:
            return 5

        # Category 4: Get Ingredient in Shot (Cost 4) - Foundational beverage step.
        if 'addingredienttoshot' in method_name_lower:
            return 4

        # Category 3 & 2: Cleaning (Cost 3 & 2) - Necessary but simpler tasks.
        if 'cleanfull' in method_name_lower:
            return 3
        if 'cleanempty' in method_name_lower:
            return 2

        # Category 1: Simple Manipulations & Actions (Cost 1) - Basic primitives.
        if 'pickup' in method_name_lower or \
           'emptyhand' in method_name_lower or \
           'putdown' in method_name_lower or \
           'pour_shaker_to_shot_action' in method_name_lower:
            return 1

        # Default for any other method (e.g., abstract task wrappers).
        return 1

    def _calculate_node_heuristic(self, node: HTNNode) -> int:
        """
        Calculates the heuristic value for a search node by looking up the
        pre-computed score for its applied method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, no heuristic cost from this step.

        method_name = node.method.name
        return self.method_scores.get(method_name, 1)  # Default score of 1 if not found.

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        The main heuristic function called by the planner during search.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self) -> str:
        return f"BarmanGemini25ProHeuristic8()"

    def __str__(self) -> str:
        return f"BarmanGemini25ProHeuristic8()"

    def __output__(self) -> str:
        return (
            f"Barman Gemini 2.5 Pro Heuristic 8 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
        )