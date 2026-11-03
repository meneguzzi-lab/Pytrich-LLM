import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini25ProHeuristic12(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Barman domain.

    Rationale:
    The Barman domain involves a highly sequential process of preparing cocktails, with significant costs
    associated with resource management, particularly cleaning containers (shakers, shots) and managing
    the two hands available for manipulation. This heuristic guides the search by assigning costs to
    decomposition methods based on their implied complexity and role in the planning process.

    Heuristic Strategy:
    1.  **Precomputation**: It analyzes all decomposition methods in the domain during initialization.
        For each method, it computes a score and stores it.
    2.  **Cost-Based Scoring**: The score is a small floating-point number representing the estimated
        cost or complexity of the operation the method performs. Methods that lead to more actions
        or handle resource-intensive tasks (like cleaning) receive a higher penalty.
    3.  **Name-Based Analysis**: The heuristic leverages the descriptive names of the methods (e.g.,
        'CleanFullShot', 'AddIngredientToShaker', 'MakeCocktailNull') to infer their function and cost.
        - **Null Methods**: Methods ending in 'Null' represent an already-achieved goal and have a cost of 0.
        - **Cleaning**: Cleaning is heavily penalized, especially cleaning a 'Full' container, as this
          requires multiple primitive actions (grasp, empty, clean).
        - **Ingredient Preparation**: Adding ingredients is another costly, multi-step sub-process.
        - **Composite Tasks**: High-level tasks like 'MakeAndPourCocktail' are given the highest penalty
          as they decompose into the largest number of sub-tasks.
    4.  **Admissibility-Awareness**: By using small, non-negative values, the heuristic aims to guide the
        search effectively without significantly violating admissibility, prioritizing better solutions
        without ruling out optimal ones.
    """
    def __init__(self, barman_gemini25pro_heuristic_12):
        # Penalty values are calibrated to reflect the relative cost of operations.
        # Cleaning is the most expensive non-composite operation.
        self.clean_penalty = 0.4
        # Ingredient preparation involves multiple steps (cleaning shot, filling, pouring).
        self.ingredient_penalty = 0.6
        # Shaking and pouring are single, final actions.
        self.shake_pour_penalty = 0.2
        # Basic manipulations are the cheapest operations.
        self.manipulation_penalty = 0.1
        # High-level tasks get a large penalty to reflect their decomposition complexity.
        self.composite_task_penalty = 1.0
        
        super().__init__(name="barman_gemini25pro_heuristic_12")
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute method characteristics.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precomputes a score for each decomposition method based on its name and structure,
        which indicates its role and complexity in the Barman domain.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name
            name_lower = method_name.lower()
            
            subtask_count = 0
            if hasattr(method, 'task_network'):
                subtask_count = len(method.task_network)

            # A small base score proportional to the number of subtasks.
            score = subtask_count * 0.05

            # --- Score based on method name, which implies its function ---
            if 'null' in name_lower:
                score = 0.0
            elif 'cleanfull' in name_lower:
                score += self.clean_penalty * 1.5
            elif 'cleanempty' in name_lower:
                score += self.clean_penalty
            elif 'clean' in name_lower:
                score += self.clean_penalty * 0.8
            elif 'addingredienttoshaker' in name_lower:
                score += self.ingredient_penalty
            elif 'addingredienttoshot' in name_lower or 'fill-shot' in name_lower:
                score += self.ingredient_penalty * 0.7
            elif 'shake' in name_lower or 'pour' in name_lower or 'empty-' in name_lower:
                score += self.shake_pour_penalty
            elif 'pickup' in name_lower or 'grasp' in name_lower or 'drop' in name_lower or 'emptyhand' in name_lower:
                score += self.manipulation_penalty

            # --- Penalize high-level composite methods ---
            if 'makecocktail' in name_lower and 'null' not in name_lower:
                score += self.composite_task_penalty
            if 'makeandpourcocktail' in name_lower and 'null' not in name_lower:
                score += self.composite_task_penalty * 1.2

            # Store the computed score for quick lookup during search.
            self.method_scores[method_name] = {
                'score': score,
                'subtask_count': subtask_count
            }

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculate the heuristic value for a given node based on the precomputed
        score of the method applied to reach this node's state.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        
        # Default to a small cost for unknown methods.
        score = self.method_scores.get(method_name, {}).get('score', 0.1)

        return score

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main heuristic function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        h_value = max(h_value, 0.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic12(clean={self.clean_penalty}, ingredient={self.ingredient_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Barman Gemini 2.5 Pro Heuristic 12 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tClean Penalty: {self.clean_penalty}\n"
            f"\tIngredient Penalty: {self.ingredient_penalty}\n"
            f"\tShake/Pour Penalty: {self.shake_pour_penalty}\n"
            f"\tManipulation Penalty: {self.manipulation_penalty}\n"
            f"\tComposite Task Penalty: {self.composite_task_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
