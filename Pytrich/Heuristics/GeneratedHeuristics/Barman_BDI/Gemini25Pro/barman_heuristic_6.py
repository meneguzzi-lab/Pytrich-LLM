import time
from Pytrich.Heuristics.heuristic import Heuristic

class BarmanGemini25ProHeuristic6(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Barman domain.

    This heuristic evaluates the complexity of decomposition methods based on the
    types of subtasks they generate. It pre-computes a score for each method
    in the domain, assigning small penalties for actions related to cleaning,
    manipulation (grasping/dropping), and ingredient handling.

    The key ideas are:
    1.  **Pre-computation:** Analyze all domain methods once during initialization
        to create a fast lookup table of method scores. This minimizes overhead
        during search.
    2.  **Task Categorization:** Subtasks are classified into categories (e.g.,
        'clean', 'manipulate', 'add_ingredient', 'main_action') with associated
        penalties. This captures the cost structure of the domain, where tasks
        like cleaning and ingredient preparation are significant contributors to
        plan length.
    3.  **Admissibility:** The heuristic values are kept small and represent a
        cost for choosing a decomposition rather than a full plan-cost estimate.
        This helps guide the search towards simpler decompositions without
        sacrificing optimality.
    4.  **Base Case Recognition:** "Null" methods, which apply when a goal is
        already satisfied, are identified and assigned a score of 0, correctly
        representing that no work is needed.
    """
    def __init__(self, barman_gemini25pro_heuristic_6=None):
        """
        Initializes the BarmanGemini25ProHeuristic.

        Args:
            barman_gemini25pro_heuristic_6 (dict, optional): A dictionary of parameters.
                Expected keys: 'clean_penalty', 'manipulation_penalty',
                'ingredient_penalty', 'main_action_penalty'.
                Defaults to None, which uses default penalty values.
        """
        if barman_gemini25pro_heuristic_6 is None:
            barman_gemini25pro_heuristic_6 = {}

        self.clean_penalty = barman_gemini25pro_heuristic_6.get('clean_penalty', 0.25)
        self.manipulation_penalty = barman_gemini25pro_heuristic_6.get('manipulation_penalty', 0.1)
        self.ingredient_penalty = barman_gemini25pro_heuristic_6.get('ingredient_penalty', 0.3)
        self.main_action_penalty = barman_gemini25pro_heuristic_6.get('main_action_penalty', 0.15)
        
        super().__init__(name="barman_gemini25pro_heuristic_6")
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing scores for all methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain model and assigns a
        heuristic score to each based on its subtasks.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            score = 0

            # Null methods represent an already-achieved goal, so their cost is 0.
            if "null" in method_name.lower():
                self.method_scores[method_name] = 0
                continue

            if hasattr(method, 'task_network'):
                for subtask in method.task_network:
                    subtask_name = subtask.name.lower()
                    
                    # Penalize methods that require cleaning.
                    if 'clean' in subtask_name:
                        score += self.clean_penalty
                    # Penalize methods involving hand/object manipulation.
                    elif any(k in subtask_name for k in ['holding', 'handempty', 'grasp', 'drop', 'ontable']):
                        score += self.manipulation_penalty
                    # Penalize methods for adding ingredients, a core multi-step process.
                    elif 'contains' in subtask_name:
                        score += self.ingredient_penalty
                    # Penalize core cocktail-making actions.
                    elif any(k in subtask_name for k in ['shake', 'pour', 'fill', 'empty']):
                        score += self.main_action_penalty

            # Store the computed score. Heuristic values must be non-negative.
            self.method_scores[method_name] = max(0, score)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        """
        # If the node doesn't have a method, it's likely a primitive action or initial task.
        if not hasattr(node, 'method') or not node.method:
            return 0

        method_name = node.method.name
        # Look up the pre-computed score. Use a default if not found.
        return self.method_scores.get(method_name, 0.5)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return (f"BarmanGemini25ProHeuristic6(clean={self.clean_penalty}, "
                f"manipulation={self.manipulation_penalty}, "
                f"ingredient={self.ingredient_penalty})")

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        """
        Returns a string with summary information about the heuristic.
        """
        return (
            f"Barman Gemini 2.5 Pro Heuristic 6 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tClean Penalty: {self.clean_penalty}\n"
            f"\tManipulation Penalty: {self.manipulation_penalty}\n"
            f"\tIngredient Penalty: {self.ingredient_penalty}\n"
            f"\tMain Action Penalty: {self.main_action_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
