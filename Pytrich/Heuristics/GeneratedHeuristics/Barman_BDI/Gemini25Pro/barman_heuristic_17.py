import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini25ProHeuristic17(Heuristic):
    """
    A domain-dependent heuristic for the Barman domain.

    This heuristic estimates the cost to complete a plan by statically analyzing
    the decomposition methods. It assigns costs based on the type of operation
    a method represents, penalizing overhead actions like cleaning and manipulation
    while assigning a base cost to productive actions like adding ingredients or
    serving a cocktail.

    The core idea is that the most significant costs in the Barman domain arise
    from resource management (cleaning containers, managing the two hands).
    By penalizing methods that involve these costly operations, the heuristic
    guides the search towards more direct and efficient plans.
    """
    def __init__(self, barman_gemini25pro_heuristic_17, cleaning_penalty=1.5, manipulation_penalty=0.5, subtask_penalty=0.1):
        """
        Initializes the heuristic with weights for different actions.

        Args:
            barman_gemini25pro_heuristic_17: Dummy parameter to fulfill a specific request.
            cleaning_penalty (float): The base cost for cleaning a container.
            manipulation_penalty (float): The base cost for hand/object manipulation (grasp, drop).
            subtask_penalty (float): A small cost added for each subtask in a method.
        """
        # The 'barman_gemini25pro_heuristic_17' parameter is included to match the specific request,
        # but it is not used in the heuristic's logic.
        super().__init__(name="barman_gemini25pro_heuristic_17")

        # --- Heuristic penalties --- 
        # These weights are tuned to reflect the relative cost of operations.
        # Cleaning is the most expensive overhead.
        self.cleaning_penalty = cleaning_penalty
        # Picking up and putting down items is necessary but less costly.
        self.manipulation_penalty = manipulation_penalty
        # A small penalty for complexity (number of sub-steps).
        self.subtask_penalty = subtask_penalty
        
        # --- Pre-computation storage --- 
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic and precompute method scores for efficiency.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyze all methods in the domain to assign a base score.
        This is performed once at the beginning of the search to avoid redundant calculations.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            self.method_scores[method.name] = self._score_method_static(method)
            
    def _score_method_static(self, method):
        """
        Statically scores a method based on its name and structure.
        This function encodes the domain-specific knowledge.
        """
        method_name = method.name
        method_lower = method_name.lower()
        score = 0.0

        # 'Null' methods correspond to tasks whose preconditions are already met.
        # They represent zero-cost progress and should have a heuristic value of 0.
        if 'null' in method_lower:
            return 0.0

        # --- Penalize Overhead Actions ---
        # Cleaning is a major source of overhead. Cleaning a full container is 
        # more costly because it requires an 'empty' action first.
        if 'cleanfull' in method_lower:
            score += self.cleaning_penalty * 2.0
        elif 'cleanempty' in method_lower:
            score += self.cleaning_penalty
        
        # Basic manipulation (releasing a hand, putting an item down) is necessary
        # overhead that doesn't directly create the cocktail.
        if 'emptyhand' in method_lower or 'putdown' in method_lower:
            score += self.manipulation_penalty

        # --- Assign Base Cost for Productive Actions ---
        # These actions are part of the core workflow.
        if 'makecocktail' in method_lower:  # The main method that involves shaking
            score += 1.0
        elif 'addingredient' in method_lower:
            score += 0.8
        elif 'fill-shot' in method_lower: # A sub-step of adding an ingredient
            score += 0.4
        elif 'dopourshakertoshot' in method_lower: # Final serving step
            score += 0.4
        elif 'pickup' in method_lower: # Grasping is fundamental manipulation
            score += self.manipulation_penalty

        # --- Add Complexity Penalty from Subtasks ---
        # A method with more subtasks is generally more complex and will likely
        # lead to a longer plan.
        if hasattr(method, 'task_network'):
            score += len(method.task_network) * self.subtask_penalty

        return score

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a given search node by looking up
        the pre-computed score of the method that generated it.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0
        
        method_name = node.method.name
        # Default to a score of 1.0 if a method was not found during pre-computation.
        return self.method_scores.get(method_name, 1.0)

    def __call__(self, parent_node, node):
        """
        The main heuristic function called by the planner during search.
        """
        h_value = self._calculate_node_heuristic(node)
        
        # The heuristic must be non-negative.
        h_value = max(h_value, 0.0)
        
        super().update_info(h_value)
        return h_value
        
    def __repr__(self):
        return f"BarmanGemini25ProHeuristic17(cleaning={self.cleaning_penalty}, manip={self.manipulation_penalty})"

    def __str__(self):
        return f"BarmanGemini25ProHeuristic17(cleaning={self.cleaning_penalty}, manip={self.manipulation_penalty})"

    def __output__(self):
        return (
            f"Barman Gemini 2.5 Pro Heuristic 17 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Scored: {len(self.method_scores)}\n"
            f"\tCleaning Penalty: {self.cleaning_penalty}\n"
            f"\tManipulation Penalty: {self.manipulation_penalty}\n"
            f"\tSubtask Penalty: {self.subtask_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
