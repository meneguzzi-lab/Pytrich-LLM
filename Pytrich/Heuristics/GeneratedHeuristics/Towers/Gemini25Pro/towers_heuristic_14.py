import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic14(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi (towers) domain.

    This heuristic guides the search by assigning small costs to methods based on their
    role in the typical recursive solution to the Towers of Hanoi problem. It operates
    under the assumption that the planner is searching for a decomposition of the
    top-level 'shiftTower' task.

    The heuristic categorizes methods into three types:
    1.  BASE_CASE: Methods that terminate a recursive branch (e.g., 'exchangeClear').
        These are rewarded with a small bonus (negative cost) as they represent progress
        towards completing a sub-problem.
    2.  PRIMITIVE_PROXY: Methods that directly decompose into a primitive action (e.g., 'move').
        These are given a small penalty, representing one unit of work in the final plan.
    3.  RECURSIVE_STEP: Methods that continue the recursion by creating more sub-tasks.
        These are given a smaller penalty, as they expand the search tree but are necessary
        for the solution.

    By differentiating methods in this way, the heuristic provides a gentle guide for the
    search algorithm to prefer decompositions that lead to terminating base cases or
    productive primitive actions, while slightly discouraging deeper recursion, helping
    with tie-breaking and search focus.
    """

    def __init__(self, towers_gemini25pro_heuristic_14=None, move_penalty=0.1, recursion_penalty=0.05, base_case_bonus=-0.01):
        super().__init__(name="towers_gemini25pro_heuristic_14")
        self.config = towers_gemini25pro_heuristic_14
        self.move_penalty = move_penalty
        self.recursion_penalty = recursion_penalty
        self.base_case_bonus = base_case_bonus
        self.method_info = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-analyzing the methods in the domain model.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes and categorizes all decomposition methods in the domain.
        This information is cached for fast lookups during the search.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            info = {'type': 'UNKNOWN'}

            # Categorize methods based on their names, which reflect their function in the Hanoi algorithm
            if 'exchangeClear' in method_name:
                # This is the base case for the 'exchange' subproblem when a tower is empty.
                info['type'] = 'BASE_CASE'
            elif 'newMethod21' in method_name:
                # This method directly decomposes 'move_abstract' into the primitive 'move' action.
                info['type'] = 'PRIMITIVE_PROXY'
            elif any(s in method_name for s in ['shiftTower', 'selectDirection', 'rotateTower', 'exchangeLR', 'exchangeRL']):
                # These methods represent the recursive steps of the Hanoi algorithm.
                info['type'] = 'RECURSIVE_STEP'
            
            self.method_info[method_name] = info

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        The value is determined by scoring the method used to generate this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, so no associated cost
        
        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Assigns a score to a method based on its pre-computed category.
        Lower scores are better.
        """
        method_name = method.name
        info = self.method_info.get(method_name, {'type': 'UNKNOWN'})
        method_type = info['type']

        score = 0
        if method_type == 'BASE_CASE':
            # Reward reaching a base case, as it terminates a recursive branch.
            score = self.base_case_bonus 
        elif method_type == 'PRIMITIVE_PROXY':
            # Penalize methods that lead to a primitive action, representing one unit of work.
            score = self.move_penalty
        elif method_type == 'RECURSIVE_STEP':
            # Slightly penalize methods that continue the recursion, expanding the plan.
            score = self.recursion_penalty
        else: # UNKNOWN methods
            # Assign a default small penalty for any uncategorized methods.
            score = self.recursion_penalty

        return max(0, score)  # Heuristic cost must be non-negative

    def __call__(self, parent_node, node):
        """
        The main entry point for the heuristic evaluation.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value
        
    def __repr__(self):
        return f"TowersGemini25ProHeuristic14(move_penalty={self.move_penalty}, recursion_penalty={self.recursion_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 14 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_info)}\n"
            f"\tMove Penalty: {self.move_penalty}\n"
            f"\tRecursion Penalty: {self.recursion_penalty}\n"
            f"\tBase Case Bonus: {self.base_case_bonus}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
        )
