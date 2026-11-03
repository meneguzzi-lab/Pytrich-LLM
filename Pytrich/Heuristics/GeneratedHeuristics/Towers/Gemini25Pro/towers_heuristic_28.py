import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic28(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic guides the planner's search for decompositions by assigning static
    costs to different methods. The core idea is to prioritize methods that represent
    base cases in the recursion or are closer to primitive actions, giving them a lower
    heuristic value (cost).

    The scoring reflects the typical recursive structure of the Tower of Hanoi solution:
    - Terminal methods (like 'exchangeClear') are cheapest, as they complete a sub-problem.
    - Methods leading directly to actions ('newMethod21' -> 'move') are next cheapest.
    - Recursive methods that break down the problem ('m-rotateTower', 'exchangeLR') are assigned
      higher costs, reflecting the larger sub-problems they generate.

    This approach is computationally efficient, requiring only a one-time analysis of the
    domain's methods during initialization. It guides the planner to find solutions by
    favoring more direct and terminal decomposition paths.
    """
    def __init__(self, towers_gemini25pro_heuristic_28, name="towers_gemini_25_pro_heuristic_28"):
        super().__init__(name=name)
        # The 'towers_gemini25pro_heuristic_28' parameter is specified by the task prompt.
        # It is stored but not used, as its purpose is not defined.
        self.config = towers_gemini25pro_heuristic_28
        self.method_scores = {}
        self.preprocessing_time = 0.0

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
        Analyzes all decomposition methods from the model and stores a pre-calculated
        heuristic score for each one.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            score = self._score_method_statically(method.name)
            self.method_scores[method.name] = score

    def _score_method_statically(self, method_name):
        """
        Assigns a static score to a method based on its role in the Hanoi recursion.
        Lower scores are better, indicating a more preferred method. The values are kept
        small to be admissible-like for cost-based search algorithms.
        """
        name = method_name.lower()
        
        # Base Case: A terminal method that completes a recursive sub-problem.
        if 'exchangeclear' in name:
            return 0.1
        
        # Near-Primitive: Decomposes a task directly into a primitive 'move' action.
        elif 'newmethod21' in name:
            return 0.2
            
        # Recursive Base Case: The base case for the 'selectDirection' recursion.
        elif 'selecteddirection' in name:
            return 0.3
            
        # Recursive Step: The recursive step for navigating down a stack of rings.
        elif 'm-selectdirection' in name:
            return 0.4
            
        # Core Recursion: The main recursive steps of the Hanoi algorithm.
        elif 'm-rotatetower' in name or 'exchangelr' in name or 'exchangerl' in name:
            return 0.5
            
        # Top-Level Task: The initial decomposition of the main problem.
        elif 'm-shifttower' in name:
            return 0.6
            
        # Default score for any unrecognized methods.
        else:
            return 1.0

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        The value is based on the pre-computed score of the method used for decomposition.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0
        
        method_name = node.method.name
        return self.method_scores.get(method_name, 1.0) # Default to 1.0 if not found

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value
        
    def __repr__(self):
        return f"TowersGemini25ProHeuristic28(name={self.name})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        """
        Returns a formatted string with information about the heuristic's configuration.
        """
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 28 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
