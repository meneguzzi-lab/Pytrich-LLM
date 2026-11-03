import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic23(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic is designed to guide the planner by providing a static preference
    for certain decomposition methods over others. Since the Towers of Hanoi problem
    has a well-defined recursive structure, the heuristic prioritizes methods that
    represent base cases of the recursion or lead directly to primitive actions (moves).
    Conversely, it assigns a higher cost (a small penalty) to methods that deepen
    the recursion, encouraging the planner to "bottom out" the recursive calls first.

    The heuristic operates without inspecting the current state of the world, focusing
    solely on the static properties of the chosen decomposition method. This makes it
    computationally very efficient and scalable, regardless of the number of rings
    in the problem instance.

    Heuristic Strategy:
    - Lowest cost (0.0): Base case methods that terminate a recursive sub-problem (e.g., 'exchangeClear').
    - Low cost (0.1): Methods that directly decompose into a primitive 'move' action.
    - Medium cost (0.2-0.3): Methods that are part of the core recursive algorithm but are not the primary recursive call.
    - Higher cost (0.4): Methods that represent the main recursive step, deepening the search.
    """

    def __init__(self, towers_gemini25pro_heuristic_23=None):
        """
        Initializes the heuristic.

        :param towers_gemini25pro_heuristic_23: Optional configuration object (not used).
        """
        super().__init__(name="towers_gemini25pro_heuristic_23")
        self.config = towers_gemini25pro_heuristic_23
        self.method_scores = {}
        self.preprocessing_time = 0
        self.model = None

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic with the planning model and pre-computes method scores.
        This method is called once before the search begins.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_scores()
        self.preprocessing_time = time.time() - start_time

        h_value = self.__call__(None, initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_scores(self):
        """
        Assigns a static score to each known method based on its role in the Towers of Hanoi
        recursive algorithm. The scores are stored for quick lookup during the search.
        """
        score_map = {
            'exchangeClear': 0.0,       # Base case: terminates a sub-problem
            'newMethod21': 0.1,         # Primitive action wrapper: performs a move
            'selectedDirection': 0.2,   # Base case for the 'selectDirection' recursion
            'm-rotateTower': 0.2,       # Leads directly to a move and an exchange
            'exchangeLR': 0.3,          # Recursive step for 'exchange'
            'exchangeRL': 0.3,          # Recursive step for 'exchange'
            'm-selectDirection': 0.4,   # Main recursive step for 'selectDirection'
            'm-shiftTower': 0.5         # Top-level task, highest abstract cost
        }

        methods = []
        if self.model:
            if hasattr(self.model, 'methods'):
                methods.extend(self.model.methods)
            if hasattr(self.model, 'decompositions'):
                methods.extend(self.model.decompositions)
            if hasattr(self.model, 'abstract_tasks'):
                for task in self.model.abstract_tasks:
                    if hasattr(task, 'decompositions'):
                        methods.extend(task.decompositions)

        if not methods:
            return

        unique_methods = {m.name: m for m in methods}.values()

        for method in unique_methods:
            method_name = method.name
            base_name = method_name.split()[0]
            score = score_map.get(base_name, 0.6)  # Default score for unknown methods
            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node by looking up the
        score of the method used to generate this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0  # Initial node or nodes not resulting from a decomposition

        method_name = node.method.name
        return self.method_scores.get(method_name, 0.6) # Use default for safety

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner for each node.
        """
        h_value = self._calculate_node_heuristic(node)
        # Ensure the heuristic value is small to remain admissible and useful for tie-breaking
        h_value = min(h_value, 1.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic23()"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        """
        Returns a formatted string with information about the heuristic's configuration and performance.
        """
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 23 Info:\n"
            f"\tName: {self.name}\n"
            f"\tStrategy: Penalize recursive methods, reward base cases and primitives.\n"
            f"\tMethods scored: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
