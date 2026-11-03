import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic4(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi (towers) domain.

    This heuristic operates without direct access to the world state, a common
    constraint in some HTN planner frameworks. Instead, it analyzes the
    decomposition method being applied at a given search node. It assigns
    costs to methods based on their role in the recursive Tower of Hanoi algorithm.

    Rationale:
    The Towers of Hanoi solution is inherently recursive. The cost of solving a
    problem of size N is dominated by the two recursive calls to solve a
    problem of size N-1. This heuristic guides the planner by assigning:
    - A high cost to methods that represent a recursive step (e.g., 'rotateTower', 'exchangeLR/RL'),
      signaling a large amount of work to follow.
    - A medium cost to methods that lead directly to a primitive 'move' action.
    - A low cost to setup or auxiliary methods.
    - A zero cost to methods that represent the base case of the recursion (e.g., 'exchangeClear'),
      encouraging the planner to terminate recursive branches.

    This approach creates a cost gradient that helps the planner prefer shorter
    decomposition paths and correctly identify the expensive recursive parts of
    the plan, improving search efficiency without compromising admissibility (the
    heuristic costs are small, constant values, well below the true cost of the
    resulting sub-plans).
    """

    def __init__(self, towers_gemini25pro_heuristic_4=0.1):
        """
        Initializes the heuristic.

        :param towers_gemini25pro_heuristic_4: A base cost unit, conceptually tied to the
                                              cost of a single 'move' action.
        """
        super().__init__(name="towers_gemini25pro_heuristic_4")
        self.base_move_cost = towers_gemini25pro_heuristic_4

        # Define relative costs for different types of methods
        self.costs = {
            'recursive_step': self.base_move_cost * 3,  # For methods like rotateTower, exchangeLR/RL
            'primitive_action_setup': self.base_move_cost, # For newMethod21 which sets up a move
            'auxiliary': self.base_move_cost / 2, # For setup methods like shiftTower, selectDirection
            'base_case': 0.0 # For terminal methods like exchangeClear
        }
        self.method_scores = {}
        self.preprocessing_time = 0.0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-computing scores for each method in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the model and assigns a heuristic
        score based on its name and role in the planning problem.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name.lower()
            score = self.costs['auxiliary'] # Default score

            if 'exchangeClear'.lower() in method_name:
                score = self.costs['base_case']
            elif 'rotateTower'.lower() in method_name or \
                 'exchangeLR'.lower() in method_name or \
                 'exchangeRL'.lower() in method_name:
                score = self.costs['recursive_step']
            elif 'newMethod21'.lower() in method_name:
                # This method directly decomposes into the 'move' action
                score = self.costs['primitive_action_setup']
            elif 'shiftTower'.lower() in method_name or \
                 'selectDirection'.lower() in method_name:
                score = self.costs['auxiliary']
            
            self.method_scores[method.name] = score

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        The value is based on the pre-computed score of the method used
        to generate this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        # Look up the pre-computed score for the method.
        # Default to a base cost if the method is somehow not found.
        return self.method_scores.get(node.method.name, self.base_move_cost)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic4(base_cost={self.base_move_cost})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 4 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tBase move cost: {self.base_move_cost}\n"
            f"\tRecursive step cost: {self.costs['recursive_step']}\n"
            f"\tBase case cost: {self.costs['base_case']}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f}s\n"
        )
