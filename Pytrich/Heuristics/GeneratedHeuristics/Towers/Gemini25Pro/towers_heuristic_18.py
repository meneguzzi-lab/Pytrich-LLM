import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic18(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi domain.
    
    This heuristic operates by statically analyzing the decomposition methods
    of the HDDL domain. It does not perform state-space analysis but instead
    assigns a small, fixed cost to each method based on its role in the
    problem's recursive structure. The goal is to guide the search towards
    methods that either terminate the recursion (base cases) or execute
    primitive actions.

    The scoring rationale is as follows:
    - Base Case Methods (e.g., 'exchangeClear'): These complete a recursive branch
      and are assigned the lowest cost (0.0) as they represent progress towards
      a solution state.
    - Action-Wrapping Methods (e.g., 'newMethod21'): Methods that directly decompose
      into a primitive 'move' action are given a very low cost, as they are the
      only way to alter the world state.
    - Recursive Step Methods (e.g., 'm-rotateTower', 'exchangeLR'): These methods
      are the core of the Tower of Hanoi algorithm, performing a move and a
      recursive call. They are assigned a moderate cost.
    - Recursive Setup Methods (e.g., 'm-selectDirection'): Methods that primarily
      manage the flow of recursion without directly causing a move are given a
      slightly higher cost.

    By preferring methods closer to primitive actions or termination, the heuristic
    aims to encourage the planner to build the solution in a direct manner,
    providing gentle guidance without the computational overhead of a full
    state-based heuristic. The costs are kept small to behave like an admissible
    heuristic, making it suitable for optimal search algorithms like A*.
    """
    def __init__(self, towers_gemini25pro_heuristic_18=0.1, name="TowersGemini25ProHeuristic18"):
        super().__init__(name=name)
        self.base_cost = towers_gemini25pro_heuristic_18
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-computing scores for all methods in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_scores()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_scores(self):
        """
        Iterates through all decomposition methods in the model and assigns a
        pre-computed score based on a static analysis of their names and roles.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            score = self._score_method_statically(method)
            self.method_scores[method.name] = score

    def _score_method_statically(self, method) -> float:
        """
        Assigns a static cost to a method based on its name.
        Lower scores are assigned to methods that are closer to termination
        or primitive actions.
        """
        method_name = method.name.lower()

        # Base case for the 'exchange' task, represents termination of a sub-problem.
        if 'exchangeclear' in method_name:
            return 0.0

        # This method is a direct wrapper for the primitive 'move' action.
        if 'newmethod21' in method_name:
            return self.base_cost * 1

        # These methods perform a move and a recursive call. They are the workhorses.
        if 'rotatetower' in method_name or 'exchangelr' in method_name or 'exchangerl' in method_name:
            return self.base_cost * 2

        # This method selects which sub-tower to move, leading into the recursion.
        if 'selecteddirection' in method_name:
            return self.base_cost * 3
            
        # This is the recursive call for selecting a direction.
        if 'm-selectdirection' in method_name:
            return self.base_cost * 4

        # This is the top-level task method that initiates the whole process.
        if 'shifttower' in method_name:
            return self.base_cost * 5

        # Default cost for any unrecognized methods.
        return self.base_cost * 10

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0
        
        method_name = node.method.name
        return self.method_scores.get(method_name, self.base_cost * 10)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic18(base_cost={self.base_cost})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 18 ({self.name}):\n"
            f"\tType: Static Method-Based Cost\n"
            f"\tBase Cost Multiplier: {self.base_cost}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
