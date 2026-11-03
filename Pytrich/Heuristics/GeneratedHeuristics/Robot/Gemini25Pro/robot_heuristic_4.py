import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic4(Heuristic):
    """
    A domain-dependent heuristic for the HDDL 'robot' domain.

    This heuristic guides the search by assigning a small cost to each decomposition
    method based on the type of action it initiates. The core idea is to prioritize
    actions that directly contribute to the main goal of delivering packages.

    The cost hierarchy is as follows (lower is better):
    1. `finished`: Completing the plan (cost 0).
    2. `putdown`: Delivering a package to its goal location.
    3. `pickup`: Starting the delivery process for a misplaced package.
    4. `move`/`open`: Auxiliary actions for navigation.

    By assigning lower costs to more productive methods, the heuristic encourages the
    planner to explore promising paths first, such as completing an ongoing delivery
    before starting a new one or moving unnecessarily. The heuristic values are kept
    small and constant to act as a gentle guide without risking inadmissibility
    or excessive computational overhead.
    """
    def __init__(self, putdown_cost=0.1, pickup_cost=0.2, move_cost=0.3, open_cost=0.3, name="robot_gemini25pro_heuristic_4"):
        """
        Initializes the heuristic with configurable costs for different action types.
        """
        super().__init__(name=name)
        self.putdown_cost = putdown_cost
        self.pickup_cost = pickup_cost
        self.move_cost = move_cost
        self.open_cost = open_cost
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
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
        Analyzes all decomposition methods in the model and assigns a score based on
        the type of operation they represent.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name.lower()
            
            # The order of checks is important, from most to least specific/preferred.
            if 'finished' in method_name:
                score = 0.0
            elif 'putdown' in method_name:
                score = self.putdown_cost
            elif 'pickup' in method_name:
                score = self.pickup_cost
            elif 'open' in method_name:
                score = self.open_cost
            elif 'move' in method_name:
                score = self.move_cost
            else:
                # Default cost for any unrecognized method, set to highest.
                score = max(self.move_cost, self.open_cost)

            self.method_scores[method.name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        The value is based on the pre-computed score of the method used for
        the decomposition at this node.
        """
        # If the node represents a primitive action or has no method, its heuristic contribution is 0.
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        
        # Return the pre-computed score, or a default value if not found.
        return self.method_scores.get(method_name, self.move_cost)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        
        # Ensure the heuristic value is small to maintain admissibility.
        h_value = min(h_value, 1.0)
        
        super().update_info(h_value)
        return h_value
    
    def __repr__(self):
        return (f"RobotGemini25ProHeuristic4(putdown={self.putdown_cost}, pickup={self.pickup_cost}, "
                f"move={self.move_cost}, open={self.open_cost})")

    def __str__(self):
        return repr(self)

    def __output__(self):
        return (
            f"Robot Gemini 2.5 Pro Heuristic 4 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPutdown Cost: {self.putdown_cost}\n"
            f"\tPickup Cost: {self.pickup_cost}\n"
            f"\tMove Cost: {self.move_cost}\n"
            f"\tOpen Cost: {self.open_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
