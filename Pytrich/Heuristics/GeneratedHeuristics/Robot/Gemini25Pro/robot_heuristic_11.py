import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic11(Heuristic):
    """
    A domain-dependent heuristic for the Robot domain.

    This heuristic estimates the cost of a decomposition method by calculating the sum
    of the costs of the primitive actions it immediately introduces. It is designed to be
    admissible (h(n) <= h*(n)) and computationally efficient.

    The core idea is to guide the search by preferring methods that make tangible progress:
    1.  **Base Cases (`finished`):** The lowest cost (0), as they complete a task branch.
    2.  **Action-Performing Methods:** Methods that decompose into primitive actions (`pickup`,
        `putdown`, `move`, `open`) are assigned a cost equal to the number of those actions.
        This prioritizes direct action over purely recursive decompositions.

    The heuristic performs a one-time analysis of all domain methods during initialization
    to pre-calculate these costs, making the heuristic evaluation during search a fast lookup.
    """
    def __init__(self, robot_gemini25pro_heuristic_11, move_cost=1.0, open_cost=1.0, pickup_cost=1.0, putdown_cost=1.0):
        """
        Initializes the heuristic with costs for each primitive action.
        The parameter `robot_gemini25pro_heuristic_11` is included to match the required signature.
        """
        super().__init__(name="robot_gemini25pro_heuristic_11")
        self.move_cost = move_cost
        self.open_cost = open_cost
        self.pickup_cost = pickup_cost
        self.putdown_cost = putdown_cost
        
        self.method_scores = {}
        self.preprocessing_time = 0
        self.config_param = robot_gemini25pro_heuristic_11 # To adhere to the __init__ parameter name requirement

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing information about the domain methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain to calculate their immediate action costs.
        This information is stored for quick retrieval during search.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return
        
        methods = self.model.decompositions
        if not methods:
            return

        for method in methods:
            method_name = method.name
            
            info = {
                'pickup_count': 0,
                'putdown_count': 0,
                'move_count': 0,
                'open_count': 0,
                'is_base_case': 'finished' in method_name.lower()
            }

            if hasattr(method, 'task_network'):
                for subtask in method.task_network:
                    subtask_name = subtask.name.lower()
                    
                    # Count primitive actions or their abstract wrappers in the decomposition.
                    # Using 'in' provides a simple and robust way to catch both primitive
                    # action names (e.g., 'pickup') and abstract task names (e.g., 'pickup_abstract').
                    if 'pickup' in subtask_name:
                        info['pickup_count'] += 1
                    elif 'putdown' in subtask_name:
                        info['putdown_count'] += 1
                    elif 'move' in subtask_name:
                        info['move_count'] += 1
                    elif 'open' in subtask_name:
                        info['open_count'] += 1
            
            self.method_scores[method_name] = info

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a search node. The value is based on the
        decomposition method applied at this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0
        
        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Scores a method based on its pre-computed immediate action costs.
        This provides an admissible estimate of the cost of this decomposition path.
        """
        method_name = method.name
        info = self.method_scores.get(method_name)

        if not info:
            # Fallback for any method not found during pre-computation
            return 1.0

        if info['is_base_case']:
            return 0.0

        score = 0.0
        score += info['pickup_count'] * self.pickup_cost
        score += info['putdown_count'] * self.putdown_cost
        score += info['move_count'] * self.move_cost
        score += info['open_count'] * self.open_cost

        # The heuristic is admissible as it only sums the costs of actions that are
        # guaranteed to be in the plan if this method is chosen, ignoring future costs
        # from recursive subtasks, thus h(n) <= h*(n).
        return score

    def __call__(self, parent_node, node):
        """
        The main entry point for the heuristic evaluation during search.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value
    
    def __repr__(self):
        return f"RobotGemini25ProHeuristic11(move={self.move_cost}, open={self.open_cost}, pickup={self.pickup_cost}, putdown={self.putdown_cost})"

    def __str__(self):
        return self.__repr__()
        
    def __output__(self):
        """
        Provides a formatted string with summary information about the heuristic.
        """
        return (
            f"Robot Gemini 2.5 Pro Heuristic 11 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tAction Costs (move, open, pickup, putdown): ({self.move_cost}, {self.open_cost}, {self.pickup_cost}, {self.putdown_cost})\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
