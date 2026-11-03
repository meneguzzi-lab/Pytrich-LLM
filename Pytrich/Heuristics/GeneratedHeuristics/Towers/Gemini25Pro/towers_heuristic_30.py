import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Towers of Hanoi (towers) domain.
# This heuristic is designed to guide the search by evaluating the cost of method decompositions.
# Since state access is not available in the heuristic function, it relies on a static analysis
# of the domain's methods to estimate the complexity of the sub-plan that will be generated.
class TowersGemini25ProHeuristic30(Heuristic):
    """
    Heuristic for the Towers of Hanoi domain.

    Rationale:
    The Towers of Hanoi problem has a well-defined recursive structure. The optimal solution involves a sequence of moves
    determined by the number of rings to shift. The cost of solving a problem with N rings is 2^N - 1 moves.
    This heuristic approximates the cost of a decomposition without inspecting the state. It does so by assigning
    a cost to each method based on the complexity of its subtasks.

    1.  **Action Cost**: The fundamental operation is the 'move' action. Any method that directly leads to a move
        is assigned a base cost. The `newMethod21` which wraps the `move` action is a key indicator.
    2.  **Recursion Cost**: The core of the Towers of Hanoi solution lies in recursion. Methods that introduce
        recursive calls to solve a sub-problem (e.g., methods with `rotateTower` or `exchange` in their subtasks)
        are penalized more heavily. This reflects the fact that these decompositions lead to a longer sequence of actions.
    3.  **Base Case Bonus**: Methods that represent a base case for the recursion (e.g., `exchangeClear`, which has no subtasks)
        are the most desirable as they terminate a branch of the search. They are assigned a cost of 0.

    By pre-computing a 'complexity score' for each method, the heuristic can quickly provide an estimate
    of the remaining work for a given decomposition choice, guiding the planner towards more efficient solutions.
    The heuristic values are kept small and positive to ensure admissibility for optimal search algorithms like A*.
    """
    def __init__(self, towers_gemini25pro_heuristic_30, move_penalty=0.1, recursive_penalty=0.2, name="towers_gemini25pro_heuristic_30"):
        super().__init__(name=name)
        # Penalties for different types of operations found in method subtasks
        self.move_penalty = move_penalty  # Cost for a single ring move
        self.recursive_penalty = recursive_penalty # Additional cost for a recursive call (e.g., rotateTower)
        self.method_scores = {}
        self.preprocessing_time = 0

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
        Analyzes all methods in the domain and assigns a static complexity score to each.
        This score is based on the nature of the method's subtasks.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            score = 0
            
            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network
            
            # If a method has no subtasks, it's a terminal/base case.
            if not subtasks:
                # e.g., 'exchangeClear' is a base case for the recursion.
                score = 0
            else:
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)
                    
                    subtask_lower = subtask_name.lower()
                    
                    # Penalize methods that result in a physical move.
                    if 'move' in subtask_lower:
                        score += self.move_penalty
                    
                    # Penalize methods that involve recursive calls more heavily.
                    # 'rotateTower' and 'exchange' are the recursive components of the solution.
                    if 'rotatetower' in subtask_lower or 'exchange' in subtask_lower:
                        score += self.recursive_penalty

            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node by looking up the
        pre-computed score of the method used for decomposition at that node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, no cost incurred.

        method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)
        
        # Return the pre-computed score for the method. Default to 0 if not found.
        return self.method_scores.get(method_name, 0)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        # Ensure the heuristic is admissible (non-negative).
        h_value = max(h_value, 0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic30(move_penalty={self.move_penalty}, recursive_penalty={self.recursive_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 30 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMove Penalty: {self.move_penalty}\n"
            f"\tRecursive Penalty: {self.recursive_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
