import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic12(Heuristic):
    """
    A domain-dependent heuristic for the Robot domain.

    This heuristic evaluates the promise of a decomposition method by assigning costs
    based on the actions involved in its subtasks. The core idea is to prioritize
    methods that directly contribute to the main goal (placing packages) while
    penalizing preparatory or navigational methods.

    - The 'finished' method is the most preferred (cost 0) as it terminates a goal-achieving branch.
    - Methods involving 'putdown' are highly encouraged with a bonus, as they complete a package delivery.
    - Methods involving 'pickup' are seen as productive steps and have a low cost.
    - Methods for 'move' and 'open' are necessary but are just means to an end, so they receive a higher cost.

    This approach guides the planner to focus on completing package deliveries one by one
    and helps in making efficient choices between moving, opening doors, or handling packages.
    """
    def __init__(self, move_penalty=0.3, open_penalty=0.3, pickup_penalty=0.2, putdown_bonus=0.4, robot_gemini25pro_heuristic_12="robot_gemini25pro_heuristic_12"):
        super().__init__(name=robot_gemini25pro_heuristic_12)
        self.move_penalty = move_penalty
        self.open_penalty = open_penalty
        self.pickup_penalty = pickup_penalty
        self.putdown_bonus = putdown_bonus
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute method characteristics.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precompute information about methods for faster scoring during search.
        This static analysis is based on the structure of methods and their subtasks.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name
            
            pickup_count = 0
            putdown_count = 0
            move_count = 0
            open_count = 0
            
            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network
            
            if subtasks:
                for subtask in subtasks:
                    subtask_name_lower = subtask.name.lower()
                    
                    if 'pickup' in subtask_name_lower:
                        pickup_count += 1
                    if 'putdown' in subtask_name_lower:
                        putdown_count += 1
                    if 'move' in subtask_name_lower:
                        move_count += 1
                    if 'open' in subtask_name_lower:
                        open_count += 1
            
            self.method_scores[method_name] = {
                'pickup_count': pickup_count,
                'putdown_count': putdown_count,
                'move_count': move_count,
                'open_count': open_count,
                'is_termination': 'finished' in method_name.lower(),
                'is_goal_achieving': 'putdown' in method_name.lower()
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a node based on its decomposition method.
        The heuristic estimates the cost of applying a particular method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Scores a method based on its name and precomputed characteristics.
        Lower scores indicate more promising methods (lower estimated cost).
        """
        method_name = method.name
        
        info = self.method_scores.get(method_name)
        if not info:
            return 0.5  # Default cost for unknown methods

        if info['is_termination']:
            return 0.0  # 'finished' method has zero cost as it completes a plan branch.

        score = 0.0
        
        # Add costs based on the types of actions in the method's subtasks.
        score += info['pickup_count'] * self.pickup_penalty
        score += info['move_count'] * self.move_penalty
        score += info['open_count'] * self.open_penalty
        
        # Apply a bonus for methods that achieve a final goal (putdown).
        if info['is_goal_achieving']:
            score += info['putdown_count'] * 0.1  # A small intrinsic cost for the action itself.
            score -= self.putdown_bonus
            
        return max(score, 0.0)

    def __call__(self, parent_node, node):
        """
        Calculate the heuristic value for the given node. This is the main entry point.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini25ProHeuristic12(move={self.move_penalty}, open={self.open_penalty}, pickup={self.pickup_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Robot Gemini 2.5 Pro Heuristic 12 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tMove Penalty: {self.move_penalty}\n"
            f"\tOpen Penalty: {self.open_penalty}\n"
            f"\tPickup Penalty: {self.pickup_penalty}\n"
            f"\tPutdown Bonus: {self.putdown_bonus}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
