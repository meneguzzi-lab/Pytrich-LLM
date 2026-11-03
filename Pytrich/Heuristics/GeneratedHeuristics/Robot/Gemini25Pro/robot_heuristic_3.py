import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic3(Heuristic):
    """
    A domain-dependent heuristic for the Robot (package delivery) domain.

    This heuristic guides the search by assigning costs to decomposition methods
    based on the type of actions they introduce. The core idea is to prioritize
    methods that make direct progress on the goal of delivering packages, while
    assigning higher costs to preparatory actions like moving or opening doors.

    Rationale:
    1. Goal-Oriented Prioritization: The primary goal is to move packages to their
       destinations. Therefore, 'putdown' actions are the most valuable as they
       complete a subgoal. 'pickup' actions are the next most valuable as they
       initiate a delivery.
    2. Cost of Enabling Actions: Actions like 'move' and 'open' do not directly
       progress the task of package delivery but are necessary enablers. They are
       penalized more heavily to encourage the planner to finish a delivery once
       a package is held, rather than moving unnecessarily.
    3. Hierarchy Awareness: The heuristic evaluates methods, which are the core of
       the hierarchical structure. By scoring decompositions, it directly influences
       the high-level strategy of the planner.
    4. Efficiency: The heuristic precomputes scores for each method in the domain
       during an initialization phase. During the search, calculating the heuristic
       value for a node is a fast dictionary lookup, imposing minimal overhead.
    5. Admissibility: The heuristic values are small, constant costs assigned to
       choices in the search space, not estimates of remaining plan length. They act
       as tie-breakers and guides. By keeping the values low (e.g., < 1.0, the cost
       of a single action), we encourage admissibility for optimal search algorithms
       like A*.
    """

    def __init__(self, move_penalty=0.3, open_penalty=0.2, pickup_penalty=0.1, putdown_penalty=0.05, robot_gemini25pro_heuristic_3=None):
        super().__init__(name="robot_gemini25pro_heuristic_3")
        self.move_penalty = move_penalty
        self.open_penalty = open_penalty
        self.pickup_penalty = pickup_penalty
        self.putdown_penalty = putdown_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-analyzing the domain's methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain model to create a
        score profile for each one based on the subtasks they contain.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

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
                    # Subtasks can be Operators or AbstractTasks. We check their names.
                    subtask_name = subtask.name.lower()
                    if 'pickup' in subtask_name:
                        info['pickup_count'] += 1
                    elif 'putdown' in subtask_name:
                        info['putdown_count'] += 1
                    elif 'move' in subtask_name:
                        info['move_count'] += 1
                    elif 'open' in subtask_name:
                        info['open_count'] += 1
            
            self.method_scores[method_name] = info

    def _calculate_node_heuristic(self, node: HTNNode):
        """
        Calculates the heuristic value for a given search node.
        The value is based on the method used to decompose the task at this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, no cost

        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Scores a method based on its precomputed characteristics.
        Lower scores are better.
        """
        score = 0
        method_name = method.name

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            
            if info['is_base_case']:
                return 0  # Goal completion method has zero cost

            # Assign costs based on the penalties for each action type found
            score += info['putdown_count'] * self.putdown_penalty
            score += info['pickup_count'] * self.pickup_penalty
            score += info['open_count'] * self.open_penalty
            score += info['move_count'] * self.move_penalty
        else:
            # Fallback logic if a method was not precomputed (should not happen)
            method_name_lower = method_name.lower()
            if 'finished' in method_name_lower:
                return 0
            if 'putdown' in method_name_lower:
                score += self.putdown_penalty
            if 'pickup' in method_name_lower:
                score += self.pickup_penalty
            if 'open' in method_name_lower:
                score += self.open_penalty
            if 'move' in method_name_lower:
                score += self.move_penalty
        
        return max(score, 0)

    def __call__(self, parent_node: HTNNode, node: HTNNode):
        """
        The main heuristic function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        
        # Keep values small to act as a guide rather than a strict cost estimate,
        # which helps with admissibility in optimal search.
        h_value = min(h_value, 1.0)
        
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return (f"RobotGemini25ProHeuristic3(move={self.move_penalty}, open={self.open_penalty}, "
                f"pickup={self.pickup_penalty}, putdown={self.putdown_penalty})")

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Robot Gemini 2.5 Pro Heuristic (robot_gemini25pro_heuristic_3) info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMove Penalty: {self.move_penalty}\n"
            f"\tOpen Penalty: {self.open_penalty}\n"
            f"\tPickup Penalty: {self.pickup_penalty}\n"
            f"\tPutdown Penalty: {self.putdown_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
