import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini25ProHeuristic19(Heuristic):
    """
    A domain-dependent heuristic for the HDDL ROVER domain.

    Design Rationale:
    1.  **Domain Analysis**: The ROVER domain is heavily dominated by navigation. Most high-level tasks
        (`get_soil_data`, `get_image_data`, etc.) require multiple navigation steps to move rovers
        between waypoints for sampling, calibration, imaging, and communication.
        Therefore, the primary driver of plan cost and complexity is the number of `navigate` actions.

    2.  **Heuristic Strategy**: The heuristic estimates the cost of a decomposition choice (a method)
        by analyzing its subtasks. It aims to guide the planner towards methods that are more efficient,
        i.e., those that achieve tasks with fewer actions, particularly fewer navigation actions.
        It works as a cost-to-go estimator for a single decomposition step.

    3.  **Preprocessing**: To ensure the heuristic is fast, it pre-computes a score for every method
        in the domain during an `initialize` phase. This score is based on the types of subtasks
        (primitive actions vs. abstract tasks) the method introduces.

    4.  **Scoring Logic**: The score for a method is a weighted sum of its subtasks:
        - **Navigation Penalty (`nav_penalty`)**: Each `navigate` action in a method's subtasks adds a
          significant penalty. This is set higher than other actions to reflect its higher cost and
          impact on the plan.
        - **Action Penalty (`action_penalty`)**: Other primitive actions (`sample`, `calibrate`, `drop`, etc.)
          add a standard penalty. Control-flow actions like `nop`, `visit`, and `unvisit` are ignored as
          they do not correspond to real-world costs.
        - **Task Penalty (`task_penalty`)**: Decomposing into further abstract tasks adds a small penalty,
          favoring methods that are closer to producing a primitive plan (i.e., shorter decomposition depth).

    5.  **Admissibility**: By setting all penalties to 1.0, the heuristic would be admissible, as it would count
        the minimum number of real actions introduced by a method. The default penalties are set higher
        to provide more aggressive (greedy) guidance, which is often effective for finding solutions quickly,
        even if not strictly optimal.

    6.  **Scalability**: The preprocessing approach ensures that the heuristic's runtime per node is O(1) (a dictionary lookup),
        making it highly efficient and scalable to larger problems with more tasks.
    """
    def __init__(self, rover_gemini25pro_heuristic_19="rover_gemini25pro_heuristic_19", nav_penalty=1.5, action_penalty=1.0, task_penalty=0.2):
        """
        Initializes the heuristic.
        :param rover_gemini25pro_heuristic_19: The name for this heuristic instance.
        :param nav_penalty: The cost assigned to a 'navigate' primitive action.
        :param action_penalty: The cost assigned to other non-navigation primitive actions.
        :param task_penalty: The cost assigned to decomposing into a new abstract task.
        """
        super().__init__(name=rover_gemini25pro_heuristic_19)
        self.nav_penalty = nav_penalty
        self.action_penalty = action_penalty
        self.task_penalty = task_penalty
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
        Analyzes all decomposition methods in the model and calculates a complexity score for each.
        The score is based on a weighted count of the method's subtask types.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            
            nav_count = 0
            action_count = 0
            task_count = 0
            
            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network
            
            # Methods with no subtasks or only 'nop' have zero cost.
            is_nop_method = not subtasks or all("nop" in str(s.name).lower() for s in subtasks)
            if is_nop_method:
                 self.method_scores[method_name] = {'score': 0.0}
                 continue

            for subtask in subtasks:
                subtask_name = str(subtask.name).lower()
                
                # Distinguish between primitive Operators and abstract AbstractTasks.
                # AbstractTasks have a 'decompositions' attribute, Operators do not.
                if not hasattr(subtask, 'decompositions'): 
                    # This is a primitive action (Operator).
                    if 'navigate' in subtask_name:
                        nav_count += 1
                    # Ignore search-control actions, as they don't have a real-world cost.
                    elif any(ctrl in subtask_name for ctrl in ['nop', 'visit', 'unvisit']):
                        pass 
                    else:
                        action_count += 1
                else: 
                    # This is an abstract task.
                    task_count += 1
            
            # Calculate the weighted score for the method.
            score = (nav_count * self.nav_penalty +
                     action_count * self.action_penalty +
                     task_count * self.task_penalty)
            
            self.method_scores[method_name] = {'score': score}

    def _calculate_node_heuristic(self, node):
        """
        Retrieves the pre-calculated heuristic value for the method used to generate this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0
        
        method_name = node.method.name
        # Return the pre-computed score. Default to 1.0 if method is somehow not found.
        return self.method_scores.get(method_name, {'score': 1.0}).get('score', 1.0)

    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for a given search node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini25ProHeuristic19(nav={self.nav_penalty}, action={self.action_penalty}, task={self.task_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Rover Gemini 25 Pro Heuristic 19 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tNavigation Penalty: {self.nav_penalty}\n"
            f"\tAction Penalty: {self.action_penalty}\n"
            f"\tTask Penalty: {self.task_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
