import time
from Pytrich.Heuristics.heuristic import Heuristic

class BarmanGemini25ProHeuristic16(Heuristic):
    """
    A domain-dependent heuristic for the Barman domain.

    Rationale:
    The Barman domain involves a sequence of preparation, mixing, and serving tasks.
The primary costs in any plan come from a few key action types:
    1. Cleaning: Containers (shakers, shots) must be cleaned before use, which is pure overhead.
    2. Manipulation: Picking up and putting down items (grasp, drop) with limited hands.
       Efficient plans minimize redundant manipulations.
    3. Transfer: Pouring ingredients or cocktails. These are productive actions central to the goal.
    4. Shaking: The key value-adding step in creating a cocktail.

    This heuristic works by pre-computing a cost for each decomposition method in the domain.
The cost is an estimate of the number and type of primitive actions that will be executed as a result of choosing that method. By assigning different penalties to different action types, the heuristic can guide the search towards more efficient plans.

    - Higher penalties are given to non-productive or high-effort tasks like cleaning.
    - Medium penalties are given to manipulation actions to encourage efficient use of hands.
    - Lower penalties are assigned to productive actions like pouring and shaking.

    The heuristic is admissible because the assigned penalties are small, constant values that are less than the true cost of any single action (typically 1). Therefore, the heuristic value is always a lower bound on the actual cost to complete the plan.
    The main computation is performed once at initialization, making the heuristic very fast during the search process.
    """
    def __init__(self, barman_gemini25pro_heuristic_16="barman_gemini25pro_heuristic_16", clean_penalty=0.4, manipulation_penalty=0.2, transfer_penalty=0.1, shake_penalty=0.1):
        super().__init__(name=barman_gemini25pro_heuristic_16)
        # Penalties for different types of actions, kept small for admissibility
        self.clean_penalty = clean_penalty          # Cleaning is often overhead
        self.manipulation_penalty = manipulation_penalty  # Grasping/dropping items
        self.transfer_penalty = transfer_penalty    # Pouring/filling are productive steps
        self.shake_penalty = shake_penalty          # Shaking is a key productive step
        
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing scores for all methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precomputes a score for each decomposition method based on the cost
        of the primitive actions in its subtasks. This avoids recalculating
        scores during search.
        """
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)
        
        for method in methods:
            method_name = method.name
            score = 0
            
            # Null methods represent goals that are already satisfied and have zero cost.
            if "Null" in method_name:
                self.method_scores[method_name] = 0
                continue

            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'subtasks'):
                subtasks = method.subtasks

            # Score the method based on the weighted sum of its subtasks' action types
            for subtask in subtasks:
                subtask_name = subtask.name.lower()
                
                # Penalize cleaning actions
                if 'clean-' in subtask_name or 'empty-' in subtask_name:
                    score += self.clean_penalty
                
                # Penalize manipulation actions (picking up/putting down)
                elif 'grasp' in subtask_name or 'drop' in subtask_name:
                    score += self.manipulation_penalty

                # Add a smaller cost for productive transfer actions
                elif 'pour-' in subtask_name or 'fill-' in subtask_name:
                    score += self.transfer_penalty
                
                # Add a cost for the main cocktail creation step
                elif 'shake' in subtask_name:
                    score += self.shake_penalty

            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node. The value is based on
        the pre-computed score of the method used to decompose the current task.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, no additional cost

        method_name = node.method.name
        
        # Return the pre-computed score for the method. Default to 0 if not found.
        return self.method_scores.get(method_name, 0)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic16(clean={self.clean_penalty}, manip={self.manipulation_penalty}, transfer={self.transfer_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Barman Gemini 2.5 Pro Heuristic 16 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tClean Penalty: {self.clean_penalty}\n"
            f"\tManipulation Penalty: {self.manipulation_penalty}\n"
            f"\tTransfer Penalty: {self.transfer_penalty}\n"
            f"\tShake Penalty: {self.shake_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
