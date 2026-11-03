import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGemini25ProHeuristic14(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic works by performing a static analysis of the domain's methods
    during initialization. It assigns a cost to each method based on the complexity
    and type of its subtasks, without needing to access the dynamic state of the world
    during the search. This approach is necessary as the heuristic function in this
    framework is not provided with the current world state.

    The core ideas are:
    1.  **Penalize Preparatory Work**: Methods that decompose into tasks for clearing
        blocks are given a higher penalty. Clearing a block is often a prerequisite
        for a desired action and indicates a more complex state to resolve.
    2.  **Cost for Actions**: Methods that decompose into primitive actions like `stack`,
        `unstack`, `pickup`, and `putdown` are assigned a small, base cost.
    3.  **Reward Termination**: Methods that are base cases (i.e., have no subtasks)
        are given a small bonus (negative cost) to encourage the planner to terminate
        recursive sub-problems.
    4.  **Static Analysis**: All costs are pre-computed and stored in a dictionary,
        mapping method names to their heuristic scores, ensuring fast lookups during
        the search.
    """
    def __init__(self, blocksworld_gemini25pro_heuristic_14):
        """
        Initializes the heuristic with configurable penalties and bonuses.

        Args:
            blocksworld_gemini25pro_heuristic_14 (dict): A dictionary containing heuristic parameters.
                Expected keys are 'move_penalty', 'clear_penalty', 'goal_task_penalty',
                and 'base_case_bonus'.
        """
        super().__init__(name="blocksworld_gemini25pro_heuristic_14")
        # Use .get() to provide default values for robustness
        self.move_penalty = blocksworld_gemini25pro_heuristic_14.get('move_penalty', 0.1)
        self.clear_penalty = blocksworld_gemini25pro_heuristic_14.get('clear_penalty', 0.3)
        self.goal_task_penalty = blocksworld_gemini25pro_heuristic_14.get('goal_task_penalty', 0.2)
        self.base_case_bonus = blocksworld_gemini25pro_heuristic_14.get('base_case_bonus', 0.05)

        self.model = None
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing scores for all methods in the model.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain model to assign a static
        heuristic score based on their structure and subtasks.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            score = 0
            subtask_count = 0
            move_count = 0
            clear_count = 0

            # Analyze the primary task this method decomposes
            # Tasks involving clearing are penalized the most.
            task_name = method.compound_task.name.lower() if hasattr(method, 'compound_task') else ""
            if 'clear' in task_name:
                score += self.clear_penalty
            elif 'put_on' in task_name or 'do_on' in task_name:
                score += self.goal_task_penalty

            # Analyze the method's subtasks
            if hasattr(method, 'task_network'):
                subtasks = method.task_network
                subtask_count = len(subtasks)

                if subtask_count == 0:
                    # This is a base case method, reward it.
                    score -= self.base_case_bonus
                else:
                    for subtask in subtasks:
                        subtask_name = subtask.name.lower()
                        # Count types of subtasks to estimate complexity
                        if any(term in subtask_name for term in ['unstack', 'stack', 'pickup', 'putdown']):
                            move_count += 1
                        elif 'clear' in subtask_name:
                            clear_count += 1
            
            # Combine penalties to get the final score for this method
            total_score = (
                score +
                (move_count * self.move_penalty) +
                (clear_count * self.clear_penalty)
            )

            self.method_scores[method_name] = max(0, total_score)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        The value is retrieved from the pre-computed scores.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, so no estimated cost

        method_name = node.method.name
        # Return the pre-computed score, with a default penalty if not found
        return self.method_scores.get(method_name, self.goal_task_penalty)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        Ensures the heuristic value is small to maintain admissibility.
        """
        h_value = self._calculate_node_heuristic(node)
        # Clamp the value to a maximum of 1.0 to ensure it acts as a small, admissible guide
        h_value = min(h_value, 1.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic14(penalties={{'move': {self.move_penalty}, 'clear': {self.clear_penalty}}})"

    def __str__(self):
        return f"BlocksworldGemini25ProHeuristic14"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 14 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tClear penalty: {self.clear_penalty}\n"
            f"\tGoal Task penalty: {self.goal_task_penalty}\n"
            f"\tBase Case bonus: {self.base_case_bonus}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
