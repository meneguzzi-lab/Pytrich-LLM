import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGemini25ProHeuristic6(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic provides guidance to the planner by statically analyzing the
    structure of decomposition methods. Since direct state inspection during search is
    often limited or computationally expensive within some planner frameworks, this
    heuristic pre-computes a "complexity" score for each method in the domain.

    The scoring is based on the following principles:
    1.  **Simplicity is better:** Methods with fewer subtasks are preferred as they
        likely represent shorter paths to a solution.
    2.  **Preparatory actions have a cost:** Subtasks related to clearing blocks
        (e.g., 'unstack', 'clear', 'pick-up') add to the heuristic cost, as they
        represent necessary but non-goal-achieving work.
    3.  **Goal-achieving actions are good:** Subtasks that directly place a block
        in its goal position (e.g., 'stack', 'put-on') are rewarded with a small
        heuristic bonus (a reduction in cost).
    4.  **Base cases are ideal:** Methods that represent a terminal condition (i.e.,
        the task is already achieved) have a zero cost.

    The heuristic values are kept small to guide the search without compromising
    admissibility, making it suitable for optimal or near-optimal planning.
    """
    def __init__(self, blocksworld_gemini25pro_heuristic_6={"subtask_penalty": 0.1, "clear_penalty": 0.2, "stack_bonus": 0.05}):
        """
        Initializes the BlocksworldGemini25ProHeuristic.
        :param blocksworld_gemini25pro_heuristic_6: A dictionary of weights for scoring.
        """
        super().__init__(name="blocksworld_gemini25pro_heuristic_6")
        self.penalties = blocksworld_gemini25pro_heuristic_6
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
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
        Statically analyzes all methods in the domain to assign a complexity score.
        This score is pre-calculated and stored for fast retrieval during planning.
        """
        if not self.model:
            return

        # Accommodate different model structures for accessing methods/decompositions
        methods = []
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            score = 0.0

            # Get the list of subtasks for the current method
            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network

            # If a method has no subtasks, it's a base case with zero cost.
            if not subtasks:
                self.method_scores[method_name] = 0.0
                continue

            # Penalize based on the number of subtasks
            score += len(subtasks) * self.penalties.get("subtask_penalty", 0.1)

            for subtask in subtasks:
                subtask_name = ""
                if hasattr(subtask, 'name'):
                    subtask_name = subtask.name
                elif hasattr(subtask, 'task'):
                    subtask_name = str(subtask.task)
                else:
                    subtask_name = str(subtask)
                
                subtask_lower = subtask_name.lower()

                # Penalize methods that require clearing blocks
                if any(op in subtask_lower for op in ['clear', 'unstack', 'pick-up']):
                    score += self.penalties.get("clear_penalty", 0.2)

                # Reward methods that perform the final stacking action
                if any(op in subtask_lower for op in ['stack', 'put-on']):
                    score -= self.penalties.get("stack_bonus", 0.05)

            # Store the final computed score, ensuring it's non-negative
            self.method_scores[method_name] = max(0, score)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        This is done by looking up the pre-computed score of the method
        that generated this node.
        """
        # If the node was not generated by a method (e.g., initial node), heuristic is 0
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)

        # Return the pre-computed score, defaulting to a neutral value if not found
        return self.method_scores.get(method_name, 0.5)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        # Ensure the heuristic value is small to act as a gentle guide
        h_value = min(h_value, 1.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic6(penalties={self.penalties})"

    def __str__(self):
        return f"BlocksworldGemini25ProHeuristic6 with penalties: {self.penalties}"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 6 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tPenalties: {self.penalties}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
