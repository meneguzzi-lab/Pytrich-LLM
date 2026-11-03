import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGemini25ProHeuristic0(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Design Rationale:
    This heuristic guides the HTN planner by evaluating the complexity of decomposition methods.
    It operates statically on the domain structure, without accessing the planner's state during search.
    The core idea is to prefer simpler decompositions over more complex or recursive ones.

    Key principles:
    1.  **Base Cases are Optimal:** Methods with no subtasks represent a completed goal or a trivial case.
        They are assigned a score of 0, making them highly preferred.
    2.  **Primitives over Abstracts:** Methods that decompose into primitive actions (e.g., 'stack', 'pickup')
        are closer to a final plan than those decomposing into other abstract tasks. A smaller penalty
        is assigned to primitive subtasks.
    3.  **Penalize Recursive Structures:** The 'clear' task in Blocksworld can lead to deep recursion
        as the planner needs to dismantle towers. An extra penalty is added to methods that
        generate 'clear' subtasks to reflect their potential cost.

    The heuristic values are kept small to act as an informed tie-breaker for the search algorithm,
    guiding it towards what are syntactically simpler plans, without overriding the primary cost function.
    """
    def __init__(self, blocksworld_gemini25pro_heuristic_0):
        super().__init__(name="blocksworld_gemini25pro_heuristic_0")
        
        # Heuristic parameters for scoring method complexity
        self.primitive_penalty = 0.1  # Cost for a subtask that is a primitive action
        self.abstract_penalty = 0.3   # Cost for a subtask that is an abstract task
        self.clear_penalty = 0.2      # Additional penalty for 'clear' tasks
        self.base_case_score = 0.0    # Score for methods that are base cases (no subtasks)

        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic and precompute scores for each method in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyze all methods in the domain model to assign a static complexity score.
        This score estimates the method's complexity based on its decomposition.
        """
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'methods'):
            methods = self.model.methods

        # Define primitive actions for the Blocksworld domain
        primitive_actions = {'pickup', 'putdown', 'stack', 'unstack'}
        
        for method in methods:
            method_name = method.name
            score = 0
            
            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'subtasks'):
                subtasks = method.subtasks

            if not subtasks:
                # Base case method (e.g., goal already true).
                score = self.base_case_score
            else:
                for subtask in subtasks:
                    # Extract the action/task name from the full subtask name string
                    subtask_name_parts = subtask.name.split()
                    subtask_action = subtask_name_parts[0].strip('()').lower() if subtask_name_parts else ""

                    # Check if the subtask is a known primitive action
                    if subtask_action in primitive_actions:
                        score += self.primitive_penalty
                    else:  # Assumed to be an abstract task
                        score += self.abstract_penalty
                        # Add specific penalty for the 'clear' subproblem
                        if 'clear' in subtask_action:
                             score += self.clear_penalty
            
            self.method_scores[method_name] = max(0, score)

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a node by looking up its method's precomputed score.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        
        # Return the precomputed score, with a default penalty if not found.
        return self.method_scores.get(method_name, 1.0)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return (f"BlocksworldGemini25ProHeuristic0(primitive_penalty={self.primitive_penalty}, "
                f"abstract_penalty={self.abstract_penalty})")

    def __str__(self):
        return "BlocksworldGemini25ProHeuristic0"
        
    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 0 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tStrategy: Penalize complex decompositions based on subtask type.\n"
            f"\tPrimitive Subtask Penalty: {self.primitive_penalty}\n"
            f"\tAbstract Subtask Penalty: {self.abstract_penalty}\n"
            f"\t'Clear' Task Extra Penalty: {self.clear_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
