import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Operator, AbstractTask

class BlocksworldGemini25ProHeuristic18(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic works by pre-analyzing all the decomposition methods in the domain
    during an initialization phase. It assigns a score to each method based on the
    types of subtasks (primitive and abstract) it introduces.

    The core ideas are:
    1.  **Penalize Clearing Actions**: Methods that are designed to clear blocks off other
        blocks are considered costly because they don't directly contribute to the final
        goal configuration. They are necessary steps, but represent effort to be expended.
        Subtasks like 'clear', 'unstack', and 'putdown' are penalized.
    2.  **Reward Stacking Actions**: Methods that perform the final 'stack' action to place
        a block in its goal position are considered productive. They are assigned a lower cost.
    3.  **Favor Base Cases**: Methods that represent a terminal condition (i.e., the goal
        of the task is already satisfied) have an empty task network. These are given a
        bonus (negative cost) to strongly signal that this path is complete.

    The heuristic values are kept small and fractional to guide the search admissibly (or
    near-admissibly) without overpowering the planner's actual step costs.
    """
    def __init__(self, blocksworld_gemini25pro_heuristic_18={'stack_cost': 0.1, 'clear_cost': 0.3, 'base_case_score': 0.0}):
        super().__init__(name="blocksworld_gemini25pro_heuristic_18")
        # Heuristic parameters: costs for different types of operations
        self.stack_cost = blocksworld_gemini25pro_heuristic_18.get('stack_cost', 0.1)
        self.clear_cost = blocksworld_gemini25pro_heuristic_18.get('clear_cost', 0.3)
        self.base_case_score = blocksworld_gemini25pro_heuristic_18.get('base_case_score', 0.0)

        self.method_scores = {}
        self.preprocessing_time = 0.0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing scores for all methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_scores()
        self.preprocessing_time = time.time() - start_time
        
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_scores(self):
        """
        Analyzes every decomposition method in the domain model and assigns a heuristic score.
        The score estimates the cost of applying that method.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            score = 0.0

            # 1. Handle base cases: If a method has no subtasks, it's a terminal case.
            if not method.task_network:
                score = self.base_case_score
                self.method_scores[method_name] = score
                continue

            # 2. Analyze subtasks to determine the method's purpose and cost.
            for subtask in method.task_network:
                subtask_name = subtask.name.lower()
                
                # A. Score based on primitive actions (Operators)
                if isinstance(subtask, Operator):
                    if 'stack' in subtask_name:
                        # 'stack' is a productive action, contributing directly to the goal.
                        score += self.stack_cost
                    elif 'unstack' in subtask_name or 'putdown' in subtask_name:
                        # 'unstack' and 'putdown' are usually for clearing blocks.
                        score += self.clear_cost / 2
                    elif 'pickup' in subtask_name:
                        # 'pickup' is part of a productive move.
                        score += self.stack_cost / 2
                
                # B. Score based on abstract tasks
                elif isinstance(subtask, AbstractTask):
                    # Penalize methods that create 'clear' subgoals, as they are costly.
                    if 'clear' in subtask_name:
                        score += self.clear_cost
                    # 'do_put_on' is a recursive, goal-directed task.
                    elif 'do_put_on' in subtask_name:
                        score += self.stack_cost

            # 3. Store the final, non-negative score for the method.
            self.method_scores[method_name] = max(0, score)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        """
        # The heuristic value is the pre-computed cost of the method that generated this node.
        if not hasattr(node, 'method') or not node.method:
            return 0.0  # No method applied, e.g., initial task network node.

        method_name = node.method.name
        # Use a default cost if a method was somehow not pre-processed.
        return self.method_scores.get(method_name, 0.5)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        
        # Ensure the heuristic value is small to guide the search without being inadmissible.
        # A value of 1.0 is a safe upper bound as primitive actions have a cost of at least 1.
        h_value = min(h_value, 1.0)
        
        super().update_info(h_value)
        return h_value
        
    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic18(stack_cost={self.stack_cost}, clear_cost={self.clear_cost})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        """
        Returns a formatted string with information about the heuristic's configuration and performance.
        """
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 18 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tStacking Cost Penalty: {self.stack_cost}\n"
            f"\tClearing Cost Penalty: {self.clear_cost}\n"
            f"\tBase Case Score: {self.base_case_score}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
