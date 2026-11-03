import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGemini20FlashHeuristic15(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_15=1, ontable_penalty=1, on_penalty=1, clear_penalty=1, handempty_penalty=1, name="blocksworld_gemini20flash_heuristic_15"):  # Corrected parameter name
        super().__init__(name=name)
        self.ontable_penalty = ontable_penalty
        self.on_penalty = on_penalty
        self.clear_penalty = clear_penalty
        self.handempty_penalty = handempty_penalty
        self.preprocessing_time = 0
        self.method_scores = {}

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'abstract_tasks'):
            for task in self.model.abstract_tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            subtask_count = 0

            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
                subtask_count = len(subtasks)

            self.method_scores[method_name] = {
                'subtask_count': subtask_count
            }

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'state') or not node.state:
            return 0

        state = node.state
        h_value = 0
        goal_state = self.model.goals # Assuming self.model is accessible

        # Count missing facts in current state compared to goal state
        missing_facts = 0

        for goal_fact_id in goal_state:
             if not (state & (1 << goal_fact_id)):
                missing_facts +=1

        h_value += missing_facts

        if hasattr(node, 'method') and node.method:
            method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)
            if method_name in self.method_scores:
                h_value += self.method_scores[method_name]['subtask_count'] * 0.1  # Slightly penalize complex methods


        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic(ontable_penalty={self.ontable_penalty}, on_penalty={self.on_penalty}, clear_penalty={self.clear_penalty}, handempty_penalty={self.handempty_penalty})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic(ontable_penalty={self.ontable_penalty}, on_penalty={self.on_penalty}, clear_penalty={self.clear_penalty}, handempty_penalty={self.handempty_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 15 info:\n"
            f"\tName: {self.name}\n"
            f"\tOntable penalty: {self.ontable_penalty}\n"
            f"\tOn penalty: {self.on_penalty}\n"
            f"\tClear penalty: {self.clear_penalty}\n"
            f"\tHandempty penalty: {self.handempty_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
