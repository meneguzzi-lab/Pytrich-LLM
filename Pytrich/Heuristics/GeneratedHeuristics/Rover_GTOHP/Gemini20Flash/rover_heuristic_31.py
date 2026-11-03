import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic31(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_31="RoverGemini20FlashHeuristic31"): # all lower case init parameter name
        super().__init__(name=rover_gemini20flash_heuristic_31)
        self.task_weights = {}
        self.fact_weights = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_task_fact_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_task_fact_info(self):
        """Precompute the frequency of tasks and facts in methods and actions."""
        self.task_weights = {}
        self.fact_weights = {}

        for task in self.model.abstract_tasks:
            self.task_weights[task.name] = 0.5

        for fact in self.model.facts:
            self.fact_weights[fact.name] = 0.1

    def _calculate_node_heuristic(self, node: HTNNode):
        """Estimate the distance to the goal from the current node."""
        h_value = 0

        if node.uncompleted_tasks:
            for task in node.uncompleted_tasks:
                task_name = task.name if hasattr(task, 'name') else str(task)
                h_value += self.task_weights.get(task_name, 0.5)

        # Analyze state and goal facts for additional heuristic guidance.
        state_facts = self.model.state_explicit_repr(node.state)
        for goal_fact in self.model.goals:
            if goal_fact not in state_facts:
                h_value += self.fact_weights.get(goal_fact, 0.2)  # Reward known facts.

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic31()"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic31()"

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 31 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
