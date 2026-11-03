import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotGemini20FlashHeuristic30(Heuristic):
    def __init__(self, robot_gemini20flash_heuristic_30=1.0, name="RobotGemini20FlashHeuristic30"):
        super().__init__(name=name)
        self.holding_penalty = robot_gemini20flash_heuristic_30
        self.preprocessing_time = 0
        self.fact_dict = {}

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_facts_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_facts_info(self):
        for fact in self.model.facts:
            self.fact_dict[fact.name] = fact

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'state') or not node.state:
            return 0

        h_value = 0
        state = node.state

        # Check for holding predicate and apply penalty
        for bit_pos in range(state.bit_length()):
            if state & (1 << bit_pos):
                fact = self.model.get_component(bit_pos)
                if fact.name.startswith('holding'):
                    h_value += self.holding_penalty

        # Estimate distance to goal location for held objects
        if hasattr(node, 'task') and node.task:
            task_str = str(node.task)

            if 'achieve-goals' in task_str:
                # Iterate through facts in the current state
                for bit_pos in range(state.bit_length()):
                    if state & (1 << bit_pos):
                        fact = self.model.get_component(bit_pos)
                        # If holding an object, evaluate distance to goal
                        if fact.name.startswith('holding'):
                            obj = fact.name.split(' ')[1]  # Extract object name
                            # Search for the corresponding goal_in fact
                            goal_fact = None
                            for f in self.model.facts:
                                if f.name.startswith('goal_in') and obj in f.name:
                                    goal_fact = f
                                    break

                            if goal_fact:
                                # Add a small bonus if already in the correct location
                                current_location = None
                                for f in self.model.facts:
                                    if f.name.startswith('in') and obj in f.name and f in self.model.initial_state:
                                        current_location = f.name.split(' ')[2]
                                        break

                                if current_location and goal_fact.name.split(' ')[2] == current_location:
                                    h_value -= 0.1

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini20FlashHeuristic30(holding_penalty={self.holding_penalty})"

    def __str__(self):
        return f"RobotGemini20FlashHeuristic30(holding_penalty={self.holding_penalty})"

    def __output__(self):
        return (
            f"RobotGemini20FlashHeuristic30 info:\n"
            f"\tName: {self.name}\n"
            f"\tHolding penalty: {self.holding_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
