import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic29(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_29=1.0):
        super().__init__()
        self.h_weight = rover_gemini20flash_heuristic_29
        self.drop_penalty = 1
        self.sample_penalty = 1
        self.navigate_penalty = 1
        self.communicate_penalty = 1
        self.calibrate_penalty = 1
        self.image_penalty = 1
        self.visit_penalty = 0.1

    def initialize(self, model, initial_node):
        self.model = model
        self.goal_facts = set()
        for goal in model.goals:
            self.goal_facts.add(goal.name)
        initial_h_value = self.compute_heuristic(initial_node)
        return super().initialize(model, initial_h_value)

    def compute_heuristic(self, node: HTNNode):
        h_value = 0
        needed_effects = set()
        for goal in self.model.goals:
            if goal.name not in node.state:
                needed_effects.add(goal.name)

        h_value += len(needed_effects)
        if node.operator:
            op_name = node.operator.name
            if 'drop' in op_name:
                h_value += self.drop_penalty
            elif 'sample' in op_name:
                h_value += self.sample_penalty
            elif 'navigate' in op_name:
                h_value += self.navigate_penalty
            elif 'communicate' in op_name:
                h_value += self.communicate_penalty
            elif 'calibrate' in op_name:
                h_value += self.calibrate_penalty
            elif 'take_image' in op_name:
                h_value += self.image_penalty
            elif 'visit' in op_name or 'unvisit' in op_name:
                h_value += self.visit_penalty
        return h_value * self.h_weight

    def __call__(self, parent_node, node):
        h_value = self.compute_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic29(h_weight={self.h_weight})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic29(h_weight={self.h_weight})\n"
