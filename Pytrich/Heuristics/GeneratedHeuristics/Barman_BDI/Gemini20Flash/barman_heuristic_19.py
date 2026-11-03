import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic19(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_19: int = 1, name="BarmanGemini20FlashHeuristic19"):
        super().__init__(name=name)
        self.clean_shot_cost = 2
        self.clean_shaker_cost = 3
        self.add_ingredient_cost = 2
        self.pour_cost = 1
        self.shake_cost = 4
        self.holding_cost = 1
        self.ontable_cost = 1
        self.handempty_cost = 1
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.facts = model.facts
        self.operators = model.operators
        self.abstract_tasks = model.abstract_tasks
        self.decompositions = model.decompositions
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node.
        This heuristic estimates the cost of achieving the remaining tasks by
        considering the number of cleanings, ingredient additions, pouring operations,
        shaking operations and object manipulations needed.
        """
        h_value = 0
        task_network = node.task_network
        state = node.state

        for task in task_network:
            if isinstance(task, Operator):
                task_name = task.name.lower()
                if 'clean-shot' in task_name:
                    h_value += self.clean_shot_cost
                elif 'clean-shaker' in task_name:
                    h_value += self.clean_shaker_cost
                elif 'fill-shot' in task_name or 'pour-shot-to-clean-shaker' in task_name or 'pour-shot-to-used-shaker' in task_name:
                    h_value += self.add_ingredient_cost
                elif 'pour-shaker-to-shot' in task_name:
                    h_value += self.pour_cost
                elif 'shake' in task_name:
                    h_value += self.shake_cost
                elif 'grasp' in task_name or 'drop' in task_name:
                    h_value += self.holding_cost #Object Manipulation
                else:
                    h_value += 1
            elif isinstance(task, AbstractTask):
                task_name = task.name.lower()
                if 'achieveclean' in task_name:
                    if 'shot' in task_name:
                        h_value += self.clean_shot_cost
                    elif 'shaker' in task_name:
                        h_value += self.clean_shaker_cost
                elif 'achievecontainsshotingredient' in task_name or 'achievecontainsshakeringredient' in task_name:
                    h_value += self.add_ingredient_cost
                elif 'achieveholding' in task_name or 'achieveontable' in task_name:
                    h_value += self.ontable_cost
                elif 'achievehandempty' in task_name:
                    h_value += self.handempty_cost
                else:
                     h_value += 1
            else:
                h_value += 1

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic19()"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic19()"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 19 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tClean Shot Cost: {self.clean_shot_cost}\n"
            f"\tClean Shaker Cost: {self.clean_shaker_cost}\n"
            f"\tAdd Ingredient Cost: {self.add_ingredient_cost}\n"
            f"\tPour Cost: {self.pour_cost}\n"
            f"\tShake Cost: {self.shake_cost}\n"
        )
