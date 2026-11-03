import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic29(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_29=1.0, name="BarmanGemini20FlashHeuristic29"):
        super().__init__(name=name)
        self.clean_shot_cost = barman_gemini20flash_heuristic_29
        self.clean_shaker_cost = barman_gemini20flash_heuristic_29
        self.fill_shot_cost = barman_gemini20flash_heuristic_29
        self.pour_cost = barman_gemini20flash_heuristic_29
        self.shake_cost = barman_gemini20flash_heuristic_29
        self.grasp_cost = barman_gemini20flash_heuristic_29
        self.drop_cost = barman_gemini20flash_heuristic_29
        self.achieve_contains_cost = barman_gemini20flash_heuristic_29
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculates a heuristic value for a given node in the search space.
        This heuristic estimates the cost to achieve the remaining tasks based on the current state.
        It considers the number of 'clean', 'contains', and 'ontable' predicates that need to be achieved.
        """
        h_value = 0

        # Estimate cost for AchieveContainsShotCocktail tasks
        contains_shot_cocktail_tasks = 0
        if node.task and node.task.name == "AchieveContainsShotCocktail":
           contains_shot_cocktail_tasks +=1
        h_value += contains_shot_cocktail_tasks * self.achieve_contains_cost


        # Penalize if a shot is not clean
        if node.task and node.task.name == "AchieveCleanShot":
            h_value += self.clean_shot_cost  # Cost for cleaning a shot

        # Penalize if a shaker is not clean
        if node.task and node.task.name == "AchieveCleanShaker":
            h_value += self.clean_shaker_cost  # Cost for cleaning a shaker
        #Base actions
        if node.operator and node.operator.name == "fill-shot":
            h_value += self.fill_shot_cost  # Cost for fill shot
        if node.operator and node.operator.name == "pour-shaker-to-shot":
            h_value += self.pour_cost  # Cost for pouring
        if node.operator and node.operator.name == "shake":
            h_value += self.shake_cost  # Cost for shaking
        if node.operator and node.operator.name == "grasp":
            h_value += self.grasp_cost  # Cost for grasp
        if node.operator and node.operator.name == "drop":
            h_value += self.drop_cost  # Cost for drop

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic29(clean_shot_cost={self.clean_shot_cost}, clean_shaker_cost={self.clean_shaker_cost}, fill_shot_cost={self.fill_shot_cost})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic29(clean_shot_cost={self.clean_shot_cost}, clean_shaker_cost={self.clean_shaker_cost}, fill_shot_cost={self.fill_shot_cost})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 29 info:\n"
            f"\tName: {self.name}\n"
            f"\tClean Shot Cost: {self.clean_shot_cost}\n"
            f"\tClean Shaker Cost: {self.clean_shaker_cost}\n"
            f"\tFill Shot Cost: {self.fill_shot_cost}\n"
            f"\tPour Cost: {self.pour_cost}\n"
            f"\tShake Cost: {self.shake_cost}\n"
            f"\tGrasp Cost: {self.grasp_cost}\n"
            f"\tDrop Cost: {self.drop_cost}\n"
            f"\tAchieve Contains Cost: {self.achieve_contains_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
