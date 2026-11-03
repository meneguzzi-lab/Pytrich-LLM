import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic14(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_14=1.0, name="BarmanGemini20FlashHeuristic14"):
        super().__init__(name=name)
        self.clean_shot_cost = barman_gemini20flash_heuristic_14  # Cost for cleaning a shot
        self.fill_shot_cost = barman_gemini20flash_heuristic_14 # Cost for filling a shot
        self.make_cocktail_cost = barman_gemini20flash_heuristic_14  # Cost for making a cocktail
        self.pour_cost = barman_gemini20flash_heuristic_14 # Cost to pour cocktail
        self.move_cost = barman_gemini20flash_heuristic_14 # Cost to move
        self.clean_shaker_cost = barman_gemini20flash_heuristic_14 # Cost to clean shaker
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node.
        This heuristic estimates the cost to achieve the goals based on the tasks
        in the node's task network. It considers the costs of cleaning shots and shakers,
        filling shots with ingredients, making cocktails, and pouring cocktails into shots.
        """
        h_value = 0

        if hasattr(node, 'task_network') and node.task_network:
            for task in node.task_network:
                task_name = task.name if hasattr(task, 'name') else str(task)

                if 'AchieveCleanShot' in task_name:
                    h_value += self.clean_shot_cost  # Increment cost for each clean shot task
                elif 'AchieveContainsShotIngredient' in task_name:
                    h_value += self.fill_shot_cost # Increment cost for each fill shot task
                elif 'AchieveContainsShotCocktail' in task_name:
                     h_value += self.make_cocktail_cost # Increment cost for make cocktail
                elif 'DoPourShakerToShot' in task_name:
                    h_value += self.pour_cost # Increment cost for each pour
                elif 'AchieveCleanShaker' in task_name:
                    h_value += self.clean_shaker_cost # Increment cost for clean shaker
                elif 'AchieveHolding' in task_name:
                    h_value += self.move_cost # Increment cost for holding actions

        return h_value

    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic14(clean_shot_cost={self.clean_shot_cost}, fill_shot_cost={self.fill_shot_cost}, make_cocktail_cost={self.make_cocktail_cost}, pour_cost={self.pour_cost}, move_cost={self.move_cost}, clean_shaker_cost={self.clean_shaker_cost})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic14(clean_shot_cost={self.clean_shot_cost}, fill_shot_cost={self.fill_shot_cost}, make_cocktail_cost={self.make_cocktail_cost}, pour_cost={self.pour_cost}, move_cost={self.move_cost}, clean_shaker_cost={self.clean_shaker_cost})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 14 info:\n"
            f"\tName: {self.name}\n"
            f"\tClean Shot Cost: {self.clean_shot_cost}\n"
            f"\tFill Shot Cost: {self.fill_shot_cost}\n"
            f"\tMake Cocktail Cost: {self.make_cocktail_cost}\n"
            f"\tPour Cost: {self.pour_cost}\n"
            f"\tMove Cost: {self.move_cost}\n"
            f"\tClean Shaker Cost: {self.clean_shaker_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
