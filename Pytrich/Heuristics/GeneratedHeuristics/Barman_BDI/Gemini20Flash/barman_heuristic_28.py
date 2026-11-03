import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic28(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_28=1.0, name="BarmanGemini20FlashHeuristic28"):
        super().__init__(name=name)
        self.clean_container_reward = barman_gemini20flash_heuristic_28
        self.contains_reward = barman_gemini20flash_heuristic_28
        self.holding_reward = barman_gemini20flash_heuristic_28
        self.handempty_reward = barman_gemini20flash_heuristic_28
        self.ontable_reward = barman_gemini20flash_heuristic_28
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value based on the current node's task network.
        This heuristic estimates the cost based on achieving certain predicates,
        such as 'clean', 'contains', 'holding', 'handEmpty', and 'onTable'.
        """
        h_value = 0
        if not isinstance(node, HTNNode): # added safety check
            return 0

        if not hasattr(node, 'task_network') or node.task_network is None:
            return 0

        for task in node.task_network:
            task_str = str(task)
            if 'AchieveCleanShaker' in task_str or 'AchieveCleanShot' in task_str:
                h_value += self.clean_container_reward
            elif 'AchieveContainsShakerCocktail' in task_str or 'AchieveContainsShotCocktail' in task_str or 'AchieveContainsShakerIngredient' in task_str or 'AchieveContainsShotIngredient' in task_str:
                h_value += self.contains_reward
            elif 'AchieveHolding' in task_str:
                h_value += self.holding_reward
            elif 'AchieveHandEmpty' in task_str:
                h_value += self.handempty_reward
            elif 'AchieveOnTable' in task_str:
                h_value += self.ontable_reward

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic28(clean_container={self.clean_container_reward}, contains={self.contains_reward}, holding={self.holding_reward}, handempty={self.handempty_reward}, ontable={self.ontable_reward})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic28(clean_container={self.clean_container_reward}, contains={self.contains_reward}, holding={self.holding_reward}, handempty={self.handempty_reward}, ontable={self.ontable_reward})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 28 info:\n"
            f"\tName: {self.name}\n"
            f"\tClean container reward: {self.clean_container_reward}\n"
            f"\tContains reward: {self.contains_reward}\n"
            f"\tHolding reward: {self.holding_reward}\n"
            f"\tHandEmpty reward: {self.handempty_reward}\n"
            f"\tOnTable reward: {self.ontable_reward}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
