import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for barman domain
class BarmanGemini20FlashHeuristic11(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_11=1.0, name="BarmanGemini20FlashHeuristic11"):
        super().__init__(name=name)
        self.clean_shot_weight = barman_gemini20flash_heuristic_11
        self.clean_shaker_weight = barman_gemini20flash_heuristic_11
        self.contains_weight = barman_gemini20flash_heuristic_11
        self.holding_weight = barman_gemini20flash_heuristic_11
        self.ontable_weight = barman_gemini20flash_heuristic_11
        self.handempty_weight = barman_gemini20flash_heuristic_11
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node based on the state
        and the task network. This heuristic estimates the cost of achieving
        the remaining goals by considering the number of clean shots and shakers
        needed, the number of contains relationships that need to be established,
        and object manipulation requirements (holding, ontable, handempty).
        """
        state = node.state
        tn = node.task_network

        # Initialize heuristic value
        h = 0

        # Count the number of AchieveContainsShotCocktail tasks in the task network
        achieve_contains_shot_cocktail_count = sum(1 for task in tn if hasattr(task, 'name') and task.name == 'AchieveContainsShotCocktail')
        h += achieve_contains_shot_cocktail_count * self.contains_weight

        # Count the number of AchieveCleanShot tasks in the task network
        achieve_clean_shot_count = sum(1 for task in tn if hasattr(task, 'name') and task.name == 'AchieveCleanShot')
        h += achieve_clean_shot_count * self.clean_shot_weight

        # Count the number of AchieveCleanShaker tasks in the task network
        achieve_clean_shaker_count = sum(1 for task in tn if hasattr(task, 'name') and task.name == 'AchieveCleanShaker')
        h += achieve_clean_shaker_count * self.clean_shaker_weight

        # Count the number of AchieveHolding tasks in the task network
        achieve_holding_count = sum(1 for task in tn if hasattr(task, 'name') and task.name == 'AchieveHolding')
        h += achieve_holding_count * self.holding_weight

        # Count the number of AchieveOnTable tasks in the task network
        achieve_ontable_count = sum(1 for task in tn if hasattr(task, 'name') and task.name == 'AchieveOnTable')
        h += achieve_ontable_count * self.ontable_weight

        # Count the number of AchieveHandEmpty tasks in the task network
        achieve_handempty_count = sum(1 for task in tn if hasattr(task, 'name') and task.name == 'AchieveHandEmpty')
        h += achieve_handempty_count * self.handempty_weight


        return h

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic11(clean_shot_weight={self.clean_shot_weight}, clean_shaker_weight={self.clean_shaker_weight}, contains_weight={self.contains_weight}, holding_weight={self.holding_weight}, ontable_weight={self.ontable_weight}, handempty_weight={self.handempty_weight})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic11(clean_shot_weight={self.clean_shot_weight}, clean_shaker_weight={self.clean_shaker_weight}, contains_weight={self.contains_weight}, holding_weight={self.holding_weight}, ontable_weight={self.ontable_weight}, handempty_weight={self.handempty_weight})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 11 info:\n"
            f"\tName: {self.name}\n"
            f"\tClean shot weight: {self.clean_shot_weight}\n"
            f"\tClean shaker weight: {self.clean_shaker_weight}\n"
            f"\tContains weight: {self.contains_weight}\n"
            f"\tHolding weight: {self.holding_weight}\n"
            f"\tOn table weight: {self.ontable_weight}\n"
            f"\tHand empty weight: {self.handempty_weight}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
