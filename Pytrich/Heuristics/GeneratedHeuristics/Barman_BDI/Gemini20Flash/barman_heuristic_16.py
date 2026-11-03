import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for barman domain.
class BarmanGemini20FlashHeuristic16(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_16=1, name="BarmanGemini20FlashHeuristic16"):
        super().__init__(name=name)
        self.achieve_contains_shot_cocktail_weight = 5
        self.achieve_clean_shot_weight = 3
        self.achieve_contains_shaker_cocktail_weight = 4
        self.do_pour_shaker_to_shot_weight = 2
        self.achieve_on_table_weight = 1
        self.achieve_holding_weight = 1
        self.achieve_clean_shaker_weight = 3
        self.achieve_contains_shaker_ingredient_weight = 4
        self.achieve_hand_empty_weight = 1
        self.barman_gemini20flash_heuristic_16 = barman_gemini20flash_heuristic_16
        self.method_scores = {}
        self.preprocessing_time = 0

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
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            self.method_scores[method_name] = self._calculate_method_score(method_name)

    def _calculate_method_score(self, method_name):
        score = 0
        method_lower = method_name.lower()

        if 'makeandpourcocktail' in method_lower:
            score += 5
        elif 'makecocktail' in method_lower:
            score += 4
        elif 'addingredient' in method_lower:
            score += 3
        elif 'clean' in method_lower:
            score += 2
        elif 'pickup' in method_lower or 'putdown' in method_lower:
            score += 1
        return score

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)
        if method_name in self.method_scores:
            return self.method_scores[method_name] * self.barman_gemini20flash_heuristic_16
        else:
            return self._calculate_task_heuristic(node)  # Fallback to task-based heuristic

    def _calculate_task_heuristic(self, node):
        if not hasattr(node, 'task') or not node.task:
            return 0

        task_name = node.task.name if hasattr(node.task, 'name') else str(node.task)
        task_lower = task_name.lower()

        if 'achievecontainsshotcocktail' in task_lower:
            return self.achieve_contains_shot_cocktail_weight * self.barman_gemini20flash_heuristic_16
        elif 'achievecleanshot' in task_lower:
            return self.achieve_clean_shot_weight * self.barman_gemini20flash_heuristic_16
        elif 'achievecontainsshakercocktail' in task_lower:
            return self.achieve_contains_shaker_cocktail_weight * self.barman_gemini20flash_heuristic_16
        elif 'dopourshakertoshot' in task_lower:
            return self.do_pour_shaker_to_shot_weight * self.barman_gemini20flash_heuristic_16
        elif 'achieveontable' in task_lower:
            return self.achieve_on_table_weight * self.barman_gemini20flash_heuristic_16
        elif 'achieveholding' in task_lower:
            return self.achieve_holding_weight * self.barman_gemini20flash_heuristic_16
        elif 'achievecleanshaker' in task_lower:
            return self.achieve_clean_shaker_weight * self.barman_gemini20flash_heuristic_16
        elif 'achievecontainsshakeringredient' in task_lower:
            return self.achieve_contains_shaker_ingredient_weight * self.barman_gemini20flash_heuristic_16
        elif 'achievehandempty' in task_lower:
            return self.achieve_hand_empty_weight * self.barman_gemini20flash_heuristic_16
        else:
            return 0  # Default heuristic value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic16(barman_gemini20flash_heuristic_16={self.barman_gemini20flash_heuristic_16})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic16(barman_gemini20flash_heuristic_16={self.barman_gemini20flash_heuristic_16})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 16 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tAchieve Contains Shot Cocktail Weight: {self.achieve_contains_shot_cocktail_weight}\n"
            f"\tAchieve Clean Shot Weight: {self.achieve_clean_shot_weight}\n"
            f"\tAchieve Contains Shaker Cocktail Weight: {self.achieve_contains_shaker_cocktail_weight}\n"
            f"\tDo Pour Shaker To Shot Weight: {self.do_pour_shaker_to_shot_weight}\n"
        )
