import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic4(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_4=1.0, name="BarmanGemini20FlashHeuristic4"):
        super().__init__(name=name)
        self.clean_shot_cost = barman_gemini20flash_heuristic_4  # Cost for cleaning a shot
        self.fill_shot_cost = barman_gemini20flash_heuristic_4  # Cost for filling a shot
        self.clean_shaker_cost = barman_gemini20flash_heuristic_4  # Cost for cleaning shaker
        self.pour_cost = barman_gemini20flash_heuristic_4  # Cost for pouring
        self.shake_cost = barman_gemini20flash_heuristic_4 # cost for shake action
        self.holding_cost = barman_gemini20flash_heuristic_4 # cost for holding
        self.ontable_cost = barman_gemini20flash_heuristic_4 # cost for putting on the table
        self.handempty_cost = barman_gemini20flash_heuristic_4 # cost for emptying hand
        self.make_cocktail_cost = barman_gemini20flash_heuristic_4 # cost for make cocktail
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

            clean_count = 0
            fill_count = 0
            pour_count = 0
            shake_count = 0
            load_count = 0
            unload_count = 0
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
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)

                    subtask_lower = subtask_name.lower()

                    if 'clean' in subtask_lower:
                        clean_count += 1
                    if 'fill' in subtask_lower:
                        fill_count += 1
                    if 'pour' in subtask_lower:
                        pour_count += 1
                    if 'shake' in subtask_lower:
                        shake_count += 1
                    if 'load' in subtask_lower:
                        load_count += 1
                    if 'unload' in subtask_lower:
                        unload_count += 1

            self.method_scores[method_name] = {
                'clean_count': clean_count,
                'fill_count': fill_count,
                'pour_count': pour_count,
                'shake_count': shake_count,
                'load_count': load_count,
                'unload_count': unload_count,
                'subtask_count': subtask_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'is_base_case': self._is_base_case_method(method_name)
            }

    def _is_goal_achieving_method(self, method_name):
        return 'achievecontainsshotcocktail' in method_name.lower()

    def _is_base_case_method(self, method_name):
        base_indicators = ['Null']
        return any(indicator in method_name.lower() for indicator in base_indicators)

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0

        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['clean_count'] * self.clean_shot_cost
            score += info['fill_count'] * self.fill_shot_cost
            score += info['pour_count'] * self.pour_cost
            score += info['shake_count'] * self.shake_cost

            if info['is_goal_achieving']:
                score -= self.make_cocktail_cost

            if info['is_base_case']:
                score -= self.ontable_cost
        else:
            # Provide a default score if the method is not precomputed.
            score = self.clean_shot_cost + self.fill_shot_cost + self.pour_cost

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic4(clean_shot_cost={self.clean_shot_cost}, fill_shot_cost={self.fill_shot_cost}, shake_cost = {self.shake_cost})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic4(clean_shot_cost={self.clean_shot_cost}, fill_shot_cost={self.fill_shot_cost}, shake_cost = {self.shake_cost})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 4 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tClean shot cost: {self.clean_shot_cost}\n"
            f"\tFill shot cost: {self.fill_shot_cost}\n"
            f"\tShake cost: {self.shake_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
