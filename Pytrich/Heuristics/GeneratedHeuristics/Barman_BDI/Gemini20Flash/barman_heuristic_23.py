import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic23(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_23=1.0, name="BarmanGemini20FlashHeuristic23"):
        super().__init__(name=name)
        self.achieve_contains_shot_cocktail_cost = barman_gemini20flash_heuristic_23
        self.clean_container_bonus = 0.1
        self.hand_empty_bonus = 0.05
        self.holding_bonus = 0.05
        self.ontable_penalty = 0.02
        self.preprocessing_time = 0
        self.method_scores = {}

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
            pour_count = 0
            pickup_count = 0
            drop_count = 0
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
                    if 'pour' in subtask_lower:
                        pour_count += 1
                    if 'grasp' in subtask_lower:
                        pickup_count += 1
                    if 'drop' in subtask_lower:
                        drop_count += 1

            self.method_scores[method_name] = {
                'clean_count': clean_count,
                'pour_count': pour_count,
                'pickup_count': pickup_count,
                'drop_count': drop_count,
                'subtask_count': subtask_count,
                'is_achieve_contains_shot_cocktail': 'achievecontainsshotcocktail' in method_name.lower(),
                'is_make_cocktail_method': 'makecocktail' in method_name.lower(),
                'complexity_score': clean_count + pour_count + pickup_count + drop_count
            }

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

            score += info['clean_count'] * self.clean_container_bonus
            score += info['pour_count'] * 0.1  # small cost for pouring
            score -= info['pickup_count'] * self.holding_bonus # small reward for pickup
            score += info['drop_count'] * self.ontable_penalty

            if info['is_achieve_contains_shot_cocktail']:
                score += self.achieve_contains_shot_cocktail_cost

            if info['is_make_cocktail_method']:
                score += 0.2 # small cost for makecocktail methods

            score += info['subtask_count'] * 0.01 # tiny cost for each subtask

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic23(achieve_contains_shot_cocktail_cost={self.achieve_contains_shot_cocktail_cost})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic23(achieve_contains_shot_cocktail_cost={self.achieve_contains_shot_cocktail_cost})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 23 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tAchieve Contains Shot Cocktail Cost: {self.achieve_contains_shot_cocktail_cost}\n"
            f"\tClean Container Bonus: {self.clean_container_bonus}\n"
            f"\tHand Empty Bonus: {self.hand_empty_bonus}\n"
            f"\tHolding Bonus: {self.holding_bonus}\n"
            f"\tOntable Penalty: {self.ontable_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
