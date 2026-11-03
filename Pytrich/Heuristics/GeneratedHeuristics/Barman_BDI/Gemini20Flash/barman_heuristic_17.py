import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic17(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_17=1.0, name="BarmanGemini20FlashHeuristic17"):
        super().__init__(name=name)
        self.shot_penalty = 0.5
        self.shaker_penalty = 0.7
        self.ingredient_penalty = 0.3
        self.clean_penalty = 0.2
        self.holding_penalty = 0.1
        self.pour_penalty = 0.4
        self.shake_penalty = 0.6
        self.ontable_penalty = 0.1
        self.empty_penalty = 0.1

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
            shot_count = 0
            shaker_count = 0
            ingredient_count = 0
            clean_count = 0
            holding_count = 0
            pour_count = 0
            shake_count = 0
            subtask_count = 0
            ontable_count = 0
            empty_count = 0

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

                    if 'shot' in subtask_lower:
                        shot_count += 1
                    if 'shaker' in subtask_lower:
                        shaker_count += 1
                    if 'ingredient' in subtask_lower:
                        ingredient_count += 1
                    if 'clean' in subtask_lower:
                        clean_count += 1
                    if 'holding' in subtask_lower:
                        holding_count += 1
                    if 'pour' in subtask_lower:
                        pour_count += 1
                    if 'shake' in subtask_lower:
                        shake_count += 1
                    if 'ontable' in subtask_lower:
                        ontable_count += 1
                    if 'empty' in subtask_lower:
                        empty_count += 1

            self.method_scores[method_name] = {
                'shot_count': shot_count,
                'shaker_count': shaker_count,
                'ingredient_count': ingredient_count,
                'clean_count': clean_count,
                'holding_count': holding_count,
                'pour_count': pour_count,
                'shake_count': shake_count,
                'subtask_count': subtask_count,
                'ontable_count': ontable_count,
                'empty_count': empty_count,
                'is_achieve_contains_shot_cocktail': self._is_achieve_contains_shot_cocktail(method_name),
                'is_transport_method': self._is_transport_method(method_name),
                'is_base_case': self._is_base_case_method(method_name),
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': shot_count + shaker_count + ingredient_count + clean_count + holding_count + pour_count + shake_count
            }

    def _is_achieve_contains_shot_cocktail(self, method_name):
        return 'achievecontainsshotcocktail' in method_name.lower()

    def _is_transport_method(self, method_name):
        transport_indicators = ['pour', 'fill', 'grasp', 'drop']
        return any(indicator in method_name.lower() for indicator in transport_indicators)

    def _is_base_case_method(self, method_name):
        base_indicators = ['null']
        return any(indicator in method_name.lower() for indicator in base_indicators)

    def _is_goal_achieving_method(self, method_name):
        goal_indicators = ['makeandpourcocktail']
        return any(indicator in method_name.lower() for indicator in goal_indicators)

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

            score += info['shot_count'] * self.shot_penalty
            score += info['shaker_count'] * self.shaker_penalty
            score += info['ingredient_count'] * self.ingredient_penalty
            score += info['clean_count'] * self.clean_penalty
            score += info['holding_count'] * self.holding_penalty
            score += info['pour_count'] * self.pour_penalty
            score += info['shake_count'] * self.shake_penalty
            score += info['ontable_count'] * self.ontable_penalty
            score += info['empty_count'] * self.empty_penalty
            score += info['subtask_count'] * 0.01

            if info['is_goal_achieving']:
                score -= 0.1

            if info['is_base_case']:
                score -= 0.05

            if info['is_achieve_contains_shot_cocktail']:
                score -= 0.2

        if 'makeandpourcocktail' in method_lower:
            score -= 0.15
        elif 'addingredient' in method_lower:
            score += 0.05
        elif 'clean' in method_lower:
            score += 0.03
        elif 'pickup' in method_lower or 'putdown' in method_lower:
            score += 0.02
        elif 'handempty' in method_lower:
            score += 0.01

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        h_value = min(h_value, 1.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic17()"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic17()"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 17 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tShot penalty: {self.shot_penalty}\n"
            f"\tShaker penalty: {self.shaker_penalty}\n"
            f"\tIngredient penalty: {self.ingredient_penalty}\n"
            f"\tClean penalty: {self.clean_penalty}\n"
            f"\tHolding penalty: {self.holding_penalty}\n"
            f"\tPour penalty: {self.pour_penalty}\n"
            f"\tShake penalty: {self.shake_penalty}\n"
            f"\tOnTable penalty: {self.ontable_penalty}\n"
            f"\tEmpty penalty: {self.empty_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
