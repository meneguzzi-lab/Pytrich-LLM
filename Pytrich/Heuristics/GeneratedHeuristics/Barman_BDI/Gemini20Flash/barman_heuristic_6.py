import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini20FlashHeuristic6(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_6: str = "BarmanGemini20FlashHeuristic6"):
        super().__init__(name=barman_gemini20flash_heuristic_6)
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
            add_ingredient_count = 0
            pour_count = 0
            shake_count = 0
            hand_op_count = 0 # Hand-related operations like pickup, drop, empty
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
                    if 'addingredient' in subtask_lower or 'fill-shot' in subtask_lower or 'pour-shot-to' in subtask_lower:
                        add_ingredient_count += 1
                    if 'pour-shaker-to-shot' in subtask_lower:
                        pour_count += 1
                    if 'shake' in subtask_lower:
                        shake_count += 1
                    if 'pickup' in subtask_lower or 'drop' in subtask_lower or 'emptyhand' in subtask_lower or 'grasp' in subtask_lower:
                        hand_op_count += 1

            self.method_scores[method_name] = {
                'clean_count': clean_count,
                'add_ingredient_count': add_ingredient_count,
                'pour_count': pour_count,
                'shake_count': shake_count,
                'hand_op_count': hand_op_count,
                'subtask_count': subtask_count,
                'is_achieve_cocktail': 'AchieveContainsShotCocktail'.lower() in method_name.lower(),
                'is_make_cocktail': 'MakeCocktail'.lower() in method_name.lower(),
                'is_add_ingredient': 'AddIngredient'.lower() in method_name.lower(),
                'is_clean_method': 'Clean'.lower() in method_name.lower(),
                'is_hand_operation': 'Holding'.lower() in method_name.lower() or 'OnTable'.lower() in method_name.lower() or 'HandEmpty'.lower() in method_name.lower(),
                'complexity_score': clean_count + add_ingredient_count + pour_count + shake_count + hand_op_count
            }

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0  # Start with 0 cost, penalize from there

        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            score += info['clean_count'] * 0.2  # Cleaning is generally good but has a small cost
            score += info['add_ingredient_count'] * 0.3 # Adding ingredients requires actions
            score += info['pour_count'] * 0.25  # Pouring requires setup and execution
            score += info['shake_count'] * 0.4  # Shaking is a key step
            score += info['hand_op_count'] * 0.1 # Hand operations have setup costs.

            # Bonuses (negative cost) for key achievement methods
            if info['is_achieve_cocktail']:
                score -= 0.5 # Big bonus since its the main achieve goal
            if info['is_make_cocktail']:
                score -= 0.3  # Making cocktail is a high level goal

            # Minor bonus for ingredient adding/cleaning etc.
            if info['is_add_ingredient']:
                score -= 0.1
            if info['is_clean_method']:
                score -= 0.15
            if info['is_hand_operation']:
                score -= 0.05
        else:
            score += 1 # Big penalty if the method is not recognized

        return max(score, 0) # Ensure no negative heuristics

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic6()"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic6()"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 6 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
