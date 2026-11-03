import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic2(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_2="barman_gemini20flash_heuristic_2"):
        super().__init__(name=barman_gemini20flash_heuristic_2)
        self.preprocessing_time = 0
        self.method_scores = {}

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute method characteristics.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precompute information about methods for faster scoring.
        """
        if not self.model:
            return

        # Try different possible attributes for methods
        methods = []
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'tasks'):
            # Get methods from abstract tasks
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count different types of subtasks
            clean_count = 0
            pour_count = 0
            add_count = 0
            shake_count = 0
            grasp_count = 0
            subtask_count = 0

            # Try different possible attributes for subtasks
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
                    if 'add' in subtask_lower:
                        add_count += 1
                    if 'shake' in subtask_lower:
                        shake_count += 1
                    if 'grasp' in subtask_lower:
                        grasp_count += 1

            self.method_scores[method_name] = {
                'clean_count': clean_count,
                'pour_count': pour_count,
                'add_count': add_count,
                'shake_count': shake_count,
                'grasp_count': grasp_count,
                'subtask_count': subtask_count,
                'is_make_cocktail_method': self._is_make_cocktail_method(method_name),
                'is_add_ingredient_method': self._is_add_ingredient_method(method_name),
                'is_clean_method': self._is_clean_method(method_name),
                'is_pour_method': self._is_pour_method(method_name),
                'is_holding_method': self._is_holding_method(method_name),
                'is_ontable_method': self._is_ontable_method(method_name),
                'complexity_score': clean_count + pour_count + add_count + shake_count + grasp_count
            }

    def _is_make_cocktail_method(self, method_name):
        return 'makecocktail' in method_name.lower()

    def _is_add_ingredient_method(self, method_name):
        return 'addingredient' in method_name.lower()

    def _is_clean_method(self, method_name):
        return 'clean' in method_name.lower()

    def _is_pour_method(self, method_name):
        return 'pour' in method_name.lower()

    def _is_holding_method(self, method_name):
        return 'holding' in method_name.lower()
    
    def _is_ontable_method(self, method_name):
        return 'ontable' in method_name.lower()


    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0  # Start with 0 cost
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        # Use precomputed info if available
        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Penalize complexity (more subtasks)
            score += info['subtask_count'] * 0.01

            # Reward actions that make progress towards goals
            if info['is_make_cocktail_method']:
                score -= 0.15  # Encourage making cocktails
            if info['is_add_ingredient_method']:
                score -= 0.10  # Encourage adding ingredients

            # Discourage actions related to setup that might be redundant
            if info['is_clean_method']:
                score += 0.05
            if info['is_pour_method']:
                score += 0.02
            if info['is_holding_method']:
                score += 0.03
            if info['is_ontable_method']:
                score += 0.01

        else:
            # Default penalty for unknown methods
            score += 0.05

        return max(score, 0)


    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic2(name='{self.name}')"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic2(name='{self.name}')"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 2 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
