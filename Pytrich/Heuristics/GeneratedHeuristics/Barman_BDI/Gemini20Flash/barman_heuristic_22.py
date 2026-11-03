import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic22(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_22=1.0, name="BarmanGemini20FlashHeuristic22"):
        super().__init__(name=name)
        self.pour_penalty = 0.5
        self.clean_penalty = 0.3
        self.grasp_penalty = 0.2
        self.fill_penalty = 0.4
        self.shake_penalty = 0.6  # Increased shake penalty
        self.ontable_bonus = -0.1  # Added bonus for being on the table
        self.handempty_bonus = -0.1 # Added bonus for having a hand empty
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
        elif hasattr(self.model, 'abstract_tasks'):
            for task in self.model.abstract_tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            pour_count = 0
            clean_count = 0
            grasp_count = 0
            fill_count = 0
            shake_count = 0
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

                    if 'pour' in subtask_lower:
                        pour_count += 1
                    elif 'clean' in subtask_lower:
                        clean_count += 1
                    elif 'grasp' in subtask_lower:
                        grasp_count += 1
                    elif 'fill' in subtask_lower:
                        fill_count += 1
                    elif 'shake' in subtask_lower:
                        shake_count += 1

            self.method_scores[method_name] = {
                'pour_count': pour_count,
                'clean_count': clean_count,
                'grasp_count': grasp_count,
                'fill_count': fill_count,
                'shake_count': shake_count,
                'subtask_count': subtask_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': pour_count + clean_count + grasp_count + fill_count + shake_count
            }

    def _is_goal_achieving_method(self, method_name):
        return 'achievecontainsshotcocktail' in method_name.lower()

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

            score += info['pour_count'] * self.pour_penalty
            score += info['clean_count'] * self.clean_penalty
            score += info['grasp_count'] * self.grasp_penalty
            score += info['fill_count'] * self.fill_penalty
            score += info['shake_count'] * self.shake_penalty  # Applying the higher shake penalty

            if info['is_goal_achieving']:
                score -= 0.2 # small bonus for achieving the goal

        # Bonuses for ontable and handempty
        if hasattr(node, 'state'):
            state = node.state
            ontable_count = sum(1 for fact in self.model.facts if 'ontable' in fact.name.lower() and fact.name in self.model.state_explicit_repr(state))
            handempty_count = sum(1 for fact in self.model.facts if 'handempty' in fact.name.lower() and fact.name in self.model.state_explicit_repr(state))

            score += ontable_count * self.ontable_bonus
            score += handempty_count * self.handempty_bonus

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic22(pour={self.pour_penalty}, clean={self.clean_penalty}, grasp={self.grasp_penalty}, fill={self.fill_penalty}, shake={self.shake_penalty})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic22(pour={self.pour_penalty}, clean={self.clean_penalty}, grasp={self.grasp_penalty}, fill={self.fill_penalty}, shake={self.shake_penalty})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 22 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPour penalty: {self.pour_penalty}\n"
            f"\tClean penalty: {self.clean_penalty}\n"
            f"\tGrasp penalty: {self.grasp_penalty}\n"
            f"\tFill penalty: {self.fill_penalty}\n"
            f"\tShake penalty: {self.shake_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
