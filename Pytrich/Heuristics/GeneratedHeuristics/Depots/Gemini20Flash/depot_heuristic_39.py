import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotGemini20FlashHeuristic39(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_39=1.0, name="depot_gemini20flash_heuristic_39"):
        super().__init__(name=name)
        self.load_penalty = depot_gemini20flash_heuristic_39
        self.move_penalty = depot_gemini20flash_heuristic_39
        self.lift_penalty = depot_gemini20flash_heuristic_39
        self.drop_penalty = depot_gemini20flash_heuristic_39
        self.hoist_penalty = depot_gemini20flash_heuristic_39
        self.unhoist_penalty = depot_gemini20flash_heuristic_39
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
        elif hasattr(self.model, 'abstract_tasks'):
            for abstract_task in self.model.abstract_tasks:
                if hasattr(abstract_task, 'decompositions'):
                    methods.extend(abstract_task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            load_count = 0
            move_count = 0
            lift_count = 0
            drop_count = 0
            hoist_count = 0
            unhoist_count = 0
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

                    if 'load' in subtask_lower:
                        load_count += 1
                    if 'move' in subtask_lower:
                        move_count += 1
                    if 'lift' in subtask_lower:
                        lift_count += 1
                    if 'drop' in subtask_lower:
                        drop_count += 1
                    if 'hoist' in subtask_lower:
                        hoist_count += 1
                    if 'unhoist' in subtask_lower:
                        unhoist_count += 1

            self.method_scores[method_name] = {
                'load_count': load_count,
                'move_count': move_count,
                'lift_count': lift_count,
                'drop_count': drop_count,
                'hoist_count': hoist_count,
                'unhoist_count': unhoist_count,
                'subtask_count': subtask_count,
                'is_move_method': self._is_move_method(method_name),
                'is_load_method': self._is_load_method(method_name),
                'is_goal_method': self._is_goal_method(method_name)
            }

    def _is_move_method(self, method_name):
        return 'move' in method_name.lower()

    def _is_load_method(self, method_name):
        return 'load' in method_name.lower() or 'unload' in method_name.lower()

    def _is_goal_method(self, method_name):
        return 'achieve' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0

        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['load_count'] * self.load_penalty
            score += info['move_count'] * self.move_penalty
            score += info['lift_count'] * self.lift_penalty
            score += info['drop_count'] * self.drop_penalty
            score += info['hoist_count'] * self.hoist_penalty
            score += info['unhoist_count'] * self.unhoist_penalty

            if info['is_move_method']:
                score += self.move_penalty / 2
            if info['is_load_method']:
                score += self.load_penalty / 2
            if info['is_goal_method']:
                score -= 0.01 #Small bonus for goal achieving methods

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic39(load={self.load_penalty}, move={self.move_penalty}, lift={self.lift_penalty}, drop={self.drop_penalty}, hoist={self.hoist_penalty}, unhoist={self.unhoist_penalty})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic39(load={self.load_penalty}, move={self.move_penalty}, lift={self.lift_penalty}, drop={self.drop_penalty}, hoist={self.hoist_penalty}, unhoist={self.unhoist_penalty})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 39 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tLoad penalty: {self.load_penalty}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tLift penalty: {self.lift_penalty}\n"
            f"\tDrop penalty: {self.drop_penalty}\n"
            f"\tHoist penalty: {self.hoist_penalty}\n"
            f"\tUnhoist penalty: {self.unhoist_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
