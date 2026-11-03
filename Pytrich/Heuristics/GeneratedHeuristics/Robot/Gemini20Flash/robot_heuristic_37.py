import time
from typing import List, Union
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Operator, AbstractTask, Decomposition


class RobotGemini20FlashHeuristic37(Heuristic):
    def __init__(self, robot_gemini20flash_heuristic_37: int = 1, name: str = "robot_gemini20flash_heuristic_37") -> None:
        super().__init__(name=name)
        self.pickup_penalty = robot_gemini20flash_heuristic_37  # Penalty for each pickup operation
        self.move_penalty = robot_gemini20flash_heuristic_37  # Penalty for each move operation
        self.putdown_penalty = robot_gemini20flash_heuristic_37 # Penalty for each putdown operation
        self.open_penalty = robot_gemini20flash_heuristic_37  # Penalty for each open operation

        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode) -> float:
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self) -> None:
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
            
            pickup_count = 0
            move_count = 0
            putdown_count = 0
            open_count = 0
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

                    if 'pickup' in subtask_lower:
                        pickup_count += 1
                    elif 'move' in subtask_lower:
                        move_count += 1
                    elif 'putdown' in subtask_lower:
                        putdown_count += 1
                    elif 'open' in subtask_lower:
                        open_count += 1

            self.method_scores[method_name] = {
                'pickup_count': pickup_count,
                'move_count': move_count,
                'putdown_count': putdown_count,
                'open_count': open_count,
                'subtask_count': subtask_count
            }

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """Calculates the heuristic value for a given node."""
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method: Union[Decomposition, Operator], node: HTNNode) -> float:
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['pickup_count'] * self.pickup_penalty
            score += info['move_count'] * self.move_penalty
            score += info['putdown_count'] * self.putdown_penalty
            score += info['open_count'] * self.open_penalty

        return score

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini20FlashHeuristic(pickup={self.pickup_penalty}, move={self.move_penalty}, putdown={self.putdown_penalty}, open={self.open_penalty})"

    def __str__(self):
        return f"RobotGemini20FlashHeuristic(pickup={self.pickup_penalty}, move={self.move_penalty}, putdown={self.putdown_penalty}, open={self.open_penalty})"

    def __output__(self):
        return (
            f"Robot Gemini 2.0 Flash Heuristic 37 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPickup penalty: {self.pickup_penalty}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tPutdown penalty: {self.putdown_penalty}\n"
            f"\tOpen penalty: {self.open_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
