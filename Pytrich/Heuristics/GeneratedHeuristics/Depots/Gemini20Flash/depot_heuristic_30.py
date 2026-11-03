import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotGemini20FlashHeuristic30(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_30=1, name="DepotGemini20FlashHeuristic30"):
        super().__init__(name=name)
        self.refuel_penalty = 5  # Cost of refuelling
        self.move_penalty = 1    # Cost of moving
        self.lift_penalty = 2    # Cost of lifting/dropping
        self.deliver_penalty = 3 # cost to deliver a crate
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

            refuel_count = 0
            move_count = 0
            lift_count = 0
            deliver_count = 0

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

                    if 'refuel' in subtask_lower:
                        refuel_count += 1
                    if 'move' in subtask_lower:
                        move_count += 1
                    if any(lift_term in subtask_lower for lift_term in ['lift', 'drop']):
                        lift_count += 1
                    if 'deliver' in subtask_lower:
                        deliver_count += 1

            self.method_scores[method_name] = {
                'refuel_count': refuel_count,
                'move_count': move_count,
                'lift_count': lift_count,
                'deliver_count': deliver_count,
                'subtask_count': subtask_count,
                'is_goal_method': self._is_goal_method(method_name),
                'is_task_decomposition': self._is_task_decomposition(method_name),
                'complexity_score': refuel_count + move_count + lift_count + deliver_count,
            }


    def _is_goal_method(self, method_name):
        return 'achieve' in method_name.lower()

    def _is_task_decomposition(self, method_name):
        return 'm' in method_name.lower() and method_name[0].isdigit()

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

            score += info['refuel_count'] * self.refuel_penalty
            score += info['move_count'] * self.move_penalty
            score += info['lift_count'] * self.lift_penalty
            score += info['deliver_count'] * self.deliver_penalty

            if info['is_goal_method']:
                score += self.deliver_penalty

        if 'move' in method_lower:
            score += self.move_penalty

        if 'refuel' in method_lower:
            score += self.refuel_penalty

        if 'lift' in method_lower or 'drop' in method_lower:
            score += self.lift_penalty

        return score

    def __call__(self, parent_node, node):

        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic30(refuel={self.refuel_penalty}, move={self.move_penalty}, lift={self.lift_penalty}, deliver={self.deliver_penalty})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic30(refuel={self.refuel_penalty}, move={self.move_penalty}, lift={self.lift_penalty}, deliver={self.deliver_penalty})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 30 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tRefuel penalty: {self.refuel_penalty}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tLift penalty: {self.lift_penalty}\n"
            f"\tDeliver penalty: {self.deliver_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
