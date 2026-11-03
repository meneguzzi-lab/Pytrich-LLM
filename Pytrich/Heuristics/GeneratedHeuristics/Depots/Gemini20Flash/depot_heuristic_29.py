import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for depot domain
class DepotGemini20FlashHeuristic29(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_29=1, name="DepotGemini20FlashHeuristic29"):
        super().__init__(name=name)
        self.name = name
        self.refuel_penalty = 5  # Cost for refuelling actions
        self.move_penalty = 2    # Cost for moving vehicles
        self.load_unload_penalty = 3 # Cost for loading/unloading operations
        self.preprocessing_time = 0
        self.method_scores = {}


    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # Initial heuristic value
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
            # Get methods from abstract tasks
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count operations in subtasks
            refuel_count = 0
            move_count = 0
            load_unload_count = 0
            subtask_count = 0

            #Get subtasks from method
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
                    if any(op in subtask_lower for op in ['load', 'unload']):
                        load_unload_count += 1

            self.method_scores[method_name] = {
                'refuel_count': refuel_count,
                'move_count': move_count,
                'load_unload_count': load_unload_count,
                'subtask_count': subtask_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': refuel_count + move_count + load_unload_count
            }

    def _is_goal_achieving_method(self, method_name):
        # Methods that directly contribute to achieving the goals
        goal_indicators = ['achieve', 'deliver']
        return any(indicator in method_name.lower() for indicator in goal_indicators)

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        # Use precomputed info if available
        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            #Penalties for operations
            score += info['refuel_count'] * self.refuel_penalty
            score += info['move_count'] * self.move_penalty
            score += info['load_unload_count'] * self.load_unload_penalty

            # Bonus for goal achieving methods
            if info['is_goal_achieving']:
                score -= 2  # Small bonus

        #Heuristic based on method name
        if 'refuel' in method_lower:
            score += self.refuel_penalty
        elif 'move' in method_lower:
            score += self.move_penalty
        elif any(op in method_lower for op in ['load', 'unload']):
            score += self.load_unload_penalty

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic29(refuel_penalty={self.refuel_penalty}, move_penalty={self.move_penalty}, load_unload_penalty={self.load_unload_penalty})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic29(refuel_penalty={self.refuel_penalty}, move_penalty={self.move_penalty}, load_unload_penalty={self.load_unload_penalty})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 29 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tRefuel penalty: {self.refuel_penalty}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tLoad/Unload penalty: {self.load_unload_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
