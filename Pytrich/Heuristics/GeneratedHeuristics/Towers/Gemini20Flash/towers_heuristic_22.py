import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic22(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_22=1.0, name="TowersGemini20FlashHeuristic22"):  # Corrected init parameter name
        super().__init__(name=name)
        self.m_distance_penalty = towers_gemini20flash_heuristic_22 # how much to value getting rings to the correct tower
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()  # Precompute method scores
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """Precompute information about methods for faster scoring."""
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

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            
            move_count = 0
            exchange_count = 0
            rotate_count = 0

            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition


            if subtasks:
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)
                    
                    if 'move_abstract' in subtask_name.lower() or 'move' in subtask_name.lower():
                        move_count += 1
                    if 'exchange' in subtask_name.lower():
                        exchange_count += 1
                    if 'rotate' in subtask_name.lower():
                        rotate_count += 1

            self.method_scores[method_name] = {
                'move_count': move_count,
                'exchange_count': exchange_count,
                'rotate_count': rotate_count,
                'is_base_case': 'exchangeClear' in method_name,
                'complexity_score': move_count + exchange_count + rotate_count
            }

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied yet

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['move_count'] * 0.1
            score += info['exchange_count'] * 0.2
            score += info['rotate_count'] * 0.15

            if info['is_base_case']:
                score -= 0.3  # Encourage base cases

        # Distance to Goal heuristic based on rings not at the correct tower
        rings_misplaced = self._calculate_misplaced_rings(node)
        score += rings_misplaced * self.m_distance_penalty # Factor in the misplaced rings distance

        return max(score, 0)

    def _calculate_misplaced_rings(self, node):
        """Calculates a heuristic based on the number of rings not on the goal tower."""
        state = node.state
        goal_tower = None
        misplaced_rings = 0
        if self.model is None or self.model.facts is None:
            return 0

        # Find which tower the goal is located at, if not defined then return 0
        for fact in self.model.facts:
            if "goal_on" in fact.name:
                goal_tower = fact.name.split()[-1]
                break
        if goal_tower is None:
            return 0
        
        # Count the number of misplaced rings, this becomes are heuristic
        for fact in self.model.facts:
            if "on" in fact.name and fact.name.split()[-1] != goal_tower:  # Count rings on the wrong tower
                misplaced_rings += 1

        return misplaced_rings

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic22(distance_penalty={self.m_distance_penalty})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic22(distance_penalty={self.m_distance_penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 22 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tDistance penalty: {self.m_distance_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
