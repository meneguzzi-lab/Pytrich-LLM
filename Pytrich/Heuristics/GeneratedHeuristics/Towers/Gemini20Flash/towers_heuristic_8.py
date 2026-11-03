import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic8(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_8=1, name="TowersGemini20FlashHeuristic8"):
        super().__init__(name=name)
        self.move_penalty = 1  # Cost for each move action
        self.method_penalty = 0.1  # Cost for each method application
        self.unplaced_ring_penalty = 5  # High cost for each ring not in its goal position
        self.preprocessing_time = 0
        self.method_scores = {}

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_heuristic(initial_node)
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
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            move_count = 0
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

                    if 'move' in subtask_lower:
                        move_count += 1

            self.method_scores[method_name] = {
                'move_count': move_count,
                'subtask_count': subtask_count
            }

    def _calculate_heuristic(self, node):
        """Calculate heuristic value for a node."""
        state = node.state
        h_value = 0

        # Count rings not at their goal locations (high penalty)
        h_value += self._calculate_unplaced_rings_penalty(state)

        # Add a small penalty for each method application
        if hasattr(node, 'method') and node.method:
            h_value += self._score_method(node.method)

        return h_value

    def _calculate_unplaced_rings_penalty(self, state):
        """Counts rings not at their goal locations and applies a high penalty."""
        unplaced_rings = 0
        if self.model:
            for fact in self.model.facts:
                if 'on' in fact.name and 'goal' not in fact.name:
                    ring, obj = self._extract_ring_and_obj(fact.name)
                    if ring and obj:
                        goal_fact_name = f'(goal_on {ring} {obj})'
                        goal_fact = next((f for f in self.model.facts if f.name == goal_fact_name), None)
                        if goal_fact is not None and not (state & (1 << goal_fact.global_id)):
                            unplaced_rings += 1
        return unplaced_rings * self.unplaced_ring_penalty

    def _score_method(self, method):
        """Score a method based on its name and precomputed information."""
        score = self.method_penalty  # Base penalty for applying a method
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            score += self.method_scores[method_name]['move_count'] * self.move_penalty
            # Consider subtask count for refinement complexity
            score += self.method_scores[method_name]['subtask_count'] * 0.05

        if 'move' in method_name.lower():
            score += self.move_penalty

        return score

    def _extract_ring_and_obj(self, fact_string):
        """Extract ring and object names from a fact string."""
        parts = fact_string.split()
        if len(parts) == 3:
            ring = parts[1] if 'r' in parts[1] else None
            obj = parts[2] if 't' in parts[2] or 'r' in parts[2] else None
            return ring, obj
        return None, None

    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic8(move_penalty={self.move_penalty}, method_penalty={self.method_penalty}, unplaced_ring_penalty={self.unplaced_ring_penalty})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic8(move_penalty={self.move_penalty}, method_penalty={self.method_penalty}, unplaced_ring_penalty={self.unplaced_ring_penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 8 info:\n"
            f"\tName: {self.name}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tMethod penalty: {self.method_penalty}\n"
            f"\tUnplaced ring penalty: {self.unplaced_ring_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
