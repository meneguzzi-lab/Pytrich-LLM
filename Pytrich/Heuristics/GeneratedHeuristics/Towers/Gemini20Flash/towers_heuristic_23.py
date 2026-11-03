import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic23(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_23=1, name="TowersGemini20FlashHeuristic23"):
        super().__init__(name=name)
        self.move_penalty = 1  # Penalty for each move action
        self.task_penalty = 0.5  # Penalty for each task decomposition
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.goal_state = model.goals
        self.facts = model.facts
        self._precompute_ring_positions(model)
        self.preprocessing_time = time.time() - start_time
        initial_h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, initial_h_value)

    def _precompute_ring_positions(self, model):
        """Precomputes initial and goal positions of rings for heuristic calculation."""
        self.initial_ring_positions = {}
        self.goal_ring_positions = {}

        for fact in model.facts:
            if fact.name.startswith('on'):
                parts = fact.name.split()
                ring = parts[1]
                position = parts[2]
                self.initial_ring_positions[ring] = position
            elif fact.name.startswith('goal_on'):
                parts = fact.name.split()
                ring = parts[1]
                position = parts[2]
                self.goal_ring_positions[ring] = position

    def _calculate_heuristic(self, node):
        """Calculates the heuristic value for a given node.
        This heuristic estimates the remaining cost based on the number of misplaced rings
        and the number of moves required to bring them to their goal positions.
        """
        misplaced_rings = 0
        total_moves = 0

        # Iterate through each ring and check if it's in the correct position
        for ring, initial_pos in self.initial_ring_positions.items():
            goal_pos = self.goal_ring_positions.get(ring)

            if goal_pos is None:
                continue #Ring not in goal, skip

            if initial_pos != goal_pos:
                misplaced_rings += 1
                # Estimate the number of moves to correct the ring's position (simplified).
                total_moves += 1  # Each misplaced ring requires at least one move

        # Add a penalty for each task decomposition in the current task network (if available).
        task_network_penalty = 0
        if hasattr(node, 'task_network') and node.task_network:
            task_network_penalty = len(node.task_network) * self.task_penalty

        # Combine the misplaced rings, moves, and task network penalties to form the heuristic.
        heuristic_value = misplaced_rings * self.move_penalty + total_moves * self.move_penalty + task_network_penalty

        return heuristic_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic23(move_penalty={self.move_penalty}, task_penalty={self.task_penalty})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic23(move_penalty={self.move_penalty}, task_penalty={self.task_penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 23 info:\n"
            f"\tName: {self.name}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tTask penalty: {self.task_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
