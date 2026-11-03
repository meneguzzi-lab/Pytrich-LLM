import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic20(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_20=1.0, name="TowersGemini20FlashHeuristic20"):
        super().__init__(name=name)
        self.towers_gemini20flash_heuristic_20 = towers_gemini20flash_heuristic_20
        self.preprocessing_time = 0
        self.goal_rings = set()

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_goal_rings()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_goal_rings(self):
        """Identifies rings that need to be on the goal tower."""
        for fact in self.model.facts:
            if 'goal_on' in fact.name:
                parts = fact.name.split()
                ring = parts[1]
                self.goal_rings.add(ring)

    def _calculate_heuristic(self, node):
        """Calculates the heuristic value for a given node.

        The heuristic estimates the number of rings that are not on the goal tower
        and adds a penalty if the 'shiftTower' task is not yet achieved.
        """
        state = node.state
        goal_tower = 't3'
        misplaced_rings = 0

        for fact in self.model.facts:
            if 'on' in fact.name:
                parts = fact.name.split()
                ring = parts[1]
                tower = parts[2]

                if ring in self.goal_rings and tower != goal_tower:
                    misplaced_rings += 1

        # Add a penalty if the main task 'shiftTower' is not yet achieved.  This
        # encourages the planner to focus on completing the main task first.
        shift_tower_achieved = True  # Assume achieved unless proven otherwise
        for task in node.task_network:
           if hasattr(task, 'name'):
             if 'shiftTower' in task.name:
                shift_tower_achieved = False
                break
           elif hasattr(task, 'task'):
               if hasattr(task.task, 'name'):
                   if 'shiftTower' in task.task.name:
                       shift_tower_achieved = False
                       break

        penalty = 0 if shift_tower_achieved else 1  # Example penalty value

        heuristic_value = misplaced_rings + penalty * self.towers_gemini20flash_heuristic_20

        return heuristic_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic20(towers_gemini20flash_heuristic_20={self.towers_gemini20flash_heuristic_20})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic20(towers_gemini20flash_heuristic_20={self.towers_gemini20flash_heuristic_20})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 20 info:\n"
            f"\tName: {self.name}\n"
            f"\tGoal Rings: {len(self.goal_rings)}\n"
            f"\tPenalty: {self.towers_gemini20flash_heuristic_20}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
