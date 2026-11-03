import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic32(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_32: str = "TowersGemini20FlashHeuristic"):  # Corrected parameter name
        super().__init__(name=towers_gemini20flash_heuristic_32)
        self.preprocessing_time = 0
        self.goal_rings = set()
        self.start_tower = None
        self.goal_tower = None
        self.num_rings = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_goal_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_goal_info(self):
        """Precomputes information about the goal state for heuristic calculations."""
        self.goal_rings = set()
        self.start_tower = None
        self.goal_tower = None

        # Identify goal rings and target tower
        for fact in self.model.facts:
            if 'goal_on' in fact.name:
                ring = fact.name.split()[1]
                tower = fact.name.split()[2]
                self.goal_rings.add(ring)
                self.goal_tower = tower  # Assuming all rings are to be moved to the same tower

        # Determine initial tower (assuming all rings start on the same tower)
        for fact in self.model.facts:
            if 'on' in fact.name and fact.name.split()[1] in self.goal_rings:
                self.start_tower = fact.name.split()[2]
                break

        self.num_rings = len(self.goal_rings)

    def _calculate_heuristic(self, node):
        """Calculates the heuristic value for a given node.

        This heuristic estimates the number of moves required to transfer all rings
        from the starting tower to the goal tower.
        """
        if self.start_tower is None or self.goal_tower is None:
            return 0  # Cannot determine heuristic if start or goal tower is unknown

        # Base case: No rings to move
        if self.num_rings == 0:
            return 0

        # Simplified heuristic: Number of rings * constant cost per ring.
        # This provides a basic estimate of the problem complexity.
        heuristic_value = self.num_rings * 2  # Each ring needs at least 2 moves (to/from temp tower)

        # Check if the node's method moves rings to the goal.
        if hasattr(node, 'method') and node.method:
            method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)
            if 'move' in method_name.lower():
                heuristic_value -= 1 #Give a small boost if it includes a move action

        return heuristic_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic32(start_tower={self.start_tower}, goal_tower={self.goal_tower}, num_rings={self.num_rings})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic32(start_tower={self.start_tower}, goal_tower={self.goal_tower}, num_rings={self.num_rings})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 32 info:\n"
            f"\tName: {self.name}\n"
            f"\tStart tower: {self.start_tower}\n"
            f"\tGoal tower: {self.goal_tower}\n"
            f"\tNumber of rings: {self.num_rings}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
