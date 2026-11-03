import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini20FlashHeuristic13(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_13=1, name="TowersGemini20FlashHeuristic13"):
        super().__init__(name=name)
        self.towers_gemini20flash_heuristic_13 = towers_gemini20flash_heuristic_13
        self.preprocessing_time = 0
        self.goal_rings = set()
        self.initial_tower = None
        self.target_tower = None

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_goal_rings()
        self._determine_initial_and_target_towers(model)
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_goal_rings(self):
        """Identifies rings that need to be moved to achieve the goal."""
        for fact in self.model.goals:
            if 'on' in fact.name and 'goal_on' in fact.name:
                parts = fact.name.split()
                ring = parts[1]
                self.goal_rings.add(ring)

    def _determine_initial_and_target_towers(self, model):
        """Determines the initial and target towers based on the initial state and goal."""
        initial_rings = []
        target_rings = []
        for fact in model.initial_state:
            if 'on' in fact.name and 'towerTop' in fact.name:
                parts = fact.name.split()
                initial_rings.append(parts[1])

        for fact in model.goals:
            if 'on' in fact.name and 'goal_on' in fact.name:
                parts = fact.name.split()
                target_rings.append(parts[3])

        if initial_rings and target_rings:
            self.initial_tower = initial_rings[0]
            self.target_tower = target_rings[0]


    def _calculate_heuristic(self, node):
        """Calculates the heuristic value for a given node.

        This heuristic estimates the cost of moving the rings from the initial tower
        to the target tower. It counts the number of 'move' actions and adds a penalty
        for each additional ring that needs to be moved.
        """
        rings_to_move = len(self.goal_rings)
        heuristic_value = rings_to_move # Minimum moves needed.

        if hasattr(node, 'method') and node.method:
            method_name = node.method.name.lower()
            if 'move' in method_name:
                heuristic_value += 0.5 # Add to ensure search explores all nodes
            elif 'exchange' in method_name:
                 heuristic_value += 0.25 # encourage exchange operations

        return heuristic_value


    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic13(towers_gemini20flash_heuristic_13={self.towers_gemini20flash_heuristic_13})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic13(towers_gemini20flash_heuristic_13={self.towers_gemini20flash_heuristic_13})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 13 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tInitial Tower: {self.initial_tower}\n"
            f"\tTarget Tower: {self.target_tower}\n"
            f"\tRings to move count: {len(self.goal_rings)}\n"
        )
