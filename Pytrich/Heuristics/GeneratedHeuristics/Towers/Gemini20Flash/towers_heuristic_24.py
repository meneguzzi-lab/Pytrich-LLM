import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic24(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_24=1.0, name="TowersGemini20FlashHeuristic24"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.goal_tower = None
        self.initial_tower = None
        self.rings = []
        self.max_rings = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model

        self._analyze_problem()
        self.preprocessing_time = time.time() - start_time
        initial_heuristic_value = self._calculate_heuristic(initial_node)

        return super().initialize(model, initial_heuristic_value)

    def _analyze_problem(self):
        """Analyzes the problem instance to extract relevant information.
        Identifies the goal tower, initial tower, and the rings involved.
        """
        # Find goal tower
        for fact in self.model.facts:
            if fact.name.startswith("goal_on"):
                parts = fact.name.split()
                self.goal_tower = parts[2]
                break
        # Find initial tower
        for fact in self.model.facts:
            if fact.name.startswith("on"):
                parts = fact.name.split()
                self.initial_tower = parts[2]
                break

        # Find all rings
        for fact in self.model.facts:
            if fact.name.startswith("smallerThan"):
               parts = fact.name.split()
               if parts[1] not in self.rings:
                  self.rings.append(parts[1])
        self.max_rings = len(self.rings)


    def _calculate_heuristic(self, node):
        """Calculates the heuristic value for a given node.

        The heuristic estimates the number of moves required to transfer all rings
        from the initial tower to the goal tower. The estimate increases with number of rings.
        """
        if self.goal_tower is None or self.initial_tower is None:
            return 0  # Cannot estimate if towers are not identified

        # If initial and goal are the same, then the problem is easy (return 0) 
        if self.goal_tower == self.initial_tower:
            return 0
        else:
           return self.max_rings


    def __call__(self, parent_node, node):
        """Calculates and returns the heuristic value for the given node.
        """
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic24(goal_tower={self.goal_tower}, initial_tower={self.initial_tower})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic24(goal_tower={self.goal_tower}, initial_tower={self.initial_tower})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 24 info:\n"
            f"\tName: {self.name}\n"
            f"\tGoal Tower: {self.goal_tower}\n"
            f"\tInitial Tower: {self.initial_tower}\n"
            f"\tNumber of Rings: {self.max_rings}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
