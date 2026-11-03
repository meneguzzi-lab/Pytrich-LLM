import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic17(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_17 = "towers_gemini20flash_heuristic_17", misplaced_rings_penalty=1, blocking_rings_penalty=2):
        super().__init__(name=towers_gemini20flash_heuristic_17)
        self.misplaced_rings_penalty = misplaced_rings_penalty
        self.blocking_rings_penalty = blocking_rings_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

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
            self.method_scores[method_name] = {}

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node in the Towers of Hanoi problem.

        The heuristic estimates the remaining cost based on the number of misplaced rings
        and the number of rings blocking the placement of other rings on the target tower.

        Args:
            node (HTNNode): The node to evaluate.

        Returns:
            int: The estimated cost to reach the goal state.
        """

        state = node.state
        goal = self.model.goals
        facts = self.model.facts

        misplaced_rings = 0
        blocking_rings = 0

        # Find the target tower.
        target_tower = None
        for fact in goal:
            if 't3' in fact.name and 'on' not in fact.name and 'smallerThan' not in fact.name and 'towerTop' not in fact.name and 'goal_on' not in fact.name:
                target_tower = fact.name.split(' ')[-1] # Extract target tower name
                break

        if not target_tower:
            return 0  # No target tower found.

        # Iterate through the rings.  Find rings in the goal.
        goal_ring_locations = {}
        for fact in facts:
            if 'goal_on' in fact.name:
                parts = fact.name.split()
                ring = parts[1]
                location = parts[2]
                goal_ring_locations[ring] = location

        # Determine misplaced and blocking rings
        current_ring_locations = {}
        for fact in facts:
             if 'on' in fact.name and 'goal_on' not in fact.name:
                parts = fact.name.split()
                ring = parts[1]
                location = parts[2]
                current_ring_locations[ring] = location

        for ring, location in current_ring_locations.items():
            if ring in goal_ring_locations:
                if location != goal_ring_locations[ring]:
                    misplaced_rings += 1
            else:
                misplaced_rings += 1

        return (self.misplaced_rings_penalty * misplaced_rings) + (self.blocking_rings_penalty * blocking_rings)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic17(misplaced={self.misplaced_rings_penalty}, blocking={self.blocking_rings_penalty})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic17(misplaced={self.misplaced_rings_penalty}, blocking={self.blocking_rings_penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 17 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMisplaced rings penalty: {self.misplaced_rings_penalty}\n"
            f"\tBlocking rings penalty: {self.blocking_rings_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
