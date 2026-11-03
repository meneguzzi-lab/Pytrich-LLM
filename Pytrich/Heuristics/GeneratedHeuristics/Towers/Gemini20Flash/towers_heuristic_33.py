import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for towers domain. Aims to estimate the number of moves required
# to achieve the goal state, considering the hierarchical nature of the problem.
class TowersGemini20FlashHeuristic33(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_33=1, name="TowersGemini20FlashHeuristic33"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.tower_count = 3  # Number of towers

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute information.
        """
        start_time = time.time()
        self.model = model

        # Precompute goal facts for heuristic calculations
        self.goal_facts = set()
        for goal_fact in self.model.goals:
            self.goal_facts.add(goal_fact.name)

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node.

        This heuristic estimates the number of moves required to achieve the goal by:
        1. Counting the number of misplaced rings.
        2. Adding a penalty for each ring that needs to be moved to the target tower.

        The heuristic considers the goal predicates 'on' and 'goal_on'.
        """
        state = node.state
        misplaced_rings = 0
        rings_to_move = 0

        # Get state facts as names
        state_facts = self.model.state_explicit_repr(state)

        # Count misplaced rings based on 'on' and 'goal_on' predicates
        for goal_fact in self.goal_facts:
            if 'on' in goal_fact:
                if goal_fact not in state_facts:
                    misplaced_rings += 1

        # Estimate additional moves needed based on smallerThan relationships
        # This part needs more context about the 'smallerThan' predicate to be more effective.
        # For now, just add a small penalty per ring.
        rings_to_move = misplaced_rings  # Each misplaced ring requires at least one move.

        # Combines the misplaced rings and the number of moves required
        heuristic_value = misplaced_rings + rings_to_move

        return heuristic_value

    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for a given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic33()"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic33()"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 33 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
