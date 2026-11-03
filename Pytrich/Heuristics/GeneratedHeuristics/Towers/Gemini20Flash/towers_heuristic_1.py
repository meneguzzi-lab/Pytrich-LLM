import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic1(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_1: str = "TowersGemini20FlashHeuristic1", misplaced_penalty: float = 1.0, move_penalty: float = 0.1):
        super().__init__(name=towers_gemini20flash_heuristic_1)
        self.misplaced_penalty = misplaced_penalty
        self.move_penalty = move_penalty
        self.model = None
        self.goal_on_predicates = set()
        self.preprocessing_time = 0

    def initialize(self, model, initial_h_value):
        start_time = time.time()
        self.model = model
        self._extract_goal_on_predicates()
        self.preprocessing_time = time.time() - start_time
        return super().initialize(model, initial_h_value)

    def _extract_goal_on_predicates(self):
        """Extracts the goal_on predicates from the goal state."""
        if self.model and self.model.goals:
            for goal_fact_id in self.model.goals:
                goal_fact = self.model.get_component(goal_fact_id)
                if 'goal_on' in goal_fact.name:
                    self.goal_on_predicates.add(goal_fact.name)

    def _calculate_misplaced_rings(self, state_bitwise):
        """Calculates the number of misplaced rings based on the goal state."""
        misplaced_rings = 0
        if not self.model:
            return 0

        state_facts = set()
        for bit_pos in range(state_bitwise.bit_length()):
            if state_bitwise & 1 << bit_pos:
                fact = self.model.get_component(bit_pos)
                state_facts.add(fact.name)


        for goal_on in self.goal_on_predicates:
            if goal_on not in state_facts:
                misplaced_rings += 1
        return misplaced_rings

    def _calculate_move_cost(self, method):
        """Estimates the move cost based on the method being applied."""
        if not method:
            return 0.0
        method_name = method.name if hasattr(method, 'name') else str(method)
        if 'move' in method_name.lower():
            return self.move_penalty
        return 0.0

    def __call__(self, parent_node, node):
        """Calculates the heuristic value for a given node."""
        state_bitwise = node.state
        misplaced_rings = self._calculate_misplaced_rings(state_bitwise)
        misplaced_cost = misplaced_rings * self.misplaced_penalty
        move_cost = self._calculate_move_cost(node.method)
        h_value = misplaced_cost + move_cost
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic1(misplaced_penalty={self.misplaced_penalty}, move_penalty={self.move_penalty})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic1(misplaced_penalty={self.misplaced_penalty}, move_penalty={self.move_penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 1 info:\n"
            f"\tName: {self.name}\n"
            f"\tMisplaced ring penalty: {self.misplaced_penalty}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tGoal ON Predicates: {len(self.goal_on_predicates)}\n"
        )
