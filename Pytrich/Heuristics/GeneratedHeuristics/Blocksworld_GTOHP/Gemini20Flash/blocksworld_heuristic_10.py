import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Blocksworld domain.
# This heuristic estimates the number of 'on' goals that are not yet satisfied.
# It also considers the cost of moving blocks to achieve these goals.
class BlocksworldGemini20FlashHeuristic10(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_10=1, name="BlocksworldGemini20FlashHeuristic10"):  # Corrected parameter name
        super().__init__(name=name)
        self.misplaced_block_penalty = 1.0
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node.
        The heuristic estimates the number of unsatisfied 'on' goals and adds a penalty
        for each misplaced block that needs to be moved to satisfy the goals.
        """
        state_bitwise = node.state
        goal_count = 0
        for goal_fact in self.model.goals:
            # Check if the goal is an 'on' predicate
            if 'on' in goal_fact.name.lower():
                # Check if the goal is satisfied in the current state
                if not (state_bitwise & (1 << goal_fact.global_id)):
                    goal_count += 1
        return goal_count * self.misplaced_block_penalty

    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic(misplaced_block_penalty={self.misplaced_block_penalty})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic(misplaced_block_penalty={self.misplaced_block_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 10 info:\n"
            f"\tName: {self.name}\n"
            f"\tMisplaced block penalty: {self.misplaced_block_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
