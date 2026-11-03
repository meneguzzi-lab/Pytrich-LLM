import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Blocksworld domain.
# This heuristic considers the number of blocks out of place and the number of putdown/pickup operations needed.
class BlocksworldGemini20FlashHeuristic3(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_3: str = "BlocksworldGemini20FlashHeuristic3", misplaced_block_penalty: float = 1.0,  unstack_penalty: float = 1.5, stack_penalty: float = 1.5):
        super().__init__(name=blocksworld_gemini20flash_heuristic_3)
        self.misplaced_block_penalty = misplaced_block_penalty
        self.unstack_penalty = unstack_penalty
        self.stack_penalty = stack_penalty
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.goal_literals = self._extract_goal_literals()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _extract_goal_literals(self):
        """Extracts the goal literals from the model."""
        goal_literals = set()
        for goal_id in self.model.goals:
            goal_fact = self.model.get_component(goal_id)
            goal_literals.add(goal_fact.name)
        return goal_literals

    def _calculate_heuristic(self, node):
        """Calculates the heuristic value for a given node."""
        state = node.state
        state_facts = self.model.state_explicit_repr(state)

        misplaced_blocks = 0
        for goal in self.goal_literals:
            if goal not in state_facts:
                misplaced_blocks += 1

        # Estimate actions needed for misplaced blocks (simplified)
        heuristic_value = (misplaced_blocks * self.misplaced_block_penalty)

        # Penalize methods involving unstacking and stacking
        if hasattr(node, 'method') and node.method:
            method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)
            if 'unstack' in method_name.lower() or 'pick-up' in method_name.lower():
                heuristic_value += self.unstack_penalty
            if 'stack' in method_name.lower() or 'put-down' in method_name.lower():
                heuristic_value += self.stack_penalty

        return heuristic_value

    def __call__(self, parent_node, node):
        """Calculates and returns the heuristic value for a node."""
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic3(misplaced_block_penalty={self.misplaced_block_penalty}, unstack_penalty={self.unstack_penalty}, stack_penalty={self.stack_penalty})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic3(misplaced_block_penalty={self.misplaced_block_penalty}, unstack_penalty={self.unstack_penalty}, stack_penalty={self.stack_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 3 info:\n"
            f"\tName: {self.name}\n"
            f"\tMisplaced Block Penalty: {self.misplaced_block_penalty}\n"
            f"\tUnstack Penalty: {self.unstack_penalty}\n"
            f"\tStack Penalty: {self.stack_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tGoal Literals: {len(self.goal_literals)}\n"
        )
