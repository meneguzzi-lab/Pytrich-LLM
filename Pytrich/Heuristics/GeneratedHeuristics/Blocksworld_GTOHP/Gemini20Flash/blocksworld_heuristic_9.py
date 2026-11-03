import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGemini20FlashHeuristic9(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_9: str = "blocksworld_gemini20flash_heuristic_9"):
        super().__init__(name=blocksworld_gemini20flash_heuristic_9)
        self.preprocessing_time = 0
        self.goal_literals = set()
        self.block_objects = set()

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_goal_and_block_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_goal_and_block_info(self):
        """Extract goal literals and block objects for heuristic computation."""
        for goal_literal in self.model.goals:
            self.goal_literals.add(goal_literal)

        for fact in self.model.facts:
            if fact.name != 'handempty' and 'block' in fact.name:
                self.block_objects.add(fact)

    def _calculate_node_heuristic(self, node):
        """Calculate the heuristic value for a given node."""
        state_bitwise = node.state
        state_literals = set()

        for i in range(state_bitwise.bit_length()):
            if (state_bitwise >> i) & 1:
                for fact in self.model.facts:
                    if fact.global_id == i:
                        state_literals.add(fact)

        misplaced_blocks = 0
        for goal_literal in self.goal_literals:
            if goal_literal not in state_literals:
                misplaced_blocks += 1

        # Task network heuristic.  If there are tasks to be done still, penalize
        task_penalty = 0
        if len(node.tn) > 0:
            task_penalty = 1

        return misplaced_blocks + task_penalty

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic()"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic()"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 9 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tGoal literals: {len(self.goal_literals)}\n"
            f"\tBlock objects: {len(self.block_objects)}\n"
        )
