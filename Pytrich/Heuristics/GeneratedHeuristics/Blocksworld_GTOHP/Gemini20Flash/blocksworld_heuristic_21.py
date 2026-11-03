import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from typing import List, Union


class BlocksworldGemini20FlashHeuristic21(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_21=1, name="BlocksworldGemini20FlashHeuristic21"):
        super().__init__(name=name)
        self.misplaced_block_penalty = 1  # Cost for each misplaced block
        self.task_penalty = 0.5  # Cost for each remaining task
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic with the model and precomputes any necessary information.
        """
        start_time = time.time()
        self.model = model
        self.goal_literals = self._extract_goal_literals()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _extract_goal_literals(self):
        """
        Extracts the goal literals from the model.
        Assumes goals are simple 'on' predicates for blocks.
        """
        goal_literals = set()
        for goal in self.model.goals:
            # Assuming goal is a fact representing (on block1 block2)
            goal_literals.add(self.model.facts[goal].name)
        return goal_literals

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node.
        Considers misplaced blocks and remaining tasks.
        """
        state = node.state
        state_literals = [self.model.facts[bit_pos].name for bit_pos in range(state.bit_length()) if state & 1 << bit_pos]
        misplaced_blocks = self._count_misplaced_blocks(state_literals)
        remaining_tasks = len(node.open_tasks) # Open tasks are remaining tasks

        heuristic_value = (misplaced_blocks * self.misplaced_block_penalty) + \
                          (remaining_tasks * self.task_penalty)

        return heuristic_value

    def _count_misplaced_blocks(self, state_literals):
        """
        Counts the number of blocks that are not in their goal positions.
        """
        misplaced_count = 0
        for goal_literal in self.goal_literals:
            if goal_literal not in state_literals:
                misplaced_count += 1
        return misplaced_count

    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic21(misplaced_penalty={self.misplaced_block_penalty}, task_penalty={self.task_penalty})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic21(misplaced_penalty={self.misplaced_block_penalty}, task_penalty={self.task_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 21 info:\n"
            f"\tName: {self.name}\n"
            f"\tMisplaced block penalty: {self.misplaced_block_penalty}\n"
            f"\tTask penalty: {self.task_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
