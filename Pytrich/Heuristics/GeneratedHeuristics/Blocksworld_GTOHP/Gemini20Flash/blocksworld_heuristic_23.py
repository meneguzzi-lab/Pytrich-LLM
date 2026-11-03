import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGemini20FlashHeuristic23(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_23 = "blocksworld_gemini20flash_heuristic_23", misplaced_block_penalty=1, action_cost=1):
        super().__init__(name=blocksworld_gemini20flash_heuristic_23)
        self.misplaced_block_penalty = misplaced_block_penalty
        self.action_cost = action_cost
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node in the Blocksworld planning problem.

        The heuristic estimates the cost to achieve the goal state from the current state.
        It considers the number of misplaced blocks (blocks not in their goal position) and
        a fixed cost for each action required to move the blocks.
        """
        state = node.state
        goal = self.model.goals

        misplaced_blocks = self._count_misplaced_blocks(state, goal)
        heuristic_value = (misplaced_blocks * self.misplaced_block_penalty) + (self.action_cost * misplaced_blocks) #each move action has cost
        return heuristic_value

    def _count_misplaced_blocks(self, state, goal):
        """
        Counts the number of blocks that are not in their goal positions.

        This function iterates through the goal state and checks if each 'on' fact
        is present in the current state. If a goal fact is missing, it increments
        the count of misplaced blocks.
        """
        misplaced_blocks = 0
        for goal_fact in self.model.goals:
            if goal_fact not in state:
                misplaced_blocks += 1
        return misplaced_blocks

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic23(misplaced_block_penalty={self.misplaced_block_penalty}, action_cost={self.action_cost})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic23(misplaced_block_penalty={self.misplaced_block_penalty}, action_cost={self.action_cost})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 23 info:\n"
            f"\tName: {self.name}\n"
            f"\tMisplaced block penalty: {self.misplaced_block_penalty}\n"
            f"\tAction cost: {self.action_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
