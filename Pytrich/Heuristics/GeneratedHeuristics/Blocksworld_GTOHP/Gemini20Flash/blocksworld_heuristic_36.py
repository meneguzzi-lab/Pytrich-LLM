import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for blocksworld domain
class BlocksworldGemini20FlashHeuristic36(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_36=1, name="blocksworld_gemini20flash_heuristic_36"):
        super().__init__(name=name)
        self.misplaced_blocks_penalty = 2
        self.unblocked_penalty = 1
        self.goal_blocks = set()
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute goal state.
        """
        start_time = time.time()
        self.model = model
        self._precompute_goal_blocks()
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_goal_blocks(self):
        """
        Precompute blocks that are part of the goal configuration.
        """
        for goal in self.model.goals:
            if goal.startswith('(on '):
                parts = goal.split()
                block1 = parts[1]
                self.goal_blocks.add(block1)

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node. This heuristic focuses on the number of misplaced
        blocks with regards to the final goal arrangement and number of clear blocks
        """
        state = node.state
        misplaced_blocks = 0
        unblocked_blocks = 0

        # Extract block positions from the state
        block_positions = {}
        for fact_id in range(state.bit_length()):
            if (state & (1 << fact_id)):
                fact_name = self.model.facts[fact_id].name
                if fact_name.startswith('(on '):
                    parts = fact_name.split()
                    block1 = parts[1]
                    block2 = parts[2].replace(')', '')
                    block_positions[block1] = block2
                elif fact_name.startswith('(ontable '):
                    parts = fact_name.split()
                    block = parts[1].replace(')', '')
                    block_positions[block] = 'ontable'
                elif fact_name.startswith('(clear '):
                    parts = fact_name.split()
                    block = parts[1].replace(')', '')
                    unblocked_blocks +=1
        # Evaluate misplaced blocks based on goal state
        for goal in self.model.goals:
            if goal.startswith('(on '):
                parts = goal.split()
                block1 = parts[1]
                block2 = parts[2].replace(')', '')
                if block1 in block_positions:
                    if block_positions[block1] != block2:
                        misplaced_blocks += 1
                else:
                    misplaced_blocks += 1 # Block not found at all



        # Penalize misplaced blocks and unblocked blocks
        heuristic_value = (misplaced_blocks * self.misplaced_blocks_penalty) + (unblocked_blocks * self.unblocked_penalty) 
        return heuristic_value

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic36(misplaced_blocks_penalty={self.misplaced_blocks_penalty}, unblocked_penalty={self.unblocked_penalty})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic36(misplaced_blocks_penalty={self.misplaced_blocks_penalty}, unblocked_penalty={self.unblocked_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 36 info:\n"
            f"\tName: {self.name}\n"
            f"\tMisplaced blocks penalty: {self.misplaced_blocks_penalty}\n"
            f"\tUnblocked penalty: {self.unblocked_penalty}\n"
            f"\tGoal blocks: {len(self.goal_blocks)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
