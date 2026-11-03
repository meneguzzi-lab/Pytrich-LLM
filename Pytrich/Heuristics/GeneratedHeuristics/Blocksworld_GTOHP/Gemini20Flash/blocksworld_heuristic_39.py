import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGemini20FlashHeuristic39(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_39: str = "blocksworld_gemini20flash_heuristic_39"):
        super().__init__(name=blocksworld_gemini20flash_heuristic_39)
        self.goal_blocks = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_goal_blocks()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_goal_blocks(self):
        """
        Extract the goal state from the model and store the 'on' relationships.
        """
        self.goal_blocks = {}
        for fact in self.model.goals:
            fact_str = str(fact.name)
            if fact_str.startswith('on'):
                parts = fact_str.split()
                if len(parts) == 3:
                    block1 = parts[1]
                    block2 = parts[2]
                    self.goal_blocks[block1] = block2

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a given node.

        The heuristic estimates the number of 'puton' operations needed to achieve the goal state.
        It counts the number of blocks that are not in their goal positions.
        """
        state_blocks = self._extract_state_blocks(node)
        misplaced_blocks = 0

        for block, on_block in self.goal_blocks.items():
            if block not in state_blocks or state_blocks[block] != on_block:
                misplaced_blocks += 1

        return misplaced_blocks

    def _extract_state_blocks(self, node):
        """
        Extract the current block positions from the node's state.
        """
        state_blocks = {}
        if isinstance(node, HTNNode):
           state = node.state
        else:
            return state_blocks

        for bit_pos in range(state.bit_length()):
            if state & (1 << bit_pos):
                fact = self.model.get_component(bit_pos)
                fact_str = str(fact.name)

                if fact_str.startswith('on'):
                    parts = fact_str.split()
                    if len(parts) == 3:
                        block1 = parts[1]
                        block2 = parts[2]
                        state_blocks[block1] = block2
                elif fact_str.startswith('ontable'):
                    parts = fact_str.split()
                    if len(parts) == 2:
                        block = parts[1]
                        state_blocks[block] = 'ontable'
        return state_blocks

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic39()"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic39()"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 39 info:\n"
            f"\tName: {self.name}\n"
            f"\tGoal blocks: {len(self.goal_blocks)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
