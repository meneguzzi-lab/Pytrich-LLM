import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for Blocksworld domain
class BlocksworldGemini20FlashHeuristic2(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_2="blocksworld_gemini20flash_heuristic_2", misplaced_block_penalty=1, action_cost=1):
        super().__init__(name=blocksworld_gemini20flash_heuristic_2)
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
        Calculates the heuristic value for a given node.
        This heuristic estimates the number of blocks that are out of place
        compared to the goal state and adds a penalty for each action.
        """
        goal_state = self.model.goals
        current_state = self._get_current_state(node)
        
        misplaced_blocks = self._count_misplaced_blocks(current_state, goal_state)
        
        # Estimate actions needed. In blocksworld, each misplaced block may require at least
        # one putdown and one pickup, thus we multiply by 2.
        estimated_actions = misplaced_blocks * 2

        # Adding a penalty based on node depth may help guide the search towards
        # solutions earlier, assuming lower depth implies closer to solution. 
        # However this should be done carefully so as not to violate admissibility
        # In HTN planning, the number of primitive tasks is not the depth, so this is safe.
        #node_depth_penalty = node.depth if hasattr(node, "depth") else 0

        heuristic_value = (misplaced_blocks * self.misplaced_block_penalty) + (estimated_actions * self.action_cost)

        return heuristic_value

    def _get_current_state(self, node):
        """
        Retrieves the current state facts from the node.
        """
        if hasattr(node, 'state'):
          return node.state
        else:
          return set()

    def _count_misplaced_blocks(self, current_state, goal_state):
        """
        Counts the number of blocks that are not in their goal positions.
        """
        misplaced_count = 0
        for goal_fact in goal_state:
            if goal_fact not in current_state:
                misplaced_count += 1
        return misplaced_count

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic2(misplaced_block_penalty={self.misplaced_block_penalty}, action_cost={self.action_cost})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic2(misplaced_block_penalty={self.misplaced_block_penalty}, action_cost={self.action_cost})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 2 info:\n"
            f"\tName: {self.name}\n"
            f"\tMisplaced block penalty: {self.misplaced_block_penalty}\n"
            f"\tAction cost: {self.action_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
