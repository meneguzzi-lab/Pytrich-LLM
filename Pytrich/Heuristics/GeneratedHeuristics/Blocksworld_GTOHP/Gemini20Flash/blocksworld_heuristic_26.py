import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for Blocksworld domain.
class BlocksworldGemini20FlashHeuristic26(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_26="blocksworld_gemini20flash_heuristic_26", misplaced_blocks_weight=1.0, action_cost=1.0):
        super().__init__(name=blocksworld_gemini20flash_heuristic_26)
        self.misplaced_blocks_weight = misplaced_blocks_weight
        self.action_cost = action_cost
        self.goal = None
        self.initial_state = None
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.goal = model.goals
        self.initial_state = model.initial_state
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """Calculates the heuristic value for a given node."""
        state = node.state
        misplaced_blocks = self._count_misplaced_blocks(state)
        return self.misplaced_blocks_weight * misplaced_blocks + self.action_cost * node.g

    def _count_misplaced_blocks(self, state):
        """Counts the number of blocks that are not in their goal positions."""
        misplaced_count = 0
        goal_facts = self.model.facts
        state_facts = [self.model.get_component(fact_id) for fact_id in range(state.bit_length()) if state & 1 << fact_id]

        # Filter goal facts to keep only 'on' predicates.
        relevant_goal_facts = {fact.name for fact in goal_facts if fact.name.startswith('on')}

        # Convert state facts to a set of their names for faster lookup
        state_fact_names = {fact.name for fact in state_facts}

        # Iterate through the relevant goal facts and check if they are present in the state.
        for goal_fact in relevant_goal_facts:
            if goal_fact not in state_fact_names:
                misplaced_count += 1

        return misplaced_count

    def __call__(self, parent_node, node):
        """Heuristic function called during search."""
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic26(misplaced_blocks_weight={self.misplaced_blocks_weight}, action_cost={self.action_cost})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic26(misplaced_blocks_weight={self.misplaced_blocks_weight}, action_cost={self.action_cost})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 26 info:\n"
            f"\tName: {self.name}\n"
            f"\tMisplaced blocks weight: {self.misplaced_blocks_weight}\n"
            f"\tAction cost: {self.action_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
