import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGemini20FlashHeuristic24(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_24=1, name="BlocksworldGemini20FlashHeuristic24"):
        super().__init__(name=name)
        self.name = name
        self.preprocessing_time = 0
        self.ontable_cost = blocksworld_gemini20flash_heuristic_24

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_heuristic(self, node):
        """Calculates the heuristic value for a given node."""
        state = node.state
        goal_facts = self.model.goals
        heuristic_value = 0

        for goal in goal_facts:
            goal_fact_name = self.model.facts[goal].name

            if goal not in state:
                # Check if the goal is an 'on' fact or an 'ontable' fact.
                if goal_fact_name.startswith('on'):
                    # Estimate cost based on number of blocks that need to be moved.
                    parts = goal_fact_name.split()
                    block1 = parts[1]
                    block2 = parts[2]

                    # This is a simplified estimate. A more accurate heuristic would
                    # need to traverse the blocks currently on top of block1 and block2.
                    heuristic_value += 1 # Simple increment for each 'on' goal not met.

                elif goal_fact_name.startswith('ontable'):
                    heuristic_value += self.ontable_cost # Cost of moving a block onto the table

                elif goal_fact_name == 'handempty':
                     heuristic_value += 0.5 # Assign some cost to ensure robot frees hands

                else:
                     heuristic_value += 1 # Default cost for other missing facts

        return heuristic_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic24(ontable_cost={self.ontable_cost})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic24(ontable_cost={self.ontable_cost})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 24 info:\n"
            f"\tName: {self.name}\n"
            f"\tOntable Cost: {self.ontable_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )