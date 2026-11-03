import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGemini20FlashHeuristic25(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_25, name="blocksworld_gemini20flash_heuristic_25"):
        super().__init__(name=name)
        self.name = name
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.fact_dict = {fact.name: fact for fact in model.facts}
        self.goal_state = set()
        for goal in model.goals:
            self.goal_state.add(model.facts[goal].name)
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        state = set()
        # Extract state from bitwise representation
        for bit_pos in range(node.state.bit_length()):
            if node.state & (1 << bit_pos):
                try:
                    fact = self.model.facts[bit_pos]
                    state.add(fact.name)
                except IndexError:
                    pass

        # Calculate the number of unsatisfied goals
        unsatisfied_goals = self.goal_state - state

        # In Blocksworld, each unsatisfied 'on' goal typically requires at least one 'put_on' action.
        # We can also add a small penalty for each 'clear' goal that is not met to account for blocks needing to be moved.
        heuristic_value = len(unsatisfied_goals)

        return heuristic_value


    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic(name={self.name})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic(name={self.name})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash  Heuristic 25 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
