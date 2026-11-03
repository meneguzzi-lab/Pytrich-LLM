import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic23(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_23=1.0):
        super().__init__(name='RoverGemini20FlashHeuristic23')
        self.base_cost = rover_gemini20flash_heuristic_23
        self.fact_weights = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_fact_weights()
        self.preprocessing_time = time.time() - start_time
        initial_h_value = self._calculate_heuristic(initial_node.state)

        return super().initialize(model, initial_h_value)

    def _precompute_fact_weights(self):
        # Initialize weights for facts based on their relevance to the goal
        for fact in self.model.facts:
            fact_name = fact.name
            if 'communicated' in fact_name:
                self.fact_weights[fact_name] = self.base_cost  # Goals are important
            elif 'have_' in fact_name:
                self.fact_weights[fact_name] = self.base_cost*0.75  # Intermediate goals
            elif 'at ' in fact_name:
                self.fact_weights[fact_name] = self.base_cost*0.5  # Location facts
            else:
                self.fact_weights[fact_name] = self.base_cost*0.25  # Other supporting facts

    def _calculate_heuristic(self, state):
        h_value = 0
        for bit_pos in range(state.bit_length()):
            if state & 1 << bit_pos:
                fact = self.model.facts[bit_pos]
                h_value += self.fact_weights.get(fact.name, 0)  # Access weight via fact name
        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic23(base_cost={self.base_cost})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic23(base_cost={self.base_cost})"

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 23 info:\n"
            f"\tName: {self.name}\n"
            f"\tBase cost: {self.base_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tFact weights: {len(self.fact_weights)}\n"
        )
