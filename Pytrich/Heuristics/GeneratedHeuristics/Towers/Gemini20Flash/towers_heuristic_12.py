import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the towers domain
class TowersGemini20FlashHeuristic12(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_12=1, name="TowersGemini20FlashHeuristic12"):
        super().__init__(name=name)
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.goal_positions = self._extract_goal_positions()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _extract_goal_positions(self):
        """Extracts the desired 'on' positions from the goal state."""
        goal_positions = {}
        for fact in self.model.goals:
            fact_str = str(fact)
            if 'goal_on' in fact_str:
                parts = fact_str.split()
                ring = parts[1]
                target = parts[2]
                goal_positions[ring] = target
        return goal_positions

    def _calculate_node_heuristic(self, node):
        """Calculates the heuristic value for a given node.
        This heuristic estimates the number of 'move' actions required to achieve the goal state.
        It counts the number of rings that are not in their goal positions.
        """
        state = node.state
        heuristic_value = 0
        rings_in_wrong_positions = 0
        if state is not None:
          current_positions = {}
          for i in range(state.bit_length()):
            if (state >> i) & 1:
              fact = self.model.get_component(i)
              fact_str = str(fact)
              if 'on' in fact_str and 'goal_on' not in fact_str:
                  parts = fact_str.split()
                  ring = parts[1]
                  current = parts[2]
                  current_positions[ring] = current
          for ring, target in self.goal_positions.items():
              if ring in current_positions and current_positions[ring] != target:
                rings_in_wrong_positions += 1
        heuristic_value = rings_in_wrong_positions
        return heuristic_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return f"Towers Gemini 2.0 Flash Heuristic 12 info:\n" \
               f"\tName: {self.name}\n" \
               f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
