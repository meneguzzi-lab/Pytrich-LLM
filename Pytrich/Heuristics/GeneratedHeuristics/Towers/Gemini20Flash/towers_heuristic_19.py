import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic19(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_19: str = "towers_gemini20flash_heuristic19"):
        super().__init__(name=towers_gemini20flash_heuristic_19)
        self.preprocessing_time = 0
        self.num_rings = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.num_rings = self._count_rings(model)
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node, model)

        return super().initialize(model, h_value)

    def _count_rings(self, model):
        count = 0
        for fact in model.facts:
            if 'RING' in str(fact.name).upper():
                count += 1
        return count

    def _calculate_node_heuristic(self, node: HTNNode, model: Model):
        """
        Calculates the heuristic value for a given node in the search tree.

        This heuristic estimates the remaining cost to solve the Towers of Hanoi problem
        by considering the number of rings that are not on the goal tower.

        The heuristic also accounts for abstract tasks, giving a small penalty for each
        to encourage the planner to decompose them efficiently.
        """
        heuristic_value = 0

        goal_state = set()
        for fact in model.facts:
          if 'goal_on' in fact.name:
            goal_state.add(fact)

        current_state = set()
        if isinstance(node.state, set):
            current_state = node.state
        else:
          for fact_id in range(node.state.bit_length()):
            if node.state & (1 << fact_id):
              fact = model.get_component(fact_id)
              current_state.add(fact)

        rings_on_goal = 0
        for goal_fact in goal_state:
            ring = goal_fact.name.split()[1]  # Extract the ring name
            ring_name = f'(on {ring} ?o - OBJ)'
            
            for current_fact in current_state:
              if ring_name == current_fact.name:
                rings_on_goal += 1

        rings_not_on_goal = self.num_rings - rings_on_goal

        heuristic_value += rings_not_on_goal  # Each misplaced ring adds to the cost

        # Penalize abstract tasks to encourage efficient decomposition
        if node.task:
            heuristic_value += 0.1  # Small penalty for each abstract task

        return heuristic_value

    def __call__(self, parent_node, node):
        model = parent_node.model if parent_node else node.model
        h_value = self._calculate_node_heuristic(node, model)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic19()"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic19()"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 19 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tNumber of rings: {self.num_rings}\n"
        )
