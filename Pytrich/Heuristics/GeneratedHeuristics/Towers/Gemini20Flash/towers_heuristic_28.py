import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic28(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_28=1, name="TowersGemini20FlashHeuristic28"):
        super().__init__(name=name)
        self.name = name
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.goal_facts = model.goals
        self.towers = []
        self.rings = []
        for fact in model.facts:
          if 'TOWER' in fact.name:
            self.towers.append(fact.name.split(' - ')[0])
          if 'RING' in fact.name:
            self.rings.append(fact.name.split(' - ')[0])
        self.n_rings = len(self.rings)
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        '''
        Calculates the heuristic value of a given node.
        The heuristic is based on the number of rings that are not on the goal tower.
        And also the number of rings that are out of order.
        '''

        state = node.state
        state_facts = self.model.state_explicit_repr(state)
        goal_tower = None

        # Find the goal tower. It's the one that has goal_on facts
        for fact in state_facts:
            if 'goal_on' in fact:
                goal_tower = fact.split(' ')[2]
                break

        if goal_tower is None:
            return self.n_rings # Worst case scenario, no goal tower

        misplaced_rings = 0
        out_of_order_rings = 0
        current_ring_location = {}
        goal_ring_location = {}

        # Find current and goal ring location
        for fact in state_facts:
          if 'on' in fact:
            parts = fact.split(' ')
            ring = parts[1]
            tower = parts[2]
            current_ring_location[ring] = tower

        for fact in state_facts:
            if 'goal_on' in fact:
                parts = fact.split(' ')
                ring = parts[1]
                tower = parts[2]
                goal_ring_location[ring] = tower

        # calculate misplaced rings
        for ring in self.rings:
          if ring in current_ring_location and ring in goal_ring_location:
              if current_ring_location[ring] != goal_ring_location[ring]:
                  misplaced_rings += 1
          else:
              misplaced_rings += 1 # ring not found in state, add one

        # calculate out of order rings
        ordered = True
        current_tower = 't1' #start at the base tower to calculate rings out of order
        max_ring = 1000

        # Check rings for being out of order
        for fact in state_facts:
            if 'on' in fact:
                parts = fact.split(' ')
                ring = parts[1]
                tower = parts[2]
                num = int(ring[1:])

                if num < max_ring:
                    max_ring = num
                else:
                    out_of_order_rings +=1

        return misplaced_rings + out_of_order_rings

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"{self.name}()"

    def __str__(self):
        return f"{self.name}()"

    def __output__(self):
        return (
            f"{self.name} info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
