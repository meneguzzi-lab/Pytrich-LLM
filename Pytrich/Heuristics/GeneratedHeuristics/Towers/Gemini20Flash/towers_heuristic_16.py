import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic16(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_16=1, name="TowersGemini20FlashHeuristic16"):
        super().__init__(name=name)
        self.move_penalty = 1  # Cost for each move action
        self.method_application_penalty = 0.1  # Cost for applying a method
        self.goal_distance_penalty = 2 # Cost related to the distance of the rings from the goal
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.goal_state = initial_node.state & model.goals
        self.goal_on_facts = self._extract_goal_on_facts(model)
        self.num_rings = self._extract_num_rings(model)
        self.start_tower = self._extract_start_tower(model)
        self.target_tower = self._extract_target_tower(model)
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _extract_goal_on_facts(self, model):
        goal_on_facts = set()
        for fact in model.facts:
            if 'goal_on' in fact.name:
                goal_on_facts.add(fact)
        return goal_on_facts

    def _extract_num_rings(self, model):
        num_rings = 0
        for fact in model.facts:
            if 'RING' in str(fact.name).upper():
                num_rings+=1
        return num_rings

    def _extract_start_tower(self, model):
        for fact in model.facts:
            if 'on' in fact.name and 't1' in fact.name:  #heuristic assumes start is always t1
                return 't1'

    def _extract_target_tower(self, model):
         for fact in model.facts:
            if 'goal_on' in fact.name and 't3' in fact.name: #heuristic assumes target is always t3
                return 't3'

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a given node.

        The heuristic estimates the cost of moving rings from the start tower to the target tower,
        taking into account the number of rings, the current state, and the goal state.
        """
        state = node.state

        #Base heuristic cost is 0
        heuristic_cost = 0

        #Estimate misplaced rings
        misplaced_rings = 0

        #Check each goal fact
        for goal_fact in self.goal_on_facts:
            #check if the goal fact is not achieved in current state, and if not then increment the counter
            if not (state & (1 << goal_fact.global_id)):
                misplaced_rings+=1

        #Calculate the heuristic cost based on the number of misplaced rings.
        heuristic_cost += self.goal_distance_penalty * misplaced_rings

        #If the node expanded a method, add a method application penalty
        if hasattr(node, 'method') and node.method is not None:
            heuristic_cost += self.method_application_penalty

        return heuristic_cost

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic16(move_penalty={self.move_penalty}, method_penalty={self.method_application_penalty}, goal_distance_penalty={self.goal_distance_penalty})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic16(move_penalty={self.move_penalty}, method_penalty={self.method_application_penalty},  goal_distance_penalty={self.goal_distance_penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 16 info:\n"
            f"\tName: {self.name}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tMethod application penalty: {self.method_application_penalty}\n"
            f"\tGoal distance penalty: {self.goal_distance_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tNumber of rings: {self.num_rings}\n"
        )
