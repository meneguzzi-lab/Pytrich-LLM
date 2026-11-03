from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import sys

# BOTTLENECK ANALYSIS:
# The primary bottleneck in the Barman_BDI domain is parameter-binding symmetry, specifically the choice of which 'shot' 
# to use for transferring ingredients. With many identical shots, the branching factor grows exponentially. 
# A secondary bottleneck is the singleton resource constraint of the shaker and hands.

# ADMISSIBILITY SKETCHES:
# Term 1 (Task Complexity): Sum of minimum action costs per task in the network. Each task must decompose 
# into at least its atomic components (e.g., pouring requires cleaning, holding, etc.), providing a lower bound.
# Term 2 (Unsatisfied Goals): Each 'contains(shot, cocktail)' goal not yet in the state requires a minimum sequence 
# of cocktail preparation (~20 actions), independent of current network tasks if they don't overlap.
# ADMISSIBILITY: We use max(Term1, Term2) and then add tie-breaking penalties for symmetry breaking.

class BarmanBDIGemini3FlashHeuristic17(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_17="BarmanHeuristic"):
        super().__init__(barman_gemini3flash_heuristic_17)
        self.task_costs = {
            "AchieveContainsShotCocktail": 20,
            "AchieveContainsShakerCocktail": 15,
            "AchieveContainsShakerIngredient": 10,
            "AchieveContainsShotIngredient": 8,
            "AchieveCleanShaker": 5,
            "AchieveCleanShot": 5,
            "AchieveHolding": 2,
            "AchieveOnTable": 2,
            "AchieveHandEmpty": 2,
            "DoPourShakerToShot": 5
        }
        self.goal_shots = set()
        self.shot_order = {}
        self.all_hands = set()
        self.preferred_hand = None
        self.goal_fact_ids = []

    def _parse_item(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        for char in ["[", "("]:
            idx = name.find(char)
            if idx > 0:
                prefix = name[:idx]
                for end_char in ["]", ")"]:
                    end_idx = name.find(end_char)
                    if end_idx > idx:
                        args_str = name[idx+1:end_idx]
                        args = [a.strip() for a in args_str.split(",") if a.strip()]
                        return prefix, args
                return prefix, []
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.goal_fact_ids = [f.global_id for f in model.facts if (model.goals >> f.global_id) & 1]
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            pred, args = self._parse_item(gn)
            if pred == "contains" and args:
                self.goal_shots.add(args[0])
        
        shot_set = set()
        for f in model.facts:
            p, args = self._parse_item(f.name)
            if args:
                for a in args:
                    if "shot" in a.lower():
                        shot_set.add(a)
                if p == "handEmpty":
                    self.all_hands.add(args[0])
        
        sorted_shots = sorted(list(shot_set))
        self.shot_order = {s: i for i, s in enumerate(sorted_shots)}
        
        sorted_hands = sorted(list(self.all_hands))
        if sorted_hands:
            self.preferred_hand = sorted_hands[0]

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        task_sum = 0
        penalty = 0
        
        for task in task_network:
            prefix, args = self._parse_item(task.name)
            task_sum += self.task_costs.get(prefix, 1)
            
            # SYMMETRY BREAKING HINTS IMPLEMENTATION
            is_final_goal_task = prefix in ["AchieveContainsShotCocktail", "DoPourShakerToShot", "pour-shaker-to-shot"]
            
            for arg in args:
                # Goal-shot penalty: 500 for using goal-shots in intermediate tasks
                if arg in self.goal_shots and not is_final_goal_task:
                    penalty += 500
                # Working-shot ordering: canonicalize selection via index-based penalty
                if arg in self.shot_order:
                    penalty += self.shot_order[arg] * 100
                # Hand preference: favor preferred hand
                if self.preferred_hand and arg in self.all_hands and arg != self.preferred_hand:
                    penalty += 50

        # Unachieved goals term
        unachieved_cost = 0
        for g_id in self.goal_fact_ids:
            if not ((state >> g_id) & 1):
                unachieved_cost += 20

        return max(task_sum, unachieved_cost) + penalty

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return int(h_value)
