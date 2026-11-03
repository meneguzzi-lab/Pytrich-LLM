from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# BOTTLENECK ANALYSIS:
# The primary bottleneck in the Barman_BDI domain is the serialization of shaker and hand usage. 
# Only one shaker exists, and it must be cleaned, filled with two separate ingredients (via shot glasses), 
# shaken, and then poured, before it can be reused. This creates a tight resource constraint. 
# Additionally, symmetry among interchangeable shot glasses causes massive state-space explosion.

# ADMISSIBILITY SKETCHES:
# 1. Task Network Cost (h_tasks): Sum of minimal actions required to decompose/execute each task in the network. 
#    Since each task must eventually be replaced by at least one action (or zero if Null), this is a lower bound 
#    if costs are chosen conservatively (e.g., 1 per primitive action).
# 2. Goal Progress: The number of 'contains' cocktail goals not yet present in the task network or the state. 
#    Each missing cocktail requires a sequence of at least 10-12 actions.

class BarmanBDIGemini3FlashHeuristic19(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_19="barman_gemini3flash_heuristic_19"):
        super().__init__(name=barman_gemini3flash_heuristic_19)
        self.component_costs = {}
        self.component_penalties = {}
        self.goal_shots = set()
        self.all_shots = []
        self.shot_map = {}

    def _parse_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        if "[" in name and name.endswith("]"):
            bracket_idx = name.find("[")
            basename = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return basename, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Identify Goal Shots and All Shots
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_name(g_name)
            if pred == "contains":
                for a in args:
                    if "shot" in a.lower():
                        self.goal_shots.add(a)
        
        shots_found = set()
        for f in model.facts:
            _, args = self._parse_name(f.name)
            for a in args:
                if "shot" in a.lower():
                    shots_found.add(a)
        self.all_shots = sorted(list(shots_found))
        self.shot_map = {name: i for i, name in enumerate(self.all_shots)}

        # 2. Precompute Costs and Penalties for all Abstract Tasks and Operators
        base_costs = {
            "AchieveContainsShotCocktail": 15,
            "AchieveContainsShakerCocktail": 10,
            "AchieveContainsShakerIngredient": 5,
            "AchieveContainsShotIngredient": 3,
            "AchieveCleanShot": 2,
            "AchieveCleanShaker": 2,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "DoPourShakerToShot": 1
        }

        # Process Abstract Tasks
        for abt in model.abstract_tasks:
            basename, args = self._parse_name(abt.name)
            cost = base_costs.get(basename, 1)
            penalty = self._compute_penalty(basename, args)
            self.component_costs[abt.global_id] = cost
            self.component_penalties[abt.global_id] = penalty

        # Process Operators
        for op in model.operators:
            basename, args = self._parse_name(op.name)
            cost = 1
            penalty = self._compute_penalty(basename, args)
            self.component_costs[op.global_id] = cost
            self.component_penalties[op.global_id] = penalty

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute_penalty(self, basename, args):
        penalty = 0
        # Goal-shot penalty (P6): Intermediary tasks using goal shots
        is_intermediary = basename in ["AchieveContainsShotIngredient", "fill-shot", "pour-shot-to-clean-shaker", "pour-shot-to-used-shaker"]
        for a in args:
            # Symmetry breaking: Shot ordering (P6)
            if a in self.shot_map:
                penalty += 100 * self.shot_map[a]
                if is_intermediary and a in self.goal_shots:
                    penalty += 500
            # Hand preference (P6)
            if "right" in a.lower():
                penalty += 50
        return penalty

    def _compute(self, state, task_network):
        h = 0
        for task in task_network:
            # Use precomputed global_id lookup for O(1) in call
            h += self.component_costs.get(task.global_id, 1)
            h += self.component_penalties.get(task.global_id, 0)
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
