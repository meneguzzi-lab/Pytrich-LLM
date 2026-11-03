from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# BOTTLENECK ANALYSIS:
# The primary bottleneck in Barman_BDI is the container preparation cycle (cleaning, filling, pouring, shaking).
# Each cocktail requires a specific sequence of actions involving a single shaker and multiple shots,
# and since there are only two hands, the planner must serialize the cleaning and transferring of ingredients.
# Efficiency is gained by minimizing unnecessary drops/grasps and ensuring containers are cleaned only when needed.

# ADMISSIBILITY SKETCHES:
# 1. Task Cost (h_tasks): Each abstract task in the network must eventually decompose into a sequence of primitive actions.
#    By assigning a pre-computed minimum number of actions for each task type, we obtain an admissible lower bound.
# 2. Hand/Container State (h_state): Actions like 'drop' or 'clean' are mandatory if hands are full or containers are dirty
#    before certain tasks can proceed. These add unavoidable costs to the plan length.

class BarmanBDIGemini3FlashHeuristic1(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_1="barman_gemini3flash_heuristic_1"):
        super().__init__(barman_gemini3flash_heuristic_1)
        self.task_min_costs = {
            "AchieveContainsShotCocktail": 10,
            "AchieveContainsShakerCocktail": 7,
            "AchieveContainsShakerIngredient": 3,
            "AchieveContainsShotIngredient": 2,
            "AchieveCleanShot": 1,
            "AchieveCleanShaker": 1,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 1
        }
        self.clean_ids = []
        self.hand_empty_ids = []
        self.holding_ids = []
        self.used_ids = []

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "clean":
                self.clean_ids.append(f.global_id)
            elif pred == "handEmpty":
                self.hand_empty_ids.append(f.global_id)
            elif pred == "holding":
                self.holding_ids.append(f.global_id)
            elif pred == "used":
                self.used_ids.append(f.global_id)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        # Term 1: Task Network Minimum Costs
        h_tasks = 0
        for task in task_network:
            # task can be an AbstractTask or an Operator
            name = task.name
            # Strip HDDL-style parameters if they exist in name string (e.g., "task_name ?x")
            base_name = name.split(" ")[0] if " " in name else name
            
            if base_name in self.task_min_costs:
                h_tasks += self.task_min_costs[base_name]
            else:
                # If it's a primitive operator not in the dict, cost is 1
                h_tasks += 1

        # Term 2: Mandatory State Transitions
        # If a hand is not empty, we might need a 'drop' action (+1)
        # If a container is 'used' but not 'clean', we might need a 'clean' action (+1)
        h_state = 0
        
        # Check hands
        hands_full = 0
        for he_id in self.hand_empty_ids:
            if not ((state >> he_id) & 1):
                hands_full += 1
        h_state += hands_full

        # Check for dirty containers (used and not clean)
        # This is a very rough admissible estimate
        # In Barman, 'used' implies 'not clean'
        for u_id in self.used_ids:
            if (state >> u_id) & 1:
                h_state += 1

        # Tie-breaking: small penalty for number of tasks to prefer shorter networks
        tie_breaker = len(task_network) * 0.01

        return int(h_tasks + h_state + tie_breaker)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
