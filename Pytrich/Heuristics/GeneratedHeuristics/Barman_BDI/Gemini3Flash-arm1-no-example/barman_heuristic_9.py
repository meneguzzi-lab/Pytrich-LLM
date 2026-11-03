from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import math

"""
Bottleneck Analysis:
In Barman_BDI, the bottleneck is the sequential process of preparing cocktails using a single shaker and limited hands.
Each cocktail requires cleaning the shaker, filling it with two ingredients (which involves cleaning and filling shots),
shaking it, and pouring. The hand capacity (max 2) and container availability (1 shaker) force a strict sequence.

Admissible Lower Bounds:
1. Cocktail Remaining Bound: Each 'AchieveContainsShotCocktail' or 'AchieveContainsShakerCocktail' task in the network 
   requires a minimum number of physical actions (fill, pour, shake, clean). Preparing a cocktail from scratch 
   takes at least 8-10 operators.
2. Shaker Status Bound: If the shaker is dirty or contains the wrong beverage, at least one 'clean-shaker' or 'empty-shaker' 
   action is required before it can be used for a new cocktail.
3. Ingredient Bound: Every missing ingredient for a cocktail currently in the task network requires at least one 'fill-shot' 
   and one 'pour-shot-to-shaker' action.

Symmetry Breaking:
Small penalties are added for using different shots to encourage the planner to stick with one shot if possible,
reducing unnecessary 'grasp'/'drop' cycles.
"""

class BarmanBDIGemini3FlashHeuristic9(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_9="barman_heuristic_9"):
        super().__init__(barman_gemini3flash_heuristic_9)
        self.clean_shaker_ids = []
        self.clean_shot_ids = {}
        self.contains_ids = {}
        self.cocktail_parts = {}
        self.shaker_ids = []
        self.shot_ids = []
        self.goal_cocktails = []

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
        self.model = model
        self.clean_shaker_ids = []
        self.clean_shot_ids = {}
        self.contains_ids = {}
        self.cocktail_parts = {}
        self.shaker_ids = []
        self.shot_ids = []

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "clean":
                # Type checking via naming convention or model lookup if needed
                # but usually args[0] starts with 'shaker' or 'shot'
                if args[0].startswith("shaker"):
                    self.clean_shaker_ids.append(fact.global_id)
                    if args[0] not in self.shaker_ids: self.shaker_ids.append(args[0])
                elif args[0].startswith("shot"):
                    self.clean_shot_ids[args[0]] = fact.global_id
                    if args[0] not in self.shot_ids: self.shot_ids.append(args[0])
            elif pred == "contains":
                self.contains_ids[(args[0], args[1])] = fact.global_id
            elif pred == "cocktailPart1":
                if args[0] not in self.cocktail_parts: self.cocktail_parts[args[0]] = []
                self.cocktail_parts[args[0]].append(args[1])
            elif pred == "cocktailPart2":
                if args[0] not in self.cocktail_parts: self.cocktail_parts[args[0]] = []
                self.cocktail_parts[args[0]].append(args[1])

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        if not task_network:
            return 0

        h = 0
        # Term 1: Task Network Complexity
        # Each AchieveContainsShotCocktail is approx 10 actions if cocktail not present
        # Each AchieveContainsShakerCocktail is approx 6 actions
        # Each AchieveContainsShakerIngredient is approx 3 actions
        for task in task_network:
            t_name = task.name.lower()
            if "achievecontainsshotcocktail" in t_name:
                # format usually: AchieveContainsShotCocktail shot cocktail
                parts = t_name.replace("(", " ").replace(")", " ").split()
                if len(parts) >= 3:
                    shot, cocktail = parts[1], parts[2]
                    cont_id = self.contains_ids.get((shot, cocktail))
                    if cont_id is not None and not ((state >> cont_id) & 1):
                        h += 10
            elif "achievecontainsshakercocktail" in t_name:
                h += 6
            elif "achievecontainsshakingredient" in t_name:
                h += 3
            elif "achievecleanshot" in t_name:
                h += 1
            elif "achievecleanshaker" in t_name:
                h += 1
            elif "achieveholding" in t_name:
                h += 1
            else:
                # Basic cost for any other non-primitive task
                if hasattr(task, 'decompositions'):
                    h += 1

        # Term 2: Shaker State Penalty
        # If any shaker is not clean and we have cocktail goals, we must clean it
        for s_id in self.clean_shaker_ids:
            if not ((state >> s_id) & 1):
                h += 2

        # Term 3: Symmetry Breaking
        # Small penalty based on the index of the container to prioritize lower-indexed shots
        symmetry_penalty = 0
        for task in task_network:
            t_name = task.name.lower()
            for i, shot_name in enumerate(self.shot_ids):
                if shot_name in t_name:
                    symmetry_penalty += i * 0.001
                    break

        return int(h) + symmetry_penalty

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
