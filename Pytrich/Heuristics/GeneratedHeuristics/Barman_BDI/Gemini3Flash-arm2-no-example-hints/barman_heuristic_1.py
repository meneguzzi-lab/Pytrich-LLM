from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain Bottleneck Analysis:
The primary bottleneck in the Barman domain is the parameter-binding symmetry among interchangeable shot containers. 
Since any available shot can be used to transfer ingredients, the branching factor explodes (N shots ^ K ingredients). 
Furthermore, using a shot designated as a goal container for an intermediary transfer requires redundant cleaning steps.

Admissibility Sketch:
1. Task Cost Term: Assigns a minimum number of required actions to each task in the network (e.g., 10 for a cocktail).
   This is a lower bound because decomposition to primitive actions cannot result in fewer actions than the physical sequence.
2. State Satisfaction: If a task's target state (e.g., clean container) is already true, its cost is reduced to zero.
3. Symmetry Breaking: Goal-shot penalties (500), hand preference (50), and shot-order penalties (100*idx) break 
   permutations without violating the search for a valid plan.
"""

class BarmanBDIGemini3FlashHeuristic1(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_1="BarmanBDIGemini3FlashHeuristic1"):
        super().__init__(barman_gemini3flash_heuristic_1)
        self.goal_shots = {}  # shot -> cocktail
        self.shot_order = {}  # shot -> penalty_val
        self.task_info = {}   # global_id -> (base_cost, shot_arg, cocktail_arg, is_right_hand)
        self.fact_map = {}    # pred -> args -> global_id
        
        self.TASK_COSTS = {
            "AchieveContainsShotCocktail": 10,
            "AchieveContainsShakerCocktail": 7,
            "AchieveContainsShakerIngredient": 3,
            "AchieveCleanShot": 2,
            "AchieveCleanShaker": 2,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 1,
            "MakeAndPourCocktail": 10,
            "MakeCocktail": 7,
            "AddIngredientToEmptyShaker": 3,
            "AddIngredientToUsedShaker": 3,
            "AddIngredientToShot": 2,
            "CleanFullShot": 2,
            "CleanEmptyShot": 2,
            "CleanEmptyShaker": 2,
            "CleanFullShaker": 2,
            "PickUp": 1,
            "EmptyHand": 1,
            "PutDown": 1,
            "pour_shaker_to_shot_action": 1
        }

    def _parse_name(self, name):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = tuple(args_str.split(",")) if args_str else ()
            return predicate, args
        return name, ()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map Facts
        for f in model.facts:
            pred, args = self._parse_name(f.name)
            if pred not in self.fact_map: self.fact_map[pred] = {}
            self.fact_map[pred][args] = f.global_id

        # 2. Identify Goal Shots
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for g in goal_names:
            pred, args = self._parse_name(g)
            if pred == "contains" and len(args) == 2:
                self.goal_shots[args[0]] = args[1]

        # 3. Shot Ordering (Consistency)
        all_shots = sorted([args[0] for pred, facts in self.fact_map.items() if pred == "clean" for args in facts])
        for idx, s in enumerate(all_shots):
            self.shot_order[s] = idx * 100

        # 4. Pre-process Tasks (Abstract and Operators)
        for task_list in [model.abstract_tasks, model.operators]:
            for t in task_list:
                prefix, args = self._parse_name(t.name)
                base_cost = self.TASK_COSTS.get(prefix, 1)
                shot = None
                cocktail = None
                is_right = False
                
                if len(args) > 0:
                    # In Barman, shot is almost always arg[0] in relevant tasks
                    if "shot" in args[0]: shot = args[0]
                    if "cocktail" in args[-1]: cocktail = args[-1]
                    for arg in args: 
                        if arg == "right": is_right = True
                
                # Task state-satisfaction mapping
                fact_id = -1
                if prefix == "AchieveContainsShotCocktail" and len(args) >= 2: 
                    fact_id = self.fact_map.get("contains", {}).get((args[0], args[1]), -1)
                elif prefix == "AchieveCleanShot" and len(args) >= 1:
                    fact_id = self.fact_map.get("clean", {}).get((args[0],), -1)
                elif prefix == "AchieveHandEmpty" and len(args) >= 1:
                    fact_id = self.fact_map.get("handEmpty", {}).get((args[0],), -1)

                self.task_info[t.global_id] = (base_cost, shot, cocktail, is_right, fact_id)

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _compute(self, state, task_network):
        h = 0
        for task in task_network:
            info = self.task_info.get(task.global_id)
            if not info: continue
            
            base_cost, shot, cocktail, is_right, fact_id = info
            
            # Admissible base cost (skip if already satisfied in state)
            if fact_id != -1 and (state >> fact_id) & 1:
                pass
            else:
                h += base_cost
            
            # Symmetry Breaking
            if shot:
                h += self.shot_order.get(shot, 0)
                if shot in self.goal_shots:
                    # Penalize using a goal shot for something other than its intended cocktail
                    is_valid_goal_task = (cocktail == self.goal_shots[shot])
                    if not is_valid_goal_task:
                        h += 500
            
            if is_right:
                h += 50
                
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
