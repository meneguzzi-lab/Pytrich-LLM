import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic9(Heuristic):
    """
    Domain-dependent heuristic for the Barman_BDI domain.

    1. Bottleneck Identification:
       The primary bottleneck is the Shaker. Making a cocktail requires a non-parallelizable sequence 
       of actions: cleaning, filling with two ingredients (via shots), shaking, and pouring out.
       Since only one cocktail can be processed by a shaker at a time, the sum of actions 
       required for all outstanding cocktail goals is an admissible lower bound.

    2. Admissible Lower Bounds:
       - Shaker Bottleneck Cost (State-based): For each goal (contains shot cocktail), we count the 
         unavoidable remaining actions (fill-shot, pour-to-shaker, shake, pour-to-shot). 
         Summing these is admissible as shaker-time cannot be shared between different cocktails.
       - Task Network Cost: Each abstract task or action in the network represents a commitment 
         to future work. We sum the minimum decomposition depth for each item.

    3. Guidance/Tie-breaking:
       Small penalties for containers that are not clean or hands that are not empty help guide 
       the search toward 'prepared' states.
    """

    def __init__(self, barman_gemini3flash_heuristic_9=None, name="BarmanBDIGemini3FlashHeuristic9"):
        super().__init__(name=name)
        self.goal_cocktails = []  # List of (shot, cocktail)
        self.cocktail_parts = {}  # cocktail -> [ing1, ing2]
        self.fact_ids = defaultdict(dict)  # pred -> args_tuple -> global_id
        self.ing_in_any_shot_ids = defaultdict(list)  # ing -> [fact_id, ...]
        self.shakers = []
        self.shots = []
        
        self.task_costs = {
            "AchieveContainsShotCocktail": 6,
            "AchieveContainsShakerCocktail": 4,
            "AchieveContainsShakerIngredient": 2,
            "AchieveContainsShotIngredient": 1,
            "DoPourShakerToShot": 1,
            "AchieveCleanShaker": 1,
            "AchieveCleanShot": 1,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1
        }

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map all facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            self.fact_ids[pred][tuple(args)] = fact.global_id
            
            if pred == "cocktailPart1" or pred == "cocktailPart2":
                cocktail, ingredient = args[0], args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = []
                self.cocktail_parts[cocktail].append(ingredient)
            
            if pred == "contains":
                container, beverage = args[0], args[1]
                # We'll identify shots and shakers later, but we can index ingredient-in-shot facts now
                # This will be refined after we know which containers are shots

        # 2. Extract Goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for goal_name in goal_fact_names:
            pred, args = self._parse_fact_name(goal_name)
            if pred == "contains" and len(args) == 2:
                self.goal_cocktails.append((args[0], args[1]))

        # 3. Identify Objects
        # We can infer types from fact signatures or just use the model's structure
        # Here we look for containers that appear in shakerLevel or clean-shaker vs clean-shot
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "shakerLevel":
                if args[0] not in self.shakers: self.shakers.append(args[0])
            if pred == "used" and args[0] not in self.shots:
                # This is a bit heuristic; normally we'd check the type 'shot' from model
                if args[0] not in self.shakers and args[0] not in self.shots: 
                    self.shots.append(args[0])
        
        # If shot list is empty, fill from contains goals
        if not self.shots:
            self.shots = list(set(g[0] for g in self.goal_cocktails))

        # 4. Precompute ingredient-in-shot fact IDs
        for ing_list in self.cocktail_parts.values():
            for ing in ing_list:
                for shot in self.shots:
                    fid = self.fact_ids['contains'].get((shot, ing))
                    if fid is not None:
                        self.ing_in_any_shot_ids[ing].append(fid)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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

    def _get_task_base_name(self, name):
        if '[' in name:
            return name[:name.find('[')]
        return name

    def _compute_state_h(self, state):
        h = 0
        for target_shot, cocktail in self.goal_cocktails:
            # Goal: (contains shot cocktail)
            goal_fid = self.fact_ids['contains'].get((target_shot, cocktail), -1)
            if goal_fid != -1 and (state >> goal_fid) & 1:
                continue
            
            # Need at least one pour to target shot
            h += 1
            
            # Is the cocktail ready in a shaker?
            in_shaker_ready = False
            for shaker in self.shakers:
                c_fid = self.fact_ids['contains'].get((shaker, cocktail), -1)
                s_fid = self.fact_ids['shaked'].get((shaker,), -1)
                if c_fid != -1 and s_fid != -1:
                    if (state >> c_fid) & 1 and (state >> s_fid) & 1:
                        in_shaker_ready = True
                        break
            
            if in_shaker_ready:
                continue
                
            # Need at least one shake
            h += 1
            
            # Ingredients logic
            for ing in self.cocktail_parts.get(cocktail, []):
                # In Shaker?
                ing_in_shaker = False
                for shaker in self.shakers:
                    is_fid = self.fact_ids['contains'].get((shaker, ing), -1)
                    if is_fid != -1 and (state >> is_fid) & 1:
                        ing_in_shaker = True
                        break
                if ing_in_shaker:
                    continue
                
                # Need to pour from shot to shaker
                h += 1
                
                # In any shot?
                ing_in_shot = False
                for fid in self.ing_in_any_shot_ids.get(ing, []):
                    if (state >> fid) & 1:
                        ing_in_shot = True
                        break
                if not ing_in_shot:
                    # Need to fill shot
                    h += 1
        return h

    def _compute_network_h(self, task_network):
        h = 0
        for task in task_network:
            base_name = self._get_task_base_name(task.name)
            h += self.task_costs.get(base_name, 1)
        return h

    def _compute(self, state, task_network):
        state_h = self._compute_state_h(state)
        network_h = self._compute_network_h(task_network)
        return max(state_h, network_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
