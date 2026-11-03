from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict

class BarmanBDIGemini3FlashHeuristic10(Heuristic):
    """
    Domain-dependent heuristic for the Barman_BDI domain.

    Bottleneck Analysis:
    The single shaker (shaker1) is the primary bottleneck. Every cocktail requires a strict sequential process:
    Clean Shaker -> Fill Shot (Ing1) -> Pour Shot to Shaker -> Fill Shot (Ing2) -> Pour Shot to Shaker -> Shake -> Pour to Shot.
    Since there is typically only one shaker, all cocktail goals must be processed largely in serial.

    Admissibility Sketch:
    This heuristic sums the minimum remaining actions for each unsatisfied goal cocktail. 
    Each cocktail goal 'contains(shot, cocktail)' requires a sequence of actions (fill, pour, shake, pour) 
    that are strictly required. Summing these is admissible because the shaker can only process one 
    cocktail at a time, making the total plan length at least the sum of these sequential steps.
    """

    def __init__(self, barman_gemini3flash_heuristic_10=None, name="BarmanBDIGemini3FlashHeuristic10"):
        super().__init__(name=name)
        self.goal_shot_cocktail = {} # shot_name -> cocktail_name
        self.recipe = defaultdict(list) # cocktail_name -> [ing1, ing2]
        self.shakers = []
        self.shots = []
        
        # Fact Maps
        self.contains_id = {} # (container, beverage) -> global_id
        self.clean_id = {} # container -> global_id
        self.empty_id = {} # container -> global_id
        self.shaked_id = {} # shaker -> global_id
        self.unshaked_id = {}
        self.holding_id = {} # (hand, container) -> global_id
        self.hand_empty_id = {} # hand -> global_id
        
        # Reverse maps for efficient lookup
        self.ing_in_shots_ids = defaultdict(list) # ingredient -> [global_ids of contains(shot, ing)]

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse recipe and objects from static facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            
            if pred == "cocktailPart1" or pred == "cocktailPart2":
                self.recipe[args[0]].append(args[1])
            elif pred == "shakerLevel":
                if args[0] not in self.shakers: self.shakers.append(args[0])
            elif pred == "clean":
                if args[0].startswith("shot") and args[0] not in self.shots: self.shots.append(args[0])
                elif args[0].startswith("shaker") and args[0] not in self.shakers: self.shakers.append(args[0])
            
            # Map facts for bitwise checking
            if pred == "contains":
                self.contains_id[(args[0], args[1])] = fact.global_id
            elif pred == "clean":
                self.clean_id[args[0]] = fact.global_id
            elif pred == "empty":
                self.empty_id[args[0]] = fact.global_id
            elif pred == "shaked":
                self.shaked_id[args[0]] = fact.global_id
            elif pred == "unshaked":
                self.unshaked_id[args[0]] = fact.global_id
            elif pred == "holding":
                self.holding_id[(args[0], args[1])] = fact.global_id
            elif pred == "handEmpty":
                self.hand_empty_id[args[0]] = fact.global_id

        # 2. Map goal cocktails to their target shots
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for goal_name in goal_fact_names:
            pred, args = self._parse_fact_name(goal_name)
            if pred == "contains" and args[0].startswith("shot"):
                self.goal_shot_cocktail[args[0]] = args[1]

        # 3. Pre-index ingredient locations in shots
        for cocktail, ingredients in self.recipe.items():
            for ing in ingredients:
                for shot in self.shots:
                    fid = self.contains_id.get((shot, ing))
                    if fid is not None:
                        self.ing_in_shots_ids[ing].append(fid)

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
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts: return parts[0], parts[1:]
        return None, []

    def _is_true(self, state, fact_id):
        return (state >> fact_id) & 1 if fact_id is not None else False

    def _compute(self, state, task_network):
        total_h = 0
        shaker = self.shakers[0] if self.shakers else None
        if not shaker: return 0

        # Check each goal cocktail
        for shot, cocktail in self.goal_shot_cocktail.items():
            # If goal reached, cost is 0
            if self._is_true(state, self.contains_id.get((shot, cocktail))):
                continue
            
            # Default cost to pour from shaker to shot
            h = 1 
            
            # If cocktail is not yet shaked in the shaker
            if not (self._is_true(state, self.contains_id.get((shaker, cocktail))) and self._is_true(state, self.shaked_id.get(shaker))):
                h += 1 # Action: shake
                
                # Cost to get ingredients into shaker
                ingredients = self.recipe.get(cocktail, [])
                for ing in ingredients:
                    if self._is_true(state, self.contains_id.get((shaker, ing))):
                        continue
                    # If ingredient in any shot, need 1 action to pour; else 2 to fill and pour
                    in_shot = any(self._is_true(state, fid) for fid in self.ing_in_shots_ids.get(ing, []))
                    h += 1 if in_shot else 2
                
                # If shaker is not clean/empty, we need to prepare it
                if not self._is_true(state, self.clean_id.get(shaker)):
                    h += 1 # clean-shaker

            # Shot prep costs
            if not self._is_true(state, self.clean_id.get(shot)):
                h += 1
            if not self._is_true(state, self.empty_id.get(shot)):
                h += 1

            total_h += h

        # Tie-breaker: prefer empty hands
        hand_penalty = 0
        for fact_id in self.hand_empty_id.values():
            if not self._is_true(state, fact_id):
                hand_penalty += 0.1
        
        return int(total_h + hand_penalty)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
