import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Barman_BDI (HDDL) domain.

1. Bottleneck Identification:
   The Shaker is the primary serial bottleneck. Making a cocktail requires a specific sequence of state transitions
   (Shot-Filling -> Shaker-Pouring -> Shaking -> Shot-Pouring). Since only one cocktail can be in a shaker
   at a time, and a limited number of shakers (usually one) exist, the sequential operations on the shaker
   dominate the plan length.

2. Admissible Lower Bounds:
   - Term 1: Cocktail Progress Cost (Admissible).
     Each cocktail goal 'contains(shot, cocktail)' requires a set of actions depending on its current state.
     - If the cocktail is in the target shot: 0 actions.
     - If the cocktail is in the shaker and shaked: 1 action (pour-shaker-to-shot).
     - If both ingredients are in the shaker: 2 actions (shake, pour-shaker-to-shot).
     - If only one ingredient is in the shaker: 4 actions (fill-shot-2, pour-shot-2, shake, pour-shaker-to-shot).
     - If no ingredients are in the shaker: 6 actions (fill-shot-1, pour-shot-1, fill-shot-2, pour-shot-2, shake, pour-shaker-to-shot).
     This term is a lower bound because these physical actions are strictly necessary.

   - Term 2: Task Network Weight (Guidance).
     Every task in the HTN task network represents at least one future operator (often many more).
     Adding a small value per task helps guide the search toward simpler task networks.

3. Preprocessing:
   - Fact names are parsed once to map (predicate, args) to global_ids.
   - Cocktail recipes (part1, part2) are extracted from initial facts.
   - Goal conditions (shot, cocktail) are extracted from model.goals.

4. Bitwise Efficiency:
   - State checks use (state_bitwise >> global_id) & 1 for O(1) lookups.
"""

class BarmanBDIGemini3FlashHeuristic11(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_11=None, name="BarmanBDIGemini3FlashHeuristic11"):
        super().__init__(name=name)
        self.fact_ids = {} # (predicate, args_tuple) -> global_id
        self.goal_cocktails = [] # List of (shot, cocktail)
        self.recipes = {} # cocktail -> (ing1, ing2)
        self.shakers = []
        self.shots = []
        self.ingredients = []

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
        
        # 1. Map Facts
        temp_cocktail_part1 = {}
        temp_cocktail_part2 = {}

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_ids[(pred, tuple(args))] = fact.global_id
                
                if pred == "cocktailPart1":
                    temp_cocktail_part1[args[0]] = args[1]
                elif pred == "cocktailPart2":
                    temp_cocktail_part2[args[0]] = args[1]
                
                # Collect objects by type from fact arguments
                # This is a bit loose but effective for Barman where types are usually in names
                for arg in args:
                    if arg.startswith("shaker") and arg not in self.shakers: self.shakers.append(arg)
                    if arg.startswith("shot") and arg not in self.shots: self.shots.append(arg)
                    if arg.startswith("ingredient") and arg not in self.ingredients: self.ingredients.append(arg)

        # 2. Build Recipes
        for cocktail in temp_cocktail_part1:
            if cocktail in temp_cocktail_part2:
                self.recipes[cocktail] = (temp_cocktail_part1[cocktail], temp_cocktail_part2[cocktail])

        # 3. Extract Goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_names:
                pred, args = self._parse_fact_name(g_name)
                if pred == "contains" and len(args) == 2:
                    # Usually (shot, cocktail)
                    self.goal_cocktails.append((args[0], args[1]))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        h = 0
        
        # Term 1: Cocktail Progress (Admissible estimate of actions)
        for shot, cocktail in self.goal_cocktails:
            # Goal: (contains shot cocktail)
            cid = self.fact_ids.get(("contains", (shot, cocktail)))
            if cid is not None and (state >> cid) & 1:
                continue # Goal satisfied for this shot
            
            # If not in shot, check the progress across all shakers
            best_cocktail_cost = 7 # Cost if we haven't started
            
            ing1, ing2 = self.recipes.get(cocktail, (None, None))
            if not ing1:
                h += 5 # Fallback if recipe unknown
                continue

            for shaker in self.shakers:
                # Is cocktail shaked in this shaker?
                shaked_id = self.fact_ids.get(("shaked", (shaker,)))
                cont_cocktail_id = self.fact_ids.get(("contains", (shaker, cocktail)))
                
                if shaked_id is not None and cont_cocktail_id is not None:
                    if ((state >> shaked_id) & 1) and ((state >> cont_cocktail_id) & 1):
                        best_cocktail_cost = min(best_cocktail_cost, 1)
                        break # Optimal progress for shaker found
                
                # Are ingredients in shaker?
                c1id = self.fact_ids.get(("contains", (shaker, ing1)))
                c2id = self.fact_ids.get(("contains", (shaker, ing2)))
                
                has_ing1 = (c1id is not None) and ((state >> c1id) & 1)
                has_ing2 = (c2id is not None) and ((state >> c2id) & 1)
                
                if has_ing1 and has_ing2:
                    best_cocktail_cost = min(best_cocktail_cost, 2)
                elif has_ing1 or has_ing2:
                    best_cocktail_cost = min(best_cocktail_cost, 4)
                else:
                    # Shaker is empty or contains wrong stuff. 
                    # Check if shaker is clean (needs cleaning if used)
                    clean_id = self.fact_ids.get(("clean", (shaker,)))
                    if clean_id is not None and not ((state >> clean_id) & 1):
                        best_cocktail_cost = min(best_cocktail_cost, 8) # Extra cost for cleaning
                    else:
                        best_cocktail_cost = min(best_cocktail_cost, 6)
            
            h += best_cocktail_cost

        # Term 2: Task Network Guidance
        # Each task in HTN implies at least 1 action, usually more.
        h += len(task_network) * 1.5

        # Tie-breaker: Hand status
        # Having hands empty is generally better for starting new tasks
        for hand in ['left', 'right']:
            hid = self.fact_ids.get(("handEmpty", (hand,)))
            if hid is not None and not ((state >> hid) & 1):
                h += 0.1

        return int(h)

    def __call__(self, parent_node, node):
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
