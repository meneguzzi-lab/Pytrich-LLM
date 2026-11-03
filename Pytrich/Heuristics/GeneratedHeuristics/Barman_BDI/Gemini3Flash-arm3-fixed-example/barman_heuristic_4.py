from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict

"""
Domain-dependent heuristic for the Barman_BDI domain.

1. Bottleneck Analysis:
   The shaker is the central bottleneck. Every cocktail requires a series of sequential steps: 
   cleaning the shaker, filling it with two specific ingredients (which themselves require 
   filling shots and pouring), shaking, and finally pouring into a clean shot. 
   Since usually only one shaker exists, these processes cannot happen in parallel for different cocktails.

2. Admissible Lower Bounds:
   - Cocktail Progress Cost: For each goal cocktail (shot, cocktail), we estimate the minimal number of 
     actions remaining based on its current location and state (dispenser -> shot -> shaker -> shaken -> target shot).
   - Task Network Depth: The number of tasks remaining in the network provides a lower bound on 
     the structural work remaining in the HTN decomposition.

3. Admissibility Sketches:
   - Cocktail Progress: Each missing step (fill, pour, shake, grasp) is a mandatory physical action.
     The cost increments (2, 4, 6, 8) are conservative lower bounds for these sequences.
   - Task Network: Each task in the network must eventually be decomposed into at least one action 
     (unless it's a null-method task, which is handled by the state-based term).
"""

class BarmanBDIGemini3FlashHeuristic4(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_4=None, name="BarmanBDIGemini3FlashHeuristic4"):
        super().__init__(name=name)
        self.goal_cocktails = [] # List of (shot, cocktail)
        self.cocktail_ingredients = {} # cocktail -> (ing1, ing2)
        self.shakers = set()
        self.shots = set()
        
        # Fact mapping: predicate -> args_tuple -> global_id
        self.fact_ids = defaultdict(dict)

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
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self, model: Model):
        # 1. Identify goal requirements
        if hasattr(model, 'goals') and model.goals:
            goal_fact_names = model.state_explicit_repr(model.goals)
            for name in goal_fact_names:
                pred, args = self._parse_fact_name(name)
                if pred == "contains" and len(args) == 2:
                    self.goal_cocktails.append((args[0], args[1]))

        # 2. Extract domain structure from all grounded facts
        ing_parts = defaultdict(lambda: [None, None])
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            
            # Store fact ID for fast bitwise lookup
            self.fact_ids[pred][tuple(args)] = fact.global_id

            # Identify ingredients for cocktails
            if pred == "cocktailPart1":
                ing_parts[args[0]][0] = args[1]
            elif pred == "cocktailPart2":
                ing_parts[args[0]][1] = args[1]
            
            # Identify shakers and shots
            if pred == "shakerLevel":
                self.shakers.add(args[0])
            elif pred == "clean":
                # Initially we don't know if it's a shot or shaker, but shakerLevel helps above
                pass
            elif pred == "ontable":
                if args[0].startswith("shot"):
                    self.shots.add(args[0])
                elif args[0].startswith("shaker"):
                    self.shakers.add(args[0])

        self.cocktail_ingredients = {c: tuple(p) for c, p in ing_parts.items() if None not in p}

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess(model)
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state: int, task_network) -> int:
        # 1. Cocktail Progress Cost (State-based)
        cocktail_h = 0
        for shot, cocktail in self.goal_cocktails:
            # If goal achieved, skip
            goal_fact_id = self.fact_ids['contains'].get((shot, cocktail))
            if goal_fact_id is not None and (state >> goal_fact_id) & 1:
                continue
            
            # Step-wise cost calculation
            cost = 8 # Default: ingredients in dispensers
            
            # Check if shaked in a shaker
            found_shaked = False
            for shaker in self.shakers:
                cont_id = self.fact_ids['contains'].get((shaker, cocktail))
                shaked_id = self.fact_ids['shaked'].get((shaker,))
                if cont_id is not None and (state >> cont_id) & 1:
                    if shaked_id is not None and (state >> shaked_id) & 1:
                        found_shaked = True
                        break
            
            if found_shaked:
                cost = 2 # Just need to grasp and pour
            else:
                # Check ingredients in shaker
                ings = self.cocktail_ingredients.get(cocktail)
                if ings:
                    ings_in = 0
                    for shaker in self.shakers:
                        for ing in ings:
                            ing_id = self.fact_ids['contains'].get((shaker, ing))
                            if ing_id is not None and (state >> ing_id) & 1:
                                ings_in += 1
                    if ings_in == 2:
                        cost = 4 # Shake + Pour
                    elif ings_in == 1:
                        cost = 6 # Fill 1 + Pour 1 + Shake + Pour
            
            cocktail_h += cost

        # 2. Task Network Weight
        tn_h = len(task_network)

        # 3. Hand Constraints (Tie-breaking)
        # If hands are full but we need to grasp something else, add small penalty
        hand_penalty = 0
        for hand in ["left", "right"]:
            hand_empty_id = self.fact_ids['handEmpty'].get((hand,))
            if hand_empty_id is not None and not ((state >> hand_empty_id) & 1):
                hand_penalty += 0.1

        return max(cocktail_h, tn_h) + int(hand_penalty)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
