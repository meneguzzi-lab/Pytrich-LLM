"""
Domain-dependent heuristic for the Barman_BDI (barman_htn) domain.

Bottleneck Analysis:
The primary bottleneck in this domain is the serial nature of cocktail preparation.
Each cocktail requires a specific sequence: Clean Shaker -> Clean Shot -> Fill Shot (Ing1) -> 
Pour Shot to Shaker -> Clean Shot -> Fill Shot (Ing2) -> Pour Shot to Shaker -> Shake -> 
Pour Shaker to Shot. Since hands and shakers are limited resources, only a few of 
these actions can happen in parallel, effectively making the plan length proportional 
to the sum of actions for each cocktail.

Admissibility Sketches:
1. Cocktail Progress (Term 1): For each cocktail goal (shot, cocktail), we calculate the minimum 
number of mandatory actions remaining (pour, shake, fill, etc.) based on the current state. 
Since each cocktail must eventually occupy a shaker and be poured into a shot, this count is a 
lower bound on the actions required.
2. Task Network Depth (Term 2): The HTN task network length represents remaining abstract tasks 
and operators. Every non-trivial task decomposes into at least one action. This is used for 
guidance and tie-breaking.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic7(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_7=None, name="BarmanBDIGemini3FlashHeuristic7"):
        super().__init__(name=name)
        self.goal_cocktails = [] # List of (shot, cocktail)
        self.cocktail_parts = {}  # cocktail -> (ing1, ing2)
        self.fact_ids = {}        # (predicate, tuple(args)) -> global_id
        self.shakers = []
        self.shots = []
        self.hands = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map all facts to IDs for bitwise checking
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_ids[(pred, tuple(args))] = fact.global_id
                if pred == "shaker": self.shakers.append(args[0])
                if pred == "shot": self.shots.append(args[0])
                if pred == "hand": self.hands.append(args[0])

        # 2. Extract static cocktail recipes from initial state or facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "cocktailPart1":
                cocktail, ing = args[0], args[1]
                if cocktail not in self.cocktail_parts: self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][0] = ing
            elif pred == "cocktailPart2":
                cocktail, ing = args[0], args[1]
                if cocktail not in self.cocktail_parts: self.cocktail_parts[cocktail] = [None, None]
                self.cocktail_parts[cocktail][1] = ing

        # 3. Determine goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for goal_name in goal_fact_names:
            pred, args = self._parse_fact_name(goal_name)
            if pred == "contains" and len(args) == 2:
                # Check if it's a (shot, cocktail) goal
                # This is a bit simplified, usually identified by type in HDD
                if args[0] in self.shots and args[1] in self.cocktail_parts:
                    self.goal_cocktails.append((args[0], args[1]))

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

    def _is_true(self, state_bitwise: int, predicate: str, *args):
        fid = self.fact_ids.get((predicate, tuple(args)))
        if fid is not None:
            return (state_bitwise >> fid) & 1
        return False

    def _compute(self, state: int, task_network) -> int:
        total_cost = 0

        # Term 1: Serial Cocktail Completion Cost
        for shot, cocktail in self.goal_cocktails:
            if self._is_true(state, "contains", shot, cocktail):
                continue

            # Minimum cost to finish this cocktail
            c_cost = 1 # The final pour-shaker-to-shot is always required

            # Is the cocktail already prepared in any shaker?
            in_shaker = False
            for shaker in self.shakers:
                if self._is_true(state, "contains", shaker, cocktail):
                    in_shaker = True
                    if not self._is_true(state, "shaked", shaker):
                        c_cost += 1 # Needs shaking
                    break
            
            if not in_shaker:
                # Must prepare in shaker
                c_cost += 2 # shake + pour-shaker
                ings = self.cocktail_parts.get(cocktail, [])
                for ing in ings:
                    if ing is None: continue
                    
                    # Is ingredient in a shaker?
                    ing_in_shaker = False
                    for shaker in self.shakers:
                        if self._is_true(state, "contains", shaker, ing):
                            ing_in_shaker = True
                            break
                    
                    if not ing_in_shaker:
                        c_cost += 1 # pour-shot-to-shaker
                        # Is ingredient in a shot?
                        ing_in_shot = False
                        for s in self.shots:
                            if self._is_true(state, "contains", s, ing):
                                ing_in_shot = True
                                break
                        if not ing_in_shot:
                            c_cost += 1 # fill-shot
                            # If no shot is clean, we'll eventually need to clean one
                            any_clean_shot = any(self._is_true(state, "clean", s) for s in self.shots)
                            if not any_clean_shot:
                                c_cost += 1 # clean-shot
                
                # If shaker is used and doesn't contain a component, it must be cleaned
                # We'll just check if any shaker is clean/empty
                any_ready_shaker = any(self._is_true(state, "clean", sh) and self._is_true(state, "empty", sh) for sh in self.shakers)
                if not any_ready_shaker:
                    c_cost += 1

            total_cost += c_cost

        # Term 2: Hand logistics
        # If hands are full, it might cost actions to empty them
        hands_full = 0
        for h in self.hands:
            if not self._is_true(state, "handEmpty", h):
                hands_full += 1
        
        # Guidance term: Task Network size
        # Each task in the network is a unit of work
        tn_cost = len(task_network)

        return int(total_cost + (0.1 * hands_full) + (0.01 * tn_cost))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
