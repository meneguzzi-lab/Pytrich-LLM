from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

"""
Domain-dependent heuristic for the Barman_BDI domain.

Bottleneck Analysis:
The shaker is the primary serializing resource. Only one cocktail can be prepared 
at a time because the shaker must be cleaned, filled with specific ingredients, 
shaked, and then poured. Each cocktail requires a minimum of 6 actions: 
2x (fill-shot + pour-shot) + shake + pour-shaker-to-shot.

Admissibility Sketches:
1. Cocktail Completion Term: For each cocktail goal not yet in its target shot, 
   we count the remaining actions in the shaker-sequence (cleaning, ingredient 
   delivery, shaking, pouring). Since only one cocktail uses the shaker at a time, 
   this sum is a lower bound on the total plan length.
2. Container Maintenance: Dirty shots and shakers must be cleaned before they 
   can be reused. Each cleaning is at least 1 action.
"""

class BarmanBDIGemini3FlashHeuristic1(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_1=None, name="BarmanBDIGemini3FlashHeuristic1"):
        super().__init__(name=name)
        self.barman_gemini3flash_heuristic_1 = barman_gemini3flash_heuristic_1
        self.cocktail_parts = {}  # cocktail -> [ingr1, ingr2]
        self.goal_cocktails = []  # List of (shot, cocktail)
        self.fact_ids = {} # predicate -> args_tuple -> global_id
        self.shakers = []
        self.shots = []

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
        # Precompute static data and fact IDs
        temp_parts = {}
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            # Store fact IDs for fast lookup
            self.fact_ids.setdefault(pred, {})[tuple(args)] = fact.global_id
            
            if pred == "cocktailPart1":
                temp_parts.setdefault(args[0], [None, None])[0] = args[1]
            elif pred == "cocktailPart2":
                temp_parts.setdefault(args[0], [None, None])[1] = args[1]
            elif pred == "shakerLevel":
                if args[0] not in self.shakers: self.shakers.append(args[0])
            elif pred == "ontable" and "shot" in args[0]:
                if args[0] not in self.shots: self.shots.append(args[0])

        self.cocktail_parts = {k: v for k, v in temp_parts.items() if v[0] and v[1]}
        
        # Extract goals
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "contains" and len(args) == 2:
                self.goal_cocktails.append((args[0], args[1]))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _is_true(self, state, pred, args):
        fid = self.fact_ids.get(pred, {}).get(tuple(args))
        if fid is None: return False
        return (state >> fid) & 1

    def _compute(self, state, task_network):
        total_h = 0
        
        # 1. State-based Cocktail Progress (Admissible sum over serial bottleneck)
        # For each goal not satisfied, estimate the remaining shaker-related actions.
        for shot, cocktail in self.goal_cocktails:
            if self._is_true(state, "contains", [shot, cocktail]):
                continue
            
            # Minimum cost to get a cocktail to a shot
            cost = 0
            shaker = self.shakers[0] if self.shakers else None
            
            if shaker:
                if self._is_true(state, "contains", [shaker, cocktail]):
                    if self._is_true(state, "shaked", [shaker]):
                        cost = 1 # pour-shaker-to-shot
                    else:
                        cost = 2 # shake + pour
                else:
                    # Ingredients check
                    parts = self.cocktail_parts.get(cocktail, [])
                    needed_ingr_count = 0
                    for ingr in parts:
                        if not self._is_true(state, "contains", [shaker, ingr]):
                            needed_ingr_count += 1
                    
                    # Each missing ingredient is fill-shot (1) + pour-shot (1) = 2 actions
                    # Then shake (1) + pour (1)
                    cost = (needed_ingr_count * 2) + 2
                    
                    # Check if shaker needs cleaning to start this cocktail
                    if not self._is_true(state, "clean", [shaker]) and not self._is_true(state, "empty", [shaker]):
                        cost += 1
            else:
                cost = 6 # Default fallback
            
            # Shot cleaning cost
            if not self._is_true(state, "clean", [shot]):
                cost += 1
            
            total_h += cost

        # 2. Hand availability (Minor constraint)
        # If we need to clean or pour, we need at least one hand empty or holding the target.
        # This is usually covered by the action costs above, but we ensure it's not zero if tasks remain.
        if task_network and total_h == 0:
            total_h = len(task_network)

        # 3. Symmetry breaking (Tie-breaker only)
        # Penalize dirty containers slightly to encourage cleaning early.
        dirty_penalty = 0
        for s in self.shots:
            if not self._is_true(state, "clean", [s]):
                dirty_penalty += 0.01
        
        return int(total_h) + (0 if total_h == 0 else dirty_penalty)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return int(h_val)
