from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class BarmanBDIGemini3FlashHeuristic11(Heuristic):
    """
    Domain-Dependent Heuristic for Barman_BDI.
    
    Bottleneck: Parameter-binding symmetry among multiple interchangeable shot glasses 
    and hands serializes the construction of cocktails.
    
    Admissibility Sketches:
    1. Cocktail-State Lower Bound: For each cocktail goal, it must at minimum be filled, 
       shaked, and poured. If it is not even in the shaker, at least 4 actions (2 fills, 
       1 shake, 1 pour) are required. This is a lower bound as it ignores cleaning and hand-swapping.
    2. Task Network Length: Every task in the network must be either decomposed or executed.
       A flat count of remaining tasks is a simple admissible lower bound if each step costs >= 1.
    3. Symmetry Breaking: Large penalties are added for using goal shots as intermediaries 
       or using non-preferred hands. These are not strictly part of the physical cost but 
       drastically reduce the branching factor.
    """

    def __init__(self, barman_gemini3flash_heuristic_11="barman_bdi_heuristic_11"):
        super().__init__(barman_gemini3flash_heuristic_11)
        self.goal_shots = set()
        self.cocktail_goals = {} # cocktail -> target_shot
        self.ingredients_for_cocktail = {} # cocktail -> [ing1, ing2]
        self.shot_ids = {} # shot_name -> int index
        self.hand_ids = {} # hand_name -> int index
        self.fact_map = {} # predicate -> arg_tuple -> global_id
        self.pref_hand = None

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
        # 1. Parse facts and build mapping
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred:
                if pred not in self.fact_map: self.fact_map[pred] = {}
                self.fact_map[pred][tuple(args)] = f.global_id
            
            if pred == "cocktailPart1":
                c, i = args[0], args[1]
                if c not in self.ingredients_for_cocktail: self.ingredients_for_cocktail[c] = []
                self.ingredients_for_cocktail[c].append(i)
            elif pred == "cocktailPart2":
                c, i = args[0], args[1]
                if c not in self.ingredients_for_cocktail: self.ingredients_for_cocktail[c] = []
                self.ingredients_for_cocktail[c].append(i)

        # 2. Identify goal shots
        goal_names = model.state_explicit_repr(model.goals)
        for g in goal_names:
            pred, args = self._parse_fact_name(g)
            if pred == "contains":
                self.goal_shots.add(args[0])
                self.cocktail_goals[args[1]] = args[0]

        # 3. Categorize objects for symmetry breaking
        all_shots = sorted([args[0] for args in self.fact_map.get('ontable', {}).keys() if 'shot' in args[0]])
        self.shot_ids = {s: i for i, s in enumerate(all_shots)}
        
        all_hands = sorted(list(set([args[0] for args in self.fact_map.get('handEmpty', {}).keys()])))
        if all_hands: 
            self.pref_hand = all_hands[0]
            self.hand_ids = {h: i for i, h in enumerate(all_hands)}

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _is_true(self, state, pred, args):
        if pred not in self.fact_map: return False
        f_id = self.fact_map[pred].get(tuple(args))
        if f_id is None: return False
        return (state >> f_id) & 1

    def _compute(self, state, task_network):
        if not task_network: return 0
        
        h = len(task_network)
        
        # Cocktail Progress Lower Bound
        for cocktail, target_shot in self.cocktail_goals.items():
            if self._is_true(state, "contains", [target_shot, cocktail]):
                continue
            
            # Check shaker status
            shaker_has = False
            shaker_shaked = False
            # Optimization: assumes shaker1 since most problems use one shaker
            for shaker_args in self.fact_map.get('shaked', {}).keys():
                shaker = shaker_args[0]
                if self._is_true(state, "contains", [shaker, cocktail]):
                    shaker_has = True
                    if self._is_true(state, "shaked", [shaker]):
                        shaker_shaked = True
                    break
            
            if shaker_shaked:
                h += 2 # Pour + 1
            elif shaker_has:
                h += 3 # Shake + Pour + 1
            else:
                h += 5 # Fill1 + Fill2 + Shake + Pour + 1

        # Symmetry Breaking Penalties from Hints
        top_task = task_network[0]
        name = top_task.name
        bracket_idx = name.find("[")
        if bracket_idx > 0:
            args = name[bracket_idx+1:-1].split(",")
            
            # Grounded args[0] is often the SHOT per hint
            # Penalize using goal shots as measurement intermediaries
            # We check if the task is a measurement task (fill-shot, pour-shot, etc)
            is_measurement = any(x in name for x in ["fill-shot", "pour-shot", "clean-shot", "AchieveContainsShotIngredient"])
            if is_measurement and args and args[0] in self.goal_shots:
                # If it's not the goal shot for the specific cocktail we want, penalty
                # However, hint suggests a flat large penalty for using goal shots as intermediaries
                h += 500
            
            # Hand preference: 50
            if self.pref_hand and "hand" in name.lower():
                # Find which arg is the hand
                for a in args:
                    if a in self.hand_ids and a != self.pref_hand:
                        h += 50
                        break
            
            # Working-shot ordering: 100 * index
            if args and args[0] in self.shot_ids and args[0] not in self.goal_shots:
                h += 100 * self.shot_ids[args[0]]

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
