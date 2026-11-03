from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, AbstractTask, Operator
import re

class BarmanBDIGemini3FlashHeuristic19(Heuristic):
    """
    Domain-dependent heuristic for Barman_BDI.
    Bottleneck analysis: The shaker is the central serial resource. Every cocktail must be made one-by-one 
    using a sequence of: cleaning, filling shots, pouring to shaker, shaking, and pouring to the goal shot.
    
    Admissible lower bounds:
    1. cocktail_completion_cost: For each goal cocktail, count the minimum actions (fill, pour, shake, etc.) 
       remaining based on the current container status. Summed over all cocktails in the task network.
    2. container_cleaning_cost: A container (shot or shaker) that is not clean must be cleaned before use.
    3. hand_management: Tie-breaker for maintaining free hands.
    """

    def __init__(self, barman_gemini3flash_heuristic_19="BarmanBDIGemini3FlashHeuristic19"):
        super().__init__(barman_gemini3flash_heuristic_19)
        self.fids = {}
        self.cocktail_parts = {}
        self.shot_names = []
        self.shaker = "shaker1"
        self.ingredients = []
        self.hands = []
        self.ing_in_shot_fids = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # Map predicates and arguments to global IDs
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred not in self.fids:
                self.fids[pred] = {}
            self.fids[pred][tuple(args)] = f.global_id

        # Identify objects and cocktail recipes
        all_objects = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            all_objects.update(args)
            if pred == "cocktailPart1":
                if args[0] not in self.cocktail_parts: self.cocktail_parts[args[0]] = [None, None]
                self.cocktail_parts[args[0]][0] = args[1]
            if pred == "cocktailPart2":
                if args[0] not in self.cocktail_parts: self.cocktail_parts[args[0]] = [None, None]
                self.cocktail_parts[args[0]][1] = args[1]

        self.shot_names = [o for o in all_objects if "shot" in o.lower()]
        shakers = [o for o in all_objects if "shaker" in o.lower()]
        self.shaker = shakers[0] if shakers else "shaker1"
        self.ingredients = [o for o in all_objects if "ingredient" in o.lower()]
        self.hands = [o for o in all_objects if "hand" in o.lower() or "left" in o.lower() or "right" in o.lower()]

        # Pre-map ingredient-in-shot FIDs for faster checks
        for ing in self.ingredients:
            self.ing_in_shot_fids[ing] = [self.get_fid("contains", (s, ing)) for s in self.shot_names]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        if not task_network: return 0
        h = 0
        for task in task_network:
            if hasattr(task, 'decompositions'): # AbstractTask
                name, args = self._parse_task_name(task.name)
                if name == "AchieveContainsShotCocktail":
                    h += self._cocktail_cost(state, args[1], args[0])
                    if not self._check(state, "clean", (args[0],)): h += 1
                elif name == "AchieveContainsShakerCocktail":
                    h += max(0, self._cocktail_cost(state, args[1], None) - 1)
                elif name == "AchieveCleanShaker":
                    if not self._check(state, "clean", (args[0],)): h += 1
                elif name == "AchieveCleanShot":
                    if not self._check(state, "clean", (args[0],)): h += 1
                elif name == "AchieveHolding":
                    if not self._check(state, "holding", (args[0], args[1])): h += 1
                elif name == "AchieveHandEmpty":
                    if not self._check(state, "handEmpty", (args[0],)): h += 1
                else:
                    h += 1
            else: # Operator
                h += task.cost
        
        # Shaker cleaning requirement
        if h > 0 and not self._check(state, "clean", (self.shaker,)): h += 1

        # Tie-breaker for hands
        hands_full = sum(1 for hand in self.hands if not self._check(state, "handEmpty", (hand,)))
        h += 0.001 * hands_full

        return int(h) if h > 1000 else round(h, 3)

    def _cocktail_cost(self, state, cocktail, shot):
        if shot and self._check(state, "contains", (shot, cocktail)): return 0
        if self._check(state, "shaked", (self.shaker,)) and self._check(state, "contains", (self.shaker, cocktail)): return 1
        
        cost = 2 # shake + pour-to-shot
        parts = self.cocktail_parts.get(cocktail, [None, None])
        for ing in parts:
            if not ing: continue
            if not self._check(state, "contains", (self.shaker, ing)):
                cost += 1 # pour-shot-to-shaker
                if not self._ing_in_any_shot(state, ing):
                    cost += 1 # fill-shot
        return cost

    def _ing_in_any_shot(self, state, ing):
        for fid in self.ing_in_shot_fids.get(ing, []):
            if fid != -1 and (state >> fid) & 1: return True
        return False

    def _check(self, state, pred, args):
        fid = self.get_fid(pred, args)
        return fid != -1 and (state >> fid) & 1

    def get_fid(self, pred, args):
        return self.fids.get(pred, {}).get(tuple(args), -1)

    def _parse_fact_name(self, name: str):
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return predicate, args
        return name, []

    def _parse_task_name(self, task_name):
        parts = task_name.replace('[', ' ').replace(']', ' ').replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
        if not parts: return "", []
        return parts[0], parts[1:]
