from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic14(Heuristic):
    """
    Domain-Dependent Heuristic for Barman_BDI.
    
    Bottleneck Analysis:
    The central bottleneck in Barman is the Shaker and Hand availability. Every cocktail requires 
    a sequence of roughly 15-20 actions: cleaning the shaker, cleaning/filling/pouring shots for 
    at least two ingredients, shaking, and finally pouring. Since there are only 2 hands, 
    holding one container often blocks the ability to clean or fill others, leading to a serial 
    dependency on the shaker resource.

    Admissibility Sketch:
    1. Task Cost Term: Each task in the network (e.g., AchieveContainsShotCocktail) is assigned a 
       realistic lower bound of the actions required for its decomposition (e.g., 20). 
       This is a lower bound because it ignores potential costs of cleaning or hand management 
       already true in the state. State-aware checks set these costs to 0 if the condition is met.
    2. Symmetry Breaking: Penalties for using goal shots as intermediaries (500), non-preferred hands (50),
       and increasing costs for identical shots (100 * index) are added to guide search. 
       While large penalties technically deviate from pure A* admissibility, they are crucial for 
       navigating the vast symmetry of Barman instances.
    """

    def __init__(self, barman_gemini3flash_heuristic_14="barman_gemini3flash_heuristic_14"):
        super().__init__(barman_gemini3flash_heuristic_14)
        self.fact_map = {}
        self.goal_shots = set()
        self.shot_ids = {}
        self.preferred_hand = None

    def _parse_name(self, name: str):
        """Parses grounded name format: predicate[arg1,arg2]"""
        if "[" not in name: return name, []
        parts = name.split("[")
        pred = parts[0]
        if pred.startswith("+") or pred.startswith("-"): pred = pred[1:]
        args = parts[1][:-1].split(",")
        return pred, [a.strip() for a in args]

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self.fact_map = {f.name: f.global_id for f in model.facts}
        
        # Identify goal shots (where a cocktail must eventually end up)
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            p, a = self._parse_name(g_name)
            if p == "contains" and a:
                # In Barman goals, shot is usually args[0]
                self.goal_shots.add(a[0])
        
        # Identify all shots for symmetry breaking
        all_shots = []
        all_hands = []
        # Look through facts to find shots and hands
        for f in model.facts:
            p, a = self._parse_name(f.name)
            if p == "ontable":
                for arg in a:
                    if "shot" in arg.lower(): all_shots.append(arg)
            if p == "handEmpty":
                for arg in a:
                    if "hand" in arg.lower() or arg.lower() in ["left", "right"]: all_hands.append(arg)
        
        all_shots = sorted(list(set(all_shots)))
        for i, shot in enumerate(all_shots):
            self.shot_ids[shot] = i
            
        all_hands = sorted(list(set(all_hands)))
        if all_hands: 
            self.preferred_hand = all_hands[0] # Pick canonical preferred hand

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _is_true(self, state, pred, args):
        fname = f"+{pred}[{','.join(args)}]"
        fid = self.fact_map.get(fname)
        if fid is not None:
            return (state >> fid) & 1
        return 0

    def _compute(self, state, task_network):
        h = 0
        for task in task_network:
            p, a = self._parse_name(task.name)
            
            # --- 1. Task Progress Costs (State-Aware) ---
            cost = 0
            if p == "AchieveContainsShotCocktail":
                if not self._is_true(state, "contains", a): cost = 20
            elif p == "AchieveContainsShakerCocktail":
                if not self._is_true(state, "contains", a): cost = 12
            elif p == "AchieveContainsShakerIngredient":
                if not self._is_true(state, "contains", a): cost = 5
            elif p == "AchieveCleanShaker":
                if not self._is_true(state, "clean", a): cost = 3
            elif p == "AchieveCleanShot":
                if not self._is_true(state, "clean", a): cost = 3
            elif p == "AchieveHolding":
                if not self._is_true(state, "holding", a): cost = 2
            elif p == "AchieveHandEmpty":
                if not self._is_true(state, "handEmpty", a): cost = 1
            else:
                cost = 1 # Generic action cost
            
            h += cost
            
            # --- 2. Symmetry Breaking & Domain Penalties ---
            
            # A. Goal-Shot Penalty (using goal shots as intermediaries)
            # If we are doing 'AddIngredientToShot' or similar tasks on a shot that is someone's goal
            if p in ["AchieveContainsShotIngredient", "AddIngredientToShot", "fill-shot"]:
                if a and a[0] in self.goal_shots:
                    h += 500
            
            # B. Working-Shot Ordering (Breaking ties between identical shots)
            for arg in a:
                if arg in self.shot_ids:
                    h += 100 * self.shot_ids[arg]
            
            # C. Hand Preference (Halving branching by preferring one hand)
            if p == "AchieveHolding" and a:
                # args are [hand, container]
                if self.preferred_hand and a[0] != self.preferred_hand:
                    h += 50
                    
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
