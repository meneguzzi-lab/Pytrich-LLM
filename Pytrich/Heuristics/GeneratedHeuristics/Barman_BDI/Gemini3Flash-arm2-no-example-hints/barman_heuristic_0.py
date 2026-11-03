from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic0(Heuristic):
    """
    Domain-Dependent Heuristic for Barman BDI.
    Bottleneck: The Shaker resource. Only one cocktail can be prepared and shaken at a time.
    
    Admissible Bounds:
    1. Unfinished Goal Count: Each unsatisfied cocktail goal requires cleaning, filling, shaking, and pouring (approx 8-10 actions).
    2. Task Network Complexity: Each AchieveContainsShotCocktail task represents a mandatory chain of decompositions.
    
    Symmetry Breaking (Sub-unit penalties):
    - Penalize using goal-shots as buffers (500 per task).
    - symmetry-breaking ordering for working shots (100 * index).
    - Hand preference (50 per non-preferred hand use).
    """
    def __init__(self, barman_gemini3flash_heuristic_0="barman_gemini3flash_heuristic_0"):
        super().__init__(name=barman_gemini3flash_heuristic_0)
        self.goal_contains = {}
        self.goal_shots = set()
        self.working_shots = []
        self.fact_map = {}
        self.preferred_hand = None

    def _parse_name(self, name):
        if not name: return "", []
        n = name[1:] if name.startswith(("+", "-")) else name
        idx = n.find("[")
        if idx == -1: return n, []
        return n[:idx], n[idx+1:-1].split(",")

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_name(f.name)
            if pred not in self.fact_map: self.fact_map[pred] = {}
            self.fact_map[pred][tuple(args)] = f.global_id
            
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            pred, args = self._parse_name(gn)
            if pred == "contains":
                shot, cocktail = args[0], args[1]
                self.goal_contains[shot] = (cocktail, self.fact_map[pred][tuple(args)])
                self.goal_shots.add(shot)

        all_shots = sorted([args[0] for args in self.fact_map.get("empty", {}).keys() if args[0].startswith("shot")])
        self.working_shots = [s for s in all_shots if s not in self.goal_shots]
        
        hands = sorted(list(self.fact_map.get("handEmpty", {}).keys()))
        if hands: self.preferred_hand = hands[0][0]
        
        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        h = 0
        # 1. Goal satisfaction check
        for shot, (cocktail, fact_id) in self.goal_contains.items():
            if not ((state >> fact_id) & 1):
                h += 10

        # 2. Task network and symmetry breaking
        for task in task_network:
            t_name, t_args = self._parse_name(task.name)
            
            if t_name == "AchieveContainsShotCocktail": h += 10
            elif t_name == "AchieveContainsShakerCocktail": h += 6
            elif t_name == "AchieveContainsShakerIngredient": h += 3
            elif t_name == "AchieveContainsShotIngredient": h += 2
            elif t_name == "AchieveHolding":
                hand = t_args[0] if len(t_args) > 0 else None
                if hand and hand != self.preferred_hand: h += 50
                h += 1
            
            # Symmetry breaking penalties from analysis hints
            if t_name in ["AchieveContainsShotIngredient", "AchieveContainsShakerIngredient", "AddIngredientToEmptyShaker", "AddIngredientToUsedShaker"]:
                shot_arg = None
                if t_name == "AchieveContainsShotIngredient": shot_arg = t_args[0]
                elif t_name in ["AddIngredientToEmptyShaker", "AddIngredientToUsedShaker"]: shot_arg = t_args[4] if len(t_args) > 4 else None
                
                if shot_arg:
                    if shot_arg in self.goal_shots:
                        h += 500
                    elif shot_arg in self.working_shots:
                        h += 100 * self.working_shots.index(shot_arg)

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
