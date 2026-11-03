from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic2(Heuristic):
    """
    Heuristic for Barman_BDI domain.
    
    Bottleneck Analysis:
    The main bottleneck is the serial coordination of the single shaker and hands to process multi-step 
    cocktail recipes. Every cocktail requires the shaker to be cleaned, ingredients to be transferred 
    via shots, shaken, and then poured into clean target shots. 

    Admissibility Sketches:
    1. Goal-Progress Term: Each unsatisfied goal cocktail in a target shot requires at least 
       10-15 actions (clean shaker, fill shot1, pour shot1, fill shot2, pour shot2, shake, pour to shot).
    2. Task Network Complexity: Sum of minimum decomposition costs for active tasks in the network.
    
    Symmetry Breaking:
    - Goal-shot penalty (500) to prevent using target shots for other purposes.
    - Index-based penalty (100 * shot_idx) to force selection of identical shots in order.
    - Hand preference (50) to prefer the left hand over the right hand for coordination.
    """

    def __init__(self, barman_gemini3flash_heuristic_2="barman_gemini3flash_heuristic_2"):
        super().__init__(barman_gemini3flash_heuristic_2)
        self.goal_fids = []
        self.shot_to_idx = {}
        self.hands = []
        self.preferred_hand = None
        self.shot_is_goal_for = {} 
        self.cocktails = set()
        self.ingredients = set()
        self.cocktail_ingredients = {}
        self.task_costs = {
            "AchieveContainsShotCocktail": 15,
            "AchieveContainsShakerCocktail": 10,
            "AchieveContainsShakerIngredient": 5,
            "AchieveContainsShotIngredient": 4,
            "AchieveCleanShaker": 3,
            "AchieveCleanShot": 3,
            "AchieveHolding": 2,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 2
        }

    def _parse_fact_name(self, name: str):
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        all_shots = set()
        self.hands = []
        
        for fact in model.facts:
            p, args = self._parse_fact_name(fact.name)
            if p == "ontable" and "shot" in args[0]:
                all_shots.add(args[0])
            if p == "handEmpty":
                self.hands.append(args[0])
            if p == "cocktailPart1" or p == "cocktailPart2":
                self.cocktails.add(args[0])
                self.ingredients.add(args[1])
                if args[0] not in self.cocktail_ingredients:
                    self.cocktail_ingredients[args[0]] = []
                self.cocktail_ingredients[args[0]].append(args[1])

        self.hands.sort()
        if self.hands:
            self.preferred_hand = self.hands[0]
        
        sorted_shots = sorted(list(all_shots))
        self.shot_to_idx = {shot: i for i, shot in enumerate(sorted_shots)}

        self.goal_fids = []
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            p, args = self._parse_fact_name(g_name)
            if p == "contains" and args[0] in self.shot_to_idx:
                fid = -1
                for fact in model.facts:
                    if fact.name == g_name:
                        fid = fact.global_id
                        break
                if fid != -1:
                    self.goal_fids.append(fid)
                    self.shot_is_goal_for[args[0]] = args[1]

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        h_goal = sum(15 for fid in self.goal_fids if not ((state >> fid) & 1))
        
        h_tn = 0
        penalty = 0
        
        for task in task_network:
            t_name, t_args = self._parse_fact_name(task.name)
            h_tn += self.task_costs.get(t_name, 1)
            
            for arg in t_args:
                if arg in self.shot_to_idx:
                    penalty += self.shot_to_idx[arg] * 100
                    target_cocktail = self.shot_is_goal_for.get(arg)
                    if target_cocktail:
                        misused = False
                        if any(other_a in self.cocktails and other_a != target_cocktail for other_a in t_args):
                            misused = True
                        if any(other_a in self.ingredients and other_a not in self.cocktail_ingredients.get(target_cocktail, []) for other_a in t_args):
                            misused = True
                        if misused:
                            penalty += 500
                
                if arg in self.hands and arg != self.preferred_hand:
                    penalty += 50

        return max(h_goal, h_tn) + penalty
