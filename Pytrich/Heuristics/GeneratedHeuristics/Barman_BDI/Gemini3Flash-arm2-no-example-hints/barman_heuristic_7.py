from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic7(Heuristic):
    """
    Domain-Dependent Heuristic for Barman_BDI.

    BOTTLENECK: The sequential dependency between filling shots, transferring to shakers, shaking, 
    and pouring into goal shots, all while managing a two-hand constraint.

    ADMISSIBILITY SKETCHES:
    1. Goal-State Lower Bound: Each unsatisfied goal cocktail fact (contains shot cocktail) 
       requires at least one 'pour-shaker-to-shot' action. (Cost: 1)
    2. Task-Network Lower Bound: Each task in the network represents a commitment to perform 
       a sequence of actions. For example, 'AchieveContainsShotCocktail' requires at least 
       one action if the state does not already satisfy the goal. (Cost: 1 to 13)
    3. Cleaning/Emptying: If a required container is dirty or full of an incorrect beverage,
       at least one cleaning/emptying action is unavoidable. (Cost: 1)
    """

    def __init__(self, barman_gemini3flash_heuristic_7="BarmanBDIGemini3FlashHeuristic7"):
        super().__init__(name=barman_gemini3flash_heuristic_7)
        self.goal_shot_cocktails = {} # shot -> cocktail
        self.shot_priority = {} # shot -> int
        self.fact_ids = {}
        self.cocktail_recipes = {} # cocktail -> (ing1, ing2)
        self.preferred_hand = None

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
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        all_shots = set()
        self.fact_ids = {}
        
        # Pre-parse all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            self.fact_ids[f.name] = f.global_id
            
            if pred == "cocktailPart1":
                cocktail, ing = args[0], args[1]
                if cocktail not in self.cocktail_recipes: self.cocktail_recipes[cocktail] = [None, None]
                self.cocktail_recipes[cocktail][0] = ing
            elif pred == "cocktailPart2":
                cocktail, ing = args[0], args[1]
                if cocktail not in self.cocktail_recipes: self.cocktail_recipes[cocktail] = [None, None]
                self.cocktail_recipes[cocktail][1] = ing
            elif "shot" in args[0]:
                all_shots.add(args[0])

        # Find Goal Facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "contains" and "shot" in args[0]:
                self.goal_shot_cocktails[args[0]] = args[1]

        # Symmetry Breaking: Sort shots and assign priorities
        sorted_shots = sorted(list(all_shots))
        for i, shot in enumerate(sorted_shots):
            self.shot_priority[shot] = i

        # Hand Preference
        for f in model.facts:
            if "handEmpty" in f.name:
                _, args = self._parse_fact_name(f.name)
                self.preferred_hand = args[0]
                break

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _is_true(self, state, fact_name):
        fid = self.fact_ids.get(fact_name)
        if fid is None: return False
        return (state >> fid) & 1

    def _compute(self, state, task_network):
        h = 0
        
        # 1. Goal Progress Term
        for shot, cocktail in self.goal_shot_cocktails.items():
            if not self._is_true(state, f"+contains[{shot},{cocktail}]"):
                h += 1

        # 2. Task Network Analysis & Symmetry Breaking
        for task in task_network:
            t_name, t_args = self._parse_fact_name(task.name)
            
            # Base task costs (admissible estimates of remaining actions)
            if t_name == "AchieveContainsShotCocktail":
                shot, cocktail = t_args[0], t_args[1]
                if not self._is_true(state, f"+contains[{shot},{cocktail}]"):
                    h += 5
                # Symmetry Breaking: Penalize using goal shots for intermediate tasks
                if shot in self.goal_shot_cocktails and self.goal_shot_cocktails[shot] != cocktail:
                    h += 500
                h += 100 * self.shot_priority.get(shot, 0)

            elif t_name == "AchieveContainsShakerCocktail":
                shaker, cocktail = t_args[0], t_args[1]
                if not self._is_true(state, f"+contains[{shaker},{cocktail}]"):
                    h += 3

            elif t_name == "AchieveContainsShakerIngredient":
                shaker, ing = t_args[0], t_args[1]
                if not self._is_true(state, f"+contains[{shaker},{ing}]"):
                    h += 2

            elif t_name == "AchieveContainsShotIngredient":
                shot, ing = t_args[0], t_args[1]
                if not self._is_true(state, f"+contains[{shot},{ing}]"):
                    h += 2
                # Symmetry Breaking: Don't use goal shots for random ingredients
                if shot in self.goal_shot_cocktails:
                    recipe = self.cocktail_recipes.get(self.goal_shot_cocktails[shot], [])
                    if ing not in recipe:
                        h += 500
                h += 100 * self.shot_priority.get(shot, 0)

            elif t_name == "AchieveHolding":
                hand, container = t_args[0], t_args[1]
                if not self._is_true(state, f"+holding[{hand},{container}]"):
                    h += 1
                # Hand preference penalty
                if hand != self.preferred_hand:
                    h += 50

            elif t_name == "AchieveHandEmpty":
                hand = t_args[0]
                if not self._is_true(state, f"+handEmpty[{hand}]"):
                    h += 1

            elif t_name in ["AchieveCleanShot", "AchieveCleanShaker"]:
                container = t_args[0]
                if not self._is_true(state, f"+clean[{container}]"):
                    h += 1

            # Operator Cost
            elif t_name in ["fill-shot", "shake", "pour-shot-to-clean-shaker", "pour-shaker-to-shot"]:
                h += 1

        return h
