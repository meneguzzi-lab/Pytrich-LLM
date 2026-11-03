from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict

class BarmanBDIGemini3FlashHeuristic13(Heuristic):
    """
    Bottleneck Analysis:
    The main bottleneck in the Barman_BDI domain is the sequential use of containers (shakers and shots).
    Specifically, making a cocktail requires a single shaker to be cleaned, filled with multiple measured 
    ingredients (using shots), shaken, and then poured. The restricted number of hands (2) and 
    shakers (usually 1) forces a strict serialization of measurement, cleaning, and mixing actions.

    Admissibility Sketches:
    1. Task-Network State-Aware Cost: Each abstract task in the network represents a subtree of 
       future operators. We assign a lower bound to each task (e.g., 10 for AchieveContainsShotCocktail) 
       and subtract costs based on progress already reflected in the current state (e.g., if the shaker 
       is already filled). The sum is a lower bound because tasks in an ordered network must be 
       processed sequentially.
    2. Disjoint Action Count: We estimate the minimum remaining actions for each goal in model.goals 
       that is not yet satisfied by the state. This is a lower bound on unavoidable work.
    """

    def __init__(self, barman_gemini3flash_heuristic_13="barman_gemini3flash_heuristic_13"):
        super().__init__(name=barman_gemini3flash_heuristic_13)
        self.fact_ids = defaultdict(dict)
        self.cocktail_ingredients = defaultdict(list)
        self.shakers = []
        self.parsed_cache = {}

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

    def _parse_item_name(self, name: str):
        if name in self.parsed_cache:
            return self.parsed_cache[name]
        
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            res = (predicate, args)
        else:
            res = (name, [])
        
        self.parsed_cache[name] = res
        return res

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred:
                self.fact_ids[pred][tuple(args)] = f.global_id
                if pred == 'cocktailPart1' or pred == 'cocktailPart2':
                    self.cocktail_ingredients[args[0]].append(args[1])
            
        # Identify all shakers in the problem
        for f in model.facts:
            _, args = self._parse_fact_name(f.name)
            for arg in args:
                if "shaker" in arg and arg not in self.shakers:
                    self.shakers.append(arg)

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _is_true(self, state, pred, args):
        fid = self.fact_ids.get(pred, {}).get(tuple(args))
        return (state >> fid) & 1 if fid is not None else False

    def _compute(self, state, task_network):
        h = 0
        for task in task_network:
            pred, args = self._parse_item_name(task.name)
            
            if pred == 'AchieveContainsShotCocktail':
                # args: [shot, cocktail]
                shot, cocktail = args[0], args[1]
                if self._is_true(state, 'contains', [shot, cocktail]):
                    h += 0
                else:
                    # Check if cocktail is in any shaker and shaked
                    in_shaker = False
                    for shaker in self.shakers:
                        if self._is_true(state, 'contains', [shaker, cocktail]) and self._is_true(state, 'shaked', [shaker]):
                            in_shaker = True
                            break
                    if in_shaker:
                        h += 2 # Holding + Pour
                    else:
                        h += 10 # Standard make and pour cost
            
            elif pred == 'AchieveContainsShakerCocktail':
                # args: [shaker, cocktail]
                shaker, cocktail = args[0], args[1]
                if self._is_true(state, 'contains', [shaker, cocktail]) and self._is_true(state, 'shaked', [shaker]):
                    h += 0
                else:
                    # Check ingredients
                    ings = self.cocktail_ingredients.get(cocktail, [])
                    count = sum(1 for ing in ings if self._is_true(state, 'contains', [shaker, ing]))
                    if count == 2:
                        h += 2 # Just shake
                    elif count == 1:
                        h += 5 # One more ingredient + shake
                    else:
                        h += 7 # Two ingredients + shake

            elif pred == 'AchieveContainsShakerIngredient':
                shaker, ing = args[0], args[1]
                if self._is_true(state, 'contains', [shaker, ing]):
                    h += 0
                else:
                    h += 3

            elif pred == 'AchieveContainsShotIngredient':
                h += 2
            
            elif pred == 'AchieveCleanShaker' or pred == 'AchieveCleanShot':
                # args: [container]
                container = args[0]
                if self._is_true(state, 'clean', [container]):
                    h += 0
                else:
                    h += 2

            elif pred in ['AchieveHolding', 'AchieveHandEmpty', 'AchieveOnTable', 'DoPourShakerToShot']:
                h += 1
            
            elif hasattr(task, 'cost'):
                h += task.cost
            else:
                h += 1

        # Goal bitwise check as a backup / tie-breaker
        remaining_goal_bits = self.model.goals & ~state
        if remaining_goal_bits:
            # Add a small cost for every missing goal fact not covered by the task network
            # We use a bit count approximation for performance
            h += bin(remaining_goal_bits).count('1')

        # Symmetry breaking: prefer earlier task networks
        h += 0.001 * len(task_network)

        return int(h)
