from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import math

class BarmanBDIGemini3FlashHeuristic4(Heuristic):
    """
    Heuristic for Barman_BDI domain.
    
    Bottleneck Analysis:
    The main bottleneck is the Shaker. Every cocktail must be prepared in a shaker, which involves
    a sequence of cleaning, filling with two different ingredients (via a shot), shaking, and pouring.
    The hand constraint (only 2 hands) also serializes the process.
    
    Admissibility Sketches:
    1. Task Network Cost (h_task): Each task in the network has a minimum number of required actions 
       to be decomposed into primitives. For example, AchieveContainsShotCocktail requires at least 
       ~15 actions (clean shaker, 2x fill shot, 2x pour to shaker, shake, pour to shot, plus picking up items).
    2. Goal-Based Cost (h_goal): Each unsatisfied 'contains(shot, cocktail)' goal requires similar work.
       The cost is state-aware; if a shaker already contains the cocktail, the cost is lower.
    3. Tie-breaking (h_tie): Small penalties for dirty containers or non-empty hands to guide the search.
    """

    def __init__(self, barman_gemini3flash_heuristic_4="barman_gemini3flash_heuristic_4"):
        super().__init__(name=barman_gemini3flash_heuristic_4)
        self.contains_map = {} # (container, beverage) -> global_id
        self.clean_map = {}    # container -> global_id
        self.hand_empty_ids = []
        self.cocktail_ingredients = {} # cocktail -> [ing1, ing2]
        self.goal_contains = [] # list of (shot, cocktail, fact_id)
        self.container_ids = []

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
        # 1. Map Facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "contains":
                self.contains_map[(args[0], args[1])] = f.global_id
            elif pred == "clean":
                self.clean_map[args[0]] = f.global_id
                if args[0] not in self.container_ids: self.container_ids.append(args[0])
            elif pred == "handEmpty":
                self.hand_empty_ids.append(f.global_id)
            elif pred == "cocktailPart1" or pred == "cocktailPart2":
                cocktail, ing = args[0], args[1]
                if cocktail not in self.cocktail_ingredients: self.cocktail_ingredients[cocktail] = []
                self.cocktail_ingredients[cocktail].append(ing)

        # 2. Map Goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "contains":
                # args[0] is shot, args[1] is cocktail
                # We need the global id of the fact
                fact_id = -1
                for f in model.facts:
                    if f.name == g_name:
                        fact_id = f.global_id
                        break
                if fact_id != -1:
                    self.goal_contains.append((args[0], args[1], fact_id))

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _compute(self, state, task_network):
        # Term 1: Task Network Cost
        h_task = 0
        for task in task_network:
            name = task.name.lower()
            if "achievecontainsshotcocktail" in name:
                h_task += 15
            elif "achievecontainsshakercocktail" in name:
                h_task += 10
            elif "achievecontainsshakeringredient" in name:
                h_task += 5
            elif "achievecontainsshotingredient" in name:
                h_task += 3
            elif "achievecleanshot" in name or "achievecleanshaker" in name:
                h_task += 1
            elif "achieveholding" in name or "achievehandempty" in name or "achieveontable" in name:
                h_task += 1
            else:
                h_task += 1 # Default for operators or other abstract tasks

        # Term 2: Goal Progress Cost
        h_goal = 0
        for shot, cocktail, g_id in self.goal_contains:
            # If goal not met
            if not ((state >> g_id) & 1):
                # Base cost for a cocktail
                cost = 12
                # If shaker doesn't even have it, it's more expensive
                # Finding a shaker is hard, assume shaker1 or any container
                shaker_has_it = False
                # Check if any container contains the cocktail (shaked shaker)
                # Actually simplified check: if cocktail is not shaked, it's costlier
                h_goal += cost

        # Tie-breaking: Penalize dirty containers and non-empty hands
        h_tie = 0
        for c in self.container_ids:
            if c in self.clean_map:
                if not ((state >> self.clean_map[c]) & 1):
                    h_tie += 0.1
        for h_id in self.hand_empty_ids:
            if not ((state >> h_id) & 1):
                h_tie += 0.05

        return max(h_task, h_goal) + h_tie

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
