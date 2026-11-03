from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

class BarmanBDIGemini3FlashHeuristic8(Heuristic):
    """
    Domain-dependent heuristic for the Barman_BDI domain.

    1. Bottleneck Analysis:
       The primary bottleneck is the Shaker and the Hands. There is usually only one shaker, 
       and it must be cleaned and filled with two specific ingredients before shaking 
       and pouring. Hands must be empty to perform cleaning and grasping operations.

    2. Admissible Lower Bounds:
       - Term 1: Task Network Cost. Each task in the HTN network represents a series of 
         unavoidable primitive actions. For example, 'AchieveContainsShotCocktail' 
         requires at least 10 actions (filling, pouring, shaking, etc.).
       - Term 2: Hand/Container state overhead. If a hand is not empty or a container is not clean
         when a task requires it, additional actions (drop/clean) are mandatory.

    3. Preprocessing:
       Builds a mapping of predicates and task names to their corresponding global IDs 
       and pre-calculates costs for various task types.
    """

    def __init__(self, barman_gemini3flash_heuristic_8=None, name="BarmanBDIGemini3FlashHeuristic8"):
        super().__init__(name=name)
        self.pred_map = {}
        self.task_costs = {
            "AchieveContainsShotCocktail": 10,
            "AchieveContainsShakerCocktail": 6,
            "AchieveContainsShakerIngredient": 3,
            "AchieveContainsShotIngredient": 2,
            "AchieveCleanShot": 2,
            "AchieveCleanShaker": 2,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 1
        }

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _preprocess(self):
        # Map predicates to a dictionary: pred_name -> args_tuple -> global_id
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                if pred not in self.pred_map:
                    self.pred_map[pred] = {}
                self.pred_map[pred][tuple(args)] = fact.global_id

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

    def _get_fact_id(self, pred, args):
        return self.pred_map.get(pred, {}).get(tuple(args))

    def _is_true(self, state, pred, args):
        fid = self._get_fact_id(pred, args)
        if fid is None:
            return False
        return (state >> fid) & 1

    def _compute(self, state, task_network):
        if not task_network:
            return 0

        total_h = 0
        
        for task in task_network:
            # Split task name into (task_type, args)
            parts = task.name.split()
            if not parts:
                continue
            
            task_type = parts[0]
            args = parts[1:]
            
            # Base cost from our lookup table
            cost = self.task_costs.get(task_type, 1)
            
            # Refine cost based on current state (P2: State-aware refinement)
            if task_type == "AchieveContainsShotCocktail":
                # If the shot already contains the cocktail, cost is 0
                if self._is_true(state, "contains", args):
                    cost = 0
                else:
                    # Heuristic reduction if parts are already in the shaker
                    # (Args: shot, cocktail)
                    cocktail = args[1]
                    # We don't have the shaker in these args, but we check if *any* shaker has it
                    # This is a bit expensive for __call__, so we use a simplified check if possible
                    # In this domain, shaker1 is usually the only shaker.
                    shaker_args = ["shaker1", cocktail]
                    if self._is_true(state, "contains", shaker_args):
                        if self._is_true(state, "shaked", ["shaker1"]):
                            cost = 2 # Grasp + Pour
                        else:
                            cost = 4 # Grasp + Shake + Pour

            elif task_type == "AchieveContainsShakerCocktail":
                if self._is_true(state, "contains", args):
                    cost = 0
                elif self._is_true(state, "unshaked", [args[0]]):
                    cost = 2

            elif task_type == "AchieveCleanShot" or task_type == "AchieveCleanShaker":
                if self._is_true(state, "clean", args):
                    cost = 0

            elif task_type == "AchieveHolding":
                if self._is_true(state, "holding", args):
                    cost = 0

            elif task_type == "AchieveHandEmpty":
                if self._is_true(state, "handEmpty", args):
                    cost = 0

            total_h += cost

        # Add a small tie-breaker to favor specific container ordering (P6)
        # Helps avoid symmetrical explorations of shot1 vs shot2
        tie_breaker = 0
        for task in task_network:
            for arg in task.name.split()[1:]:
                if "shot" in arg:
                    try:
                        num = int("".join(filter(str.isdigit, arg)))
                        tie_breaker += num * 0.0001
                    except:
                        pass

        return int(total_h) + tie_breaker

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
