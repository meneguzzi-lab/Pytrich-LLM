import re
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic2(Heuristic):
    """
    Domain-dependent heuristic for Barman_BDI HDDL domain.
    
    Bottleneck Analysis:
    The domain is bottlenecked by Shaker and Hand availability. Shakers must be cleaned before reuse, 
    and ingredients must be measured using shots before being poured into shakers. Since there are 
    only two hands, the serialization of grasping, cleaning, filling, and pouring is the primary 
    constraint on plan length.

    Admissibility Sketch:
    1. Task Decomposition Lower Bound: Each task in the network is assigned a minimum number of 
       primitive actions (operators) required to satisfy its goal. For instance, a cocktail requires 
       at least 6 actions: 2 fills, 2 pours, 1 shake, and 1 final pour into the target shot.
    2. State-Aware Discounting: If the objective of a task is already satisfied in the current state 
       (e.g., a shot already contains the desired cocktail), the cost for that task is reduced to 0.
    3. Hand/Empty Requirement: Tasks requiring hands to be empty or containers to be held contribute 
       to the cost if those conditions are not met.
    """

    def __init__(self, barman_gemini3flash_heuristic_2="barman_gemini3flash_heuristic_2"):
        super().__init__(barman_gemini3flash_heuristic_2)
        self.pred_map = {}
        self.goal_info = []
        self.base_task_costs = {
            "AchieveContainsShotCocktail": 6,
            "AchieveContainsShakerCocktail": 5,
            "AchieveContainsShakerIngredient": 2,
            "AchieveContainsShotIngredient": 1,
            "AchieveCleanShot": 1,
            "AchieveCleanShaker": 1,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 1
        }

    def initialize(self, model: Model, initial_node: HTNNode):
        # Map all facts to their global IDs for O(1) bitwise checks
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                if pred not in self.pred_map:
                    self.pred_map[pred] = {}
                self.pred_map[pred][tuple(args)] = fact.global_id

        # Parse goal facts if they exist
        goal_names = model.state_explicit_repr(model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if pred:
                self.goal_info.append((pred, args))

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network) -> int:
        h = 0
        
        # 1. Sum of remaining task costs
        for task in task_network:
            if hasattr(task, 'cost'):
                # It's a primitive operator
                h += task.cost
            else:
                # It's an abstract task
                base_name, args = self._parse_task(task.name)
                cost = self.base_task_costs.get(base_name, 1)
                
                # State-aware cost reduction
                if base_name == "AchieveContainsShotCocktail" and len(args) == 2:
                    if self._is_true(state, "contains", args): cost = 0
                elif base_name == "AchieveContainsShakerCocktail" and len(args) == 2:
                    if self._is_true(state, "contains", args): cost = 0
                elif base_name == "AchieveContainsShakerIngredient" and len(args) == 2:
                    if self._is_true(state, "contains", args): cost = 0
                elif base_name == "AchieveContainsShotIngredient" and len(args) == 2:
                    if self._is_true(state, "contains", args): cost = 0
                elif base_name == "AchieveCleanShot" and len(args) == 1:
                    if self._is_true(state, "clean", args): cost = 0
                elif base_name == "AchieveCleanShaker" and len(args) == 1:
                    if self._is_true(state, "clean", args): cost = 0
                elif base_name == "AchieveHolding" and len(args) == 2:
                    if self._is_true(state, "holding", args): cost = 0
                elif base_name == "AchieveHandEmpty" and len(args) == 1:
                    if self._is_true(state, "handEmpty", args): cost = 0
                elif base_name == "AchieveOnTable" and len(args) == 1:
                    if self._is_true(state, "ontable", args): cost = 0
                
                h += cost

        # 2. Account for specific goal facts not mentioned in the current task network
        # This provides guidance when the task network is partially decomposed
        for pred, args in self.goal_info:
            if not self._is_true(state, pred, args):
                # Minimal cost for a missing goal fact if not already covered by task costs
                h += 1

        # 3. Tie-breaking: small penalty for task network size to favor progress
        h += int(0.001 * len(task_network))
        
        return h

    def _is_true(self, state: int, pred: str, args: list) -> bool:
        if pred in self.pred_map:
            args_tuple = tuple(args)
            if args_tuple in self.pred_map[pred]:
                fid = self.pred_map[pred][args_tuple]
                return (state >> fid) & 1
        return False

    def _parse_task(self, task_name: str):
        parts = [p for p in re.split(r'[ ,()]+', task_name) if p]
        if not parts: return "", []
        return parts[0], parts[1:]

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
