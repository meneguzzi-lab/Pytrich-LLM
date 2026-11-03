import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic17(Heuristic):
    """
    Domain-dependent heuristic for Barman_BDI HTN domain.

    Bottleneck Analysis:
    The central bottleneck in Barman is the cyclical usage of the shaker. Every cocktail requires
    the shaker to be cleaned, filled with two ingredients (via shots), shaked, and then poured. 
    Each of these steps involves multiple primitive actions (grasp, drop, clean, fill, pour, shake).

    Admissibility Sketches:
    1. Task Network Cost: Each task in the task network represents a commitment to a sequence of actions.
       By assigning a minimum possible action count (base cost) to each abstract task, we create an
       admissible lower bound. If the goal of the task is already met (checked via 'Null' methods), the cost is 0.
    2. Shaker Resource: Only one cocktail can be in the shaker at a time, forcing a serial execution for most of the process.
    """

    def __init__(self, barman_gemini3flash_heuristic_17=None, name="BarmanBDIGemini3FlashHeuristic17"):
        super().__init__(name=name)
        self.task_id_to_cost = {}
        self.task_id_to_null_fact = {}
        self.goal_fact_ids = []

        # Base costs derived from minimum possible actions to satisfy the task
        self.base_costs = {
            "AchieveContainsShotCocktail": 15,    # Clean shot + Shaker cycle
            "AchieveContainsShakerCocktail": 10,  # Clean shaker + 2x(Fill+Pour) + Shake
            "AchieveContainsShakerIngredient": 4, # Fill shot + Pour to shaker
            "AchieveContainsShotIngredient": 2,   # Clean shot + Fill shot
            "AchieveCleanShaker": 2,              # Grasp + Clean
            "AchieveCleanShot": 2,                # Grasp + Clean
            "AchieveHandEmpty": 1,                # Drop
            "AchieveHolding": 1,                  # Grasp
            "AchieveOnTable": 1,                  # Drop
            "DoPourShakerToShot": 1               # Primitive pour action
        }

        # Predicates that satisfy the 'Null' method for each task
        self.null_preds = {
            "AchieveContainsShotCocktail": "contains",
            "AchieveContainsShakerCocktail": "contains",
            "AchieveContainsShakerIngredient": "contains",
            "AchieveContainsShotIngredient": "contains",
            "AchieveCleanShaker": "clean",
            "AchieveCleanShot": "clean",
            "AchieveHandEmpty": "handEmpty",
            "AchieveHolding": "holding",
            "AchieveOnTable": "ontable"
        }

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Map (predicate, args_tuple) -> global_id for fast lookup
        fact_map = {}
        for fact in self.model.facts:
            pred, args = self._parse_name(fact.name)
            if pred:
                fact_map[(pred, tuple(args))] = fact.global_id

        # Precompute costs and null-checks for abstract tasks
        for task in self.model.abstract_tasks:
            pred, args = self._parse_name(task.name)
            if pred in self.base_costs:
                tid = task.global_id
                self.task_id_to_cost[tid] = self.base_costs[pred]
                
                if pred in self.null_preds:
                    null_pred = self.null_preds[pred]
                    if null_pred == "handEmpty":
                        # Special case: handEmpty ?x_1 in domain allows any empty hand
                        self.task_id_to_null_fact[tid] = [fid for (p, a), fid in fact_map.items() if p == "handEmpty"]
                    else:
                        # Standard case: null method precondition matches task args
                        fid = fact_map.get((null_pred, tuple(args)))
                        if fid is not None:
                            self.task_id_to_null_fact[tid] = fid

        # Precompute goal fact IDs
        if hasattr(self.model, 'goals') and self.model.goals:
            for i in range(self.model.goals.bit_length()):
                if (self.model.goals >> i) & 1:
                    self.goal_fact_ids.append(i)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_name(self, name: str):
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

    def _compute(self, state: int, task_network: list) -> int:
        h = 0
        # Sum costs of all tasks in the network
        for task in task_network:
            tid = task.global_id
            if tid in self.task_id_to_cost:
                base_cost = self.task_id_to_cost[tid]
                null_fact = self.task_id_to_null_fact.get(tid)
                
                is_satisfied = False
                if null_fact is not None:
                    if isinstance(null_fact, list):
                        # Check if any specified fact is true (for handEmpty)
                        if any((state >> fid) & 1 for fid in null_fact):
                            is_satisfied = True
                    else:
                        if (state >> null_fact) & 1:
                            is_satisfied = True
                
                if not is_satisfied:
                    h += base_cost
            else:
                # If it's an operator or unmapped task, assume cost 1
                h += 1
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
