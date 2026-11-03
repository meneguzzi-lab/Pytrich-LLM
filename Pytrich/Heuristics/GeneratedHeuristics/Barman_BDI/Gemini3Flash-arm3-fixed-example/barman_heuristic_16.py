from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

class BarmanBDIGemini3FlashHeuristic16(Heuristic):
    """
    Domain-dependent heuristic for the Barman_BDI domain.
    
    Bottleneck Analysis:
    The primary bottleneck is the sequential nature of cocktail preparation using a single shaker 
    and the requirement for containers (shots and shakers) to be clean before use. 
    Each cocktail requires multiple preparation steps: filling/pouring ingredients, shaking, 
    and final pouring, most of which require a free hand and a specific container state.
    
    Admissibility Sketches:
    1. h_task_network: This term sums the minimum unavoidable costs of each task in the 
       current network. A task like AchieveContainsShotCocktail involves at least 4 primitive 
       actions (2 pours, 1 shake, 1 pour-to-shot), provided its goal isn't already met. 
       This is additive because cocktails share the shaker and must be prepared sequentially.
    2. h_state_goals: This term checks for goal facts from the model's goal state that are not 
       yet satisfied and not currently addressed by any task in the task network. 
       Each missing goal adds a base cost of 2 actions.
    """

    def __init__(self, barman_gemini3flash_heuristic_16=None, name="BarmanBDIGemini3FlashHeuristic16"):
        super().__init__(name=name)
        self.task_to_fact = {}      # task_global_id -> fact_global_id
        self.task_to_cost = {}      # task_global_id -> estimated_min_cost
        self.goal_fact_ids = []     # List of fact_global_ids for goals
        self.fact_name_map = {}     # (pred, tuple(args)) -> fact_global_id

    def _parse_name(self, name: str):
        """Parses both fact names '+pred[a,b]' and task names 'pred a b' or '(pred a b)'."""
        name = name.strip("()+[]- ")
        if "[" in name:
            pred, rest = name.split("[")
            args = rest.rstrip("]").split(",")
            return pred, [a.strip() for a in args]
        parts = name.split()
        if not parts: return None, []
        return parts[0], [p.strip() for p in parts[1:]]

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Map all facts for O(1) lookup
        for fact in model.facts:
            pred, args = self._parse_name(fact.name)
            if pred:
                self.fact_name_map[(pred, tuple(args))] = fact.global_id

        # Task predicate mapping
        task_pred_map = {
            "AchieveContainsShotCocktail": "contains",
            "AchieveContainsShakerCocktail": "contains",
            "AchieveContainsShakerIngredient": "contains",
            "AchieveContainsShotIngredient": "contains",
            "AchieveCleanShot": "clean",
            "AchieveCleanShaker": "clean",
            "AchieveHolding": "holding",
            "AchieveHandEmpty": "handEmpty",
            "AchieveOnTable": "ontable"
        }

        task_base_costs = {
            "AchieveContainsShotCocktail": 6,
            "AchieveContainsShakerCocktail": 4,
            "AchieveContainsShakerIngredient": 2,
            "AchieveContainsShotIngredient": 1,
            "AchieveCleanShot": 1,
            "AchieveCleanShaker": 1,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1
        }

        # Precompute task costs and their target fact IDs
        for task in model.abstract_tasks:
            t_pred, t_args = self._parse_name(task.name)
            if t_pred in task_pred_map:
                f_pred = task_pred_map[t_pred]
                key = (f_pred, tuple(t_args))
                if key in self.fact_name_map:
                    self.task_to_fact[task.global_id] = self.fact_name_map[key]
                self.task_to_cost[task.global_id] = task_base_costs.get(t_pred, 1)

        # Extract goals
        if hasattr(model, 'goals') and model.goals:
            goal_names = model.state_explicit_repr(model.goals)
            for g_name in goal_names:
                pred, args = self._parse_name(g_name)
                key = (pred, tuple(args))
                if key in self.fact_name_map:
                    self.goal_fact_ids.append(self.fact_name_map[key])

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _compute(self, state: int, task_network) -> int:
        h = 0
        tasks_covering_facts = set()

        # Term 1: Task Network Costs
        for task in task_network:
            tid = task.global_id
            # If it's an abstract task we mapped
            if tid in self.task_to_fact:
                fid = self.task_to_fact[tid]
                if not ((state >> fid) & 1):
                    h += self.task_to_cost.get(tid, 1)
                    tasks_covering_facts.add(fid)
            elif tid >= self.model.iop_init and tid <= self.model.iop_end:
                # It's a primitive operator
                h += 1
            else:
                # Abstract task not in our Achieve map
                h += 1

        # Term 2: Missing Goals Cost
        for g_fid in self.goal_fact_ids:
            if not ((state >> g_fid) & 1):
                if g_fid not in tasks_covering_facts:
                    h += 2

        # Symmetry Breaking / Tie Breaking
        # Small penalty for longer task networks to prefer progress
        h_val = int(h) + (0.001 * len(task_network))
        return h_val

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
