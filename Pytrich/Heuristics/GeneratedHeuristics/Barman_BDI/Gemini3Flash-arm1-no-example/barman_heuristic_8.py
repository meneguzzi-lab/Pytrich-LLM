from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic8(Heuristic):
    """
    Heuristic for Barman_BDI domain.
    Bottleneck: Sequential container processing (cleaning -> filling -> shaking -> pouring) and limited hand resources.
    
    Admissibility Sketch:
    - Term 1 (Task Network): Sums minimum actions required to decompose each task in the network.
      A task contributes 0 if its target state (e.g., shot contains cocktail) is already achieved.
      Minimum costs are derived from the shortest possible decomposition path (e.g., ShotCocktail = 6 actions).
    - Term 2 (State check): If task network is empty, verify goal satisfaction.
    - Tie-breaking: Small penalty for each hand currently holding a container to favor states where hands are freed.
    """
    def __init__(self, barman_gemini3flash_heuristic_8="barman_gemini3flash_heuristic_8"):
        super().__init__(name=barman_gemini3flash_heuristic_8)
        self.task_to_fact = {}
        self.task_base_costs = {}
        self.hand_empty_fact_ids = []
        self.fact_map = {}

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = tuple(args_str.split(",")) if args_str else ()
                return predicate, args
        return None, ()

    def initialize(self, model: Model, initial_node: HTNNode):
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_map[(pred, args)] = fact.global_id
                if pred == "handEmpty":
                    self.hand_empty_fact_ids.append(fact.global_id)

        cost_lookup = {
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

        fact_trigger = {
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

        for task in model.abstract_tasks + model.operators:
            parts = task.name.split()
            if not parts: continue
            t_name = parts[0]
            t_args = tuple(parts[1:])
            
            self.task_base_costs[task.global_id] = cost_lookup.get(t_name, 1)
            
            if t_name in fact_trigger:
                f_pred = fact_trigger[t_name]
                f_key = (f_pred, t_args)
                if f_key in self.fact_map:
                    self.task_to_fact[task.global_id] = self.fact_map[f_key]

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _compute(self, state: int, task_network) -> int:
        if not task_network:
            return 0 if (state & self.model.goals) == self.model.goals else 1
        
        h = 0
        for task in task_network:
            tid = task.global_id
            fid = self.task_to_fact.get(tid)
            if fid is not None and (state >> fid) & 1:
                continue
            h += self.task_base_costs.get(tid, 1)
        
        # Tie-breaker: minor penalty for occupied hands
        for fid in self.hand_empty_fact_ids:
            if not ((state >> fid) & 1):
                h += 0.01
        return int(h) if h > 1000000 else h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
