from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic16(Heuristic):
    """
    Heuristic for Barman_BDI domain.
    
    Bottleneck Analysis:
    The central bottleneck is the shaker. Producing a cocktail requires a specific sequence 
    of cleaning, filling, and shaking that must pass through the shaker before reaching the shot.
    Hands are also a limited resource (2 hands) which serializes picking and pouring actions.

    Admissibility Sketch:
    1. Task Network Cost: For each task in the network, we estimate the minimum number of actions 
       required to complete it from the current state. Since these are serial steps and cannot 
       be shared (e.g., you cannot shake two different cocktails simultaneously in one shaker), 
       summing these is a lower bound.
    2. State-Aware Task Costing: If a task's goal (e.g., cocktail in shot) is already true in 
       the state, the cost is 0. If it's in the shaker and shaked, it's 1 (the pour). 
       This provides an admissible lower bound on the remaining work.
    """

    def __init__(self, barman_gemini3flash_heuristic_16="barman_gemini3flash_heuristic_16"):
        super().__init__(barman_gemini3flash_heuristic_16)
        self.fact_ids = {}
        self.task_costs = {}
        self.shakers = set()
        self.shot_penalties = {}

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
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # Map all facts to IDs
        for f in model.facts:
            pred, args = self._parse_name(f.name)
            self.fact_ids[(pred, tuple(args))] = f.global_id
            if pred == 'shakerLevel' or 'shaker' in args[0]:
                self.shakers.add(args[0])

        # Map symmetry breaking penalties for shots
        all_shots = sorted([args[0] for p, args in self.fact_ids.keys() if p == 'clean' and 'shot' in args[0]])
        self.shot_penalties = {shot: i * 0.0001 for i, shot in enumerate(all_shots)}

        # Precompute cost logic for all grounded tasks in the model
        all_tasks = model.abstract_tasks + model.operators
        for task in all_tasks:
            self.task_costs[task.global_id] = self._build_cost_fn(task.name)

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _build_cost_fn(self, task_name):
        pred, args = self._parse_name(task_name)
        
        if pred == 'AchieveContainsShotCocktail':
            shot, cocktail = args[0], args[1]
            sid = self.fact_ids.get(('contains', (shot, cocktail)), -1)
            shaker_data = []
            for shaker in self.shakers:
                shaker_data.append((
                    self.fact_ids.get(('contains', (shaker, cocktail)), -1),
                    self.fact_ids.get(('shaked', (shaker,)), -1)
                ))
            penalty = self.shot_penalties.get(shot, 0)
            return lambda state: 0 if (sid != -1 and (state >> sid) & 1) else (\
                min([1 if (scid != -1 and (state >> scid) & 1 and sshid != -1 and (state >> sshid) & 1) else (\
                2 if (scid != -1 and (state >> scid) & 1) else 6) for scid, sshid in shaker_data] + [6]) + penalty)

        elif pred == 'AchieveContainsShakerCocktail':
            shaker, cocktail = args[0], args[1]
            sid = self.fact_ids.get(('contains', (shaker, cocktail)), -1)
            shid = self.fact_ids.get(('shaked', (shaker,)), -1)
            return lambda state: 0 if (sid != -1 and (state >> sid) & 1 and shid != -1 and (state >> shid) & 1) else 4

        elif pred == 'AchieveContainsShakerIngredient':
            shaker, ingredient = args[0], args[1]
            sid = self.fact_ids.get(('contains', (shaker, ingredient)), -1)
            return lambda state: 0 if (sid != -1 and (state >> sid) & 1) else 2

        elif pred == 'AchieveCleanShaker' or pred == 'AchieveCleanShot':
            container = args[0]
            cid = self.fact_ids.get(('clean', (container,)), -1)
            return lambda state: 0 if (cid != -1 and (state >> cid) & 1) else 2

        elif pred == 'AchieveHolding':
            container = args[1]
            hid = -1
            for h_args in [ (a, container) for p, a in self.fact_ids.keys() if p == 'holding' and a[1] == container]:
                hid = self.fact_ids.get(('holding', h_args), -1)
            return lambda state: 0 if (hid != -1 and (state >> hid) & 1) else 1

        elif pred == 'AchieveHandEmpty' or pred == 'AchieveOnTable':
            return lambda state: 1

        # Action costs
        return lambda state: 1

    def _compute(self, state, task_network):
        if not task_network: return 0
        total_h = 0
        for task in task_network:
            cost_fn = self.task_costs.get(task.global_id)
            if cost_fn:
                total_h += cost_fn(state)
            else:
                total_h += 1
        return int(total_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
