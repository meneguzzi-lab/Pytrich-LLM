from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic16(Heuristic):
    """
    Domain-Dependent Heuristic for Barman_BDI.
    
    BOTTLENECK: Parameter-binding symmetry among identical shots and hands. 
    The domain has N shots that are interchangeable as intermediaries, causing N^2K branching.
    
    ADMISSIBLE BOUNDS:
    1. Task Network Cost: Minimum actions required to decompose/execute all tasks currently in the network.
    2. State-Aware Refinement: If a task's objective (e.g., container clean) is already met in the current state, its cost is reduced.
    
    SYMMETRY BREAKING (HINTS):
    - Penalize using goal shots as intermediaries (Penalty: 500).
    - Penalize non-preferred hands to halve branching (Penalty: 50).
    - Assign increasing penalties to shots to break symmetry (Penalty: 100 * index).
    """

    def __init__(self, barman_gemini3flash_heuristic_16="barman_gemini3flash_heuristic_16"):
        super().__init__(barman_gemini3flash_heuristic_16)
        self.task_costs = {
            "AchieveContainsShotCocktail": 20,
            "AchieveContainsShakerCocktail": 15,
            "AchieveContainsShakerIngredient": 10,
            "AchieveContainsShotIngredient": 10,
            "AchieveCleanShaker": 5,
            "AchieveCleanShot": 5,
            "AchieveHolding": 2,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 5,
            "clean-shaker": 1, "clean-shot": 1, "drop": 1, "empty-shaker": 1,
            "empty-shot": 1, "fill-shot": 1, "grasp": 1, "pour-shaker-to-shot": 1,
            "pour-shot-to-clean-shaker": 1, "pour-shot-to-used-shaker": 1, "shake": 1
        }
        self.goal_shots = set()
        self.all_hands = []
        self.non_preferred_hand = None
        self.shot_order = {}
        self.clean_fact_ids = {}
        self.empty_fact_ids = {}
        self.hand_empty_fact_ids = {}
        self.holding_fact_ids = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for f_name in goal_fact_names:
            p, a = self._parse_fact_name(f_name)
            if p == "contains":
                self.goal_shots.add(a[0])

        all_hands = set()
        all_shots = set()
        for abt in model.abstract_tasks:
            if "AchieveCleanShot" in abt.name:
                _, args = self._parse_task(abt.name)
                if args: all_shots.add(args[0])
        for op in model.operators:
            if "grasp" in op.name or "drop" in op.name:
                _, args = self._parse_task(op.name)
                if args: all_hands.add(args[0])

        sorted_hands = sorted(list(all_hands))
        self.all_hands = sorted_hands
        if len(sorted_hands) > 1: self.non_preferred_hand = sorted_hands[1]

        sorted_shots = sorted(list(all_shots))
        self.shot_order = {s: i for i, s in enumerate(sorted_shots)}

        for f in model.facts:
            p, a = self._parse_fact_name(f.name)
            if p == "clean": self.clean_fact_ids[a[0]] = f.global_id
            elif p == "empty": self.empty_fact_ids[a[0]] = f.global_id
            elif p == "handEmpty": self.hand_empty_fact_ids[a[0]] = f.global_id
            elif p == "holding" and len(a) >= 2: self.holding_fact_ids[(a[0], a[1])] = f.global_id

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        h = 0
        for task in task_network:
            t_name, t_args = self._parse_task(task.name)
            base = self.task_costs.get(t_name, 1)
            
            if t_args:
                if t_name == "AchieveCleanShot" or t_name == "AchieveCleanShaker":
                    if (state >> self.clean_fact_ids.get(t_args[0], 0)) & 1: base = 0
                elif t_name == "AchieveHandEmpty":
                    if (state >> self.hand_empty_fact_ids.get(t_args[0], 0)) & 1: base = 0
                elif t_name == "AchieveHolding" and len(t_args) >= 2:
                    if (state >> self.holding_fact_ids.get((t_args[0], t_args[1]), 0)) & 1: base = 0
            
            h += base
            for arg in t_args:
                if arg in self.shot_order:
                    h += 100 * self.shot_order[arg]
                    if arg in self.goal_shots:
                        if t_name in ["AchieveContainsShakerIngredient", "AchieveContainsShotIngredient"]:
                            h += 500
                if arg == self.non_preferred_hand:
                    h += 50
        return int(h)

    def _parse_task(self, name):
        if "[" not in name: return name, []
        idx = name.find("[")
        t_name = name[:idx]
        args = name[idx+1:-1].split(",")
        return t_name, args

    def _parse_fact_name(self, name):
        name = name.strip()
        if name.startswith(("+", "-")): name = name[1:]
        if "[" not in name: return name, []
        idx = name.find("[")
        pred = name[:idx]
        args = name[idx+1:-1].split(",")
        return pred, args
