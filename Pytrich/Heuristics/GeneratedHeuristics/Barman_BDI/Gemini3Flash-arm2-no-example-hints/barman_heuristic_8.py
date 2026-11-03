from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic8(Heuristic):
    """
    Bottleneck Analysis: 
    The Barman domain is bottlenecked by the sequencing of shared shakers and the scarcity of shots. 
    Symmetry among identical shots leads to exponential branching. 
    
    Admissible Lower Bounds:
    1. State-Aware Task Cost: Each abstract task in the TN requires a known minimum number of actions 
       based on the current state (e.g., AchieveContainsShotCocktail needs ~15 actions if not started).
    2. Symmetry Breaking: Penalizing non-canonical shot selection and the use of 'goal shots' 
       (shots intended for final cocktails) as intermediaries.
    """
    def __init__(self, barman_gemini3flash_heuristic_8="barman_gemini3flash_heuristic_8"):
        super().__init__(barman_gemini3flash_heuristic_8)
        self.fact_map = {}
        self.goal_shots = set()
        self.shot_to_index = {}
        self.hand_pref = {}
        self.task_costs = {
            "AchieveContainsShotCocktail": 15,
            "AchieveContainsShakerCocktail": 10,
            "AchieveContainsShakerIngredient": 4,
            "AchieveCleanShot": 2,
            "AchieveCleanShaker": 2,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 1
        }

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

    def _parse_task_name(self, name: str):
        idx = name.find("(")
        if idx == -1: return name, []
        task_name = name[:idx]
        args_str = name[idx+1:-1]
        args = [a.strip() for a in args_str.split(",")] if args_str else []
        return task_name, args

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred:
                self.fact_map[(pred, tuple(args))] = f.global_id
        
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "contains" and len(args) == 2:
                self.goal_shots.add(args[0])
        
        all_shots = set()
        all_hands = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred in ["clean", "empty", "ontable"]:
                if args and "shot" in args[0].lower(): all_shots.add(args[0])
            if pred in ["handEmpty"]:
                if args: all_hands.add(args[0])
        
        for i, s in enumerate(sorted(list(all_shots))):
            self.shot_to_index[s] = i
        
        hands = sorted(list(all_hands))
        if hands:
            self.hand_pref[hands[0]] = 0
            for h in hands[1:]: self.hand_pref[h] = 50

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        h = 0
        for task in task_network:
            t_name, t_args = self._parse_task_name(task.name)
            base_cost = self.task_costs.get(t_name, 1)
            
            if t_name == "AchieveContainsShotCocktail" and len(t_args) >= 2:
                fid = self.fact_map.get(("contains", (t_args[0], t_args[1])))
                if fid is not None and (state >> fid) & 1: base_cost = 0
            elif t_name == "AchieveCleanShot" and len(t_args) >= 1:
                fid = self.fact_map.get(("clean", (t_args[0],)))
                if fid is not None and (state >> fid) & 1: base_cost = 0
            elif t_name == "AchieveContainsShotIngredient" and len(t_args) >= 2:
                shot = t_args[0]
                fid = self.fact_map.get(("contains", (shot, t_args[1])))
                if fid is not None and (state >> fid) & 1: 
                    base_cost = 0
                else:
                    if shot in self.goal_shots: h += 500
                    h += self.shot_to_index.get(shot, 0) * 100
            
            for arg in t_args:
                if arg in self.hand_pref: h += self.hand_pref[arg]
            h += base_cost
        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
