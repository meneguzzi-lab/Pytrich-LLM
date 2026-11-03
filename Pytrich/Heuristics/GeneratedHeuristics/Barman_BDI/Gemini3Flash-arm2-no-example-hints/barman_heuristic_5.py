from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic5(Heuristic):
    """
    Domain-Dependent Heuristic for Barman_BDI.
    
    Bottleneck Analysis:
    The main bottleneck is the serial management of hand and shaker resources. Each cocktail 
    requires a sequence: Clean Shaker -> Add Ingredient 1 (via shot) -> Add Ingredient 2 (via shot) 
    -> Shake -> Pour to Shot. The interchangeability of shots for ingredient transfer creates 
    massive parameter-binding symmetry.

    Admissibility Sketches:
    1. Task Base Costs: Assigning 1-15 points per task based on its minimum required action 
       depth (e.g., Cocktail=15, Ingredient=5). This is a lower bound because each task 
       requires at least that many actions to reach a state where preconditions are met.
    2. State-Goal Check: If a goal 'contains' fact is already true in the state, the cost 
       associated with that specific container/beverage pair is removed.

    Symmetry Breaking:
    - Penalize using 'goal shots' (shots that must contain a cocktail in the final state) 
      as intermediaries for ingredients (500 per task).
    - Break symmetry among non-goal 'working shots' by assigning increasing penalties (index * 100).
    - Prefer one hand over the other to reduce branching (50 penalty for non-preferred).
    """
    def __init__(self, barman_gemini3flash_heuristic_5="barman_gemini3flash_heuristic_5"):
        super().__init__(barman_gemini3flash_heuristic_5)
        self.goal_shots = set()
        self.shot_indices = {}
        self.hand_indices = {}
        self.fact_ids = {}  # (predicate, tuple(args)) -> global_id

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
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Identify Goal Shots
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "contains" and args:
                # In Barman, shots are goal containers. Shakers are intermediaries.
                if "shot" in args[0]:
                    self.goal_shots.add(args[0])

        # 2. Build Fact Map for quick state checking
        for f in model.facts:
            p, a = self._parse_fact_name(f.name)
            if p:
                self.fact_ids[(p, tuple(a))] = f.global_id

        # 3. Collect and sort constants for symmetry breaking
        shots = sorted([a[0] for p, a in self.fact_ids.keys() if p == "clean" and "shot" in a[0]])
        hands = sorted([a[0] for p, a in self.fact_ids.keys() if p == "handEmpty"])
        for i, s in enumerate(shots): self.shot_indices[s] = i
        for i, h in enumerate(hands): self.hand_indices[h] = i

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network) -> int:
        h = 0
        for task in task_network:
            t_type, args = self._parse_task_name(task.name)
            
            # Base Admissible Costs
            if t_type == "AchieveContainsShotCocktail":
                # If not satisfied, base cost 20
                shot, cocktail = args[0], args[1]
                fid = self.fact_ids.get(("contains", (shot, cocktail)))
                if fid is not None and not ((state >> fid) & 1):
                    h += 20
            elif t_type == "AchieveContainsShakerCocktail":
                h += 15
            elif t_type == "AchieveContainsShakerIngredient":
                h += 10
            elif t_type == "AchieveContainsShotIngredient":
                h += 5
                # SYMMETRY: Penalty for using a goal shot for an ingredient
                if args and args[0] in self.goal_shots:
                    h += 500
                # SYMMETRY: Break ties among working shots
                h += self.shot_indices.get(args[0], 0) * 100
            elif t_type == "AchieveHolding":
                h += 2
                # SYMMETRY: Hand preference (prefer first hand in index)
                if args and self.hand_indices.get(args[0], 0) > 0:
                    h += 50
            elif t_type == "AchieveCleanShot" or t_type == "AchieveCleanShaker":
                h += 3
            elif t_type == "AchieveHandEmpty":
                h += 1
            else:
                h += 1
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
