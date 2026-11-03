from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic3(Heuristic):
    """
    Domain-Dependent Heuristic for Barman_BDI.

    BOTTLENECK ANALYSIS:
    The Shaker is the central serializing bottleneck. Every cocktail requires a strict sequence: 
    Clean -> Fill Ing1 (via shot) -> Fill Ing2 (via shot) -> Shake -> Pour. 
    Hands (2) and Shots (N) are resources that must be managed to keep the shaker moving.

    ADMISSIBILITY SKETCH:
    1. Task Cost: Each task in the task network represents a set of unavoidable actions. 
       We sum the minimum action-costs for each task using a state-aware lookup. 
    2. Goal Cost: For each 'contains' goal in the model not yet satisfied, a cost is added.

    SYMMETRY BREAKING (Hints):
    - Parameter-binding symmetry: Assigning index-based penalties to shots breaks N-way branching.
    - Goal-shot penalty: Penalizing using goal-destined shots as intermediaries (500 per task).
    - Hand preference: Halving branching by preferring one hand over another (50 per task).
    """

    def __init__(self, barman_gemini3flash_heuristic_3="barman_gemini3flash_heuristic_3"):
        super().__init__(barman_gemini3flash_heuristic_3)
        self.goal_shots = {}  # shot_name -> cocktail_name
        self.all_shots = []
        self.hand_preference = "right"  # Non-preferred hand for penalty
        self.fact_map = {}  # (pred, tuple(args)) -> global_id
        self.goal_ids = []
        self.task_memo = {}
        self.task_min_costs = {
            "AchieveContainsShotCocktail": 15,
            "AchieveContainsShakerCocktail": 10,
            "AchieveContainsShakerIngredient": 5,
            "AchieveContainsShotIngredient": 3,
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
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _parse_task_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def _memoized_parse_task(self, name):
        if name not in self.task_memo:
            self.task_memo[name] = self._parse_task_name(name)
        return self.task_memo[name]

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred:
                self.fact_map[(pred, tuple(args))] = f.global_id

        # Identify goal shots from initial task network
        for task in initial_node.task_network:
            name, args = self._memoized_parse_task(task.name)
            if name == "AchieveContainsShotCocktail" and len(args) >= 2:
                self.goal_shots[args[0]] = args[1]

        # Identify all shots for symmetry breaking (alphabetical index)
        shots_set = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "clean" and args and "shot" in args[0]:
                shots_set.add(args[0])
        self.all_shots = sorted(list(shots_set))

        # Precompute state-based goal IDs
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "contains":
                f_id = self.fact_map.get((pred, tuple(args)))
                if f_id is not None:
                    self.goal_ids.append(f_id)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        h = 0

        # 1. State-based goal distance
        for g_id in self.goal_ids:
            if not ((state >> g_id) & 1):
                h += 15

        # 2. Task network analysis
        for task in task_network:
            t_name, t_args = self._memoized_parse_task(task.name)

            # State-aware cost pruning
            satisfied = False
            if t_name == "AchieveCleanShot" and t_args:
                f_id = self.fact_map.get(("clean", (t_args[0],)))
                if f_id is not None and (state >> f_id) & 1: satisfied = True
            elif t_name == "AchieveCleanShaker" and t_args:
                f_id = self.fact_map.get(("clean", (t_args[0],)))
                if f_id is not None and (state >> f_id) & 1: satisfied = True
            elif t_name == "AchieveHandEmpty" and t_args:
                f_id = self.fact_map.get(("handEmpty", (t_args[0],)))
                if f_id is not None and (state >> f_id) & 1: satisfied = True
            elif t_name == "AchieveHolding" and len(t_args) > 1:
                f_id = self.fact_map.get(("holding", (t_args[0], t_args[1])))
                if f_id is not None and (state >> f_id) & 1: satisfied = True
            elif t_name == "AchieveOnTable" and t_args:
                f_id = self.fact_map.get(("ontable", (t_args[0],)))
                if f_id is not None and (state >> f_id) & 1: satisfied = True

            if satisfied:
                continue

            # Accumulate base cost
            h += self.task_min_costs.get(t_name, 1)

            # 3. Symmetry Breaking and Bottleneck Penalties
            shot_found = None
            for arg in t_args:
                if arg in self.all_shots:
                    shot_found = arg
                    break

            if shot_found:
                # Penalty for using a goal shot for anything other than its final cocktail
                is_final_delivery = (t_name == "AchieveContainsShotCocktail" or t_name == "pour-shaker-to-shot")
                if shot_found in self.goal_shots and not is_final_delivery:
                    h += 500
                
                # Symmetry breaking: increasing penalties for working shots
                h += self.all_shots.index(shot_found) * 100

            # Hand preference penalty (Halve branching)
            if any(self.hand_preference in arg for arg in t_args):
                h += 50

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
