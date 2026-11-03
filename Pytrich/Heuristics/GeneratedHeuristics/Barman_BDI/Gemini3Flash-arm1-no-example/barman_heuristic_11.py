from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic11(Heuristic):
    """
    Heuristic for the Barman_BDI domain.

    1. Bottleneck Identification:
       The core bottleneck is the sequential lifecycle of the shaker. For every cocktail, the shaker must 
       typically be cleaned, filled with two distinct ingredients (via shots), shaken, and then poured. 
       Because shakers and hands are limited resources, this creates a strict serial dependency for 
       each cocktail requirement.

    2. Admissible Lower Bounds:
       - h_tasks: Estimates the minimum number of actions to satisfy each task in the current task network. 
         For example, 'AchieveContainsShotCocktail' requires at least 4-6 actions (fill, pour, fill, pour, shake, pour). 
         This is a lower bound because it ignores hand-swapping and cleaning overhead.
       - h_goals: Counts the number of goal facts (from model.goals) not yet satisfied in the state.
         Each unsatisfied 'contains' or 'clean' goal requires at least one action.
       - h_composite: returns max(h_tasks, h_goals) + small tie-breakers.
    """

    def __init__(self, barman_gemini3flash_heuristic_11="BarmanBDIGemini3FlashHeuristic11"):
        super().__init__(barman_gemini3flash_heuristic_11)
        self.pred_map = {}
        self.shakers = set()
        self.goal_facts = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        # Precompute predicate mappings for O(1) bitwise checks
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.pred_map[(pred, tuple(args))] = fact.global_id
                if pred == "shakerLevel":
                    self.shakers.add(args[0])

        # Precompute goal requirements
        if model.goals:
            goal_strings = self.model.state_explicit_repr(model.goals)
            for g_str in goal_strings:
                p, a = self._parse_fact_name(g_str)
                if p:
                    self.goal_facts.append((p, tuple(a)))

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        # Term 1: Task Network Cost (Primary guidance in HTN)
        h_tasks = 0
        for task in task_network:
            t_type, t_args = self._parse_name(task.name)
            h_tasks += self._estimate_task_cost(state, t_type, t_args)

        # Term 2: Goal Progress (Ensures we move towards the final bitwise state)
        h_goals = 0
        for p, a in self.goal_facts:
            if not self._is_true(state, p, a):
                h_goals += 1

        # Admissible composition: max of independent bounds
        return max(h_tasks, h_goals)

    def _estimate_task_cost(self, state, t_type, t_args):
        if t_type == "AchieveContainsShotCocktail":
            if len(t_args) < 2: return 1
            if self._is_true(state, "contains", [t_args[0], t_args[1]]): return 0
            # Check if any shaker is already ready
            for shaker in self.shakers:
                if self._is_true(state, "contains", [shaker, t_args[1]]) and self._is_true(state, "shaked", [shaker]):
                    return 2 # holding + pour
            return 6 # clean + fill*2 + shake + pour

        elif t_type == "AchieveContainsShakerCocktail":
            if len(t_args) < 2: return 1
            if self._is_true(state, "contains", [t_args[0], t_args[1]]) and self._is_true(state, "shaked", [t_args[0]]): 
                return 0
            return 4

        elif t_type == "AchieveCleanShot" or t_type == "AchieveCleanShaker":
            if len(t_args) >= 1 and self._is_true(state, "clean", [t_args[0]]): return 0
            return 1

        elif t_type == "AchieveHolding":
            if len(t_args) >= 2 and self._is_true(state, "holding", [t_args[0], t_args[1]]): return 0
            return 1

        elif t_type == "AchieveHandEmpty":
            if len(t_args) >= 1 and self._is_true(state, "handEmpty", [t_args[0]]): return 0
            return 1

        return 1

    def _is_true(self, state, pred, args):
        fid = self.pred_map.get((pred, tuple(args)))
        if fid is not None:
            return (state >> fid) & 1
        return 0

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

    def _parse_name(self, name: str):
        name = name.strip()
        if "[" in name:
            parts = name.split("[")
            return parts[0], parts[1].rstrip("]").split(",")
        if "(" in name:
            parts = name.split("(")
            return parts[0], parts[1].rstrip(")").split(",")
        return name, []
