from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic6(Heuristic):
    """
    Domain-dependent heuristic for Barman_BDI.

    Bottleneck Analysis:
    The central bottleneck in the Barman domain is the Shaker. Only one shaker is typically 
    available, and it must go through a strict cycle: clean -> fill Part1 -> fill Part2 -> shake -> pour -> clean.
    This serialization of work means that the depth of the task network is a strong indicator of progress.

    Admissibility Sketches:
    1. Task Progress Term (h_tasks): Each task in the network represents a mandatory sequence of future actions.
       Assigning conservative lower bounds (e.g., 10 for a full cocktail, 3 for an ingredient) ensures we don't
       overestimate the number of primitive actions while providing strong guidance.
    2. Predicate Goal Term (h_goals): If the model specifies bitwise goals, each unsatisfied goal requires at 
       least one primitive action (admissible).
    """

    def __init__(self, barman_gemini3flash_heuristic_6="barman_gemini3flash_heuristic_6"):
        super().__init__(barman_gemini3flash_heuristic_6)
        self.clean_ids = {}
        self.contains_ids = {}
        self.holding_ids = {}
        self.empty_ids = {}
        self.goal_predicates = []

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

    def initialize(self, model: Model, initial_node: HTNNode):
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "clean":
                self.clean_ids[args[0]] = fact.global_id
            elif pred == "contains":
                self.contains_ids[(args[0], args[1])] = fact.global_id
            elif pred == "holding":
                self.holding_ids[(args[0], args[1])] = fact.global_id
            elif pred == "empty":
                self.empty_ids[args[0]] = fact.global_id

        # Map bitwise goals
        self.goal_predicates = []
        goal_names = model.state_explicit_repr(model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "contains":
                self.goal_predicates.append((self.contains_ids.get((args[0], args[1])), 10))
            elif pred == "clean":
                self.goal_predicates.append((self.clean_ids.get(args[0]), 2))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        h_tasks = 0
        for task in task_network:
            name_parts = task.name.split()
            task_type = name_parts[0]
            args = name_parts[1:]

            # Conservative state-aware task costing
            if task_type == "AchieveContainsShotCocktail":
                # If goal already satisfied in state, cost is 0, else 10
                shot, cocktail = args[0], args[1]
                fid = self.contains_ids.get((shot, cocktail))
                if fid is not None and not ((state >> fid) & 1):
                    h_tasks += 10
            elif task_type == "AchieveContainsShakerCocktail":
                shaker, cocktail = args[0], args[1]
                fid = self.contains_ids.get((shaker, cocktail))
                if fid is not None and not ((state >> fid) & 1):
                    h_tasks += 7
            elif task_type == "AchieveContainsShakerIngredient":
                shaker, ing = args[0], args[1]
                fid = self.contains_ids.get((shaker, ing))
                if fid is not None and not ((state >> fid) & 1):
                    h_tasks += 3
            elif task_type == "AchieveContainsShotIngredient":
                shot, ing = args[0], args[1]
                fid = self.contains_ids.get((shot, ing))
                if fid is not None and not ((state >> fid) & 1):
                    h_tasks += 2
            elif task_type == "AchieveCleanShaker" or task_type == "AchieveCleanShot":
                cont = args[0]
                fid = self.clean_ids.get(cont)
                if fid is not None and not ((state >> fid) & 1):
                    h_tasks += 2
            elif task_type == "AchieveHolding":
                hand, cont = args[0], args[1]
                fid = self.holding_ids.get((hand, cont))
                if fid is not None and not ((state >> fid) & 1):
                    h_tasks += 1
            else:
                # Default cost for primitive operators and other tasks
                h_tasks += getattr(task, 'cost', 1)

        # Admissible Goal predicate term
        h_goals = 0
        for fid, cost in self.goal_predicates:
            if fid is not None and not ((state >> fid) & 1):
                # We use 1 to remain strictly admissible here, 
                # as tasks already capture high-level work.
                h_goals += 1

        return max(h_tasks, h_goals)
