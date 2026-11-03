from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic15(Heuristic):
    """
    Domain-dependent heuristic for Barman_BDI.
    Bottleneck Analysis:
    The bottleneck in the Barman_BDI domain is the shaker and hand availability; every cocktail requires 
    a sequential process of cleaning, filling with multiple ingredients via shots, shaking, and pouring, 
    all while constrained by having only two hands.

    Admissible Lower Bounds:
    1. Task Network Cost: Each grounded task in the network has a minimum number of required actions 
       to be completed. Summing these costs is admissible as the Barman domain tasks are largely serial.
    2. Unmet Goal Cost: Each goal fact (contains shot cocktail) not currently satisfied requires at least 
       one action (pour) but realistically many more. Counting them is a safe lower bound.
    """

    def __init__(self, barman_gemini3flash_heuristic_15="barman_gemini3flash_heuristic_15"):
        super().__init__(barman_gemini3flash_heuristic_15)
        self.task_info = {} # global_id -> (cost, fact_id_to_check)
        self.goal_ids = []
        self.fact_map = {} # (pred, args) -> global_id

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, tuple(args), prefix
        return None, (), ""

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map facts for O(1) lookup
        for fact in model.facts:
            pred, args, prefix = self._parse_fact_name(fact.name)
            if pred and prefix == "+":
                self.fact_map[(pred, args)] = fact.global_id

        # 2. Map goals
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args, prefix = self._parse_fact_name(g_name)
            if (pred, args) in self.fact_map:
                self.goal_ids.append(self.fact_map[(pred, args)])

        # 3. Precompute Task Costs and associated Fact IDs
        # Standard costs (lower bounds on actions to achieve from scratch)
        base_costs = {
            "AchieveContainsShotCocktail": 10,
            "AchieveContainsShakerCocktail": 6,
            "AchieveContainsShakerIngredient": 2,
            "AchieveContainsShotIngredient": 1,
            "AchieveCleanShaker": 1,
            "AchieveCleanShot": 1,
            "AchieveHandEmpty": 1,
            "AchieveHolding": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 1
        }

        for i in range(model.iabt_init, model.iabt_end + 1):
            task = model.get_component(i)
            parts = task.name.split()
            t_pred = parts[0]
            t_args = tuple(parts[1:])
            
            cost = base_costs.get(t_pred, 1)
            f_id = None
            
            # Map specific tasks to facts they achieve
            if t_pred in ["AchieveContainsShotCocktail", "AchieveContainsShakerCocktail", 
                        "AchieveContainsShakerIngredient", "AchieveContainsShotIngredient"]:
                f_id = self.fact_map.get(("contains", t_args))
            elif t_pred == "AchieveCleanShaker":
                f_id = self.fact_map.get(("clean", (t_args[0],)))
            elif t_pred == "AchieveCleanShot":
                f_id = self.fact_map.get(("clean", (t_args[0],)))
            elif t_pred == "AchieveHandEmpty":
                f_id = self.fact_map.get(("handEmpty", (t_args[0],)))
            elif t_pred == "AchieveHolding":
                f_id = self.fact_map.get(("holding", (t_args[0], t_args[1])))
            elif t_pred == "AchieveOnTable":
                f_id = self.fact_map.get(("ontable", (t_args[0],)))
            elif t_pred == "DoPourShakerToShot":
                # args: shaker, shot, cocktail
                f_id = self.fact_map.get(("contains", (t_args[1], t_args[2])))

            self.task_info[task.global_id] = (cost, f_id)

        # Add costs for operators
        for i in range(model.iop_init, model.iop_end + 1):
            op = model.get_component(i)
            self.task_info[op.global_id] = (1, None)

        return super().initialize(model, self._compute(initial_node.state, initial_node.task_network))

    def _compute(self, state, task_network):
        # Term 1: Task Network progress cost
        tn_cost = 0
        for task in task_network:
            tid = task.global_id
            if tid in self.task_info:
                cost, fid = self.task_info[tid]
                if fid is not None:
                    if not ((state >> fid) & 1):
                        tn_cost += cost
                else:
                    tn_cost += cost
            else:
                tn_cost += 1

        # Term 2: Global goal progress cost
        unmet_goals = 0
        for gid in self.goal_ids:
            if not ((state >> gid) & 1):
                unmet_goals += 1

        return max(tn_cost, unmet_goals)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
