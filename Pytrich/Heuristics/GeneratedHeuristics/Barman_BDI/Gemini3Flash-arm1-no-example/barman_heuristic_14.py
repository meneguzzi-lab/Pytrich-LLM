from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# BOTTLENECK ANALYSIS:
# The single Shaker (shaker1) is the primary bottleneck of the Barman_BDI domain.
# Every cocktail requires a serial process of cleaning the shaker, adding two ingredients 
# (via a clean shot), shaking, and pouring. Since there is only one shaker, these 
# processes cannot overlap for different cocktails, making the shaker clean/fill/shake cycle 
# the critical path for any plan.

# ADMISSIBILITY SKETCHES:
# 1. Task Network Cost Sum: Each grounded AbstractTask and Operator in the task network 
#    represents unavoidable work. By assigning a conservative lower bound (minimum 
#    possible actions to decompose/satisfy) to each task based on the current state 
#    (e.g., a task costs 0 if its effect is already true), we create an admissible sum.
# 2. Sequential Bottleneck: Because the shaker is a single resource, the total plan 
#    length must be at least the sum of the remaining shaker-processing costs for all 
#    unfulfilled cocktail goals.

class BarmanBDIGemini3FlashHeuristic14(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_14="barman_gemini3flash_heuristic_14"):
        super().__init__(barman_gemini3flash_heuristic_14)
        self.task_data = {}  # Map: task_id -> (done_fact_id, base_cost)
        self.fact_map = {}   # Map: (predicate, args_tuple) -> global_id

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
        if "[" in name and name.endswith("]"):
            parts = name.split("[")
            task_type = parts[0]
            args = parts[1][:-1].split(",")
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates for fast bitwise state checking
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_map[(pred, tuple(args))] = fact.global_id

        # 2. Precompute costs for abstract tasks based on grounded parameters
        # Costs are conservative lower bounds on actions required to resolve the task.
        task_cost_lookup = {
            "AchieveContainsShotCocktail": 10,
            "AchieveContainsShakerCocktail": 7,
            "AchieveContainsShakerIngredient": 3,
            "AchieveContainsShotIngredient": 2,
            "AchieveCleanShaker": 1,
            "AchieveCleanShot": 1,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 1
        }

        # Map task completion facts to task types
        task_to_fact_pred = {
            "AchieveContainsShotCocktail": "contains",
            "AchieveContainsShakerCocktail": "contains",
            "AchieveContainsShakerIngredient": "contains",
            "AchieveContainsShotIngredient": "contains",
            "AchieveCleanShaker": "clean",
            "AchieveCleanShot": "clean",
            "AchieveHandEmpty": "handEmpty",
            "AchieveHolding": "holding",
            "AchieveOnTable": "ontable"
        }

        for t in model.abstract_tasks:
            t_type, t_args = self._parse_task_name(t.name)
            base_cost = task_cost_lookup.get(t_type, 1)
            fact_id = -1
            
            pred = task_to_fact_pred.get(t_type)
            if pred:
                # Args mapping for specific completion checks
                if pred == "handEmpty":
                    f_args = (t_args[0],)
                elif pred == "holding":
                    # Achievement check for AchieveHolding is usually holding(hand, container)
                    f_args = (t_args[0], t_args[1])
                elif pred == "ontable":
                    f_args = (t_args[0],)
                elif pred == "clean":
                    f_args = (t_args[0],)
                else: # contains
                    f_args = (t_args[0], t_args[1])
                
                fact_id = self.fact_map.get((pred, f_args), -1)
            
            self.task_data[t.global_id] = (fact_id, base_cost)

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _compute(self, state, task_network):
        h_val = 0
        for task in task_network:
            # Try Abstract Task lookup
            if task.global_id in self.task_data:
                fact_id, base_cost = self.task_data[task.global_id]
                # If the goal fact is not true in the current state, add cost
                if fact_id == -1 or not ((state >> fact_id) & 1):
                    h_val += base_cost
            else:
                # Operator lookup
                h_val += getattr(task, 'cost', 1)
        
        # Add a tiny tie-breaker for network complexity to encourage decomposition
        h_val += 0.001 * len(task_network)
        return h_val

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return int(h_val)
