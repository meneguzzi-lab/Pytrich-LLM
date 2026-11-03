from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-Dependent Heuristic for the Lamps (game) domain.

Bottleneck Analysis:
The Lamps domain is a grid-based puzzle where flipping a lamp status (Y/N) propagates changes to adjacent lamps 
of the same status. The single most restrictive bottleneck is the total number of status mismatches between 
the current state and the goal. Each mismatch MUST eventually be the target of a 'flip' primitive action, 
and each flip action costs exactly 1. Additionally, the propagation mechanism involves methods with 
preconditions, which Pytrich's cost model charges as hidden operators. 

Admissibility Sketches:
1. Mismatch Term: Every lamp whose current status != goal status must be flipped. Each flip costs 1. 
   This term is a lower bound on the number of flip actions required.
2. Task Network Term: Every task in the network (except 'play') corresponds to at least one future action. 
   Specifically, a 'turn' task expands into a 'flip' and four 'propagate' tasks. Each 'propagate' 
   involves a precondition check (cost 1). Thus, a 'turn' costs at least 5 units (1 flip + 4 propagate preconds).
   Each 'propagate' task already in the network costs at least 1 unit.

Guidance Strategy:
We use max(mismatches, task_cost) or a sum of both (slightly non-admissible but better guidance) 
to drive the search toward states that resolve more mismatches or simplify the task network.
"""

class LampsGemini3FlashHeuristic0(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_0=None, name="LampsGemini3FlashHeuristic0"):
        super().__init__(name=name)
        self.goal_on_fact_ids = {}  # (x, y) -> global_id of goal fact
        self.task_min_costs = {}    # task_global_id -> min cost to resolve
        self.num_goal_lamps = 0

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name: +predicate[arg1,arg2]"""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Identify goal status for every coordinate
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            goal_ids = set()
            for f_name in goal_fact_names:
                # Find the global id of this specific fact in model.facts
                for f in self.model.facts:
                    if f.name == f_name:
                        goal_ids.add(f.global_id)
                        predicate, args = self._parse_fact_name(f.name)
                        if predicate == "on" and len(args) >= 3:
                            # key: (x, y), value: fact_id
                            self.goal_on_fact_ids[(args[0], args[1])] = f.global_id
                        break
        
        self.num_goal_lamps = len(self.goal_on_fact_ids)

        # 2. Precompute task minimum costs based on decomposition structure
        # Cost model: flip action = 1, method precondition check = 1, decomposition = 0
        # Abstract Tasks
        for task in self.model.abstract_tasks:
            tname = task.name.lower()
            if "turn" in tname:
                # turn -> flip(1) + 4 * propagate(1) = 5
                self.task_min_costs[task.global_id] = 5
            elif "propagate" in tname:
                # propagate always has a method precondition check action
                self.task_min_costs[task.global_id] = 1
            elif "play" in tname:
                # play can be end_game (0)
                self.task_min_costs[task.global_id] = 0
            else:
                self.task_min_costs[task.global_id] = 0

        # Operators
        for op in self.model.operators:
            oname = op.name.lower()
            if "flip" in oname:
                self.task_min_costs[op.global_id] = 1
            elif "precondition" in oname or "no-op" in oname:
                self.task_min_costs[op.global_id] = 1
            else:
                self.task_min_costs[op.global_id] = 1

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state_bitwise: int, task_network) -> float:
        # Term 1: Mismatch Cost
        # Every coordinate mentioned in goal that is NOT in the goal status is a mismatch.
        # Each mismatch requires a 'flip' action (cost 1).
        mismatches = 0
        for coord, goal_fact_id in self.goal_on_fact_ids.items():
            is_at_goal = (state_bitwise >> goal_fact_id) & 1
            if not is_at_goal:
                mismatches += 1

        # Term 2: Task Network Cost
        # Sum of minimum costs of all tasks currently in the network.
        tn_cost = 0
        for task in task_network:
            tn_cost += self.task_min_costs.get(task.global_id, 0)

        # Admissible combination: Each mismatch is a flip action.
        # If the flip action is already in the TN, they overlap.
        # To be safe and guide well, we return the max or a weighted sum.
        # A sum is better for guidance in Pytrich because the goal is to empty the TN.
        h = mismatches + tn_cost

        # Pruning dead ends
        # if mismatches > 0 and TN is empty, Pytrich handles it, 
        # but if play is gone and we have mismatches, it's a dead end.
        return float(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
