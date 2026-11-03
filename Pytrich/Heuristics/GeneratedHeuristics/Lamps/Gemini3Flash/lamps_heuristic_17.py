import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
In the Lamps domain, the primary bottleneck is the number of 'flip' actions required to transform 
the initial status of each grid cell (coord x, coord y) into the target status defined by the goal. 
Each flip action costs 1 unit and modifies exactly one cell. While flipping a cell can trigger 
a propagation chain (up, down, left, right), each modified cell in that chain requires its own 
primitive 'flip' action.

Admissibility Sketches:
Term 1: Incorrect Cells Count
- Each cell (x, y) has a binary status (Y/N). If a cell's current status differs from its goal 
  status, at least one 'flip' action must be applied to that specific cell to reach the goal. 
  Since every 'flip' costs 1, the count of mismatched cells is an admissible lower bound on 
  the remaining cost.
- Note: Flipping a cell multiple times (e.g., 3 times) results in the same status change as 
  flipping it once. 1 is the minimum odd number of flips needed for a change.

Term 2: Task Network Guidance (Tie-breaking)
- Plan length is also affected by the number of decomposition methods and precondition check 
  operators. While decompositions cost 0, the grounder-generated '__method_precondition_*' 
  operators cost 1. 
- A small sub-unit tie-breaker based on the task network length helps the search prioritize 
  nodes that have successfully decomposed high-level tasks into primitive actions.
"""

class LampsGemini3FlashHeuristic17(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_17=None, name="LampsGemini3FlashHeuristic17"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        # Map: (coord_x, coord_y) -> {status_string: global_fact_id}
        self.on_facts_by_coord = defaultdict(dict)
        # Map: (coord_x, coord_y) -> status_string
        self.goal_status = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start_time
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Grounded format: +predicate[arg1,arg2,...] or -predicate[arg1,arg2,...]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _preprocess(self):
        # 1. Index all 'on' facts to quickly check grid status
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                # args: [coord_x, coord_y, status]
                x, y, s = args[0], args[1], args[2]
                self.on_facts_by_coord[(x, y)][s] = fact.global_id

        # 2. Extract goal status for each cell
        if self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_fact_names:
                predicate, args = self._parse_fact_name(g_name)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    self.goal_status[(x, y)] = s

    def _compute(self, state_bitwise: int, task_network: list) -> float:
        # Term 1: Admissible lower bound - one flip per mismatched cell
        mismatch_count = 0
        for (x, y), status_map in self.on_facts_by_coord.items():
            goal_s = self.goal_status.get((x, y))
            if goal_s is None:
                continue

            # Identify current status of the cell using bitwise check
            current_s = None
            for s, fid in status_map.items():
                if (state_bitwise >> fid) & 1:
                    current_s = s
                    break
            
            if current_s is not None and current_s != goal_s:
                mismatch_count += 1

        # Term 2: Pending operator costs
        # Operators currently in the task network have a guaranteed cost of at least 1.
        # This includes primitive 'flip' actions and '__method_precondition' operators.
        pending_op_cost = 0
        for task in task_network:
            # If it's an Operator object, add its cost (typically 1 in this domain)
            if hasattr(task, 'cost'):
                pending_op_cost += task.cost

        # Combine terms: mismatch_count is the number of 'flip' actions required.
        # We add a small tie-breaker for the task network length (P6) to differentiate 
        # nodes with similar mismatch counts.
        h = float(mismatch_count + pending_op_cost + 0.001 * len(task_network))
        return h

    def __repr__(self):
        return f"LampsGemini3FlashHeuristic17(goals={len(self.goal_status)})"

    def __str__(self):
        return f"LampsGemini3FlashHeuristic17 counting mismatches for {len(self.goal_status)} cells"
