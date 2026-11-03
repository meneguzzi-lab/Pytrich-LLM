"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
The primary bottleneck in this domain is the mandatory cost associated with the propagation logic. 
Every change to the board (a flip) is either the start of a 'turn' or triggered by a 'propagate' method.
In Pytrich, every primitive action (like 'flip') and every method-precondition operator 
(automatically generated for methods with preconditions) costs 1. 
The propagation methods (propagate_up_do, propagate_up_end_field, etc.) all have preconditions,
meaning every step in a propagation chain adds at least 1 unit of cost even before the 'flip' action itself.

Admissibility Sketches:
1. State-based Term (h_state): 
   A cell (x, y) that differs from the goal status MUST be flipped at least once. 
   - A 'turn' flipping k cells costs: 1 (initial flip) + 2*(k-1) (each propagation step costs 1 for the method precon + 1 for the flip) 
     + 4 (each of the 4 propagation chains started by 'turn' must eventually hit an 'end' method, each costing 1 for its precondition).
   - Total cost for k flips in one turn: 1 + 2k - 2 + 4 = 2k + 3.
   - Total cost for K flips across m turns: 2K + 3m.
   - Since K >= Hamming Distance (number of wrong cells) and m >= 1 if K > 0, then 2*Hamming + 3 is a lower bound.

2. Task-based Term (h_tn):
   Each task currently in the Task Network must eventually be resolved into primitive actions.
   - 'turn' task: Resolves into 1 'flip' action and 4 'propagate' tasks. Each 'propagate' task costs at least 1 (precondition action).
     Min cost for 'turn' = 1 + 4 = 5.
   - 'propagate' task: Resolves using a method, all of which have a cost-1 precondition action. Min cost = 1.
   - 'flip' action: Cost = 1.
   - 'play' task: Can decompose to empty (0 cost) if goal met, or to 'turn' if board wrong. 
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class LampsGemini3FlashHeuristic15(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_15=None, name="LampsGemini3FlashHeuristic15"):
        super().__init__(name=name)
        self.goal_statuses = {}  # (x, y) -> target_status
        self.on_fact_ids = {}    # (x, y, status) -> global_id
        self.coords = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse goal state
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for name in goal_fact_names:
                pred, args = self._parse_fact_name(name)
                if pred == 'on' and len(args) == 3:
                    # format: on[x, y, status]
                    self.goal_statuses[(args[0], args[1])] = args[2]

        # 2. Build fact ID map for fast bitwise checking
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'on' and len(args) == 3:
                self.on_fact_ids[(args[0], args[1], args[2])] = fact.global_id

        self.coords = list(self.goal_statuses.keys())
        
        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

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

    def _get_task_cost(self, task, hamming):
        name = task.name.lower()
        # Primitive operator costs are usually 1 in Pytrich
        if "flip" in name:
            return 1
        if "method_precondition" in name:
            return 1
        # Abstract Task lower bounds
        if "turn" in name:
            return 5
        if "propagate" in name:
            return 1
        if "play" in name:
            # play costs 0 if board is correct (end_game), but at least 5 if wrong (must do a turn)
            return 5 if hamming > 0 else 0
        return 0

    def _compute(self, state_bitwise: int, task_network) -> float:
        # Term 1: Hamming Distance based cost
        hamming = 0
        for coord in self.coords:
            target_status = self.goal_statuses[coord]
            fact_id = self.on_fact_ids.get((coord[0], coord[1], target_status))
            if fact_id is not None:
                # Check if target status bit is NOT set
                if not ((state_bitwise >> fact_id) & 1):
                    hamming += 1
        
        # Admissible cost based on mandatory propagation checks: 2*K + 3 per turn
        h_state = (2 * hamming + 3) if hamming > 0 else 0

        # Term 2: Task Network committed cost
        h_tn = 0
        for task in task_network:
            h_tn += self._get_task_cost(task, hamming)

        # Use max to maintain admissibility while capturing both types of pending work
        return float(max(h_state, h_tn))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
