"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck: Each cell that differs from the goal must be flipped at least once
(cost 1 per flip). Propagation can flip multiple cells in one turn, but the
center flip is always required. Additionally, each 'turn' task decomposition
generates __method_precondition operators that cost 1 each.

Admissible lower bounds:
1. Number of mismatched cells: each flip fixes exactly one cell, so
   #mismatched <= #flips needed. Admissible because flips are the only
   actions that change cell status.
2. Task network estimate: remaining abstract tasks in the network imply
   a minimum number of method precondition operators and flips.
3. Combined max of the above.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic18(Heuristic):
    """
    Heuristic for the Lamps domain that estimates remaining cost based on:
    - Number of cells whose current status differs from goal status
    - Task network structure (remaining turns imply minimum costs)
    """

    def __init__(self, lamps_anthropicopus46_heuristic_18=None, name="LampsAnthropicOpus46Heuristic18"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Maps from (x, y, status) -> fact.global_id
        self.on_fact_ids = {}  # (x, y, status_str) -> global_id
        
        # Goal: list of (global_id_for_desired_status, global_id_for_undesired_status)
        # For each goal cell, we store the fact that should be true
        self.goal_fact_ids = []  # list of global_ids that must be true in goal
        
        # For mismatch counting: list of (goal_fact_global_id,) - just check if goal fact is true
        self.goal_checks = []  # list of global_ids: if bit not set, cell is mismatched
        
        # For task network analysis
        self.turn_task_ids = set()  # global_ids of 'turn' abstract tasks
        self.play_task_ids = set()  # global_ids of 'play' abstract tasks
        self.propagate_task_ids = set()  # global_ids of propagate_* abstract tasks
        
        # Operator analysis
        self.flip_op_ids = set()  # global_ids of flip operators
        self.method_precond_op_ids = set()  # global_ids of __method_precondition operators
        
        # Grid info
        self.grid_size = 0
        self.coords = []
        self.next_map = {}  # coord -> next coord
        self.prev_map = {}  # coord -> prev coord
        self.max_coord = None
        self.min_coord = None

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start_time
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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

    def _preprocess(self):
        """
        Parse all facts and build lookup structures.
        """
        # Build fact maps
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None:
                continue
            
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.on_fact_ids[(x, y, s)] = fact.global_id
            elif predicate == "next" and len(args) == 2:
                self.next_map[args[0]] = args[1]
                self.prev_map[args[1]] = args[0]
            elif predicate == "max" and len(args) == 1:
                self.max_coord = args[0]
            elif predicate == "min" and len(args) == 1:
                self.min_coord = args[0]

        # Build coordinate list
        if self.min_coord is not None:
            c = self.min_coord
            self.coords.append(c)
            while c in self.next_map:
                c = self.next_map[c]
                self.coords.append(c)
            self.grid_size = len(self.coords)

        # Parse goal facts
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    key = (x, y, s)
                    if key in self.on_fact_ids:
                        self.goal_checks.append(self.on_fact_ids[key])

        # Identify task types for task network analysis
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower()
            if 'turn' in name_lower and 'propagate' not in name_lower and 'play' not in name_lower:
                self.turn_task_ids.add(at.global_id)
            elif 'play' in name_lower:
                self.play_task_ids.add(at.global_id)
            elif 'propagate' in name_lower:
                self.propagate_task_ids.add(at.global_id)

        # Identify operator types
        for op in self.model.operators:
            name_lower = op.name.lower()
            if 'flip' in name_lower and '__method_precondition' not in name_lower:
                self.flip_op_ids.add(op.global_id)
            elif '__method_precondition' in name_lower:
                self.method_precond_op_ids.add(op.global_id)

    def _count_mismatched(self, state_bitwise: int) -> int:
        """
        Count how many goal 'on' facts are not satisfied in the current state.
        Each unsatisfied goal fact means at least one flip is needed.
        This is admissible: each flip action changes exactly one cell.
        """
        count = 0
        for gid in self.goal_checks:
            if not ((state_bitwise >> gid) & 1):
                count += 1
        return count

    def _task_network_estimate(self, task_network) -> float:
        """
        Estimate minimum cost from the task network structure.
        
        Each 'turn' in the task network will produce:
        - At least 1 flip action (the center cell)
        - Method precondition operators for do_turn and propagate methods
        
        Each 'play' task that still needs to decompose into a turn 
        costs at least what a turn costs.
        
        Propagate tasks each cost at least 0 (they can end immediately)
        but if they have preconditions met, they'll produce flips.
        """
        if not task_network:
            return 0
        
        turn_count = 0
        play_count = 0
        flip_count = 0
        method_precond_count = 0
        propagate_count = 0
        
        for task in task_network:
            gid = task.global_id
            if gid in self.turn_task_ids:
                turn_count += 1
            elif gid in self.play_task_ids:
                play_count += 1
            elif gid in self.propagate_task_ids:
                propagate_count += 1
            elif gid in self.flip_op_ids:
                flip_count += 1
            elif gid in self.method_precond_op_ids:
                method_precond_count += 1
        
        # Each turn task decomposes into: flip + 4 propagate tasks
        # The flip costs 1 action. Each propagate can terminate for free.
        # So minimum cost per turn = 1 (the center flip)
        # Plus method precondition operators: do_turn has a precondition check
        # that generates a __method_precondition operator (cost 1)
        # Actually, do_turn method itself may or may not have a method_precondition.
        # Let's be conservative: each turn costs at least 1 (flip) + 1 (method precond for do_turn)
        
        # Each play that needs a turn costs at least what a turn costs + 
        # method precond for do_play
        
        # Conservative: each turn costs at least 1 flip
        # Each pending play that must do a turn costs at least 1 flip + overhead
        
        estimate = flip_count + method_precond_count
        
        # Each turn task implies at least 1 flip
        estimate += turn_count * 1
        
        # For propagate tasks, they might need 0 flips (if they terminate)
        # So we don't add anything for them - admissible
        
        return estimate

    def _compute(self, state_bitwise: int, task_network) -> float:
        """
        Compute heuristic as max of independent admissible bounds.
        """
        # Term 1: Number of mismatched cells
        # Admissible because each flip fixes exactly one mismatch
        mismatched = self._count_mismatched(state_bitwise)
        
        # Term 2: Task network based estimate
        tn_estimate = self._task_network_estimate(task_network)
        
        # Return max of admissible bounds
        h_value = max(mismatched, tn_estimate)
        
        return h_value

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic18(grid={self.grid_size}x{self.grid_size})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic18 with {self.grid_size}x{self.grid_size} grid"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {self.grid_size}x{self.grid_size}\n"
            f"\tGoal Checks: {len(self.goal_checks)}\n"
            f"\tOn Facts: {len(self.on_fact_ids)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
