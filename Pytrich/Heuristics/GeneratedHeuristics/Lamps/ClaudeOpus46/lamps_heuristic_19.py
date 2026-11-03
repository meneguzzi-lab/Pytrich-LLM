"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
  The Lamps domain is a grid-based light-flipping puzzle. Each "turn" picks a cell (x,y),
  flips it, then propagates the flip in all 4 directions along contiguous runs of cells
  sharing the same original status. The only primitive action is `flip`, which costs 1.
  
  The key bottleneck is that each cell whose status differs between current state and goal
  must be flipped at least once (cost 1 per flip). However, a single "turn" can flip
  multiple cells (the center + propagation along 4 directions). So the number of
  differing cells is an upper bound on flips, but we need a lower bound.

Admissible Lower Bounds:
  1. **Differing-cells bound**: Count cells where current status != goal status.
     Each flip action changes exactly one cell. So the number of differing cells
     is a perfect admissible lower bound on the number of flip actions needed.
     This is admissible because each differing cell requires at least one flip action.

  2. **Task-network bound**: Count the minimum number of primitive actions that must
     appear when decomposing the remaining task network. Each `turn` task decomposes
     into at least 1 flip (the center flip) plus method precondition operators.
     We count abstract tasks that will require at least some primitives.

  3. **Method-precondition-aware bound**: The grounder produces `__method_precondition_*`
     operators that also cost 1 each. Each `do_turn` method has a precondition check
     (1 op) + 1 flip = at least 2 ops per turn. Plus `do_play` has a precondition.
     We estimate the minimum ops from the task network structure.

The heuristic returns max(differing_cells, task_network_estimate).
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic19(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.
    
    Uses the number of cells differing from goal as the primary admissible
    lower bound, combined with task-network analysis.
    """

    def __init__(self, lamps_anthropicopus46_heuristic_19=None, name="LampsAnthropicOpus46Heuristic19"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Maps for quick lookup
        self.goal_on_facts = {}       # (x, y) -> (fact_for_Y, fact_for_N, goal_status)
        self.on_facts = {}            # fact.global_id -> (x, y, status)
        self.on_fact_y = {}           # (x, y) -> global_id of on[x,y] with status Y
        self.on_fact_n = {}           # (x, y) -> global_id of on[x,y] with status N
        
        # Goal info
        self.goal_cells = []          # List of (x, y, goal_status_is_Y: bool)
        self.goal_y_mask = 0          # Bitmask: bits set for on[x,y,Y] facts that should be true in goal
        self.goal_n_mask = 0          # Bitmask: bits set for on[x,y,N] facts that should be true in goal
        
        # For checking differences efficiently
        self.y_fact_ids = []          # List of (global_id_Y, global_id_N, goal_is_Y) for each goal cell
        
        # Task network analysis
        self.turn_task_ids = set()    # global_ids of abstract tasks named 'turn'
        self.play_task_ids = set()    # global_ids of abstract tasks named 'play'
        self.flip_op_ids = set()      # global_ids of flip operators
        self.method_precond_ids = set()  # global_ids of method precondition operators
        
        # Minimum ops per turn (flip + propagation flips + method preconditions)
        # At minimum, a turn does 1 flip + some method precondition ops
        self.min_ops_per_turn = 1     # Will be refined during preprocessing
        
        # For efficient task network scanning
        self.is_primitive = {}        # global_id -> bool
        self.min_cost_cache = {}      # abstract_task global_id -> min primitive cost

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
        Build lookup structures for efficient heuristic computation.
        """
        # Parse all facts and build on-fact maps
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.on_facts[fact.global_id] = (x, y, status)
                if status == "y":
                    self.on_fact_y[(x, y)] = fact.global_id
                elif status == "n":
                    self.on_fact_n[(x, y)] = fact.global_id
        
        # Parse goal state
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, status = args[0], args[1], args[2]
                    goal_is_y = (status == "y")
                    self.goal_cells.append((x, y, goal_is_y))
        
        # Build efficient difference-checking structure
        # For each goal cell, store the fact IDs we need to check
        self.y_fact_ids = []
        for (x, y, goal_is_y) in self.goal_cells:
            y_id = self.on_fact_y.get((x, y), -1)
            n_id = self.on_fact_n.get((x, y), -1)
            if y_id >= 0 and n_id >= 0:
                self.y_fact_ids.append((y_id, n_id, goal_is_y))
        
        # Analyze operators for task network heuristic
        for op in self.model.operators:
            name = op.name.lower() if op.name else ""
            if "flip" in name and "method_precondition" not in name and "__" not in name:
                self.flip_op_ids.add(op.global_id)
            if "__method_precondition" in (op.name if op.name else ""):
                self.method_precond_ids.add(op.global_id)
        
        # Analyze abstract tasks
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower() if at.name else ""
            if "turn" in name_lower:
                self.turn_task_ids.add(at.global_id)
            if "play" in name_lower:
                self.play_task_ids.add(at.global_id)
        
        # Compute minimum cost for each abstract task via BFS over decompositions
        self._compute_min_costs()

    def _compute_min_costs(self):
        """
        Compute minimum number of primitive operators for each abstract task.
        Uses iterative fixpoint computation.
        """
        # Initialize: operators cost 1 each
        min_cost = {}
        
        for op in self.model.operators:
            min_cost[op.global_id] = 1
        
        # Initialize abstract tasks with infinity
        for at in self.model.abstract_tasks:
            min_cost[at.global_id] = float('inf')
        
        # Iterative fixpoint
        changed = True
        iterations = 0
        max_iterations = 100  # Safety bound
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for at in self.model.abstract_tasks:
                for dec in at.decompositions:
                    # Cost of this decomposition = sum of costs of subtasks
                    dec_cost = 0
                    feasible = True
                    for subtask in dec.task_network:
                        sc = min_cost.get(subtask.global_id, float('inf'))
                        if sc == float('inf'):
                            feasible = False
                            break
                        dec_cost += sc
                    
                    if feasible and dec_cost < min_cost[at.global_id]:
                        min_cost[at.global_id] = dec_cost
                        changed = True
        
        self.min_cost_cache = min_cost

    def _count_differing_cells(self, state_bitwise: int) -> int:
        """
        Count how many goal cells currently differ from their goal status.
        Each differing cell needs at least 1 flip action.
        This is admissible.
        """
        count = 0
        for (y_id, n_id, goal_is_y) in self.y_fact_ids:
            if goal_is_y:
                # Goal wants Y. Check if Y fact is currently true.
                if not ((state_bitwise >> y_id) & 1):
                    count += 1
            else:
                # Goal wants N. Check if N fact is currently true.
                if not ((state_bitwise >> n_id) & 1):
                    count += 1
        return count

    def _task_network_estimate(self, task_network) -> float:
        """
        Estimate minimum primitive actions from the task network.
        Uses precomputed minimum costs for each task type.
        """
        if not task_network:
            return 0
        
        total = 0
        for task in task_network:
            cost = self.min_cost_cache.get(task.global_id, 1)
            if cost == float('inf'):
                cost = 1  # Fallback
            total += cost
        
        return total

    def _compute(self, state_bitwise: int, task_network) -> float:
        """
        Compute the heuristic value.
        
        Returns max of:
        1. Number of differing cells (admissible: each needs >= 1 flip)
        2. Task network minimum cost estimate
        """
        # If no tasks remain, check if goal is reached
        if not task_network:
            diff = self._count_differing_cells(state_bitwise)
            return diff
        
        # Term 1: differing cells lower bound
        diff_cells = self._count_differing_cells(state_bitwise)
        
        # Term 2: task network lower bound
        tn_estimate = self._task_network_estimate(task_network)
        
        # Return max for tighter bound
        h = max(diff_cells, tn_estimate)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic19(goal_cells={len(self.goal_cells)})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic19 with {len(self.goal_cells)} goal cells"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGoal Cells: {len(self.goal_cells)}\n"
            f"\tY Facts Tracked: {len(self.y_fact_ids)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
