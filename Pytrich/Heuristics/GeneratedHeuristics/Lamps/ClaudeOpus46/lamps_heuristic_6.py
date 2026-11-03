"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
  The Lamps domain models a grid of lamps that can be Y (on) or N (off).
  A "turn" picks a cell (x,y), flips it, then propagates in all 4 directions
  flipping consecutive cells that had the same original status as (x,y).
  The bottleneck is that each turn produces exactly 1 flip for the center cell
  plus additional flips for propagation neighbors. Every cell that differs
  between current state and goal MUST be flipped at least once (cost 1 per flip).
  Additionally, each turn requires at least 1 flip (the center), plus the
  grounder inserts __method_precondition operators that also cost 1 each.

Admissible Lower Bounds:
  1. **Goal-diff count**: Count the number of cells (x,y) where the current
     status differs from the goal status. Each such cell needs at least one
     flip action (cost 1). This is admissible because each flip action changes
     exactly one cell, so we need at least this many flip actions.
  
  2. **Turn-based lower bound**: Each "turn" in the HTN produces flips for
     one center cell plus propagation. But the turn structure also involves
     method precondition operators. We can estimate a minimum number of turns
     needed. Each turn can fix at most ~(2*grid_size - 1) cells (one row + one
     column through the center). A simpler bound: count cells that need
     flipping and note each turn fixes at least 1 cell, so we need at least
     that many turns. Each turn costs at least 1 (the center flip) + overhead
     from method precondition operators. However, the goal-diff already
     captures the flip count well.

  3. **Task network awareness**: If the task network still contains abstract
     tasks (play, turn, propagate_*), we can estimate minimum remaining
     primitive actions. Each 'turn' task decomposes into at least 1 flip.
     Each 'play' containing a turn needs at least 1 flip.

Implementation:
  - In initialize(), we parse all on[x,y,s] facts, build a fast lookup
    from (x,y,status) -> fact global_id, and extract goal state requirements.
  - In __call__(), we use bitwise operations to count mismatches between
    current state and goal, and combine with task network estimates.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic6(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.
    
    Uses a multi-component max of:
      1. Goal-diff count (number of cells needing a flip)
      2. Task-network-based lower bound
    """

    def __init__(self, lamps_anthropicopus46_heuristic_6=None, name="LampsAnthropicOpus46Heuristic6"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Mapping from (x, y) -> {status_str -> fact_global_id}
        # e.g., ("l0", "l1") -> {"Y": 5, "N": 6}
        self.on_facts = {}  # (x, y, status) -> global_id
        
        # Goal requirements: list of (x, y, required_status, fact_global_id)
        self.goal_requirements = []
        
        # For fast goal-diff: we store the goal as a bitmask
        # goal_positive_mask: bits that must be 1 in the state
        # goal_negative_mask: bits that must be 0 in the state
        self.goal_positive_mask = 0
        self.goal_negative_mask = 0
        
        # For each (x,y) pair, store fact_id for Y and N status
        self.cell_facts = {}  # (x, y) -> (Y_fact_global_id, N_fact_global_id)
        
        # Grid info
        self.coords = set()
        self.grid_size = 0
        
        # Task name identification
        self.turn_task_ids = set()  # global_ids of 'turn' abstract tasks
        self.play_task_ids = set()  # global_ids of 'play' abstract tasks
        self.propagate_task_ids = set()  # global_ids of propagate_* abstract tasks
        
        # Operator identification
        self.flip_op_ids = set()  # global_ids of flip operators
        self.method_precon_op_ids = set()  # global_ids of __method_precondition operators
        
        # For the task network heuristic: minimum cost per abstract task type
        self.min_cost_per_task = {}  # global_id -> minimum cost estimate

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
        Parse all facts, build lookup structures, extract goals.
        """
        # Build fact lookups
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.on_facts[(x, y, status)] = fact.global_id
                self.coords.add(x)
                self.coords.add(y)
                
                key = (x, y)
                if key not in self.cell_facts:
                    self.cell_facts[key] = {}
                self.cell_facts[key][status] = fact.global_id
        
        self.grid_size = len(self.coords)
        
        # Extract goal requirements from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, status = args[0], args[1], args[2]
                    fact_id = self.on_facts.get((x, y, status))
                    if fact_id is not None:
                        self.goal_requirements.append((x, y, status, fact_id))
                        self.goal_positive_mask |= (1 << fact_id)
                        
                        # The opposite status fact should be 0
                        opp_status = "N" if status == "Y" else "Y"
                        opp_fact_id = self.on_facts.get((x, y, opp_status))
                        if opp_fact_id is not None:
                            self.goal_negative_mask |= (1 << opp_fact_id)
        
        # Identify abstract tasks
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower()
            if 'turn' in name_lower and 'propagate' not in name_lower and 'play' not in name_lower:
                self.turn_task_ids.add(at.global_id)
            elif 'play' in name_lower:
                self.play_task_ids.add(at.global_id)
            elif 'propagate' in name_lower:
                self.propagate_task_ids.add(at.global_id)
        
        # Identify operators
        for op in self.model.operators:
            name_lower = op.name.lower()
            if 'flip' in name_lower and '__method_precondition' not in name_lower:
                self.flip_op_ids.add(op.global_id)
            elif '__method_precondition' in name_lower:
                self.method_precon_op_ids.add(op.global_id)
        
        # Precompute: for each cell, which goal-required fact id and which
        # opposite fact id. This allows O(1) checking per cell.
        self.goal_cell_info = []  # list of (required_fact_id, opposite_fact_id)
        for (x, y, status, fact_id) in self.goal_requirements:
            opp_status = "N" if status == "Y" else "Y"
            opp_fact_id = self.on_facts.get((x, y, opp_status))
            self.goal_cell_info.append((fact_id, opp_fact_id))

    def _count_goal_diff(self, state):
        """
        Count how many cells differ from their goal status.
        Each such cell needs at least 1 flip action.
        
        Uses bitwise operations for speed.
        """
        count = 0
        for (required_fact_id, opposite_fact_id) in self.goal_cell_info:
            # Check if the required fact is NOT true in the current state
            if not ((state >> required_fact_id) & 1):
                count += 1
        return count

    def _task_network_estimate(self, task_network):
        """
        Estimate minimum remaining cost from the task network.
        
        Each element in the task network is either an Operator or an AbstractTask.
        - Operators cost 1 each (including method_precondition and noop).
        - Abstract tasks have minimum decomposition costs:
          - 'turn' must produce at least 1 flip + method precondition ops for
            propagation (at least 4 propagation tasks, each needing at least
            1 method precondition to check termination = 4 method precondition ops
            + 1 flip = 5 primitive actions minimum)
          - 'play' with do_play decomposes to turn + play, so at least the cost
            of one turn if there's work left. With end_game, it's 0.
          - 'propagate_*' tasks need at least 1 method precondition op to check
            the termination condition.
        """
        cost = 0
        for task in task_network:
            gid = task.global_id
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                # It's a primitive operator - costs 1
                cost += 1
            elif gid in self.turn_task_ids:
                # A turn task: at minimum 1 flip + 4 method_precondition for
                # the 4 propagate termination checks
                cost += 5
            elif gid in self.play_task_ids:
                # play can be end_game (0 cost) or do_play (turn + play)
                # We can't know without state, so we estimate 0 for play itself
                # (the turn within will be counted separately)
                cost += 0
            elif gid in self.propagate_task_ids:
                # Each propagate task needs at least 1 method precondition
                # to evaluate its termination condition
                cost += 1
        return cost

    def _compute(self, state, task_network):
        """
        Compute heuristic as max of multiple admissible lower bounds.
        """
        # Term 1: Goal-diff count
        # Each differing cell needs at least 1 flip action (cost 1 each)
        goal_diff = self._count_goal_diff(state)
        
        if goal_diff == 0:
            # Check if task network is empty too
            if len(task_network) == 0:
                return 0
            # Goal is reached in terms of state, but task network may not be empty
            # Still need to execute remaining tasks
            # Count remaining operators in task network
            remaining_ops = 0
            for task in task_network:
                gid = task.global_id
                if gid >= self.model.iop_init and gid <= self.model.iop_end:
                    remaining_ops += 1
            return remaining_ops
        
        # Term 2: Task network based estimate
        tn_estimate = self._task_network_estimate(task_network)
        
        # Term 3: Turn-based estimate
        # Each turn can fix at most (2 * grid_size - 1) cells (cross pattern)
        # So minimum number of turns needed is ceil(goal_diff / max_cells_per_turn)
        # Each turn costs at least 5 (1 flip + 4 propagation method preconditions)
        # But this is often weaker than goal_diff for small problems
        if self.grid_size > 0:
            max_cells_per_turn = 2 * self.grid_size - 1
            min_turns = (goal_diff + max_cells_per_turn - 1) // max_cells_per_turn
            turn_based = min_turns * 5  # 1 flip + 4 method_precondition ops per turn
        else:
            turn_based = 0
        
        return max(goal_diff, tn_estimate, turn_based)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic6(grid_size={self.grid_size})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic6 with grid_size={self.grid_size}"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {self.grid_size}\n"
            f"\tGoal Requirements: {len(self.goal_requirements)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
