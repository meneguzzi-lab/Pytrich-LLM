"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
  Each "turn" flips a chosen cell and propagates flips along 4 directions
  to consecutive cells sharing the same original status. The serializing
  constraint is that each mismatched cell (current != goal) must be flipped
  at least once, costing 1 primitive action per flip. Turns group flips but
  each turn costs at least 1 flip plus method-precondition overhead operators.

Admissible Lower Bounds:
  Term 1 (Mismatched cells): Count cells where current state differs from goal.
    Each such cell requires at least one flip action => admissible.
  Term 2 (Task network lower bound): Count minimum primitive actions remaining
    based on the task network structure. Each abstract task in the network
    that will decompose non-trivially requires at least some actions.
  Term 3 (Turn-based estimate): The number of distinct "regions" of mismatched
    cells gives a lower bound on turns needed; each turn has overhead cost.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic10(Heuristic):
    """
    Heuristic for the Lamps domain that estimates remaining cost based on:
    1. Number of cells that differ from goal (direct flip count lower bound)
    2. Task network structure analysis
    3. Turn-count estimation with overhead
    """

    def __init__(self, lamps_anthropicopus46_heuristic_10=None, name="LampsAnthropicOpus46Heuristic10"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Maps for efficient lookup
        self.on_facts = {}        # (x, y, s) -> fact.global_id
        self.goal_config = {}     # (x, y) -> desired_status ('Y' or 'N')
        self.all_cells = set()    # set of (x, y) pairs
        self.goal_bits = 0        # bitwise goal state
        
        # For each cell (x,y), store the global_ids for on[x,y,Y] and on[x,y,N]
        self.cell_y_fact_id = {}  # (x, y) -> global_id of on[x,y,Y]
        self.cell_n_fact_id = {}  # (x, y) -> global_id of on[x,y,N]
        
        # Grid structure
        self.next_map = {}        # coord -> next_coord
        self.prev_map = {}        # coord -> prev_coord  
        self.max_coords = set()
        self.min_coords = set()
        self.coords = set()
        
        # Task type identification
        self.play_task_ids = set()
        self.turn_task_ids = set()
        self.propagate_task_ids = set()
        self.flip_op_ids = set()
        self.method_precon_op_ids = set()
        
        # Precomputed goal mask for fast XOR-based mismatch counting
        self.goal_y_mask = 0  # bits set for on[x,y,Y] facts that are in goal
        self.goal_n_mask = 0  # bits set for on[x,y,N] facts that are in goal
        self.all_on_y_mask = 0  # bits for all on[x,y,Y] facts
        self.all_on_n_mask = 0  # bits for all on[x,y,N] facts

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
        """Parse all facts and build lookup structures."""
        
        # Parse all facts
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None:
                continue
            
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.on_facts[(x, y, s)] = fact.global_id
                self.all_cells.add((x, y))
                self.coords.add(x)
                self.coords.add(y)
                
                if s == "Y":
                    self.cell_y_fact_id[(x, y)] = fact.global_id
                    self.all_on_y_mask |= (1 << fact.global_id)
                elif s == "N":
                    self.cell_n_fact_id[(x, y)] = fact.global_id
                    self.all_on_n_mask |= (1 << fact.global_id)
            
            elif predicate == "next" and len(args) == 2:
                self.next_map[args[0]] = args[1]
                self.prev_map[args[1]] = args[0]
            
            elif predicate == "max" and len(args) == 1:
                self.max_coords.add(args[0])
            
            elif predicate == "min" and len(args) == 1:
                self.min_coords.add(args[0])
        
        # Parse goal state
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    self.goal_config[(x, y)] = s
                    fact_id = self.on_facts.get((x, y, s))
                    if fact_id is not None:
                        if s == "Y":
                            self.goal_y_mask |= (1 << fact_id)
                        else:
                            self.goal_n_mask |= (1 << fact_id)
        
        self.goal_bits = self.model.goals
        
        # Identify task/operator types
        for op in self.model.operators:
            name_lower = op.name.lower()
            if "flip" in name_lower and "method" not in name_lower:
                self.flip_op_ids.add(op.global_id)
            elif "__method_precondition" in name_lower or "method_precondition" in name_lower:
                self.method_precon_op_ids.add(op.global_id)
        
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower()
            if "play" in name_lower:
                self.play_task_ids.add(at.global_id)
            elif "turn" in name_lower and "propagate" not in name_lower:
                self.turn_task_ids.add(at.global_id)
            elif "propagate" in name_lower:
                self.propagate_task_ids.add(at.global_id)

    def _count_mismatched_cells(self, state_bitwise):
        """
        Count the number of cells where current state differs from goal.
        
        For each cell (x,y) in the goal, check if the desired status fact is true.
        If on[x,y,Y] is in goal but not in state, or on[x,y,N] is in goal but not
        in state, that cell is mismatched and needs at least one flip.
        """
        count = 0
        
        # Check Y goals: cells that should be Y
        # A cell is mismatched if goal wants on[x,y,Y] but state doesn't have it
        # We can check: goal_y_mask & ~state gives us Y-goal facts not in state
        mismatched_y = self.goal_y_mask & ~state_bitwise
        count += bin(mismatched_y).count('1')
        
        # Check N goals: cells that should be N  
        mismatched_n = self.goal_n_mask & ~state_bitwise
        count += bin(mismatched_n).count('1')
        
        return count

    def _count_mismatched_cells_popcount(self, state_bitwise):
        """
        Fast mismatch count using bitwise operations.
        Each mismatch requires at least 1 flip action.
        """
        # For Y-goal facts not satisfied in current state
        miss_y = self.goal_y_mask & ~state_bitwise
        # For N-goal facts not satisfied in current state  
        miss_n = self.goal_n_mask & ~state_bitwise
        
        combined = miss_y | miss_n
        return bin(combined).count('1')

    def _task_network_estimate(self, task_network):
        """
        Estimate minimum actions from the task network.
        
        Key observations:
        - A 'play' task can decompose to (turn, play) or empty.
          If there are mismatched cells, at least one turn is needed.
        - A 'turn' task decomposes to: flip + propagate_up + propagate_down + 
          propagate_left + propagate_right. That's at least 1 flip action
          plus method precondition overhead.
        - Each propagate_X can do 0+ flips.
        - Each flip operator is 1 action.
        """
        if not task_network:
            return 0
        
        estimate = 0
        for task_or_op in task_network:
            gid = task_or_op.global_id
            
            if gid in self.flip_op_ids:
                # Primitive flip action = 1
                estimate += 1
            elif gid in self.method_precon_op_ids:
                # Method precondition operators cost 1
                estimate += 1
            elif gid in self.turn_task_ids:
                # A turn does at least 1 flip + method precondition checks
                # The do_turn method has subtasks: flip + 4 propagates
                # At minimum: 1 flip action
                estimate += 1
            elif gid in self.play_task_ids:
                # play will contribute through its subtask estimation
                # Don't double count - handled via mismatch count
                pass
            elif gid in self.propagate_task_ids:
                # Propagation may or may not do flips - 0 lower bound
                pass
            elif hasattr(task_or_op, 'cost'):
                # It's a primitive operator
                estimate += 1
            # Abstract tasks not specifically identified get 0
        
        return estimate

    def _estimate_min_turns(self, state_bitwise):
        """
        Estimate the minimum number of turns needed.
        
        Each turn flips at least 1 cell. A turn consists of:
        - 1 flip action (the center cell)
        - Propagation flips (0 or more in each direction)
        
        If there are M mismatched cells, we need at least ceil(M / max_flips_per_turn)
        turns. But more importantly, each turn costs at least 1 flip action.
        
        A tighter bound: consider that each turn can fix at most (1 + 4*grid_size)
        cells in the best case. But it can also break already-correct cells.
        
        Simple lower bound: if any cells are mismatched, at least 1 turn is needed,
        and each turn costs at least 1 action.
        """
        mismatched = self._count_mismatched_cells_popcount(state_bitwise)
        if mismatched == 0:
            return 0
        
        # Each turn flips the center + potentially chains in 4 directions.
        # In the worst case for the heuristic (best case for the agent),
        # a single turn could fix many cells at once.
        # But each turn costs at least 1 flip action.
        # A rough admissible estimate: at least 1 turn needed if any mismatch.
        # Actually, we can be smarter: consider how many cells each turn can affect.
        
        n_coords = len(self.coords)
        if n_coords == 0:
            n_coords = 1
        
        # Max cells affected per turn: 1 + (n_coords-1)*4 in theory
        # but typically much less. Use a conservative upper bound on
        # cells fixable per turn to get admissible lower bound on turns.
        max_cells_per_turn = 1 + 4 * (n_coords - 1) if n_coords > 1 else 1
        
        # Each turn needs at least 1 flip
        min_turns = max(1, (mismatched + max_cells_per_turn - 1) // max_cells_per_turn)
        
        return min_turns

    def _compute(self, state_bitwise, task_network):
        """
        Compute heuristic as max of multiple admissible terms.
        """
        # Term 1: Direct mismatch count
        # Each mismatched cell needs at least 1 flip action
        mismatched = self._count_mismatched_cells_popcount(state_bitwise)
        
        # If goal is reached (no mismatches and empty task network), return 0
        if mismatched == 0 and (not task_network or len(task_network) == 0):
            return 0
        
        # Term 2: Task network lower bound
        tn_estimate = self._task_network_estimate(task_network)
        
        # Term 3: Turn-based estimate
        # Each turn that does useful work costs at least 1 flip.
        # But we also need to account for turns that undo/redo work.
        turn_estimate = self._estimate_min_turns(state_bitwise)
        
        # The mismatch count is our strongest admissible bound.
        # Each mismatched cell MUST be flipped an odd number of times (at least 1).
        # Flips happen in groups (propagation chains), but each flip is still 1 action.
        h = max(mismatched, tn_estimate, turn_estimate)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic10(cells={len(self.all_cells)})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic10 with {len(self.all_cells)} cells"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Cells: {len(self.all_cells)}\n"
            f"\tNumber of Coords: {len(self.coords)}\n"
            f"\tGoal Config Size: {len(self.goal_config)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
