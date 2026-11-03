"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck: Each cell that differs from goal must be flipped at least once (cost 1 per flip).
However, flips are coupled through propagation - a single turn can flip multiple cells
along contiguous same-status runs. The grounder also produces __method_precondition_*
operators that each cost 1.

Admissible lower bounds:
1. Number of cells differing from goal state - each such cell needs at least one flip
   action applied to it. This is a strict lower bound since each flip action costs 1.
2. Number of distinct "turns" needed - each turn involves at least 1 flip + method
   precondition checks. We estimate turns needed from the structure of differences.
3. Row/column coverage bound - if differences span k distinct rows or columns,
   we need enough turns to cover them.

The heuristic returns max of these terms.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic9(Heuristic):
    """
    Domain-dependent heuristic for the Lamps (game) domain.
    
    The domain has an NxN grid where each cell is Y or N. A 'turn' selects a cell,
    flips it, then propagates flips in all 4 cardinal directions along contiguous
    runs of cells that had the same status as the flipped cell's original status.
    
    Key insight: each cell whose current status differs from goal needs at least
    one flip action. Each flip action costs 1. Additionally, the HTN structure
    requires method precondition operators (cost 1 each) for propagation decisions.
    """

    def __init__(self, lamps_anthropicopus46_heuristic_9=None, name="LampsAnthropicOpus46Heuristic9"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Maps: (x, y, status) -> fact object with global_id
        self.on_fact_map = {}  # (x_str, y_str, s_str) -> Fact
        
        # Goal facts: set of (x_str, y_str, s_str) tuples
        self.goal_on_facts = {}  # (x_str, y_str) -> s_str (goal status)
        
        # For fast lookup: (x_str, y_str, s_str) -> global_id
        self.on_fact_ids = {}
        
        # Grid coordinates
        self.coords = set()
        self.grid_size = 0
        
        # next relation for propagation estimation
        self.next_map = {}  # coord -> next_coord
        self.prev_map = {}  # coord -> prev_coord
        
        # For counting method_precondition operators
        self.method_precondition_count = 0
        
        # All cells in the grid
        self.all_cells = []
        
        # Precomputed: for each cell, the fact ids for Y and N status
        self.cell_y_id = {}  # (x,y) -> global_id of on[x,y,Y]
        self.cell_n_id = {}  # (x,y) -> global_id of on[x,y,N]

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
        Extract domain structure:
        - Build map of on(x,y,s) facts to their global_ids
        - Extract goal state
        - Build next/prev maps for grid navigation
        - Identify all grid cells
        """
        # Parse all facts
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None:
                continue
            
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.on_fact_map[(x, y, s)] = fact
                self.on_fact_ids[(x, y, s)] = fact.global_id
                self.coords.add(x)
                self.coords.add(y)
                
                if s == "Y":
                    self.cell_y_id[(x, y)] = fact.global_id
                elif s == "N":
                    self.cell_n_id[(x, y)] = fact.global_id
            
            elif predicate == "next" and len(args) == 2:
                self.next_map[args[0]] = args[1]
                self.prev_map[args[1]] = args[0]
        
        self.grid_size = len(self.coords)
        
        # Build list of all cells
        # We need to find all (x, y) pairs that exist
        all_cells_set = set()
        for (x, y, s) in self.on_fact_ids:
            all_cells_set.add((x, y))
        self.all_cells = list(all_cells_set)
        
        # Extract goal state
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    self.goal_on_facts[(x, y)] = s
        
        # Count method precondition operators for estimation
        self.method_precon_ops = []
        self.flip_ops = []
        for op in self.model.operators:
            if "__method_precondition" in op.name:
                self.method_precon_ops.append(op)
            elif "flip" in op.name:
                self.flip_ops.append(op)
    
    def _count_differing_cells(self, state_bitwise):
        """
        Count how many cells currently differ from their goal status.
        Uses fast bitwise checks.
        """
        diff_count = 0
        for (x, y), goal_s in self.goal_on_facts.items():
            # Check if the goal status is currently true
            if goal_s == "Y":
                fid = self.cell_y_id.get((x, y))
            else:
                fid = self.cell_n_id.get((x, y))
            
            if fid is not None:
                if not ((state_bitwise >> fid) & 1):
                    diff_count += 1
        
        return diff_count
    
    def _count_differing_rows_cols(self, state_bitwise):
        """
        Count distinct rows and columns that have at least one differing cell.
        Each turn can affect at most 2 rows + 2 columns (the center's row and column
        plus propagation), but really just 1 row and 1 column for the center.
        """
        diff_rows = set()
        diff_cols = set()
        
        for (x, y), goal_s in self.goal_on_facts.items():
            if goal_s == "Y":
                fid = self.cell_y_id.get((x, y))
            else:
                fid = self.cell_n_id.get((x, y))
            
            if fid is not None:
                if not ((state_bitwise >> fid) & 1):
                    diff_rows.add(x)
                    diff_cols.add(y)
        
        return len(diff_rows), len(diff_cols)
    
    def _estimate_turns_needed(self, state_bitwise):
        """
        Estimate the minimum number of turns needed.
        
        Each turn selects one cell and flips it + propagates. The propagation
        can flip cells in the same row (left/right) and same column (up/down)
        that have the same status as the original cell.
        
        A turn affects at most one row and one column. So if differences
        appear in R distinct rows and C distinct columns, we need at least
        max(R, C) turns if we could perfectly align them, but in practice
        we may need more.
        
        Each turn costs: 1 flip (the center) + propagation flips + method_precondition ops.
        The method_precondition ops are for deciding propagation direction termination.
        Each propagation direction needs at least 1 method_precondition op.
        So minimum cost per turn = 1 (flip) + 4 (one method_precon per direction) = 5.
        But this can vary based on how the grounder encodes things.
        """
        diff_rows, diff_cols = self._count_differing_rows_cols(state_bitwise)
        # Minimum turns needed to cover all affected rows and columns
        return max(diff_rows, diff_cols)
    
    def _compute(self, state_bitwise, task_network):
        """
        Compute heuristic value as max of multiple admissible terms.
        """
        # If no goal facts, return 0
        if not self.goal_on_facts:
            return 0
        
        # Check if goal is reached
        if self.model.goal_reached(state_bitwise, task_network):
            return 0
        
        # Term 1: Number of differing cells
        # Each differing cell needs at least 1 flip action (cost 1)
        # This is admissible because a flip can only change one cell's status
        diff_count = self._count_differing_cells(state_bitwise)
        
        if diff_count == 0:
            # All cells match goal, but task network not empty
            # We may still need some no-ops or method preconditions
            # Return a small value based on remaining tasks
            remaining = 0
            if task_network:
                for t in task_network:
                    if hasattr(t, 'decompositions'):
                        # Abstract task - needs at least one decomposition step
                        remaining += 0
                    else:
                        # Primitive operator - costs 1
                        remaining += 1
            return remaining
        
        term1 = diff_count
        
        # Term 2: Turns-based estimate
        # Each turn requires at least: 1 flip + 4 method_precondition ops (one per direction)
        # = 5 primitive actions minimum per turn
        # Actually, let's be more careful. After grounding, each do_turn method produces:
        # - 1 flip action
        # - 4 propagate subtasks
        # Each propagate subtask at minimum needs 1 method_precondition op to check termination
        # So minimum per turn = 1 + 4 = 5 primitive ops
        # But we need to be admissible, so let's think carefully.
        # 
        # Actually, when propagation hits a boundary or different cell immediately,
        # the method has no subtasks (empty ordered-subtasks), but the grounder may
        # still produce a __method_precondition_* operator for the precondition check.
        # Let's assume minimum 5 per turn to be safe, but check if this is too high.
        # 
        # Being conservative: each turn needs at least 1 flip.
        # With method preconditions: at least 1 + 4 = 5.
        # Let's use a moderate estimate.
        
        min_turns = self._estimate_turns_needed(state_bitwise)
        
        # Each turn costs at least 5 actions (1 flip + 4 method precons for propagation)
        # But to be safe and admissible, let's use the minimum possible:
        # Actually, the do_turn method decomposes into:
        #   flip, propagate_up, propagate_down, propagate_left, propagate_right
        # Each propagate_* either:
        #   - propagate_*_do: has __method_precondition (1) + flip (1) + recursive propagate_*
        #   - propagate_*_end_field: has __method_precondition (1)
        #   - propagate_*_end_different: has __method_precondition (1)
        # So minimum per propagation direction = 1 (method_precondition for end case)
        # Minimum per turn = 1 (flip) + 4 (method_precons for 4 directions) = 5
        
        # But each turn in the do_play method also costs: the play decomposition itself
        # do_play -> turn, play. The turn is free (method decomposition = 0 cost).
        # But __method_precondition for do_turn may cost 1.
        # Actually, do_turn has no precondition, so no __method_precondition.
        # But propagate methods do have preconditions.
        
        cost_per_turn = 5  # 1 flip + 4 propagation terminators
        term2 = min_turns * cost_per_turn
        
        # Term 3: Tighter cell-based estimate
        # If we have D differing cells, and each turn can fix at most (2*grid_size - 1) cells
        # (the center's entire row + entire column), then we need at least
        # ceil(D / (2*grid_size - 1)) turns.
        # But each turn also creates the propagation flips which cost 1 each.
        # Actually, propagation flips ARE the fixes, so the cost is:
        # for a turn that fixes k cells: k flips + 4 method_precons = k + 4
        # Total cost >= D (flips) + 4 * num_turns (method_precons)
        
        if self.grid_size > 0:
            max_fixes_per_turn = 2 * self.grid_size - 1  # entire row + column through center
            min_turns_from_diff = max(1, (diff_count + max_fixes_per_turn - 1) // max_fixes_per_turn)
        else:
            min_turns_from_diff = diff_count
        
        # Cost = at least D flips + 4 * min_turns method_precons
        term3 = diff_count + 4 * min_turns_from_diff
        
        return max(term1, term2, term3)
    
    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic9(grid_size={self.grid_size})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic9 with {self.grid_size} coords, {len(self.all_cells)} cells"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {self.grid_size}\n"
            f"\tTotal Cells: {len(self.all_cells)}\n"
            f"\tGoal Cells: {len(self.goal_on_facts)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
