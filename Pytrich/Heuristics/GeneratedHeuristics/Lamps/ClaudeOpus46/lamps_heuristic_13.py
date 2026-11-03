"""Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
  Each turn selects a cell, flips it, then propagates flips along 4 cardinal
  directions through consecutive cells sharing the original status. Each flip
  is one primitive action costing 1. The bottleneck is that every cell whose
  status differs from its goal status must be flipped an odd number of times,
  and every cell matching its goal must be flipped an even number (including 0).
  Since each flip costs 1, the total number of flips is the plan cost.

Admissible Lower Bounds:
  Term 1 - Mismatch count: Each cell that differs from goal needs at least 1
           flip. Multiple flips may cancel, but the parity argument gives us
           that mismatched cells need odd flips >= 1. This is a lower bound
           because each flip action changes exactly one cell.
  
  Term 2 - Turn-based lower bound: Each turn requires at least 1 flip (the
           center cell) plus method precondition checks. If there are any
           mismatches, we need >= 1 turn. Each turn has overhead from
           __method_precondition operators. We estimate minimum turns needed
           and multiply by the minimum overhead per turn.
  
  Term 3 - Row/column analysis: For each row and column, count contiguous
           segments of mismatched cells. Each segment may require a separate
           turn to fix, giving a tighter bound.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic13(Heuristic):
    """
    Heuristic for the Lamps domain based on counting mismatched cells
    and analyzing the structure of required changes.
    """

    def __init__(self, lamps_anthropicopus46_heuristic_13=None, name="LampsAnthropicOpus46Heuristic13"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Maps (x, y, status) -> fact global_id for 'on' predicates
        self.on_fact_ids = {}  # (x, y, s) -> global_id
        
        # Goal state info: (x, y) -> required status
        self.goal_status = {}  # (x, y) -> 'Y' or 'N'
        
        # Grid structure
        self.coords = []  # list of coordinate names
        self.coord_order = {}  # coord_name -> integer index
        self.grid_size = 0
        
        # Next relationships
        self.next_map = {}  # coord -> next_coord
        self.prev_map = {}  # coord -> prev_coord
        self.min_coord = None
        self.max_coord = None
        
        # For fast goal checking: list of (x, y, goal_status, on_Y_gid, on_N_gid)
        self.goal_cells = []
        
        # Precomputed: for each cell, the fact id for on(x,y,Y) and on(x,y,N)
        self.on_y_ids = {}  # (x, y) -> global_id of on[x,y,Y]
        self.on_n_ids = {}  # (x, y) -> global_id of on[x,y,N]
        
        # Method precondition / overhead estimation
        self.method_precondition_count = 0
        self.min_overhead_per_turn = 0
        
        # For row/column analysis
        self.rows = defaultdict(list)  # x -> sorted list of y coords
        self.cols = defaultdict(list)  # y -> sorted list of x coords

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
        """Extract domain structure from grounded facts."""
        coords_set = set()
        
        # Parse all facts
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None:
                continue
            
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.on_fact_ids[(x, y, s)] = fact.global_id
                coords_set.add(x)
                coords_set.add(y)
                if s == "Y":
                    self.on_y_ids[(x, y)] = fact.global_id
                elif s == "N":
                    self.on_n_ids[(x, y)] = fact.global_id
            
            elif predicate == "next" and len(args) == 2:
                self.next_map[args[0]] = args[1]
                self.prev_map[args[1]] = args[0]
            
            elif predicate == "max" and len(args) == 1:
                self.max_coord = args[0]
            
            elif predicate == "min" and len(args) == 1:
                self.min_coord = args[0]
        
        # Build ordered coordinate list
        if self.min_coord is not None:
            c = self.min_coord
            idx = 0
            while c is not None:
                self.coords.append(c)
                self.coord_order[c] = idx
                idx += 1
                c = self.next_map.get(c)
            self.grid_size = len(self.coords)
        else:
            self.coords = sorted(list(coords_set))
            for i, c in enumerate(self.coords):
                self.coord_order[c] = i
            self.grid_size = len(self.coords)
        
        # Build row/column structure
        for x in self.coords:
            for y in self.coords:
                if (x, y) in self.on_y_ids or (x, y) in self.on_n_ids:
                    self.rows[x].append(y)
                    self.cols[y].append(x)
        
        # Sort by coord order
        for x in self.rows:
            self.rows[x].sort(key=lambda c: self.coord_order.get(c, 0))
        for y in self.cols:
            self.cols[y].sort(key=lambda c: self.coord_order.get(c, 0))
        
        # Parse goals from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gname in goal_fact_names:
                predicate, args = self._parse_fact_name(gname)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    self.goal_status[(x, y)] = s
        
        # Build goal_cells list for fast evaluation
        for (x, y), goal_s in self.goal_status.items():
            y_gid = self.on_y_ids.get((x, y), -1)
            n_gid = self.on_n_ids.get((x, y), -1)
            self.goal_cells.append((x, y, goal_s, y_gid, n_gid))
        
        # Count method precondition operators for overhead estimation
        self.method_precondition_count = 0
        for op in self.model.operators:
            if op.name and "method_precondition" in op.name:
                self.method_precondition_count += 1
        
        # Estimate minimum overhead per turn
        # A turn = do_turn method which has: flip + propagate_up + propagate_down + propagate_left + propagate_right
        # Each propagate direction has a method precondition check (at least 1 for the end condition)
        # do_turn itself may have a method precondition
        # Minimum: 1 flip + 4 propagate end checks = 1 + 4 method_precondition ops = 5
        # But this depends on grounding. Let's estimate conservatively.
        # Each propagate_*_end or propagate_*_end_different has a method precondition = 1 op each
        # So minimum per turn: 1 (flip) + 4 * 1 (propagate ends) = 5
        # Plus do_play has method precondition, and do_turn may have one
        # Conservative: at least 1 flip per turn
        self.min_overhead_per_turn = 5  # 1 flip + 4 propagate-end method preconditions

    def _count_mismatches(self, state):
        """Count cells whose current status differs from goal status."""
        count = 0
        for (x, y, goal_s, y_gid, n_gid) in self.goal_cells:
            if goal_s == "Y":
                # Goal wants Y, check if on[x,y,Y] is true
                if y_gid >= 0:
                    if not ((state >> y_gid) & 1):
                        count += 1
                # If y_gid < 0, fact doesn't exist, can't be satisfied
            elif goal_s == "N":
                # Goal wants N, check if on[x,y,N] is true
                if n_gid >= 0:
                    if not ((state >> n_gid) & 1):
                        count += 1
        return count

    def _get_mismatch_grid(self, state):
        """Build a dict of (x,y) -> True/False for whether cell is mismatched."""
        mismatches = {}
        for (x, y, goal_s, y_gid, n_gid) in self.goal_cells:
            is_mismatched = False
            if goal_s == "Y":
                if y_gid >= 0:
                    if not ((state >> y_gid) & 1):
                        is_mismatched = True
            elif goal_s == "N":
                if n_gid >= 0:
                    if not ((state >> n_gid) & 1):
                        is_mismatched = True
            if is_mismatched:
                mismatches[(x, y)] = True
        return mismatches

    def _estimate_min_turns(self, state):
        """Estimate minimum number of turns needed.
        
        Each turn affects a cross-shaped pattern from the center cell.
        A turn at (x,y) with original status s flips (x,y) and then
        flips consecutive cells in each direction that also have status s.
        
        Lower bound: analyze rows and columns separately.
        For each row x, look at the mismatch pattern along y coordinates.
        A single turn can fix at most a contiguous segment in one direction
        from the center. The minimum turns for a row is related to the
        number of "alternating segments" of mismatches.
        
        Simpler approach: each turn fixes at most one contiguous run per
        direction. Count isolated mismatch regions.
        """
        mismatches = self._get_mismatch_grid(state)
        if not mismatches:
            return 0
        
        # Simple estimate: each turn can fix at most a cross of cells
        # Count the number of "independent" mismatch groups
        # For a conservative lower bound, just return 1 if any mismatch exists
        # Better: count mismatched cells and divide by max cross size
        
        n = self.grid_size
        if n == 0:
            return 0
        
        # Max cells a single turn can flip: 1 + (n-1)*4 = 4n - 3
        # But typically much less. A better bound:
        # Max cells in one turn = 1 (center) + up to (n-1) in each of 4 directions = 4n-3
        # But the propagation stops at first different cell, so max is actually n in each dir
        # giving at most 1 + 4*(n-1) = 4n-3
        
        num_mismatches = len(mismatches)
        if num_mismatches == 0:
            return 0
        
        max_per_turn = max(1, 4 * n - 3)
        
        # Lower bound on turns
        min_turns = max(1, (num_mismatches + max_per_turn - 1) // max_per_turn)
        
        return min_turns

    def _row_col_segment_bound(self, state):
        """Analyze rows and columns for a tighter lower bound.
        
        For each row, count the number of contiguous segments of mismatched cells.
        A single turn can affect at most one contiguous segment per row per direction.
        So the max over all rows of segment counts gives a lower bound on turns.
        Similarly for columns.
        
        But since turns affect both rows and columns simultaneously, we take max.
        """
        mismatches = self._get_mismatch_grid(state)
        if not mismatches:
            return 0
        
        max_segments = 0
        
        # Analyze rows
        for x in self.coords:
            segments = 0
            in_segment = False
            for y in self.rows.get(x, []):
                if (x, y) in mismatches:
                    if not in_segment:
                        segments += 1
                        in_segment = True
                else:
                    in_segment = False
            max_segments = max(max_segments, segments)
        
        # Analyze columns
        for y in self.coords:
            segments = 0
            in_segment = False
            for x in self.cols.get(y, []):
                if (x, y) in mismatches:
                    if not in_segment:
                        segments += 1
                        in_segment = True
                else:
                    in_segment = False
            max_segments = max(max_segments, segments)
        
        return max_segments

    def _task_network_bound(self, task_network):
        """Estimate minimum remaining cost from the task network.
        
        Count minimum primitive actions needed to complete the task network.
        Each 'play' task needs at least 0 actions (could be end_game).
        Each 'turn' task needs at least 1 flip action + method precondition ops.
        Each 'propagate_*' task needs at least 1 method precondition op (for end check).
        Each operator in the network costs 1.
        """
        if not task_network:
            return 0
        
        count = 0
        for task in task_network:
            # Check if it's an operator (primitive) - has 'applicable' method and is in operator range
            if hasattr(task, 'pos_precons') and hasattr(task, 'add_effects') and not hasattr(task, 'decompositions'):
                # It's a primitive operator
                count += 1
            elif hasattr(task, 'decompositions'):
                # It's an abstract task - estimate minimum cost
                task_name = task.name if hasattr(task, 'name') else ''
                if 'turn' in task_name:
                    # A turn needs at least 1 flip + 4 propagate ends
                    count += self.min_overhead_per_turn
                elif 'propagate' in task_name:
                    # Propagation needs at least 1 method precondition for end
                    count += 1
                elif 'play' in task_name:
                    # play can be end_game (0 cost) - no guaranteed cost
                    pass
                # Other abstract tasks: be conservative
        return count

    def _compute(self, state, task_network):
        """Compute the heuristic value.
        
        Returns max of multiple admissible lower bounds.
        """
        # Term 1: Direct mismatch count
        # Each mismatched cell needs at least 1 flip action
        mismatch_count = self._count_mismatches(state)
        
        if mismatch_count == 0:
            # Check if task network is empty (goal might be reached)
            if not task_network:
                return 0
            # Still have tasks but no mismatches - estimate from task network
            return self._task_network_bound(task_network)
        
        term1 = mismatch_count
        
        # Term 2: Turn-based estimate
        # Each turn needs at least min_overhead_per_turn primitive actions
        # (1 flip + method precondition operators for propagation ends)
        min_turns = self._estimate_min_turns(state)
        term2 = min_turns * self.min_overhead_per_turn
        
        # Term 3: Row/column segment analysis
        # Number of contiguous mismatch segments in the worst row/column
        # gives a lower bound on number of turns needed
        segment_bound = self._row_col_segment_bound(state)
        term3 = segment_bound * self.min_overhead_per_turn
        
        # Term 4: Task network lower bound
        term4 = self._task_network_bound(task_network)
        
        result = max(term1, term2, term3, term4)
        
        return result

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic13(grid={self.grid_size}x{self.grid_size})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic13 with {self.grid_size}x{self.grid_size} grid"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {self.grid_size}x{self.grid_size}\n"
            f"\tGoal Cells: {len(self.goal_cells)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
