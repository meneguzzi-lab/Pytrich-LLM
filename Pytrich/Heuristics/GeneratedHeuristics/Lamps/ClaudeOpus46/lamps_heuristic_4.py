"""Domain-dependent heuristic for the Lamps domain.

Bottleneck: Each cell that differs from its goal status must be flipped an odd
number of times. The only action is 'flip', costing 1 each. Every flip is a
primitive action. A single 'turn' flips a center cell plus propagates flips
along 4 directions to consecutive same-status neighbors. The grounded plan
also includes __method_precondition_* operators (cost 1 each).

Admissible lower bounds:
  Term 1 (goal_diff_count): Each cell whose current status differs from goal
         must be flipped at least once. This is a lower bound on flip actions.
  Term 2 (turn_structure_cost): Each turn decomposition generates at least
         1 flip + method precondition operators. If there are any differing
         cells, we need at least 1 turn. We estimate minimum turns needed.
  Term 3 (row/col analysis): Analyze rows and columns to find minimum
         contiguous segments that need flipping, giving a tighter bound
         on the number of turns needed.

The heuristic returns max(term1, term2, term3) for admissibility.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic4(Heuristic):
    """
    Domain-dependent heuristic for the Lamps (game) domain.
    
    Uses multiple admissible lower bounds combined with max():
    1. Direct count of cells differing from goal (each needs >= 1 flip)
    2. Structural cost per turn (method preconditions + flips)
    3. Row/column segment analysis for minimum turns
    """

    def __init__(self, lamps_anthropicopus46_heuristic_4=None, name="LampsAnthropicOpus46Heuristic4"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Maps for fast lookup
        # on_fact_map: (x, y, s) -> fact.global_id
        self.on_fact_map = {}
        # goal_status: (x, y) -> required status ('Y' or 'N')
        self.goal_status = {}
        # All grid cells
        self.all_cells = set()
        # Grid coordinates
        self.coords = []
        # next relation: coord -> next_coord
        self.next_map = {}
        # prev relation: coord -> prev_coord  
        self.prev_map = {}
        self.min_coord = None
        self.max_coord = None
        # Sorted coordinate list for iteration
        self.sorted_coords = []
        # Number of coords
        self.n_coords = 0
        # coord to index mapping
        self.coord_to_idx = {}
        
        # Precomputed: for each cell (x,y), store the global_ids for on[x,y,Y] and on[x,y,N]
        self.on_Y_id = {}  # (x,y) -> global_id of on[x,y,Y]
        self.on_N_id = {}  # (x,y) -> global_id of on[x,y,N]
        
        # Cost per turn overhead (method preconditions)
        # Each turn has: flip + propagate_up + propagate_down + propagate_left + propagate_right
        # Each propagate direction has a method precondition check
        # Minimum overhead per turn = 1 (flip) + 4 (propagate end checks) = 5
        # But propagate_end methods that match end_field or end_different have preconditions
        # that are checked via __method_precondition operators
        self.min_cost_per_turn = 5  # 1 flip + 4 method precondition ops for propagate ends
        
        # For task network analysis
        self.turn_task_names = set()
        self.play_task_names = set()
        self.propagate_task_names = set()

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
        coords_set = set()
        
        # Parse all facts
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None:
                continue
            
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.on_fact_map[(x, y, s)] = fact.global_id
                self.all_cells.add((x, y))
                coords_set.add(x)
                coords_set.add(y)
                if s == "Y":
                    self.on_Y_id[(x, y)] = fact.global_id
                elif s == "N":
                    self.on_N_id[(x, y)] = fact.global_id
            
            elif predicate == "next" and len(args) == 2:
                self.next_map[args[0]] = args[1]
                self.prev_map[args[1]] = args[0]
            
            elif predicate == "max" and len(args) == 1:
                self.max_coord = args[0]
            
            elif predicate == "min" and len(args) == 1:
                self.min_coord = args[0]
        
        # Build sorted coordinate list
        if self.min_coord is not None:
            self.sorted_coords = []
            c = self.min_coord
            while c is not None:
                self.sorted_coords.append(c)
                c = self.next_map.get(c)
            self.n_coords = len(self.sorted_coords)
            for i, c in enumerate(self.sorted_coords):
                self.coord_to_idx[c] = i
        
        # Parse goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    self.goal_status[(x, y)] = s
        
        # Identify task types for task network analysis
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower()
            if 'turn' in name_lower and 'propagate' not in name_lower:
                self.turn_task_names.add(at.name)
            elif 'play' in name_lower:
                self.play_task_names.add(at.name)
            elif 'propagate' in name_lower:
                self.propagate_task_names.add(at.name)
        
        # Estimate actual minimum cost per turn by looking at method precondition operators
        # Count __method_precondition operators to understand overhead
        method_prec_count = 0
        for op in self.model.operators:
            if '__method_precondition' in op.name:
                method_prec_count += 1
        
        # Each turn decomposes into: flip + 4 propagate tasks
        # Each propagate task needs at least 1 method precondition check
        # So minimum per turn = 1 (flip) + 4 (method prec checks) = 5
        # But if propagation actually flips cells, it's more
        self.min_cost_per_turn = 5  # Conservative: 1 flip + 4 method_prec ops

    def _get_cell_status(self, state, x, y):
        """Get current status of cell (x,y): 'Y', 'N', or None."""
        y_id = self.on_Y_id.get((x, y))
        if y_id is not None and (state >> y_id) & 1:
            return 'Y'
        n_id = self.on_N_id.get((x, y))
        if n_id is not None and (state >> n_id) & 1:
            return 'N'
        return None

    def _count_differing_cells(self, state):
        """Count cells whose current status differs from goal."""
        count = 0
        for (x, y), goal_s in self.goal_status.items():
            if goal_s == 'Y':
                gid = self.on_Y_id.get((x, y))
                if gid is not None and not ((state >> gid) & 1):
                    count += 1
            elif goal_s == 'N':
                gid = self.on_N_id.get((x, y))
                if gid is not None and not ((state >> gid) & 1):
                    count += 1
        return count

    def _estimate_min_turns_row_col(self, state):
        """Estimate minimum number of turns needed using row/column analysis.
        
        For each row, find contiguous segments of cells that need flipping.
        Each such segment in a row could potentially be handled by one turn's
        horizontal propagation, but the center cell also affects the column.
        
        This gives a lower bound: for each row, the number of contiguous
        groups of differing cells is a lower bound on turns affecting that row.
        We take the maximum across all rows and columns.
        """
        if self.n_coords == 0:
            return 0
        
        # Build a diff grid
        # diff[xi][yi] = True if cell differs from goal
        n = self.n_coords
        
        # For efficiency, build diff as a flat check
        max_row_segments = 0
        max_col_segments = 0
        
        # Check each row (fixed x, varying y)
        for x in self.sorted_coords:
            segments = 0
            in_segment = False
            for y in self.sorted_coords:
                goal_s = self.goal_status.get((x, y))
                if goal_s is None:
                    if in_segment:
                        in_segment = False
                    continue
                
                differs = False
                if goal_s == 'Y':
                    gid = self.on_Y_id.get((x, y))
                    if gid is not None and not ((state >> gid) & 1):
                        differs = True
                elif goal_s == 'N':
                    gid = self.on_N_id.get((x, y))
                    if gid is not None and not ((state >> gid) & 1):
                        differs = True
                
                if differs:
                    if not in_segment:
                        segments += 1
                        in_segment = True
                else:
                    in_segment = False
            
            max_row_segments = max(max_row_segments, segments)
        
        # Check each column (fixed y, varying x)
        for y in self.sorted_coords:
            segments = 0
            in_segment = False
            for x in self.sorted_coords:
                goal_s = self.goal_status.get((x, y))
                if goal_s is None:
                    if in_segment:
                        in_segment = False
                    continue
                
                differs = False
                if goal_s == 'Y':
                    gid = self.on_Y_id.get((x, y))
                    if gid is not None and not ((state >> gid) & 1):
                        differs = True
                elif goal_s == 'N':
                    gid = self.on_N_id.get((x, y))
                    if gid is not None and not ((state >> gid) & 1):
                        differs = True
                
                if differs:
                    if not in_segment:
                        segments += 1
                        in_segment = True
                else:
                    in_segment = False
            
            max_col_segments = max(max_col_segments, segments)
        
        return max(max_row_segments, max_col_segments)

    def _count_turns_in_network(self, task_network):
        """Count the minimum number of turn tasks remaining in the task network."""
        turn_count = 0
        play_count = 0
        for task in task_network:
            if hasattr(task, 'name'):
                if task.name in self.turn_task_names:
                    turn_count += 1
                elif task.name in self.play_task_names:
                    play_count += 1
        return turn_count, play_count

    def _compute(self, state, task_network):
        """Compute the heuristic value."""
        # If goals are already satisfied and task network could be empty
        diff_count = self._count_differing_cells(state)
        
        if diff_count == 0:
            # Goals met - but we still need to process remaining task network
            # If there are turn tasks committed, they cost at least min_cost_per_turn each
            turn_count, play_count = self._count_turns_in_network(task_network)
            if turn_count == 0 and play_count == 0:
                return 0
            # If only play tasks remain, they can decompose to end_game (0 cost)
            if turn_count == 0:
                return 0
            # Committed turns must execute
            return turn_count * self.min_cost_per_turn
        
        # Term 1: Each differing cell needs at least one flip action
        term1 = diff_count
        
        # Term 2: Minimum turns * cost per turn
        # Each turn costs at least min_cost_per_turn
        # We need at least 1 turn if any cell differs
        # A single turn can flip at most (1 + up + down + left + right) cells
        # where up/down/left/right are the lengths of contiguous same-status runs
        # Maximum flips per turn in an n×n grid: 1 + 4*(n-1) = 4n-3
        # So minimum turns >= ceil(diff_count / (4*n - 3))
        if self.n_coords > 0:
            max_flips_per_turn = 4 * self.n_coords - 3
        else:
            max_flips_per_turn = 1
        
        min_turns_by_count = (diff_count + max_flips_per_turn - 1) // max_flips_per_turn
        
        # Term 3: Row/column segment analysis
        min_turns_by_segments = self._estimate_min_turns_row_col(state)
        
        # Best estimate of minimum turns
        min_turns = max(min_turns_by_count, min_turns_by_segments)
        
        # Account for committed turns in task network
        turn_count, play_count = self._count_turns_in_network(task_network)
        
        # If we have committed turns, those are definite costs
        # Plus we may need more turns via future play decompositions
        if turn_count > 0:
            # Already committed turn_count turns; may need more
            remaining_turns_needed = max(0, min_turns - turn_count)
            # Committed turns cost at least min_cost_per_turn each
            committed_cost = turn_count * self.min_cost_per_turn
            # Remaining turns also cost at least min_cost_per_turn each
            remaining_cost = remaining_turns_needed * self.min_cost_per_turn
            term2 = committed_cost + remaining_cost
        else:
            # No committed turns yet; need at least min_turns
            term2 = min_turns * self.min_cost_per_turn
        
        # Term 1 is admissible: each differing cell needs at least 1 flip
        # Term 2 is admissible: accounts for overhead per turn
        # We also check: each propagate task in the network costs at least 1 (method_prec)
        propagate_cost = 0
        for task in task_network:
            if hasattr(task, 'name') and task.name in self.propagate_task_names:
                propagate_cost += 1  # At least 1 method_precondition op
        
        # For the flip-count term, also add propagate overhead for committed tasks
        term1_with_overhead = diff_count + propagate_cost
        
        return max(term1, term2, term1_with_overhead)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic4(coords={self.n_coords})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic4 with {self.n_coords} coords, {len(self.goal_status)} goal cells"

    def __output__(self):
        return (
            f"Lamps Heuristic 4 Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {self.n_coords}x{self.n_coords}\n"
            f"\tGoal Cells: {len(self.goal_status)}\n"
            f"\tMin Cost Per Turn: {self.min_cost_per_turn}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
