"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck: Each cell that differs between current state and goal requires at least
one flip action. Even though propagation during a turn can flip multiple cells,
each propagated flip is itself a separate primitive action costing 1. Therefore,
the number of mismatched cells is an admissible lower bound on remaining cost.

Additionally, each "turn" decomposition requires at least 1 flip (the center cell)
plus any propagation flips, plus __method_precondition operators from the grounder.
We estimate the minimum number of turns needed based on connectivity analysis.

Lower bounds:
1. Mismatched cells count: Each mismatch requires >= 1 flip action. Admissible because
   every flip action changes exactly one cell, so we need >= #mismatches flip actions.
2. Turn-based lower bound: Each turn generates at least 1 flip + method precondition
   operators. We estimate minimum turns needed.
3. Task network analysis: Count minimum primitive actions from pending tasks.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic7(Heuristic):
    """
    Heuristic for the Lamps domain based on counting mismatched cells
    and estimating minimum turns/actions needed.
    """

    def __init__(self, lamps_anthropicopus46_heuristic_7=None, name="LampsAnthropicOpus46Heuristic7"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Maps for fast lookup
        self.on_facts = {}          # (x, y, s) -> fact.global_id
        self.goal_specs = []        # list of (x, y, s, global_id) for goal facts
        self.goal_on_facts = {}     # (x, y) -> required status s
        self.all_coords = set()
        self.grid_size = 0
        
        # For task network analysis
        self.turn_task_ids = set()   # global_ids of 'turn' abstract tasks
        self.play_task_ids = set()   # global_ids of 'play' abstract tasks
        self.flip_op_ids = set()     # global_ids of 'flip' operators
        self.method_precond_ids = set()  # global_ids of method precondition operators
        
        # Propagation direction info
        self.next_map = {}           # coord -> next coord
        self.prev_map = {}           # coord -> prev coord
        self.max_coord = None
        self.min_coord = None
        
        # For estimating turn cost
        self.coord_list = []         # ordered list of coordinates
        self.coord_to_idx = {}       # coord -> index in ordered list
        
        # Precomputed: number of method_precondition ops per turn
        self.method_precond_per_turn = 0

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
        
        # Parse all facts to build on_facts map
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None:
                continue
            
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.on_facts[(x, y, s)] = fact.global_id
                self.all_coords.add(x)
                self.all_coords.add(y)
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
                self.coord_list.append(c)
                self.coord_to_idx[c] = idx
                idx += 1
                c = self.next_map.get(c, None)
        self.grid_size = len(self.coord_list)
        
        # Parse goal facts
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    self.goal_on_facts[(x, y)] = s
                    gid = self.on_facts.get((x, y, s), None)
                    if gid is not None:
                        self.goal_specs.append((x, y, s, gid))
        
        # Identify task types and operator types
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower()
            if 'turn' in name_lower and 'propagate' not in name_lower and 'play' not in name_lower:
                self.turn_task_ids.add(at.global_id)
            elif 'play' in name_lower:
                self.play_task_ids.add(at.global_id)
        
        for op in self.model.operators:
            name_lower = op.name.lower()
            if 'flip' in name_lower and '__method_precondition' not in name_lower:
                self.flip_op_ids.add(op.global_id)
            if '__method_precondition' in name_lower:
                self.method_precond_ids.add(op.global_id)
        
        # Estimate method_precondition operators per turn
        # A turn decomposes into: flip + propagate_up + propagate_down + propagate_left + propagate_right
        # Each propagate has method preconditions. Count them from decompositions.
        self.method_precond_per_turn = self._estimate_method_precond_per_turn()

    def _estimate_method_precond_per_turn(self):
        """Estimate minimum method precondition operators per turn.
        
        A turn = do_turn decomposition which has:
          flip + propagate_up + propagate_down + propagate_left + propagate_right
        
        Each propagate ends with either end_field or end_different, which may have
        method preconditions. The minimum is when all 4 propagations stop immediately.
        
        Looking at the decompositions, propagate_up_end_field has precondition (max ?x),
        propagate_up_end_different has precondition (next ?x ?xx) and (not (on ?xx ?y ?s)).
        These become __method_precondition operators.
        
        The do_turn method itself may also have method preconditions.
        """
        # Count method precondition operators that appear in decomposition task networks
        min_precond_ops = 0
        for dec in self.model.decompositions:
            name_lower = dec.name.lower()
            if 'do_turn' in name_lower:
                # Count method_precondition operators in this decomposition's task network
                count = 0
                for task in dec.task_network:
                    if hasattr(task, 'name') and '__method_precondition' in task.name.lower():
                        count += 1
                if count > min_precond_ops:
                    min_precond_ops = count
        
        # Each propagation direction needs at least one end method with preconditions
        # Minimum: 4 propagation ends, each with ~1 method_precondition op
        # Plus the do_turn method_precondition
        # We'll use a conservative estimate
        return max(min_precond_ops, 1)

    def _count_mismatches(self, state_bitwise):
        """Count cells where current status differs from goal.
        
        Each mismatch requires at least one flip action to fix.
        This is admissible: each flip changes exactly one cell.
        """
        mismatches = 0
        for (x, y, s, gid) in self.goal_specs:
            # Check if goal fact (on x y s) is true in current state
            if not ((state_bitwise >> gid) & 1):
                mismatches += 1
        return mismatches

    def _estimate_min_turns(self, state_bitwise):
        """Estimate minimum number of turns needed.
        
        Each turn flips a center cell and propagates in 4 directions.
        A single turn can fix at most 1 + (grid_size-1)*4 cells, but typically much fewer.
        
        A better estimate: look at each row and column independently.
        In a row, a single turn's left/right propagation can fix a contiguous run of same-status cells.
        But the center flip and the vertical propagation are separate concerns.
        
        Simple lower bound: each turn costs at least 1 flip + method_precondition ops.
        We compute a lower bound on the number of turns from the mismatch pattern.
        """
        if self.grid_size == 0:
            return 0
        
        # For each cell that needs flipping, check if it could be covered by
        # a propagation from an adjacent mismatch. Build connected components
        # of mismatches along rows and columns.
        
        # Get mismatch set
        mismatch_cells = set()
        mismatch_current_status = {}  # (x,y) -> current status (the wrong one)
        
        for (x, y, s, gid) in self.goal_specs:
            if not ((state_bitwise >> gid) & 1):
                mismatch_cells.add((x, y))
                # The goal wants status s, but cell has the other status
                mismatch_current_status[(x, y)] = 'Y' if s == 'N' else 'N'
        
        if not mismatch_cells:
            return 0
        
        # Each turn picks one center cell and flips it, then propagates.
        # Propagation in a direction continues flipping cells that had the SAME
        # status as the center cell's OLD status.
        # 
        # So a turn centered at (cx, cy) with old status s flips:
        #   - (cx, cy) from s to other
        #   - all contiguous cells in each direction from (cx,cy) that also have status s
        #
        # For mismatches: if a group of adjacent cells along a line all have the same
        # wrong status, they could potentially be fixed by one turn centered at one end.
        # But this also affects the center's row/column in the other direction.
        #
        # Conservative lower bound: count the number of "independent groups" of mismatches.
        # Two mismatches can be in the same group if they're adjacent in the same row or column
        # AND have the same current (wrong) status.
        
        # Union-Find for grouping
        parent = {}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        
        for cell in mismatch_cells:
            parent[cell] = cell
        
        for (x, y) in mismatch_cells:
            curr_s = mismatch_current_status[(x, y)]
            # Check neighbor in same row (next y)
            ny = self.next_map.get(y, None)
            if ny and (x, ny) in mismatch_cells:
                if mismatch_current_status[(x, ny)] == curr_s:
                    union((x, y), (x, ny))
            # Check neighbor in same column (next x)
            nx = self.next_map.get(x, None)
            if nx and (nx, y) in mismatch_cells:
                if mismatch_current_status[(nx, y)] == curr_s:
                    union((x, y), (nx, y))
        
        # Count groups
        groups = set()
        for cell in mismatch_cells:
            groups.add(find(cell))
        
        return len(groups)

    def _task_network_lower_bound(self, task_network):
        """Estimate minimum primitive actions from pending task network.
        
        Each turn task will generate at least 1 flip action + some method_precondition ops.
        Each play task might generate 0 (end_game) or more.
        Each flip operator in the network counts as 1.
        Each __method_precondition operator counts as 1.
        """
        if not task_network:
            return 0
        
        h = 0
        turn_count = 0
        
        for task in task_network:
            gid = task.global_id
            if gid in self.turn_task_ids:
                # A turn needs at least 1 flip + method preconditions for propagation ends
                turn_count += 1
            elif gid in self.play_task_ids:
                # play might be end_game (0 cost) or do_play (turn + play)
                # Can't assume anything beyond 0 for the play task itself
                pass
            elif gid in self.flip_op_ids:
                h += 1
            elif gid in self.method_precond_ids:
                h += 1
            elif gid >= self.model.iop_init and gid <= self.model.iop_end:
                # Some other operator
                h += 1
        
        # Each turn requires at least 1 flip + at least 4 method preconditions
        # (one for each propagation direction's end condition)
        # Plus potentially 1 method_precondition for do_turn itself
        # Conservative: at least 1 flip per turn
        h += turn_count * 1
        
        return h

    def _compute(self, state_bitwise, task_network):
        """Compute the heuristic value."""
        
        # Term 1: Number of mismatched cells (each needs >= 1 flip action)
        mismatches = self._count_mismatches(state_bitwise)
        
        if mismatches == 0:
            # Check if task network is empty too
            if not task_network:
                return 0
            # Tasks remain but no mismatches - might need turns that cancel out
            # or method preconditions
            return self._task_network_lower_bound(task_network)
        
        # Term 2: Minimum turns estimate
        # Each turn costs at least 1 flip + method preconditions for the 4 propagation ends
        # We estimate min turns from connected components of mismatches
        min_turns = self._estimate_min_turns(state_bitwise)
        
        # Each turn involves at least:
        # - 1 flip for the center
        # - 4 propagation terminations (each may have a method_precondition op)
        # So minimum cost per turn is roughly 1 (flip) + 4 (method_preconditions) = 5
        # But we need to be careful about admissibility.
        # The flip actions are already counted in mismatches.
        # The method_precondition ops are additional.
        # So: total >= mismatches + min_turns * (cost of method_preconditions per turn)
        
        # However, we need to count method_precondition ops carefully.
        # Each propagation direction ends with a method that may have 1 method_precondition op.
        # Plus do_turn itself may have method_precondition ops.
        # Minimum: 4 method_precondition ops per turn (one per propagation end)
        # Plus potentially 1 for do_turn.
        # Let's count exactly from the decompositions.
        
        # Count minimum method_precondition ops per propagation end
        min_mp_per_prop_end = self._count_min_mp_per_prop_end()
        
        # Each turn has 4 propagation directions, each ending with at least min_mp_per_prop_end ops
        method_precond_cost = min_turns * (4 * min_mp_per_prop_end + self._count_min_mp_do_turn())
        
        term1 = mismatches  # Each mismatch = 1 flip action minimum
        term2 = mismatches + method_precond_cost  # mismatches + overhead
        term3 = self._task_network_lower_bound(task_network)
        
        return max(term1, term2, term3)

    def _count_min_mp_per_prop_end(self):
        """Count minimum method_precondition operators for a propagation end."""
        if not hasattr(self, '_cached_min_mp_prop_end'):
            min_mp = float('inf')
            for dec in self.model.decompositions:
                name_lower = dec.name.lower()
                if ('propagate' in name_lower and 
                    ('end_field' in name_lower or 'end_different' in name_lower)):
                    count = 0
                    for task in dec.task_network:
                        if hasattr(task, 'name') and '__method_precondition' in task.name.lower():
                            count += 1
                    min_mp = min(min_mp, count)
            self._cached_min_mp_prop_end = min_mp if min_mp != float('inf') else 0
        return self._cached_min_mp_prop_end

    def _count_min_mp_do_turn(self):
        """Count minimum method_precondition operators for do_turn."""
        if not hasattr(self, '_cached_min_mp_do_turn'):
            min_mp = float('inf')
            for dec in self.model.decompositions:
                name_lower = dec.name.lower()
                if 'do_turn' in name_lower:
                    count = 0
                    for task in dec.task_network:
                        if hasattr(task, 'name') and '__method_precondition' in task.name.lower():
                            count += 1
                    min_mp = min(min_mp, count)
            self._cached_min_mp_do_turn = min_mp if min_mp != float('inf') else 0
        return self._cached_min_mp_do_turn

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic7(grid={self.grid_size})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic7 with grid size {self.grid_size}"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {self.grid_size}\n"
            f"\tGoal Specs: {len(self.goal_specs)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
