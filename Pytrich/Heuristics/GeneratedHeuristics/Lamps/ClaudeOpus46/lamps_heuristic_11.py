"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
  The Lamps domain is a grid-based light-flipping puzzle. Each "turn" picks a cell (x,y),
  flips it, then propagates in all 4 directions flipping consecutive same-status neighbors.
  The bottleneck is that each turn produces exactly 1 flip for the center cell plus
  variable propagation flips. The key cost driver is the number of cells that differ
  between current state and goal state - each flip is one primitive action.

Admissible Lower Bounds:
  Term 1 (Goal-diff count): Count the number of cells where current status != goal status.
    Each such cell must be flipped at least once (1 primitive action per flip). This is
    admissible because no single action can flip more than one cell.
  
  Term 2 (Turn-based lower bound): Each "turn" operation consists of 1 center flip plus
    up to 4 directional propagations. A turn requires at minimum 1 flip action (the center).
    Plus the grounder inserts __method_precondition__ operators that also cost 1 each.
    We estimate the minimum number of turns needed. Since each turn can fix at most
    ~(2*grid_size - 1) cells in the best case (a cross pattern), we get a lower bound
    of ceil(diff_count / max_flips_per_turn). However, a simpler and often tighter bound
    is just the goal-diff count itself since each flip is 1 action.

  Term 3 (Task network analysis): Count remaining compound tasks in the task network.
    Each 'turn' task requires at least 1 primitive action. Each 'play' requires decomposition.
    We estimate minimum primitives from the task network structure.

Approach:
  We precompute:
  - A mapping from (x, y, status) -> fact global_id for O(1) state checks
  - The goal state as a set of (x, y, desired_status) triples
  - Grid structure (next relationships, min/max coords)
  
  At query time we:
  - Count goal mismatches using bitwise checks (O(grid_size^2))
  - Analyze task network for minimum remaining turns
  - Return max of terms
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic11(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.
    
    Uses goal-difference counting and task network analysis to provide
    an admissible lower bound on remaining cost.
    """

    def __init__(self, lamps_anthropicopus46_heuristic_11=None, name="LampsAnthropicOpus46Heuristic11"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Fact lookups: (x, y, status) -> global_id
        self.on_fact_id = {}
        
        # Goal info: list of (x, y, desired_status) and their fact global_ids
        self.goal_facts = []  # list of (global_id, x, y, status) for goal cells
        
        # Grid info
        self.grid_size = 0
        self.coords = []
        self.next_map = {}  # coord -> next coord
        self.prev_map = {}  # coord -> prev coord
        self.min_coord = None
        self.max_coord = None
        
        # Operator/task name caches
        self.turn_task_ids = set()  # global_ids of 'turn' abstract tasks
        self.play_task_ids = set()  # global_ids of 'play' abstract tasks
        self.flip_op_ids = set()   # global_ids of 'flip' operators
        self.method_prec_op_ids = set()  # global_ids of __method_precondition__ operators
        
        # For efficient propagation estimation
        self.coord_to_idx = {}  # coord name -> integer index
        
        # Precomputed: for each cell, what's the goal status fact global_id
        # goal_on_gid[(x,y)] = global_id of the fact +on[x,y,goal_status]
        self.goal_on_gid = {}
        # current mismatch check: for each goal cell, store the global_id we need to be true
        self.goal_check_list = []  # list of global_ids that must be true in goal

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
        
        # Build fact lookup
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None:
                continue
            
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.on_fact_id[(x, y, status)] = fact.global_id
                coords_set.add(x)
                coords_set.add(y)
            elif predicate == "next" and len(args) == 2:
                self.next_map[args[0]] = args[1]
                self.prev_map[args[1]] = args[0]
            elif predicate == "max" and len(args) == 1:
                self.max_coord = args[0]
            elif predicate == "min" and len(args) == 1:
                self.min_coord = args[0]
        
        # Build coordinate ordering
        if self.min_coord is not None:
            c = self.min_coord
            idx = 0
            while c is not None:
                self.coords.append(c)
                self.coord_to_idx[c] = idx
                idx += 1
                c = self.next_map.get(c, None)
            self.grid_size = len(self.coords)
        else:
            self.coords = sorted(list(coords_set))
            self.grid_size = len(self.coords)
            for idx, c in enumerate(self.coords):
                self.coord_to_idx[c] = idx
        
        # Parse goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, status = args[0], args[1], args[2]
                    gid = self.on_fact_id.get((x, y, status))
                    if gid is not None:
                        self.goal_check_list.append(gid)
                        self.goal_on_gid[(x, y)] = (gid, status)
        
        # Parse abstract tasks and operators for task network analysis
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower()
            if 'turn' in name_lower and 'propagate' not in name_lower:
                self.turn_task_ids.add(at.global_id)
            elif 'play' in name_lower:
                self.play_task_ids.add(at.global_id)
        
        for op in self.model.operators:
            name_lower = op.name.lower()
            if 'flip' in name_lower and '__method_precondition' not in name_lower:
                self.flip_op_ids.add(op.global_id)
            if '__method_precondition' in name_lower:
                self.method_prec_op_ids.add(op.global_id)

    def _count_goal_mismatches(self, state_bitwise: int) -> int:
        """Count how many goal facts are not satisfied in the current state."""
        count = 0
        for gid in self.goal_check_list:
            if not ((state_bitwise >> gid) & 1):
                count += 1
        return count

    def _estimate_min_turns(self, state_bitwise: int) -> int:
        """
        Estimate the minimum number of turns needed.
        
        Each turn flips a center cell plus propagates in 4 directions.
        In the best case, a single turn can fix multiple mismatched cells
        if they form a cross pattern of same status.
        
        A turn always costs at least 1 flip (the center). Propagation flips
        are additional. So each turn costs >= 1 flip action.
        
        Additionally, each turn decomposition involves __method_precondition__
        operators which also cost 1 each.
        """
        mismatches = self._count_goal_mismatches(state_bitwise)
        if mismatches == 0:
            return 0
        
        # Each turn can fix at most (2*grid_size - 1) cells in the absolute
        # best case (full cross). But more typically fewer.
        # The maximum cross size is 2*(grid_size-1) + 1 = 2*grid_size - 1
        if self.grid_size > 0:
            max_fix_per_turn = 2 * self.grid_size - 1
        else:
            max_fix_per_turn = 1
        
        # Minimum turns needed
        min_turns = (mismatches + max_fix_per_turn - 1) // max_fix_per_turn
        return min_turns

    def _task_network_estimate(self, task_network) -> float:
        """
        Estimate minimum remaining primitives from the task network.
        
        Each 'turn' task needs at least 1 flip + method precondition ops.
        Each 'play' task can decompose to empty (end_game) or to turn+play.
        Propagate tasks need 0 or more flips.
        Flip operators cost 1.
        Method precondition operators cost 1.
        """
        h = 0
        for task in task_network:
            gid = task.global_id
            if gid in self.turn_task_ids:
                # A turn always does at least 1 flip (the center)
                # Plus typically 1 method_precondition for do_turn
                h += 1
            elif gid in self.play_task_ids:
                # play can be end_game (0 cost) or do_play (turn + play)
                # We can't assume it will do anything, so contribute 0
                # But if we know there are mismatches, we know at least 1 turn is needed
                pass
            elif gid in self.flip_op_ids:
                h += 1
            elif gid in self.method_prec_op_ids:
                h += 1
            else:
                # Check if it's any operator (costs 1)
                if gid >= self.model.iop_init and gid <= self.model.iop_end:
                    h += 1
                # propagate tasks: they may or may not produce flips
                # In the best case they produce 0 flips (end immediately)
                # So we contribute 0 for admissibility
        return h

    def _compute(self, state_bitwise: int, task_network) -> float:
        """Compute heuristic value."""
        mismatches = self._count_goal_mismatches(state_bitwise)
        
        if mismatches == 0 and len(task_network) == 0:
            return 0
        
        # Term 1: Each mismatch needs at least 1 flip action
        # This is admissible: each flip action changes exactly 1 cell
        term1 = mismatches
        
        # Term 2: Task network lower bound
        # Count committed primitives and minimum from compound tasks
        term2 = self._task_network_estimate(task_network)
        
        # Term 3: Turn-based reasoning
        # Each turn costs at least 1 flip + at least 1 method_precondition
        # So minimum turns * 2 gives a lower bound on actions from turns
        # But we need to be careful: method_preconditions may not always exist
        # Safer: each turn costs at least 1 (just the center flip)
        min_turns = self._estimate_min_turns(state_bitwise)
        # Each turn has at least 1 flip action
        # Plus the do_turn method typically has a __method_precondition__ check
        # that costs 1 action. And each propagate direction that actually propagates
        # also has a method_precondition. So conservatively, 1 per turn.
        term3 = min_turns
        
        # For better guidance, also consider: if there are mismatches,
        # the play task must decompose via do_play (not end_game),
        # which means at least 1 turn is needed.
        # Each turn in do_turn has precondition checks.
        
        h = max(term1, term2, term3)
        
        return float(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic11(grid={self.grid_size})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic11 with grid size {self.grid_size}"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {self.grid_size}\n"
            f"\tGoal Cells: {len(self.goal_check_list)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
