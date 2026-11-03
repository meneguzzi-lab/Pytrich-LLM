"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck: Each cell differing from goal must be flipped (cost 1 each).
A single 'turn' flips a center cell plus propagates flips along 4 directions
through consecutive same-status neighbors. Each flip is a separate primitive
action costing 1. Additionally, the grounder produces __method_precondition_*
operators that each cost 1.

Admissible lower bounds:
1. Mismatch count: number of cells where current != goal status. Each such
   cell needs at least one flip action (cost 1). This is admissible because
   no flip action can fix more than one mismatch.
2. Turn-based overhead: each turn requires at least the center flip (1) plus
   method precondition operators for propagation decisions in 4 directions.
   We estimate minimum turns needed and multiply by overhead per turn.
3. Task network lower bound: remaining tasks in the network provide a
   structural lower bound on remaining actions.

We return max of these terms.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic1(Heuristic):
    """
    Heuristic for the Lamps domain.
    
    Computes multiple admissible lower bounds based on:
    - Number of cells mismatching the goal (each needs >= 1 flip)
    - Estimated number of turns needed (each turn has overhead from
      method precondition operators)
    - Task network structure
    """

    def __init__(self, lamps_anthropicopus46_heuristic_1=None, name="LampsAnthropicOpus46Heuristic1"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Maps for fast lookup
        # goal_status[x][y] = desired status string ('Y' or 'N')
        self.goal_status = {}
        # on_facts[(x, y, s)] = fact.global_id
        self.on_facts = {}
        # All (x, y) cells in the grid
        self.all_cells = set()
        # goal_fact_ids: set of (x, y, s) tuples from goal
        self.goal_cells = {}
        # Coordinates
        self.coords = set()
        # next relations for adjacency
        self.next_map = {}  # c -> next(c) i.e. c+1
        self.prev_map = {}  # c -> prev(c) i.e. c-1
        self.min_coord = None
        self.max_coord = None
        # Number of coords (grid dimension)
        self.grid_size = 0
        
        # Precomputed goal bitwise mask and expected bits
        # For fast mismatch counting
        self.goal_on_fact_ids = []  # list of global_ids that should be true in goal
        self.goal_off_fact_ids = []  # list of global_ids that should be false (the opposite status)
        
        # For each cell (x,y), store the global_id of on[x,y,Y] and on[x,y,N]
        self.cell_y_fact = {}  # (x,y) -> global_id of on[x,y,Y]
        self.cell_n_fact = {}  # (x,y) -> global_id of on[x,y,N]
        
        # Overhead per turn: method preconditions for propagation
        # Each turn has: flip + propagate_up + propagate_down + propagate_left + propagate_right
        # Each propagate direction has a method precondition check (cost 1)
        # Plus the do_turn method precondition
        self.overhead_per_turn = 0
        
        # For counting method_precondition operators
        self.method_precondition_count = 0
        
        # For task network analysis
        self.abstract_task_ids = set()
        self.operator_ids = set()
        self.play_task_ids = set()
        self.turn_task_ids = set()
        self.propagate_task_ids = set()
        
        # Precomputed: for each cell, which neighbor cells exist
        self.neighbors = {}  # (x,y) -> list of (nx, ny)

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
                if s == 'Y':
                    self.cell_y_fact[(x, y)] = fact.global_id
                elif s == 'N':
                    self.cell_n_fact[(x, y)] = fact.global_id
            
            elif predicate == "next" and len(args) == 2:
                self.next_map[args[0]] = args[1]
                self.prev_map[args[1]] = args[0]
            
            elif predicate == "min" and len(args) == 1:
                self.min_coord = args[0]
            
            elif predicate == "max" and len(args) == 1:
                self.max_coord = args[0]
        
        self.grid_size = len(self.coords)
        
        # Build neighbor map
        for (x, y) in self.all_cells:
            nbrs = []
            # up: next[x] = xx, neighbor is (xx, y)
            if x in self.next_map:
                xx = self.next_map[x]
                if (xx, y) in self.all_cells:
                    nbrs.append((xx, y))
            # down: prev[x] = xx, neighbor is (xx, y)  
            if x in self.prev_map:
                xx = self.prev_map[x]
                if (xx, y) in self.all_cells:
                    nbrs.append((xx, y))
            # left (actually next in y): next[y] = yy, neighbor is (x, yy)
            if y in self.next_map:
                yy = self.next_map[y]
                if (x, yy) in self.all_cells:
                    nbrs.append((x, yy))
            # right (actually prev in y): prev[y] = yy, neighbor is (x, yy)
            if y in self.prev_map:
                yy = self.prev_map[y]
                if (x, yy) in self.all_cells:
                    nbrs.append((x, yy))
            self.neighbors[(x, y)] = nbrs
        
        # Extract goal
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    self.goal_cells[(x, y)] = s
                    self.goal_on_fact_ids.append(self.on_facts[(x, y, s)])
                    # The opposite status should NOT be true
                    opp = 'N' if s == 'Y' else 'Y'
                    if (x, y, opp) in self.on_facts:
                        self.goal_off_fact_ids.append(self.on_facts[(x, y, opp)])
        
        # Analyze operators for overhead estimation
        # Count method_precondition operators and understand turn structure
        method_prec_ops = 0
        flip_ops = 0
        for op in self.model.operators:
            if '__method_precondition' in op.name:
                method_prec_ops += 1
            elif 'flip' in op.name:
                flip_ops += 1
        
        # Analyze abstract tasks
        for at in self.model.abstract_tasks:
            if 'play' in at.name:
                self.play_task_ids.add(at.global_id)
            elif 'turn' in at.name:
                self.turn_task_ids.add(at.global_id)
            elif 'propagate' in at.name:
                self.propagate_task_ids.add(at.global_id)
        
        # Estimate overhead per turn from method preconditions
        # Each turn decomposes into: flip + 4 propagate tasks
        # Each propagate task has a method with precondition check
        # The do_turn method itself may have a precondition check
        # We'll estimate based on the structure:
        # - do_turn has precondition checking (?s1 ?s2 status) -> 1 method_prec op
        # - each propagate direction: 1 method_prec op for the chosen method
        # So ~5 method_prec ops per turn + 1 flip = 6 ops minimum per turn
        # But propagation can add more flips
        # 
        # For a conservative (admissible) estimate, we use the minimum:
        # Each turn costs at least 1 (the center flip) + overhead from method precs
        # We need to figure out exact overhead from grounded model
        
        # Let's count method precondition ops per turn decomposition
        self._count_turn_overhead()

    def _count_turn_overhead(self):
        """Count the minimum number of method_precondition operators per turn.
        
        Each turn decomposes via do_turn into:
          flip + propagate_up + propagate_down + propagate_left + propagate_right
        
        Each propagate_X can end immediately (end_field or end_different) which 
        requires a method precondition check (1 op), or continue (do) which also
        requires a method precondition check (1 op) plus more flips.
        
        Minimum per turn: 1 flip + at least 1 method_prec per propagate direction
        that terminates = 1 flip + 4 method_prec ops = 5 ops
        
        But we also need to account for do_turn's own method precondition.
        Let's check the decompositions.
        """
        # Count method_precondition operators in the task network of do_turn
        # After grounding, the do_turn method's subtasks include:
        #   flip, propagate_up, propagate_down, propagate_left, propagate_right
        # Each propagate_X_end method has a precondition -> 1 __method_precondition op
        # do_turn itself has a precondition (implicitly, from the flip's precondition)
        
        # Minimum overhead per turn:
        # - 1 flip action
        # - 4 propagate directions, each terminating immediately = 4 method_prec ops
        # - do_turn method precondition = 1 method_prec op (for binding ?s1 ?s2)
        # Total minimum: 1 + 4 + 1 = 6
        # But we need to be careful - some of these may be merged in grounding
        
        # Let's be conservative and estimate from decomposition structure
        min_overhead = 0
        for dec in self.model.decompositions:
            if 'do_turn' in dec.name:
                # Count non-abstract tasks in task network
                mp_count = 0
                for task in dec.task_network:
                    if hasattr(task, 'pos_precons') and '__method_precondition' in task.name:
                        mp_count += 1
                if min_overhead == 0 or mp_count < min_overhead:
                    min_overhead = mp_count
        
        # Each propagate end also has method_prec
        prop_end_overhead = 0
        for dec in self.model.decompositions:
            if 'propagate' in dec.name and ('end' in dec.name):
                mp_count = 0
                for task in dec.task_network:
                    if hasattr(task, 'name') and '__method_precondition' in task.name:
                        mp_count += 1
                if prop_end_overhead == 0 or mp_count < prop_end_overhead:
                    prop_end_overhead = mp_count
        
        # Also count end_game overhead
        self.end_game_overhead = 0
        for dec in self.model.decompositions:
            if 'end_game' in dec.name:
                for task in dec.task_network:
                    if hasattr(task, 'name') and '__method_precondition' in task.name:
                        self.end_game_overhead += 1
                break
        
        # do_play overhead
        self.do_play_overhead = 0
        for dec in self.model.decompositions:
            if 'do_play' in dec.name:
                for task in dec.task_network:
                    if hasattr(task, 'name') and '__method_precondition' in task.name:
                        self.do_play_overhead += 1
                break
        
        # Total minimum per turn = do_turn_prec + flip + 4 * prop_end_prec + do_play_prec
        # But do_play_prec is per play iteration, not per turn
        self.overhead_per_turn = min_overhead + 4 * prop_end_overhead
        # overhead_per_turn represents the minimum method_prec ops per turn
        # (excluding the flip itself and do_play overhead)
        
        # Minimum cost per turn = 1 (flip) + overhead_per_turn (method precs)
        self.min_cost_per_turn = 1 + self.overhead_per_turn

    def _count_mismatches(self, state_bitwise):
        """Count cells where current status != goal status.
        
        This is a strict lower bound on the number of flip actions needed,
        because each flip changes exactly one cell's status.
        """
        mismatches = 0
        for fact_id in self.goal_on_fact_ids:
            if not ((state_bitwise >> fact_id) & 1):
                mismatches += 1
        return mismatches

    def _estimate_min_turns(self, state_bitwise):
        """Estimate minimum number of turns needed.
        
        A turn flips a center cell and propagates along 4 directions.
        The center flip always happens (1 flip). Propagation flips happen
        only for consecutive same-status neighbors.
        
        In the best case, a single turn can flip many cells at once
        (if they form a cross pattern of same status). But we need at
        least 1 turn for each connected component of mismatched cells
        that can't be covered by propagation from another turn.
        
        For admissibility, we use: max(1, mismatches) if mismatches > 0
        because at minimum 1 turn is needed if any mismatch exists.
        But actually, each turn flips at least 1 cell (the center),
        so we need at least ceil(mismatches / max_flips_per_turn) turns.
        
        max_flips_per_turn = 1 + 4*(grid_size-1) in the extreme case.
        For a conservative bound: at least 1 turn if mismatches > 0.
        """
        mismatches = self._count_mismatches(state_bitwise)
        if mismatches == 0:
            return 0
        
        # Each turn can flip at most 1 + up + down + left + right cells
        # where each direction can flip at most (grid_size - 1) cells
        # But realistically, propagation requires consecutive same-status
        # So we use a tighter bound
        
        # A tighter estimate: count isolated mismatched cells
        # (cells that don't have a mismatched neighbor) - each needs its own turn
        # But that's expensive to compute on every call.
        
        # Simple admissible bound: at least 1 turn per mismatch that can't
        # be reached by propagation from another mismatch.
        # For speed, just return 1 if any mismatch.
        # Actually, let's be smarter: maximum flips per turn is bounded by
        # the cross size = 1 + 4*(grid_size-1)
        
        max_flips_per_turn = 1 + 4 * (self.grid_size - 1) if self.grid_size > 0 else 1
        min_turns = max(1, -(-mismatches // max_flips_per_turn))  # ceil division
        
        return min_turns

    def _task_network_lower_bound(self, task_network):
        """Compute lower bound from remaining task network.
        
        Each primitive task (operator) in the network costs at least 1.
        Each abstract task must be decomposed into at least some primitives.
        """
        if not task_network:
            return 0
        
        cost = 0
        for task in task_network:
            tid = task.global_id
            if tid >= self.model.iop_init and tid <= self.model.iop_end:
                # Primitive operator - costs exactly 1
                cost += 1
            elif tid >= self.model.iabt_init and tid <= self.model.iabt_end:
                # Abstract task - estimate minimum decomposition cost
                if tid in self.turn_task_ids:
                    # A turn task needs at least 1 flip + overhead
                    cost += self.min_cost_per_turn
                elif tid in self.play_task_ids:
                    # Play can end immediately (end_game) or continue
                    # Minimum is 0 (end_game) + possible method_prec
                    cost += self.end_game_overhead
                elif tid in self.propagate_task_ids:
                    # Propagate can end immediately
                    # Minimum is 0 + method_prec for the end method
                    cost += 0  # Already counted in turn overhead
                else:
                    # Unknown abstract task, assume minimal cost
                    cost += 0
        
        return cost

    def _compute(self, state_bitwise, task_network):
        """Compute heuristic value as max of multiple admissible terms."""
        
        mismatches = self._count_mismatches(state_bitwise)
        
        if mismatches == 0 and (not task_network or len(task_network) == 0):
            return 0
        
        # Term 1: Raw mismatch count
        # Each mismatched cell needs at least 1 flip action (cost 1)
        # This is admissible because each flip changes exactly 1 cell
        term1 = mismatches
        
        # Term 2: Turn-based estimate
        # Each turn costs at least min_cost_per_turn
        # We need at least ceil(mismatches / max_flips_per_turn) turns
        min_turns = self._estimate_min_turns(state_bitwise)
        # Each turn also needs a do_play decomposition
        # Plus the final end_game
        term2 = mismatches + min_turns * self.overhead_per_turn
        if mismatches > 0:
            # Need at least min_turns do_play iterations + 1 end_game
            term2 += min_turns * self.do_play_overhead + self.end_game_overhead
        
        # Term 3: Task network structural bound
        term3 = self._task_network_lower_bound(task_network)
        
        return max(term1, term2, term3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic1(grid={self.grid_size}, cells={len(self.all_cells)})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic1 with {len(self.all_cells)} cells on {self.grid_size}x{self.grid_size} grid"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {self.grid_size}\n"
            f"\tTotal Cells: {len(self.all_cells)}\n"
            f"\tGoal Cells: {len(self.goal_cells)}\n"
            f"\tMin Cost Per Turn: {self.min_cost_per_turn}\n"
            f"\tOverhead Per Turn: {self.overhead_per_turn}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
