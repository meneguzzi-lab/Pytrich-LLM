"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
  The Lamps domain models a grid of lamps, each with status Y or N. The only
  action is 'flip', which toggles a lamp. A 'turn' picks a cell (x,y), flips it,
  then propagates in all 4 directions flipping consecutive cells that share the
  OLD status. Each turn thus flips a cross-shaped region of same-status cells.
  The bottleneck is that each turn produces exactly 1 flip for the center cell
  plus variable propagation flips, and turns are serialized through the
  play/turn/play loop. Each turn costs at least 1 primitive action (the center
  flip) plus __method_precondition operators.

Admissible Lower Bounds:
  Term 1 (Goal Mismatch Count): Count the number of cells where current status
    differs from goal status. Each flip action changes exactly one cell, so the
    number of mismatched cells is a lower bound on the number of flip actions
    needed. This is admissible because no single action can fix more than one
    mismatch.

  Term 2 (Turn-based lower bound): Each 'turn' in the HTN structure requires
    at least 1 flip (the center cell) plus method precondition operators. We
    count the minimum number of turns needed. Since a turn flips a cross-shaped
    contiguous region, we need at least ceil(mismatches / max_cross_size) turns,
    but more practically, each turn requires at least 2 primitive ops (the
    __method_precondition for do_turn + the center flip). We estimate the
    minimum number of turns from the task network and multiply by the minimum
    cost per turn.

  Term 3 (Task network analysis): Count remaining compound tasks that must
    produce at least one primitive action each, giving a lower bound.

The heuristic returns max(term1, term2, term3).
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic5(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.
    
    Uses multiple admissible lower bounds:
    1. Number of cells that differ from goal (each flip fixes exactly one cell)
    2. Turn-based estimate from task network structure
    3. Compound task lower bound from remaining task network
    """

    def __init__(self, lamps_anthropicopus46_heuristic_5=None, name="LampsAnthropicOpus46Heuristic5"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Maps for efficient state checking
        # on_facts[(x,y,s)] -> fact.global_id
        self.on_facts = {}  
        # goal_config[(x,y)] -> desired status ('Y' or 'N')
        self.goal_config = {}
        # All (x,y) cells in the grid
        self.all_cells = set()
        # For each cell, store the global_ids of on[x,y,Y] and on[x,y,N]
        self.cell_y_fact = {}  # (x,y) -> global_id of on[x,y,Y]
        self.cell_n_fact = {}  # (x,y) -> global_id of on[x,y,N]
        
        # Grid structure
        self.next_map = {}  # coord -> next coord
        self.prev_map = {}  # coord -> prev coord  
        self.max_coord = None
        self.min_coord = None
        self.grid_size = 0
        
        # Task name identification
        self.turn_task_ids = set()  # global_ids of 'turn' abstract tasks
        self.play_task_ids = set()  # global_ids of 'play' abstract tasks
        self.flip_op_ids = set()    # global_ids of 'flip' operators
        self.method_precond_op_ids = set()  # global_ids of __method_precondition operators
        
        # Decomposition analysis
        self.do_turn_decomp_ids = set()
        self.do_play_decomp_ids = set()
        self.end_game_decomp_ids = set()

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
        """Extract domain structure from grounded facts and build lookup maps."""
        
        # Parse all facts
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None:
                continue
            
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.on_facts[(x, y, s)] = fact.global_id
                self.all_cells.add((x, y))
                if s == "Y":
                    self.cell_y_fact[(x, y)] = fact.global_id
                elif s == "N":
                    self.cell_n_fact[(x, y)] = fact.global_id
            elif predicate == "next" and len(args) == 2:
                self.next_map[args[0]] = args[1]
                self.prev_map[args[1]] = args[0]
            elif predicate == "max" and len(args) == 1:
                self.max_coord = args[0]
            elif predicate == "min" and len(args) == 1:
                self.min_coord = args[0]
        
        # Compute grid size
        coords = set()
        for (x, y) in self.all_cells:
            coords.add(x)
            coords.add(y)
        self.grid_size = len(coords)
        
        # Extract goal configuration
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    self.goal_config[(x, y)] = s
        
        # Identify abstract tasks and operators by name
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
                self.method_precond_op_ids.add(op.global_id)
        
        # Identify decompositions
        for dec in self.model.decompositions:
            name_lower = dec.name.lower()
            if 'do_turn' in name_lower:
                self.do_turn_decomp_ids.add(dec.global_id)
            elif 'do_play' in name_lower:
                self.do_play_decomp_ids.add(dec.global_id)
            elif 'end_game' in name_lower:
                self.end_game_decomp_ids.add(dec.global_id)
        
        # Precompute the minimum number of primitive actions per turn.
        # A turn decomposes into: flip + propagate_up + propagate_down + propagate_left + propagate_right
        # The flip is 1 action. Each propagate may have a __method_precondition.
        # At minimum, a turn costs: 1 (flip) + overhead from method preconditions.
        # Let's estimate by looking at do_turn decomposition task networks.
        self.min_cost_per_turn = 1  # At least 1 flip
        
        # Count method_precondition operators in a do_turn decomposition
        for dec in self.model.decompositions:
            if dec.global_id in self.do_turn_decomp_ids:
                method_prec_count = 0
                for task in dec.task_network:
                    if task.global_id in self.method_precond_op_ids:
                        method_prec_count += 1
                # A turn costs at least 1 flip + method_prec_count
                cost = 1 + method_prec_count
                if cost > self.min_cost_per_turn:
                    self.min_cost_per_turn = cost
                break
        
        # Also count the method_precondition for do_play decomposition
        self.play_overhead = 0
        for dec in self.model.decompositions:
            if dec.global_id in self.do_play_decomp_ids:
                for task in dec.task_network:
                    if task.global_id in self.method_precond_op_ids:
                        self.play_overhead += 1
                break

    def _count_mismatches(self, state_bitwise):
        """
        Count the number of cells where current status differs from goal.
        Uses fast bitwise checks.
        """
        mismatches = 0
        for (x, y), goal_status in self.goal_config.items():
            if goal_status == "Y":
                # Goal wants Y. Check if on[x,y,Y] is true
                gid = self.cell_y_fact.get((x, y))
                if gid is not None:
                    if not ((state_bitwise >> gid) & 1):
                        mismatches += 1
                else:
                    mismatches += 1
            elif goal_status == "N":
                # Goal wants N. Check if on[x,y,N] is true
                gid = self.cell_n_fact.get((x, y))
                if gid is not None:
                    if not ((state_bitwise >> gid) & 1):
                        mismatches += 1
                else:
                    mismatches += 1
        return mismatches

    def _count_task_network_cost(self, task_network):
        """
        Estimate minimum cost from remaining task network.
        
        For each element in the task network:
        - Primitive operators cost 1 each
        - 'turn' abstract tasks cost at least min_cost_per_turn
        - 'play' abstract tasks cost 0 (they can end immediately via end_game)
        - Other abstract tasks cost at least 0 (propagation can be empty)
        """
        cost = 0
        for task in task_network:
            gid = task.global_id
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                # It's a primitive operator - costs 1
                cost += 1
            elif gid in self.turn_task_ids:
                # A turn task costs at least min_cost_per_turn
                cost += self.min_cost_per_turn
            elif gid in self.play_task_ids:
                # play can end immediately (end_game), so minimum cost is 0
                # But if there are mismatches, it must do at least one more turn
                pass
            # propagate tasks can be empty (end methods), so min cost = 0
            # except they might have method_precondition operators
        return cost

    def _estimate_turns_needed(self, mismatches):
        """
        Estimate minimum number of turns needed to fix all mismatches.
        
        Each turn flips a cross-shaped region. In the best case, a turn
        at position (x,y) can flip up to (2*grid_size - 1) cells (the entire
        row and column through (x,y) if they're all the same status).
        But realistically, each turn flips at least 1 cell and at most
        a cross. A conservative lower bound is ceil(mismatches / max_cross).
        
        However, we can be smarter: each turn changes the parity of the
        center cell and propagates. The minimum is at least 1 turn per
        isolated mismatch.
        """
        if mismatches == 0:
            return 0
        
        # Maximum cells a single turn can flip:
        # center + up to (grid_size-1) in each of 4 directions = up to 4*(grid_size-1)+1
        max_cross = 4 * (self.grid_size - 1) + 1 if self.grid_size > 0 else 1
        max_cross = max(max_cross, 1)
        
        # Lower bound: at least ceil(mismatches / max_cross) turns
        min_turns = (mismatches + max_cross - 1) // max_cross
        min_turns = max(min_turns, 1)  # At least 1 turn if any mismatch
        
        return min_turns

    def _compute(self, state_bitwise, task_network):
        """
        Compute the heuristic value using multiple admissible lower bounds.
        """
        # Term 1: Number of mismatched cells (each flip fixes exactly one)
        mismatches = self._count_mismatches(state_bitwise)
        term1 = mismatches  # Each mismatch needs at least 1 flip action
        
        if mismatches == 0 and len(task_network) == 0:
            return 0
        
        # Term 2: Turn-based estimate
        # Each turn costs at least min_cost_per_turn primitive actions
        # Plus each do_play that wraps a turn adds play_overhead
        min_turns = self._estimate_turns_needed(mismatches)
        term2 = min_turns * (self.min_cost_per_turn + self.play_overhead)
        
        # Term 3: Direct task network cost estimate
        term3 = self._count_task_network_cost(task_network)
        
        # If there are mismatches but task network has only play tasks,
        # those play tasks must decompose into turns
        if mismatches > 0:
            has_turn_in_network = False
            has_play_in_network = False
            for task in task_network:
                gid = task.global_id
                if gid in self.turn_task_ids:
                    has_turn_in_network = True
                if gid in self.play_task_ids:
                    has_play_in_network = True
            
            if has_play_in_network and not has_turn_in_network:
                # Play must decompose into at least min_turns turns
                # Each turn produces at least min_cost_per_turn + play_overhead actions
                term3 = max(term3, min_turns * (self.min_cost_per_turn + self.play_overhead))
        
        return max(term1, term2, term3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic5(grid={self.grid_size})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic5 with grid size {self.grid_size}"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {self.grid_size}\n"
            f"\tGoal Cells: {len(self.goal_config)}\n"
            f"\tMin Cost Per Turn: {self.min_cost_per_turn}\n"
            f"\tPlay Overhead: {self.play_overhead}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
