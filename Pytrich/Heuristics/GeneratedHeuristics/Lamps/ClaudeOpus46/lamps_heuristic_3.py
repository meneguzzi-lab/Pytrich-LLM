"""Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
  Each cell that differs between current state and goal state must be flipped
  at least once (1 primitive 'flip' action per cell). This gives a direct
  admissible lower bound: the number of mismatched cells.

Admissible Lower Bounds:
  Term 1 (cell_diff_count): Each cell differing from goal needs at least one
    flip action. Since flip is the only action and each flip changes exactly
    one cell, this is a perfect admissible lower bound on flip actions alone.
  
  Term 2 (turn_overhead): Each 'turn' task in the task network requires at
    least 1 flip + method precondition operators. We estimate minimum turns
    needed and account for overhead per turn.

  Term 3 (task_network_lower_bound): Analyze remaining tasks in the task
    network to estimate minimum primitive operations needed.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic3(Heuristic):
    """
    Heuristic for the Lamps domain based on counting cells that differ
    from the goal state and estimating minimum actions needed.
    
    The key insight is that each cell that differs from the goal needs
    at least one flip action, and each turn incurs overhead from
    method precondition operators and propagation task processing.
    """

    def __init__(self, lamps_anthropicopus46_heuristic_3=None, name="LampsAnthropicOpus46Heuristic3"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Maps from (x, y, status) -> fact global_id
        self.on_fact_ids = {}  # (x, y, status_str) -> global_id
        
        # Goal: list of (x, y, status_str, global_id) for goal facts
        self.goal_on_facts = []  # [(global_id, x, y, status_str), ...]
        
        # All on facts grouped by (x, y)
        self.on_facts_by_cell = {}  # (x, y) -> {status_str: global_id}
        
        # Grid dimensions
        self.grid_size = 0
        self.coords = []
        
        # Next relationships for propagation distance estimation
        self.next_map = {}  # coord -> next_coord
        self.prev_map = {}  # coord -> prev_coord
        
        # Operator and task identification
        self.flip_op_ids = set()
        self.method_precond_op_ids = set()
        
        # For task network analysis
        self.play_task_ids = set()
        self.turn_task_ids = set()
        self.propagate_task_ids = set()
        
        # Track which abstract task names map to which task types
        self.task_type_map = {}  # global_id -> 'play'|'turn'|'propagate_*'|'other'
        
        # Precomputed: for each goal on fact, what's the "wrong" status fact id
        # If goal says on(x,y,Y), the wrong fact is on(x,y,N)
        self.goal_wrong_fact_ids = []  # list of global_ids where being true means mismatch
        
        # Number of goal on-facts
        self.num_goal_on_facts = 0
        
        # Minimum cost per turn (flip + method preconditions)
        # After grounding, a turn decomposition has:
        #   flip (1 action) + propagate_up + propagate_down + propagate_left + propagate_right
        # Each propagate that ends immediately has ~1 method precondition check
        # Each propagate that continues has flip + method precondition + recursive propagate
        # Minimum per turn: 1 flip + 4 method_precondition checks (for ending propagation) = ~5
        self.min_cost_per_turn = 5  # Will be refined during preprocessing
        
        # For counting method_precondition operators in task network
        self.is_method_precond = {}  # op global_id -> bool

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
        """Extract domain structure from grounded facts and operators."""
        
        # Parse all facts
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None:
                continue
            
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.on_fact_ids[(x, y, status)] = fact.global_id
                
                if (x, y) not in self.on_facts_by_cell:
                    self.on_facts_by_cell[(x, y)] = {}
                self.on_facts_by_cell[(x, y)][status] = fact.global_id
                
                # Track coordinates
                if x not in self.coords:
                    self.coords.append(x)
            
            elif predicate == "next" and len(args) == 2:
                self.next_map[args[0]] = args[1]
                self.prev_map[args[1]] = args[0]
        
        self.grid_size = len(self.coords)
        
        # Parse goal state
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, status = args[0], args[1], args[2]
                    # Find the global_id of this goal fact
                    gid = self.on_fact_ids.get((x, y, status))
                    if gid is not None:
                        self.goal_on_facts.append((gid, x, y, status))
                        
                        # Find the "wrong" status fact
                        wrong_status = "N" if status == "Y" else "Y"
                        wrong_gid = self.on_fact_ids.get((x, y, wrong_status))
                        if wrong_gid is not None:
                            self.goal_wrong_fact_ids.append(wrong_gid)
            
            self.num_goal_on_facts = len(self.goal_on_facts)
        
        # Identify operators
        for op in self.model.operators:
            op_name = op.name.lower() if hasattr(op, 'name') else ""
            if "flip" in op_name and "method" not in op_name and "precondition" not in op_name:
                self.flip_op_ids.add(op.global_id)
                self.is_method_precond[op.global_id] = False
            elif "__method_precondition" in op_name or "method_precondition" in op_name:
                self.method_precond_op_ids.add(op.global_id)
                self.is_method_precond[op.global_id] = True
            else:
                self.is_method_precond[op.global_id] = False
        
        # Identify abstract tasks
        for at in self.model.abstract_tasks:
            at_name = at.name.lower() if hasattr(at, 'name') else ""
            if "play" in at_name:
                self.play_task_ids.add(at.global_id)
                self.task_type_map[at.global_id] = 'play'
            elif "turn" in at_name:
                self.turn_task_ids.add(at.global_id)
                self.task_type_map[at.global_id] = 'turn'
            elif "propagate" in at_name:
                self.propagate_task_ids.add(at.global_id)
                self.task_type_map[at.global_id] = 'propagate'
            else:
                self.task_type_map[at.global_id] = 'other'
        
        # Count method_precondition operators per turn to estimate overhead
        # A turn decomposes into: flip + propagate_up + propagate_down + propagate_left + propagate_right
        # Each propagation that terminates immediately has 1 method_precondition check
        # Each propagation that continues has: 1 method_precondition + 1 flip + recursive
        # Minimum per turn: 1 flip + 4 * 1 method_precondition = 5 actions
        # But we also need to account for the do_turn method precondition if it exists
        
        # Count method precondition operators associated with turn-related methods
        turn_method_precond_count = 0
        for op in self.model.operators:
            op_name = op.name if hasattr(op, 'name') else ""
            if "__method_precondition" in op_name:
                if "turn" in op_name or "propagate" in op_name:
                    turn_method_precond_count += 1
        
        # Estimate: each turn needs at minimum the flip + some method preconditions
        # From domain analysis: do_turn has no precondition, but propagate methods do
        # Minimum propagation: each direction ends (either at boundary or different status)
        # Each ending propagation needs 1 method_precondition check
        # So minimum per turn = 1 flip + 4 method_precondition (one per direction) = 5
        # But actually the do_turn itself might have a method_precondition too
        # Let's check decompositions
        
        has_turn_method_precond = False
        for dec in self.model.decompositions:
            dec_name = dec.name if hasattr(dec, 'name') else ""
            if "turn" in dec_name.lower() and "do_turn" in dec_name.lower():
                # Check if the task network starts with a method_precondition
                if dec.task_network:
                    first_task = dec.task_network[0]
                    first_name = first_task.name if hasattr(first_task, 'name') else ""
                    if "__method_precondition" in first_name:
                        has_turn_method_precond = True
        
        # Refine min_cost_per_turn
        # 1 flip + (1 if has_turn_method_precond else 0) + 4 * 1 (propagate end method preconditions)
        self.min_cost_per_turn = 1 + (1 if has_turn_method_precond else 0) + 4
        
        # Actually, let's be more careful. Look at actual decomposition task networks
        # to count minimum primitives per turn
        min_turn_primitives = float('inf')
        for dec in self.model.decompositions:
            dec_name = dec.name if hasattr(dec, 'name') else ""
            if "do_turn" in dec_name.lower():
                # Count primitives in this decomposition's immediate task network
                prim_count = 0
                abs_count = 0
                for task in dec.task_network:
                    if task.global_id >= self.model.iop_init and task.global_id <= self.model.iop_end:
                        prim_count += 1
                    else:
                        abs_count += 1
                # Each propagate abstract task has minimum 1 method precondition
                # (from the end methods that check boundary/different status)
                total_min = prim_count + abs_count  # each abstract needs at least 1 primitive
                min_turn_primitives = min(min_turn_primitives, total_min)
        
        if min_turn_primitives < float('inf'):
            self.min_cost_per_turn = max(self.min_cost_per_turn, min_turn_primitives)
        
        # Precompute the minimum primitives for propagation end
        # Look at propagate_*_end_field and propagate_*_end_different decompositions
        self.min_propagate_end_cost = 0  # Will be 0 or 1 depending on method preconditions
        for dec in self.model.decompositions:
            dec_name = dec.name if hasattr(dec, 'name') else ""
            if "propagate" in dec_name.lower() and ("end" in dec_name.lower()):
                prim_count = sum(1 for t in dec.task_network 
                               if t.global_id >= self.model.iop_init and t.global_id <= self.model.iop_end)
                if prim_count > 0:
                    self.min_propagate_end_cost = max(self.min_propagate_end_cost, prim_count)
        
        # Each propagate direction has minimum cost of min_propagate_end_cost
        # Update min_cost_per_turn: 1 flip + 4 * min_propagate_end_cost + do_turn method preconditions
        flip_count_in_turn = 0
        mp_count_in_turn = 0
        propagate_count_in_turn = 0
        for dec in self.model.decompositions:
            dec_name = dec.name if hasattr(dec, 'name') else ""
            if "do_turn" in dec_name.lower():
                for task in dec.task_network:
                    task_name = task.name if hasattr(task, 'name') else ""
                    if task.global_id >= self.model.iop_init and task.global_id <= self.model.iop_end:
                        if "flip" in task_name.lower() and "method" not in task_name.lower():
                            flip_count_in_turn += 1
                        elif "__method_precondition" in task_name:
                            mp_count_in_turn += 1
                        else:
                            # Some other primitive
                            mp_count_in_turn += 1
                    else:
                        propagate_count_in_turn += 1
                break  # Just need one do_turn decomposition
        
        if flip_count_in_turn > 0:
            self.min_cost_per_turn = flip_count_in_turn + mp_count_in_turn + propagate_count_in_turn * self.min_propagate_end_cost

    def _count_mismatched_cells(self, state_bitwise):
        """Count cells where current status differs from goal status.
        
        For each goal fact on(x,y,S), check if the 'wrong' status fact is true.
        If on(x,y,wrong_S) is true in current state, that cell is mismatched.
        """
        count = 0
        for wrong_gid in self.goal_wrong_fact_ids:
            if (state_bitwise >> wrong_gid) & 1:
                count += 1
        return count

    def _estimate_task_network_cost(self, task_network):
        """Estimate minimum cost from remaining task network.
        
        For each task in the network:
        - Primitive operators cost 1 each
        - 'play' tasks cost at least 0 (can end immediately via end_game)
        - 'turn' tasks cost at least min_cost_per_turn
        - 'propagate' tasks cost at least min_propagate_end_cost
        """
        if not task_network:
            return 0
        
        cost = 0
        for task in task_network:
            gid = task.global_id
            
            # Check if it's a primitive operator
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                cost += 1
            # Check if it's an abstract task
            elif gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                task_type = self.task_type_map.get(gid, 'other')
                if task_type == 'play':
                    # play can end immediately with end_game (0 cost)
                    # or do_play which requires turn + play
                    # We can't easily determine which, so 0 is safe lower bound
                    pass
                elif task_type == 'turn':
                    cost += self.min_cost_per_turn
                elif task_type == 'propagate':
                    cost += self.min_propagate_end_cost
                else:
                    # Unknown abstract task - assume at least 0
                    pass
        
        return cost

    def _compute(self, state_bitwise, task_network):
        """Compute heuristic value as max of multiple admissible terms."""
        
        # Term 1: Direct cell mismatch count
        # Each mismatched cell needs at least 1 flip action to fix
        # This is admissible because flip changes exactly one cell
        mismatch_count = self._count_mismatched_cells(state_bitwise)
        
        # If no mismatches, check if task network is also empty
        if mismatch_count == 0 and (not task_network or len(task_network) == 0):
            return 0
        
        # Term 2: Minimum turns needed * cost per turn
        # Each turn flips the center cell + propagation along 4 directions
        # Maximum cells flipped per turn: 1 + 4*(grid_size-1) = 4*grid_size - 3
        # But more realistically, propagation only flips consecutive same-status cells
        # A safe upper bound on max flips per turn for admissibility:
        if self.grid_size > 0:
            max_flips_per_turn = 1 + 4 * (self.grid_size - 1)
        else:
            max_flips_per_turn = 1
        
        if max_flips_per_turn > 0 and mismatch_count > 0:
            # Minimum number of turns needed
            min_turns = (mismatch_count + max_flips_per_turn - 1) // max_flips_per_turn
            turn_based_cost = min_turns * self.min_cost_per_turn
        else:
            turn_based_cost = 0
        
        # Term 3: Task network analysis
        tn_cost = self._estimate_task_network_cost(task_network)
        
        # Term 4: If there are mismatches, we need at least one turn
        # Each turn costs at least min_cost_per_turn primitives
        # And the play task adds overhead for each turn iteration
        if mismatch_count > 0:
            min_remaining_turns_for_mismatches = 1  # At least one turn needed
            # Actually we can be smarter: if many cells are mismatched and
            # they can't all be fixed in one turn due to propagation mechanics
            # For now, use the cell count as a strong lower bound
            pass
        
        # Combine with max for admissibility
        h_value = max(mismatch_count, turn_based_cost, tn_cost)
        
        return h_value

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic3(grid={self.grid_size})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic3 with grid size {self.grid_size}"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {self.grid_size}\n"
            f"\tGoal On Facts: {self.num_goal_on_facts}\n"
            f"\tMin Cost Per Turn: {self.min_cost_per_turn}\n"
            f"\tMin Propagate End Cost: {self.min_propagate_end_cost}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
