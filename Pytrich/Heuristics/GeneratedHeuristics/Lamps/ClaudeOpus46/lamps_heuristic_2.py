"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck: Each "turn" must flip at least 1 cell (the center flip) and also
pay for 4 propagation-end method_precondition operators plus the do_turn
method_precondition. Propagation flips are constrained by adjacency.
The number of cells that differ from goal gives a lower bound on flips needed.
Each turn produces at least 1 flip but can propagate to fix neighbors.

Admissible lower bounds:
1. Number of goal-mismatched cells: each flip fixes exactly one cell, so we need
   at least that many flip actions. But flips also come with overhead operators.
2. Structural turn count: if there are k mismatched cells, we need at least
   ceil(k / max_propagation_chain) turns, each costing overhead operators.
3. Task network analysis: count remaining compound tasks and estimate minimum
   primitive operators each will produce.

We return max of these terms.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic2(Heuristic):
    """
    Heuristic for the Lamps domain that estimates remaining cost based on:
    - Number of cells that differ from their goal status (minimum flips needed)
    - Minimum overhead operators from turn structure
    - Task network structure analysis
    """

    def __init__(self, lamps_anthropicopus46_heuristic_2=None, name="LampsAnthropicOpus46Heuristic2"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Maps for quick state checking
        self.goal_on_facts = {}  # (x, y, s) -> fact.global_id for on[x,y] facts in goal
        self.on_facts = {}       # (x, y, s) -> fact.global_id for all on facts
        self.goal_cells = {}     # (x, y) -> desired status s
        self.all_cells = set()   # set of (x, y) pairs
        
        # For each cell, store the fact global_ids for on[x,y,Y] and on[x,y,N]
        self.cell_status_facts = {}  # (x, y) -> {status_str -> global_id}
        
        # Grid structure
        self.grid_size = 0
        self.coords = []
        self.next_map = {}  # coord -> next coord
        self.prev_map = {}  # coord -> prev coord
        self.max_coord = None
        self.min_coord = None
        
        # Operator analysis
        self.flip_operators = []  # list of flip operators
        self.method_precondition_operators = []  # __method_precondition operators
        
        # Task type identification
        self.play_task_ids = set()
        self.turn_task_ids = set()
        self.propagate_task_ids = set()
        
        # Cost model constants (estimated from domain structure)
        # Per turn: 1 flip + at least 4 propagation-end method_precs
        # Plus do_turn method_prec, do_play method_prec
        self.min_cost_per_turn = 1  # Will be refined during preprocessing
        self.min_overhead_per_turn = 0  # Method precondition operators per turn

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
        
        # Parse all facts to find on[x,y] facts and structure
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None:
                continue
            
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.on_facts[(x, y, s)] = fact.global_id
                self.all_cells.add((x, y))
                if (x, y) not in self.cell_status_facts:
                    self.cell_status_facts[(x, y)] = {}
                self.cell_status_facts[(x, y)][s] = fact.global_id
            
            elif predicate == "next" and len(args) == 2:
                self.next_map[args[0]] = args[1]
                self.prev_map[args[1]] = args[0]
            
            elif predicate == "max" and len(args) == 1:
                self.max_coord = args[0]
            
            elif predicate == "min" and len(args) == 1:
                self.min_coord = args[0]
        
        # Build ordered coords list
        if self.min_coord is not None:
            c = self.min_coord
            self.coords = [c]
            while c in self.next_map:
                c = self.next_map[c]
                self.coords.append(c)
            self.grid_size = len(self.coords)
        
        # Parse goal to find desired status for each cell
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    self.goal_cells[(x, y)] = s
                    self.goal_on_facts[(x, y, s)] = self.on_facts.get((x, y, s))
        
        # Analyze operators to understand cost structure
        method_prec_count = 0
        flip_count = 0
        for op in self.model.operators:
            if "__method_precondition" in op.name:
                method_prec_count += 1
                self.method_precondition_operators.append(op)
            elif "flip" in op.name.lower():
                flip_count += 1
                self.flip_operators.append(op)
        
        # Analyze abstract tasks
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower()
            if "play" in name_lower:
                self.play_task_ids.add(at.global_id)
            elif "turn" in name_lower:
                self.turn_task_ids.add(at.global_id)
            elif "propagate" in name_lower:
                self.propagate_task_ids.add(at.global_id)
        
        # Estimate minimum overhead per turn from the decomposition structure
        # Each turn decomposes into: flip + propagate_up + propagate_down + propagate_left + propagate_right
        # Each propagation at minimum has a method_precondition for the end condition
        # do_turn itself may have a method_precondition
        # Count method_preconditions in the minimal decomposition path
        self._estimate_turn_overhead()

    def _estimate_turn_overhead(self):
        """Estimate minimum number of method_precondition operators per turn."""
        # In the grounded domain, each method with a precondition generates a
        # __method_precondition_* operator. A minimal turn:
        # - do_turn method: may have a method_precondition for binding s1, s2
        # - propagate_up_end (or similar): has method_precondition
        # - propagate_down_end: has method_precondition  
        # - propagate_left_end: has method_precondition
        # - propagate_right_end: has method_precondition
        # So minimum ~4-5 method_preconditions + 1 flip = 5-6 operators per turn
        
        # Let's count by looking at decompositions
        min_prop_cost = {}
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower()
            if "propagate" in name_lower:
                # Find minimum decomposition cost for this propagation task
                min_cost = float('inf')
                for dec in at.decompositions:
                    # Count primitive operators in the task network
                    prim_count = 0
                    has_recursive = False
                    for subtask in dec.task_network:
                        if hasattr(subtask, 'decompositions'):
                            # Abstract task - if it's the same propagation, it's recursive
                            has_recursive = True
                        else:
                            prim_count += 1
                    if not has_recursive:
                        # This is an end method
                        min_cost = min(min_cost, prim_count)
                if min_cost < float('inf'):
                    min_prop_cost[at.global_id] = min_cost
        
        # For turn tasks, count minimum primitives
        for at in self.model.abstract_tasks:
            if at.global_id in self.turn_task_ids:
                for dec in at.decompositions:
                    prim_count = 0
                    prop_count = 0
                    for subtask in dec.task_network:
                        if hasattr(subtask, 'decompositions'):
                            # Abstract subtask
                            if subtask.global_id in min_prop_cost:
                                prim_count += min_prop_cost[subtask.global_id]
                                prop_count += 1
                        else:
                            prim_count += 1
                    self.min_cost_per_turn = max(self.min_cost_per_turn, 1)  # At least the flip
                    self.min_overhead_per_turn = max(0, prim_count - 1)  # Overhead beyond the flip
                    break  # Use first decomposition found
        
        # If we couldn't determine from decompositions, use conservative estimate
        if self.min_overhead_per_turn == 0:
            # Conservative: at least 4 propagation ends (each with 1 method_prec)
            # But we count conservatively
            self.min_overhead_per_turn = 4
        
        self.min_cost_per_turn = 1 + self.min_overhead_per_turn

    def _count_mismatched_cells(self, state_bitwise):
        """Count cells that differ from their goal status. O(|cells|) with bit tests."""
        count = 0
        for (x, y), desired_s in self.goal_cells.items():
            fact_id = self.goal_on_facts.get((x, y, desired_s))
            if fact_id is not None:
                if not ((state_bitwise >> fact_id) & 1):
                    count += 1
        return count

    def _estimate_min_turns(self, state_bitwise):
        """
        Estimate minimum number of turns needed.
        
        Each turn flips 1 center cell + potentially propagation cells.
        In the best case, a single turn can fix a whole row/column of same-status cells.
        Maximum propagation reach is grid_size in each direction.
        So max cells fixed per turn = 1 + (grid_size-1)*4 in theory.
        But more realistically, we need to consider that propagation only works
        on consecutive same-status neighbors.
        
        Conservative lower bound: ceil(mismatched / max_cells_per_turn)
        Better: analyze mismatched cells by location for tighter bounds.
        """
        mismatched = self._count_mismatched_cells(state_bitwise)
        if mismatched == 0:
            return 0
        
        # Each turn can flip at most 1 + 4*(grid_size-1) cells
        # But that's very optimistic. A more realistic bound:
        # Each turn flips 1 center + consecutive same-status neighbors in 4 directions
        # In practice, max cells per turn is bounded by the grid size
        max_cells_per_turn = 1 + 4 * max(0, self.grid_size - 1) if self.grid_size > 0 else 1
        
        # But we need a tighter bound. Let's think about it differently:
        # Each flip in a propagation chain also changes a cell. But the key insight
        # is that propagation only flips cells that currently have the SAME status
        # as the center cell's ORIGINAL status. So after flipping the center,
        # propagation flips neighbors with the old status.
        # 
        # A better bound: each turn produces at least 1 flip action.
        # The number of flip actions needed is exactly the number of mismatched cells
        # (since each flip changes exactly one cell, and we need to change each
        # mismatched cell at least once, net).
        # But cells might be flipped multiple times (flipped and then flipped back).
        # So mismatched count is a lower bound on flip actions.
        
        return mismatched  # Each mismatched cell needs at least 1 flip

    def _task_network_estimate(self, task_network):
        """
        Estimate minimum remaining cost from the task network structure.
        
        Count tasks and estimate their minimum decomposition costs.
        """
        if not task_network:
            return 0
        
        total = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):
                # Abstract task
                if task.global_id in self.play_task_ids:
                    # play can be end_game (cost 0) or do_play (turn + play)
                    # Don't add anything here - we handle it through turn counting
                    pass
                elif task.global_id in self.turn_task_ids:
                    # A turn costs at least min_cost_per_turn
                    total += self.min_cost_per_turn
                elif task.global_id in self.propagate_task_ids:
                    # Propagation costs at least 1 (method_precondition for end)
                    total += 1
                else:
                    # Unknown abstract task, estimate at least 1
                    total += 1
            else:
                # Primitive operator - costs 1
                total += 1
        
        return total

    def _analyze_runs(self, state_bitwise):
        """
        Analyze mismatched cells to estimate minimum turns more tightly.
        
        Group mismatched cells into connected runs along rows and columns.
        Each isolated group of mismatched cells may require at least one turn.
        
        Returns a lower bound on the number of turns needed.
        """
        if not self.coords or not self.goal_cells:
            return 0
        
        mismatched = set()
        for (x, y), desired_s in self.goal_cells.items():
            fact_id = self.goal_on_facts.get((x, y, desired_s))
            if fact_id is not None:
                if not ((state_bitwise >> fact_id) & 1):
                    mismatched.add((x, y))
        
        if not mismatched:
            return 0
        
        # Each turn can affect cells in a cross pattern (one row segment + one column segment)
        # centered at the chosen cell. So each turn can fix at most one row-run and one column-run.
        # 
        # Count "independent" mismatched regions:
        # - Count mismatched cells in each row (row runs)
        # - Count mismatched cells in each column (column runs)
        # A turn affects one point on each axis, so it can fix cells along
        # at most one row and one column.
        
        # Simple approach: count rows that have mismatched cells
        # and columns that have mismatched cells
        rows_with_mismatch = set()
        cols_with_mismatch = set()
        for (x, y) in mismatched:
            rows_with_mismatch.add(x)
            cols_with_mismatch.add(y)
        
        # Lower bound: max of number of distinct rows or columns with mismatches
        # (each turn affects at most one row and one column)
        # But this is too loose. Let's just use ceil(mismatched / max_per_turn)
        
        return len(mismatched)

    def _compute(self, state_bitwise, task_network):
        """Compute heuristic value as max of multiple admissible terms."""
        
        # Term 1: Number of mismatched cells = lower bound on flip actions
        mismatched = self._count_mismatched_cells(state_bitwise)
        
        if mismatched == 0 and (not task_network or len(task_network) == 0):
            return 0
        
        # Term 2: Task network structure estimate
        tn_estimate = self._task_network_estimate(task_network)
        
        # Term 3: Turn-based estimate
        # Each turn that actually changes something costs at least min_cost_per_turn.
        # We need at least ceil(mismatched / max_propagation) turns to fix all mismatches.
        # But each turn's flip count contributes to mismatched count.
        # If we need at least T turns, and each turn costs min_cost_per_turn,
        # then total cost >= T * min_cost_per_turn.
        # But T's lower bound is tricky since one turn can fix many cells.
        # However, mismatched cells is already a lower bound on total flip actions.
        # The overhead method_preconditions add to cost.
        # 
        # If there are M mismatched cells:
        # - We need at least M flip actions (lower bound)
        # - We need at least 1 turn if M > 0 (each turn has overhead)
        # - Total cost >= M + overhead_per_turn * num_turns
        # 
        # Conservative: we need at least ceil(M / max_per_turn) turns
        # Each turn adds min_overhead_per_turn overhead operators
        
        if mismatched > 0:
            # Max cells a single turn can flip: 1 + propagation in 4 dirs
            max_per_turn = 1 + 4 * max(0, self.grid_size - 1) if self.grid_size > 0 else 1
            min_turns = max(1, -(-mismatched // max_per_turn))  # ceiling division
            
            # Cost = at least mismatched flips + overhead per turn
            turn_estimate = mismatched + min_turns * self.min_overhead_per_turn
        else:
            turn_estimate = 0
        
        # Return max of all terms
        h = max(mismatched, tn_estimate, turn_estimate)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic2(grid={self.grid_size}x{self.grid_size})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic2 with {self.grid_size}x{self.grid_size} grid"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {self.grid_size}x{self.grid_size}\n"
            f"\tGoal Cells: {len(self.goal_cells)}\n"
            f"\tMin Cost Per Turn: {self.min_cost_per_turn}\n"
            f"\tMin Overhead Per Turn: {self.min_overhead_per_turn}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
