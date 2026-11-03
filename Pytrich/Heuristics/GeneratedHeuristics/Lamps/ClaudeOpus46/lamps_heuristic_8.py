"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck: Each "turn" flips a center cell and propagates in 4 directions,
flipping consecutive same-status neighbors. The core serializing constraint is
that each turn must produce at least 1 flip action plus method-precondition
operators for the 4 propagation directions (even if they immediately terminate).

Admissible lower bounds:
1. Goal-mismatch count: Each cell that differs from goal needs at least one flip
   action. Multiple cells CAN be flipped in one turn (propagation), but each
   flip is still a separate primitive action costing 1. So the number of
   mismatched cells is a lower bound on flip actions alone.
2. Turn-overhead bound: Each turn generates at least 1 flip + overhead from
   method precondition operators for propagation tasks. If we can estimate
   minimum number of turns needed, we get a tighter bound.
3. Task-network bound: State-aware estimation of remaining task costs.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic8(Heuristic):
    """
    Heuristic for Lamps domain that combines:
    - Goal mismatch counting (how many cells differ from goal)
    - Task network analysis with state-aware cost estimation
    - Method precondition overhead estimation
    """

    def __init__(self, lamps_anthropicopus46_heuristic_8=None, name="LampsAnthropicOpus46Heuristic8"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Maps for fast state checking
        self.on_facts = {}          # (x, y, s) -> fact.global_id
        self.goal_on_specs = []     # list of (x, y, s, global_id) for goal facts
        self.goal_state = 0         # bitwise goal state
        self.goal_fact_ids = set()  # set of global_ids that appear in goal
        
        # Grid info
        self.coords = set()
        self.next_map = {}          # coord -> next coord
        self.prev_map = {}          # coord -> prev coord
        self.max_coords = set()
        self.min_coords = set()
        self.grid_size = 0
        
        # For task network analysis
        self.turn_task_ids = set()       # global_ids of 'turn' abstract tasks
        self.play_task_ids = set()       # global_ids of 'play' abstract tasks
        self.propagate_task_ids = set()  # global_ids of propagate_* abstract tasks
        self.flip_op_ids = set()         # global_ids of flip operators
        self.method_precond_op_ids = set()  # global_ids of __method_precondition operators
        
        # Precomputed: for each cell (x,y), which goal status is required
        self.goal_status = {}  # (x, y) -> 'Y' or 'N'
        
        # For fast mismatch detection: 
        # on_Y_facts[x,y] = global_id of on[x,y,Y]
        # on_N_facts[x,y] = global_id of on[x,y,N]
        self.on_Y_facts = {}
        self.on_N_facts = {}
        
        # All cells as list for iteration
        self.all_cells = []

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
                self.coords.add(x)
                self.coords.add(y)
                if s == "Y":
                    self.on_Y_facts[(x, y)] = fact.global_id
                elif s == "N":
                    self.on_N_facts[(x, y)] = fact.global_id
            elif predicate == "next" and len(args) == 2:
                self.next_map[args[0]] = args[1]
                self.prev_map[args[1]] = args[0]
            elif predicate == "max" and len(args) == 1:
                self.max_coords.add(args[0])
            elif predicate == "min" and len(args) == 1:
                self.min_coords.add(args[0])
        
        # Build coord list and grid size
        self.grid_size = len(self.coords)
        
        # Build all_cells: all (x, y) pairs that have on facts
        cell_set = set()
        for (x, y, s) in self.on_facts:
            cell_set.add((x, y))
        self.all_cells = list(cell_set)
        
        # Parse goal state
        self.goal_state = self.model.goals
        if self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gname in goal_fact_names:
                predicate, args = self._parse_fact_name(gname)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    self.goal_status[(x, y)] = s
                    gid = self.on_facts.get((x, y, s))
                    if gid is not None:
                        self.goal_on_specs.append((x, y, s, gid))
                        self.goal_fact_ids.add(gid)
        
        # Categorize abstract tasks and operators
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower()
            if 'turn' in name_lower and 'propagate' not in name_lower:
                self.turn_task_ids.add(at.global_id)
            elif 'play' in name_lower:
                self.play_task_ids.add(at.global_id)
            elif 'propagate' in name_lower:
                self.propagate_task_ids.add(at.global_id)
        
        for op in self.model.operators:
            name_lower = op.name.lower()
            if 'flip' in name_lower and '__method_precondition' not in name_lower:
                self.flip_op_ids.add(op.global_id)
            if '__method_precondition' in name_lower:
                self.method_precond_op_ids.add(op.global_id)

    def _count_mismatches(self, state_bitwise):
        """
        Count how many cells differ from goal.
        Each mismatch requires at least one flip action (cost 1).
        This is admissible since each flip changes exactly one cell.
        """
        count = 0
        for (x, y, s, gid) in self.goal_on_specs:
            # Check if the goal fact on[x,y,s] is true in current state
            if not ((state_bitwise >> gid) & 1):
                count += 1
        return count

    def _estimate_task_network_cost(self, state_bitwise, task_network):
        """
        State-aware task network cost estimation.
        
        For each task in the network:
        - play: estimate based on remaining mismatches (each turn fixes some cells)
        - turn: at least 1 flip + method precondition costs for 4 propagations
        - propagate_*: depends on how many consecutive same-status cells exist
        - flip operator: cost 1
        - method_precondition: cost 1
        """
        if not task_network:
            return 0
        
        total = 0
        mismatches = None  # lazy compute
        
        for task_item in task_network:
            gid = task_item.global_id
            
            if gid in self.play_task_ids:
                # play task: will decompose into (turn; play) or empty
                # Lower bound: need enough turns to fix all mismatches
                # Each turn does at least 1 flip, but propagation can do more
                # Minimum cost per turn: 1 (flip) + at least 4 (propagation end checks)
                # Actually, the grounder may inline method preconditions as operators
                # costing 1 each. Let's estimate conservatively.
                if mismatches is None:
                    mismatches = self._count_mismatches(state_bitwise)
                if mismatches > 0:
                    # Each turn costs at least 1 flip action.
                    # With method precondition overhead, each turn costs at least ~5
                    # (1 flip + 4 propagation-end method preconditions).
                    # But we need at least 'mismatches' flips total.
                    # Number of turns needed: at least 1 (since mismatches > 0)
                    # But total flips >= mismatches.
                    # Best estimate: max(mismatches, min_turns * overhead_per_turn)
                    # For admissibility, just use mismatches as the flip lower bound.
                    total += mismatches
                # If mismatches == 0, play can end immediately (end_game), cost 0
                break  # play encompasses all remaining work
                
            elif gid in self.turn_task_ids:
                # A turn task: 1 flip + 4 propagation subtasks
                # Minimum cost: 1 (the center flip)
                # Plus propagation costs
                total += 1  # at least the center flip
                
            elif gid in self.propagate_task_ids:
                # Propagation task: either ends immediately (0 cost, but method
                # precondition check costs 1) or does flip + recurse
                # Minimum cost when ending: method precondition operator = 1
                # But this depends on grounder. Be conservative: 0
                pass
                
            elif gid in self.flip_op_ids:
                total += 1
                
            elif gid in self.method_precond_op_ids:
                total += 1
            
            else:
                # Other operators cost 1
                # Check if it's an operator (has cost attribute)
                if hasattr(task_item, 'cost'):
                    total += 1
        
        return total

    def _compute(self, state_bitwise, task_network):
        """
        Compute heuristic as max of multiple admissible terms.
        """
        if not task_network:
            # No tasks left - check if goal is reached
            if self.model.goal_reached(state_bitwise, task_network):
                return 0
            return float('inf')
        
        # Term 1: Goal mismatch count (admissible: each mismatch needs >= 1 flip)
        mismatch_count = self._count_mismatches(state_bitwise)
        
        # Term 2: Task network based estimation
        tn_cost = self._estimate_task_network_cost(state_bitwise, task_network)
        
        # Term 3: Minimum turns estimate with overhead
        # If there are mismatches, we need at least 1 turn.
        # Each turn has overhead: the 4 propagation directions each need
        # at least a termination check. In the grounded model, these are
        # __method_precondition operators costing 1 each.
        # So minimum cost per turn ~ 1 (flip) + overhead
        # But we don't know exact overhead without analyzing grounder output.
        # Conservative: just use mismatch_count
        
        # For play tasks at the front, the mismatch count IS the lower bound
        # on remaining flips. But we also know each turn has overhead.
        # A better bound: if we need T turns, cost >= T * min_turn_cost + extra_flips
        # where min_turn_cost >= 1 and extra_flips are propagation flips.
        # Since propagation flips also count toward fixing mismatches,
        # total flips >= mismatch_count, so cost >= mismatch_count.
        
        h_value = max(mismatch_count, tn_cost)
        
        return h_value

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic8(cells={len(self.all_cells)})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic8 with {len(self.all_cells)} cells"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid cells: {len(self.all_cells)}\n"
            f"\tGoal specs: {len(self.goal_on_specs)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
