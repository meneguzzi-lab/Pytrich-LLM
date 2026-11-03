"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck: Each flip action changes exactly one cell (x,y) from status s1 to s2.
Propagation after a turn center-flip causes additional flips in 4 directions on
consecutive same-status neighbors. The cost is the total number of flip actions.

Admissible lower bounds:
  Term 1 (goal_mismatch): Number of cells where current status != goal status.
    Admissible because each flip action changes exactly one cell, so we need at
    least this many flip actions to reach the goal.
  Term 2 (task_network_lb): Lower bound from task network structure. Each 'turn'
    task requires at least 1 flip. Each 'play' with do_play decomposition needs
    at least 1 turn. Provides a structural lower bound.
  Term 3 (turn_count_lb): Since each turn does 1 center flip + propagation flips,
    and the center flip always happens, we need at least ceil(mismatches / max_propagation)
    turns, but each turn has at least 1 flip. We use a refined estimate.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic12(Heuristic):
    """
    Heuristic for the Lamps domain based on goal mismatch counting
    and task network analysis.
    """

    def __init__(self, lamps_anthropicopus46_heuristic_12=None, name="LampsAnthropicOpus46Heuristic12"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Maps for fast lookup
        self.goal_on_facts = {}       # (x, y) -> desired_status
        self.on_fact_ids = {}         # (x, y, status) -> global_id
        self.all_cells = set()        # set of (x, y) tuples
        self.status_Y = None          # string for Y status
        self.status_N = None          # string for N status
        
        # For task network analysis
        self.turn_task_ids = set()    # global_ids of 'turn' abstract tasks
        self.play_task_ids = set()    # global_ids of 'play' abstract tasks
        self.flip_op_ids = set()      # global_ids of 'flip' operators
        
        # Grid size for max propagation estimate
        self.grid_size = 0
        
        # Precomputed: for each cell, the fact global_id for on(x,y,Y) and on(x,y,N)
        self.cell_y_fact_id = {}      # (x, y) -> global_id of on[x,y,Y]
        self.cell_n_fact_id = {}      # (x, y) -> global_id of on[x,y,N]
        
        # Goal: for each cell, which status is desired? True=Y, False=N, None=don't care
        self.goal_status = {}         # (x, y) -> 'Y' or 'N'
        
        # Bit masks for quick mismatch counting
        self.goal_y_mask = 0          # bits set for on[x,y,Y] facts that should be true in goal
        self.goal_n_mask = 0          # bits set for on[x,y,N] facts that should be true in goal
        self.all_goal_y_ids = []      # list of global_ids for goal Y facts
        self.all_goal_n_ids = []      # list of global_ids for goal N facts
        
        # Method precondition operator detection
        self.method_precondition_ids = set()

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
        Build lookup structures for fast heuristic computation.
        """
        coords = set()
        
        # Parse all facts to find on[x,y,s] facts
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                coords.add(x)
                coords.add(y)
                self.on_fact_ids[(x, y, s)] = fact.global_id
                self.all_cells.add((x, y))
                if s == "Y":
                    self.cell_y_fact_id[(x, y)] = fact.global_id
                elif s == "N":
                    self.cell_n_fact_id[(x, y)] = fact.global_id
        
        self.grid_size = len(coords)
        
        # Parse goal state
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    self.goal_status[(x, y)] = s
                    gid = self.on_fact_ids.get((x, y, s))
                    if gid is not None:
                        if s == "Y":
                            self.goal_y_mask |= (1 << gid)
                            self.all_goal_y_ids.append(gid)
                        elif s == "N":
                            self.goal_n_mask |= (1 << gid)
                            self.all_goal_n_ids.append(gid)
        
        # Identify task types for task network analysis
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower()
            if 'turn' in name_lower and 'propagate' not in name_lower and 'play' not in name_lower:
                self.turn_task_ids.add(at.global_id)
            elif 'play' in name_lower:
                self.play_task_ids.add(at.global_id)
        
        # Identify flip operators and method precondition operators
        for op in self.model.operators:
            name_lower = op.name.lower()
            if 'flip' in name_lower and '__method_precondition' not in name_lower:
                self.flip_op_ids.add(op.global_id)
            if '__method_precondition' in name_lower:
                self.method_precondition_ids.add(op.global_id)

    def _count_mismatches(self, state):
        """
        Count cells where current status != goal status.
        Each goal fact that is NOT true in state represents a mismatch.
        
        This is admissible: each flip action fixes exactly one cell.
        """
        count = 0
        # Check goal Y facts - if on[x,y,Y] should be true but isn't
        for gid in self.all_goal_y_ids:
            if not ((state >> gid) & 1):
                count += 1
        # Check goal N facts - if on[x,y,N] should be true but isn't
        for gid in self.all_goal_n_ids:
            if not ((state >> gid) & 1):
                count += 1
        return count

    def _task_network_lb(self, task_network):
        """
        Lower bound from task network structure.
        
        Each turn task will produce at least 1 flip action (the center flip).
        Each play task that decomposes via do_play produces at least 1 turn.
        Each flip operator in the network costs 1.
        Each method_precondition operator costs 1.
        Each propagate task that will do a flip costs at least 1.
        
        We count:
        - Direct flip operators: cost 1 each
        - Turn tasks: at least 1 flip each (the center flip)
        - Play tasks: if there are mismatches, at least 1 turn needed
        - Method precondition operators: cost 1 each
        - Propagate tasks: we can't easily estimate without state, so 0
        """
        flip_count = 0
        turn_count = 0
        method_prec_count = 0
        
        for task in task_network:
            gid = task.global_id
            if gid in self.flip_op_ids:
                flip_count += 1
            elif gid in self.turn_task_ids:
                turn_count += 1
            elif gid in self.method_precondition_ids:
                method_prec_count += 1
            # play tasks don't directly add cost - they decompose into turns
        
        return flip_count + turn_count + method_prec_count

    def _propagation_aware_lb(self, state, mismatches):
        """
        Each turn does 1 center flip + up to (grid_size-1)*4 propagation flips.
        But propagation only flips consecutive same-status neighbors.
        
        A single turn can fix at most 1 + 4*(grid_size-1) mismatched cells
        (in the best case where all propagation targets are mismatched).
        But the center flip itself might CREATE a mismatch on the center cell.
        
        Actually, more carefully: a turn flips the center cell and then
        propagates. The center flip changes one cell. Each propagation flip
        changes one additional cell. So a single turn can change at most
        1 + 4*(grid_size - 1) cells.
        
        But we need at least ceil(mismatches / max_changes_per_turn) turns,
        and each turn costs at least 1 action (the center flip) plus
        method precondition checks.
        
        Actually, every turn also generates method_precondition operators.
        Looking at do_turn: flip + propagate_up + propagate_down + propagate_left + propagate_right
        The propagate methods have preconditions checked by __method_precondition operators.
        Each propagate direction has at least 1 method precondition check (to decide end vs continue).
        
        So each turn costs at least: 1 (flip) + 4 (one method_precondition per direction) = 5 actions minimum.
        But we should be conservative for admissibility.
        
        Actually, each propagation task (up/down/left/right) terminates with either:
        - end_field (no cost, just a method precondition check = 1 action)
        - end_different (no cost, just a method precondition check = 1 action)
        - do (1 flip + 1 recursive propagation + 1 method precondition check)
        
        So minimum cost of each propagation direction = 1 (method precondition for termination).
        Minimum cost of a turn = 1 (center flip) + 4 * 1 (propagation termination checks) = 5.
        
        Wait, but the do_turn method itself might have a method precondition check too.
        Let me be conservative and just use 1 per turn minimum for the flip.
        
        For admissibility, stick with: each turn costs >= 1 flip action.
        Number of turns >= ceil(mismatches / max_flips_per_turn).
        max_flips_per_turn = 1 + 4 * (grid_size - 1) = 4*grid_size - 3.
        """
        if mismatches == 0:
            return 0
        if self.grid_size <= 1:
            return mismatches  # only 1 cell, each turn flips exactly 1 cell
        
        # Maximum flips per turn (center + 4 directions each up to grid_size-1)
        max_flips_per_turn = 1 + 4 * (self.grid_size - 1)
        
        # Minimum turns needed
        min_turns = (mismatches + max_flips_per_turn - 1) // max_flips_per_turn
        
        # Each turn costs at least 1 (the center flip action)
        return min_turns

    def _compute(self, state, task_network):
        """
        Compute heuristic as max of multiple admissible lower bounds.
        """
        # Term 1: Goal mismatch count (admissible - each flip fixes one cell)
        mismatches = self._count_mismatches(state)
        
        # If no mismatches and we need to check if task network can terminate
        if mismatches == 0:
            # Check if there are any non-play tasks remaining that must execute
            # Play can terminate with end_game (empty subtasks)
            has_non_play = False
            for task in task_network:
                gid = task.global_id
                if gid not in self.play_task_ids:
                    has_non_play = True
                    break
            if not has_non_play:
                return 0
        
        # Term 2: Task network lower bound
        tn_lb = self._task_network_lb(task_network)
        
        # Term 3: Propagation-aware lower bound on turns needed
        prop_lb = self._propagation_aware_lb(state, mismatches)
        
        # Return max of all admissible terms
        h = max(mismatches, tn_lb, prop_lb)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic12(grid={self.grid_size})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic12 with grid size {self.grid_size}"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size: {self.grid_size}\n"
            f"\tGoal Cells: {len(self.goal_status)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
