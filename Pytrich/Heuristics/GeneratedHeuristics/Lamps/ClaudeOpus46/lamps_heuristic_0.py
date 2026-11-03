"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
  The Lamps domain is a grid-based light-flipping puzzle. Each "turn" picks a cell (x,y),
  flips it, then propagates flips in all 4 directions along contiguous runs of same-status
  cells. The bottleneck is that each turn requires at minimum 1 flip action (the center cell)
  plus additional flips for propagation. The goal requires certain cells to change status.

  The key insight: each cell whose status differs between current state and goal state must
  be flipped at least once. Each flip is one primitive action. So the number of cells that
  differ from goal is a lower bound on the number of flip actions needed.

  However, a single "turn" can flip multiple cells (the center + propagation). A turn with
  a center flip and k propagation flips costs (1 + k) actions but changes (1 + k) cells.
  So each differing cell costs at least 1 action even in the best case. But turns also
  have overhead from method precondition operators.

  Additionally, we need to account for the hierarchical structure: each turn is decomposed
  via do_turn which includes the flip + 4 propagations. The grounder produces
  __method_precondition_* operators that cost 1 each.

Admissible Lower Bounds:
  1. Goal-diff count: Count cells where current status != goal status. Each such cell
     needs at least one flip action. This is admissible because no flip action can change
     more than one cell.
  2. Turn count estimate: Each turn involves at least 1 flip + method precondition operators.
     We can estimate minimum number of turns needed. Each turn changes a contiguous cross-shaped
     region. The minimum turns needed is a lower bound.
  3. Task network lower bound: Count remaining tasks that must produce at least one primitive
     action each.

We use max(goal_diff, task_network_lb) as the heuristic.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic0(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.
    
    Combines:
    1. Goal-diff: number of cells differing from goal (admissible: each needs >= 1 flip)
    2. Task-network-aware estimate: counts minimum primitive actions from remaining tasks
    """

    def __init__(self, lamps_anthropicopus46_heuristic_0=None, name="LampsAnthropicOpus46Heuristic0"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Maps for quick goal checking
        self.goal_on_facts = {}  # fact.global_id -> True for facts that appear in goal
        self.goal_state_bits = 0  # The goal as a bitwise integer
        
        # For each cell (x,y), store the global_ids of on[x,y,Y] and on[x,y,N]
        self.cell_fact_Y = {}  # (x,y) -> global_id of on[x,y,Y]
        self.cell_fact_N = {}  # (x,y) -> global_id of on[x,y,N]
        
        # Goal status for each cell
        self.goal_status = {}  # (x,y) -> 'Y' or 'N'
        
        # All cells
        self.all_cells = set()
        
        # Operator classification
        self.flip_operator_ids = set()
        self.method_precondition_ids = set()
        
        # Abstract task classification
        self.play_task_ids = set()
        self.turn_task_ids = set()
        self.propagate_task_ids = set()
        
        # For task network estimation
        self.turn_min_cost = 1  # minimum cost of a turn (at least 1 flip)
        
        # Number of method_precondition operators per turn decomposition
        self.method_prec_per_turn = 0

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
        
        # Build fact maps
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.all_cells.add((x, y))
                if s == "Y":
                    self.cell_fact_Y[(x, y)] = fact.global_id
                elif s == "N":
                    self.cell_fact_N[(x, y)] = fact.global_id
        
        # Extract goal status for each cell
        self.goal_state_bits = self.model.goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for goal_name in goal_fact_names:
            predicate, args = self._parse_fact_name(goal_name)
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.goal_status[(x, y)] = s
                if s == "Y" and (x, y) in self.cell_fact_Y:
                    self.goal_on_facts[self.cell_fact_Y[(x, y)]] = True
                elif s == "N" and (x, y) in self.cell_fact_N:
                    self.goal_on_facts[self.cell_fact_N[(x, y)]] = True
        
        # Classify operators
        for op in self.model.operators:
            op_name = op.name.lower() if op.name else ""
            if "flip" in op_name and "method" not in op_name:
                self.flip_operator_ids.add(op.global_id)
            elif "__method_precondition" in op_name or "method_precondition" in op_name:
                self.method_precondition_ids.add(op.global_id)
        
        # Classify abstract tasks
        for at in self.model.abstract_tasks:
            at_name = at.name.lower() if at.name else ""
            if "play" in at_name:
                self.play_task_ids.add(at.global_id)
            elif "turn" in at_name and "propagate" not in at_name:
                self.turn_task_ids.add(at.global_id)
            elif "propagate" in at_name:
                self.propagate_task_ids.add(at.global_id)
        
        # Estimate method precondition cost per turn
        # Look at decompositions of turn tasks to count method_precondition operators
        for at in self.model.abstract_tasks:
            if at.global_id in self.turn_task_ids:
                for decomp in at.decompositions:
                    mp_count = 0
                    for subtask in decomp.task_network:
                        if hasattr(subtask, 'global_id') and subtask.global_id in self.method_precondition_ids:
                            mp_count += 1
                    self.method_prec_per_turn = max(self.method_prec_per_turn, mp_count)
                break
        
        # Build list of goal_fact_global_ids for fast iteration
        self.goal_cells_list = []
        for (x, y), s in self.goal_status.items():
            if s == "Y" and (x, y) in self.cell_fact_Y:
                self.goal_cells_list.append(((x, y), self.cell_fact_Y[(x, y)], s))
            elif s == "N" and (x, y) in self.cell_fact_N:
                self.goal_cells_list.append(((x, y), self.cell_fact_N[(x, y)], s))

    def _count_goal_diff(self, state_bitwise):
        """
        Count number of cells whose current status differs from goal.
        Each such cell requires at least one flip action.
        This is admissible: each flip action changes exactly one cell.
        """
        diff_count = 0
        for (x, y), fact_gid, goal_s in self.goal_cells_list:
            # Check if the goal fact is true in current state
            is_goal_true = (state_bitwise >> fact_gid) & 1
            if not is_goal_true:
                diff_count += 1
        return diff_count

    def _task_network_estimate(self, task_network):
        """
        Estimate minimum remaining primitive actions from task network.
        
        - Each 'turn' task requires at least 1 flip action + method precondition costs
        - Each 'play' task with non-empty subtasks requires at least 1 turn
        - Each 'propagate' task may require 0 or more flips
        - Each primitive operator costs 1
        
        This is admissible as it estimates minimum cost for each task.
        """
        if not task_network:
            return 0
        
        estimate = 0
        for task in task_network:
            gid = task.global_id
            if gid in self.turn_task_ids:
                # A turn has at least 1 flip + method precondition operators
                # do_turn decomposition: flip + propagate_up + propagate_down + propagate_left + propagate_right
                # The flip costs 1. Each propagate may have a method_precondition check.
                # Minimum: 1 flip action + method preconditions for the turn method itself
                estimate += 1 + self.method_prec_per_turn
            elif gid in self.play_task_ids:
                # play can be end_game (cost 0) or do_play (turn + play)
                # We can't know which, so minimum is 0
                # But we can use goal diff as a better bound for play
                pass
            elif gid in self.propagate_task_ids:
                # Propagate tasks may cost 0 (end) or 1+ (do propagation)
                # Minimum is 0 actions
                pass
            elif hasattr(task, 'applicable'):
                # It's a primitive operator
                estimate += 1
            else:
                # Unknown abstract task, assume at least 0
                pass
        
        return estimate

    def _compute(self, state_bitwise, task_network):
        """
        Compute heuristic value as max of multiple admissible bounds.
        """
        # If task network is empty and goals are met, we're done
        if not task_network:
            # Check if goals are satisfied
            if (state_bitwise & self.model.goals) == self.model.goals:
                return 0
            else:
                # Goals not met but no tasks - this shouldn't lead to a solution
                return float('inf')
        
        # Term 1: Goal diff count (admissible - each diff cell needs >= 1 flip)
        goal_diff = self._count_goal_diff(state_bitwise)
        
        # Term 2: Task network based estimate
        tn_estimate = self._task_network_estimate(task_network)
        
        # Term 3: If there are differing cells, we need at least ceil(goal_diff / max_flips_per_turn) turns
        # Each turn can flip at most (1 + 4*grid_size) cells in theory, but typically fewer.
        # A simpler bound: each turn needs at least 1 + method_prec_per_turn actions
        # So if we need D diffs, and each turn handles at most T diffs with cost >= (1 + mp) per turn,
        # total cost >= D (since each flip costs 1 and each diff needs at least 1 flip)
        # This is already captured by goal_diff.
        
        # Combine: the max of independent admissible bounds
        h = max(goal_diff, tn_estimate)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic0(cells={len(self.all_cells)})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic0 with {len(self.all_cells)} cells"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Cells: {len(self.all_cells)}\n"
            f"\tGoal Cells Tracked: {len(self.goal_cells_list)}\n"
            f"\tMethod Prec Per Turn: {self.method_prec_per_turn}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
