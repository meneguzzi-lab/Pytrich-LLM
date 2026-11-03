# Bottleneck: Single-arm serialization — only one block held at a time, clearing
# requires recursive unstacking (2 actions per block above target).
#
# Admissible lower bounds:
# T1: For each do_put_on(x,y) task where on(x,y) is not true, we need at least
#     2 actions (grab x + stack x on y). If x is not clear, add 2 per block above x.
#     If y is not clear, add 2 per block above y. This is admissible because these
#     are unavoidable physical actions.
# T2: For each do_clear(x) task where x is not clear, we need at least 2 actions
#     per block above x (unstack + put-down). Admissible as minimum clearing cost.
# T3: For each do_on_table(x) task where x is not on table, we need at least 2
#     actions (unstack + put-down). Admissible as the minimum move-to-table cost.
#
# We return max over a sum-based estimate to get strong guidance.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic15(Heuristic):
    """Domain-dependent heuristic for Blocksworld GTOHP.
    
    Uses state-aware per-task cost estimation with stack-height analysis
    to compute realistic lower bounds on remaining work.
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_15"):
        super().__init__(name)
        self.on_fact_id = {}       # (x, y) -> global_id for on[x,y]
        self.clear_fact_id = {}    # x -> global_id for clear[x]
        self.ontable_fact_id = {}  # x -> global_id for ontable[x]
        self.handempty_fact_id = None
        self.holding_fact_id = {}  # x -> global_id for holding[x]
        
        # Task type name patterns
        self.task_type_do_put_on = None
        self.task_type_do_clear = None
        self.task_type_do_on_table = None
        self.task_type_do_move = None
        
        # Maps from block -> list of blocks currently on top (for stack height)
        self.block_names = set()
        
        # Reverse on map: for each block y, which block x is on(x, y)?
        # Built at query time from state bits
        self.all_on_pairs = []  # list of ((x, y), global_id)
        
        # Abstract task name -> type mapping
        self.abstract_task_types = {}  # global_id -> (type_str, args)
        
        # Operator name -> type mapping  
        self.operator_types = {}  # global_id -> type_str

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []

    def _parse_task_name(self, name: str):
        """Parse a task/operator name to get the task type and arguments."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def _get_stack_height_above(self, block, state):
        """Count how many blocks are stacked above 'block' in the current state.
        Uses iterative chain-following to avoid recursion issues."""
        if (state >> self.clear_fact_id[block]) & 1:
            return 0
        
        height = 0
        current = block
        max_iter = len(self.block_names) + 1  # safety bound
        
        while max_iter > 0:
            max_iter -= 1
            # Find what's on top of current
            found = False
            if current in self.on_top_of:
                for candidate in self.on_top_of[current]:
                    pair_id = self.on_fact_id.get((candidate, current))
                    if pair_id is not None and (state >> pair_id) & 1:
                        height += 1
                        current = candidate
                        found = True
                        break
            if not found:
                break
        
        return height

    def _estimate_do_put_on_cost(self, x, y, state):
        """Estimate minimum actions needed for do_put_on(x, y)."""
        # Check if on(x, y) is already true
        on_xy_id = self.on_fact_id.get((x, y))
        if on_xy_id is not None and (state >> on_xy_id) & 1:
            return 0  # Already satisfied, uses m0 (nop)
        
        cost = 0
        
        # Need to clear x (if not already clear)
        if x in self.clear_fact_id:
            if not ((state >> self.clear_fact_id[x]) & 1):
                # Each block above x needs unstack + put-down = 2 actions
                h_above = self._get_stack_height_above(x, state)
                cost += 2 * max(h_above, 1)  # at least 1 block if not clear
        
        # Need to clear y (if not already clear)
        if y in self.clear_fact_id:
            if not ((state >> self.clear_fact_id[y]) & 1):
                h_above = self._get_stack_height_above(y, state)
                cost += 2 * max(h_above, 1)
        
        # Need to ensure y is on table (do_on_table) - may need unstack + put-down
        if y in self.ontable_fact_id:
            if not ((state >> self.ontable_fact_id[y]) & 1):
                cost += 2  # unstack y from whatever + put-down y
        
        # The actual move: pick-up/unstack x + stack x on y = 2 actions
        cost += 2
        
        return cost

    def _estimate_do_clear_cost(self, x, state):
        """Estimate minimum actions needed for do_clear(x)."""
        if x in self.clear_fact_id:
            if (state >> self.clear_fact_id[x]) & 1:
                return 0  # Already clear
            h_above = self._get_stack_height_above(x, state)
            return 2 * max(h_above, 1)
        return 0

    def _estimate_do_on_table_cost(self, x, state):
        """Estimate minimum actions needed for do_on_table(x)."""
        if x in self.ontable_fact_id:
            if (state >> self.ontable_fact_id[x]) & 1:
                # Check if clear too (for m3 nop method)
                if x in self.clear_fact_id and (state >> self.clear_fact_id[x]) & 1:
                    return 0
                # ontable but not clear - still need clearing first for the method
                # Actually m3 just needs clear(x), if ontable is true
                # m2 needs clear(x), handempty, not ontable
                # If ontable is true, m3 applies if clear
                if x in self.clear_fact_id and not ((state >> self.clear_fact_id[x]) & 1):
                    # Need to clear x first, but do_on_table just needs clear(x)
                    # Wait, do_on_table's precondition for m3 is just clear(x)
                    # But the task is do_on_table, and if ontable is already true
                    # then m3 applies once x is clear
                    return 0  # The task itself is nop if ontable, clearing is separate
                return 0
            else:
                # Not on table: need unstack + put-down = 2
                # But also need to be clear first
                cost = 2  # unstack + put-down
                if x in self.clear_fact_id and not ((state >> self.clear_fact_id[x]) & 1):
                    # Not clear either - but clearing is handled by the method's own logic
                    # m2 requires clear(x), so we just need those 2 actions
                    pass
                return cost
        return 0

    def _estimate_do_move_cost(self, x, y, state):
        """Estimate minimum actions for do_move(x, y)."""
        # do_move requires clear(x), clear(y), handempty
        # Then pick-up/unstack + stack = 2 actions
        return 2

    def _compute(self, state, task_network):
        """Compute heuristic value for a search node."""
        if not task_network:
            return 0
        
        total_cost = 0
        
        for task in task_network:
            task_type, args = self._parse_task_name(task.name)
            
            if task_type == "do_put_on" and len(args) >= 2:
                total_cost += self._estimate_do_put_on_cost(args[0], args[1], state)
            elif task_type == "do_clear" and len(args) >= 1:
                total_cost += self._estimate_do_clear_cost(args[0], state)
            elif task_type == "do_on_table" and len(args) >= 1:
                total_cost += self._estimate_do_on_table_cost(args[0], state)
            elif task_type == "do_move" and len(args) >= 2:
                total_cost += self._estimate_do_move_cost(args[0], args[1], state)
            elif task_type == "pick-up" or task_type == "unstack":
                total_cost += 1
            elif task_type == "put-down" or task_type == "stack":
                total_cost += 1
            elif task_type == "nop":
                pass
            else:
                # Unknown task type - conservative estimate
                total_cost += 1
        
        # Second term: goal-based distance
        goal_cost = 0
        for gname in self._goal_fact_names:
            pred, gargs = self._parse_fact_name(gname)
            if pred == "on" and len(gargs) >= 2:
                fid = self.on_fact_id.get((gargs[0], gargs[1]))
                if fid is not None and not ((state >> fid) & 1):
                    # This goal on-fact is not yet satisfied
                    # Minimum cost: 2 actions (grab + place)
                    # Plus clearing costs
                    x, y = gargs[0], gargs[1]
                    c = 2
                    if x in self.clear_fact_id and not ((state >> self.clear_fact_id[x]) & 1):
                        c += 2 * max(self._get_stack_height_above(x, state), 1)
                    if y in self.clear_fact_id and not ((state >> self.clear_fact_id[y]) & 1):
                        c += 2 * max(self._get_stack_height_above(y, state), 1)
                    goal_cost += c
        
        return max(total_cost, goal_cost)

    def initialize(self, model: Model, initial_node: HTNNode):
        """Parse all facts, build lookup maps, precompute goal structures."""
        self.model = model
        
        # Build fact ID maps
        self.on_top_of = {}  # block -> list of blocks that could be on top of it
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) >= 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.all_on_pairs.append(((args[0], args[1]), f.global_id))
                self.block_names.add(args[0])
                self.block_names.add(args[1])
                # Build reverse map: y -> list of x where on(x,y) exists
                if args[1] not in self.on_top_of:
                    self.on_top_of[args[1]] = []
                self.on_top_of[args[1]].append(args[0])
            elif pred == "clear" and len(args) >= 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.block_names.add(args[0])
            elif pred == "ontable" and len(args) >= 1:
                self.ontable_fact_id[args[0]] = f.global_id
                self.block_names.add(args[0])
            elif pred == "handempty":
                self.handempty_fact_id = f.global_id
            elif pred == "holding" and len(args) >= 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.block_names.add(args[0])
        
        # Cache goal fact names
        self._goal_fact_names = model.state_explicit_repr(model.goals)
        
        # Compute initial heuristic
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for a new search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
