# Bottleneck: Single-hand constraint (handempty) serializes all block moves.
# Clearing a block with height h above it costs 2*h actions (unstack+putdown each).
# Moving a block costs 2 actions (pickup/unstack + stack/putdown).
#
# Admissible lower bounds:
# T1: For each remaining compound task, estimate min actions from current state
#     using stack-height analysis. This is admissible because we compute the
#     minimum possible actions for each task independently.
# T2: Count of unsatisfied goal on-facts * 2 (each needs at least unstack+stack
#     or pickup+stack). Admissible because each unsatisfied on-goal needs >= 2 actions.
# T3: Task network size based estimate - remaining primitive actions in network.
#
# We return max(T1_sum_over_current_tasks, T2) for strong guidance.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.HDDL.hddl_types import Operator as OperatorClass, AbstractTask as AbstractTaskClass


class BlocksworldGTOHPAnthropicOpus46Heuristic3(Heuristic):
    """Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Uses stack-height analysis and goal-distance estimation to provide
    strong admissible lower bounds on remaining plan cost.
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_3"):
        super().__init__(name)
        self.on_fact_id = {}       # (x, y) -> global_id for on(x,y)
        self.clear_fact_id = {}    # x -> global_id for clear(x)
        self.ontable_fact_id = {}  # x -> global_id for ontable(x)
        self.handempty_fact_id = None
        self.holding_fact_id = {}  # x -> global_id for holding(x)
        self.all_blocks = set()
        
        # Goal tracking
        self.goal_on_pairs = []    # list of (x, y) pairs that are goals
        self.goal_on_fact_ids = [] # corresponding global_ids
        
        # Task name prefixes for identification
        self.task_type_cache = {}  # task global_id -> task_type string
        self.task_args_cache = {}  # task global_id -> list of block args
        
        # For stack height computation: on_fact for each (x,y)
        # We build a reverse map: block -> fact_id of on(?,block) to find what's on top
        self.on_top_of = {}  # y -> list of (x, fact_id) meaning on(x,y) might be true
        
        # Operator detection
        self.nop_op_ids = set()
        self.pickup_op_ids = set()
        self.putdown_op_ids = set()
        self.stack_op_ids = set()
        self.unstack_op_ids = set()

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
        return name, []

    def _parse_task_name(self, name: str):
        """Parse operator/task name to get type and args.
        Format examples: pick-up[b1], stack[b1,b2], do_put_on[b1,b2], nop[]
        """
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        # Try without brackets
        paren_idx = name.find("(")
        if paren_idx > 0:
            task_type = name[:paren_idx]
            args_str = name[paren_idx+1:].rstrip(")")
            args = [a.strip() for a in args_str.split(",")] if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
                if args[1] not in self.on_top_of:
                    self.on_top_of[args[1]] = []
                self.on_top_of[args[1]].append((args[0], f.global_id))
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "handempty" and len(args) == 0:
                self.handempty_fact_id = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
        
        # Build goal on-pairs
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                fid = self.on_fact_id.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_pairs.append((args[0], args[1]))
                    self.goal_on_fact_ids.append(fid)
        
        # Cache task types and args for abstract tasks
        for at in model.abstract_tasks:
            ttype, targs = self._parse_task_name(at.name)
            self.task_type_cache[at.global_id] = ttype
            self.task_args_cache[at.global_id] = targs
        
        # Cache operator types
        for op in model.operators:
            ttype, targs = self._parse_task_name(op.name)
            self.task_type_cache[op.global_id] = ttype
            self.task_args_cache[op.global_id] = targs
            if ttype == "nop":
                self.nop_op_ids.add(op.global_id)
            elif ttype == "pick-up":
                self.pickup_op_ids.add(op.global_id)
            elif ttype == "put-down":
                self.putdown_op_ids.add(op.global_id)
            elif ttype == "stack":
                self.stack_op_ids.add(op.global_id)
            elif ttype == "unstack":
                self.unstack_op_ids.add(op.global_id)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _stack_height_above(self, block, state):
        """Count how many blocks are stacked above 'block' in current state.
        Returns the height (number of blocks above).
        Uses iterative approach following the stack upward.
        """
        # If block is clear, height is 0
        if block in self.clear_fact_id:
            clear_id = self.clear_fact_id[block]
            if (state >> clear_id) & 1:
                return 0
        
        # Find what's on top of this block
        height = 0
        current = block
        visited = set()
        while current in self.on_top_of and current not in visited:
            visited.add(current)
            found_top = False
            for (x, fid) in self.on_top_of[current]:
                if (state >> fid) & 1:  # on(x, current) is true
                    height += 1
                    current = x
                    found_top = True
                    break
            if not found_top:
                break
        return height

    def _block_on_table(self, block, state):
        """Check if block is on table."""
        if block in self.ontable_fact_id:
            return (state >> self.ontable_fact_id[block]) & 1
        return 0

    def _block_is_clear(self, block, state):
        """Check if block is clear."""
        if block in self.clear_fact_id:
            return (state >> self.clear_fact_id[block]) & 1
        return 0

    def _on_holds(self, x, y, state):
        """Check if on(x,y) is true."""
        fid = self.on_fact_id.get((x, y))
        if fid is not None:
            return (state >> fid) & 1
        return 0

    def _estimate_do_clear_cost(self, block, state):
        """Estimate minimum cost to clear a block.
        Each block above costs 2 actions (unstack + put-down).
        Admissible: each block above MUST be unstacked and put somewhere.
        """
        h = self._stack_height_above(block, state)
        return 2 * h

    def _estimate_do_on_table_cost(self, block, state):
        """Estimate cost to put block on table.
        If already on table and clear: 0 (nop)
        If clear but not on table: 2 (unstack from current + put-down)
        If not clear: need to clear first + then 2 if not on table
        """
        is_clear = self._block_is_clear(block, state)
        is_ontable = self._block_on_table(block, state)
        
        if is_clear and is_ontable:
            return 0
        
        cost = 0
        if not is_clear:
            cost += self._estimate_do_clear_cost(block, state)
        if not is_ontable:
            cost += 2  # unstack + put-down
        return cost

    def _estimate_do_move_cost(self, x, y, state):
        """Estimate cost to move x onto y.
        Preconditions: clear(x), clear(y), handempty
        If on table: pick-up + stack = 2
        If not on table: unstack + stack = 2
        So always 2 actions for the move itself.
        """
        return 2

    def _estimate_do_put_on_cost(self, x, y, state):
        """Estimate cost for do_put_on(x, y).
        If on(x,y) already holds: 0 (nop)
        Otherwise: clear(x) + clear(y) + on_table(y) + move(x,y)
        """
        if self._on_holds(x, y, state):
            return 0
        
        cost = 0
        # Cost to clear x
        cost += self._estimate_do_clear_cost(x, state)
        # Cost to clear y
        cost += self._estimate_do_clear_cost(y, state)
        # Cost to put y on table (if not already)
        # But y might already be on table after clearing
        # Conservatively: if y is not on table and not clear, clearing may leave it still not on table
        # After clearing y, y is clear. If not ontable, need 2 more actions.
        # But we already counted clearing y. After clearing, is y on table?
        # y is on table only if ontable(y) is already true. Clearing doesn't change that.
        if not self._block_on_table(y, state):
            cost += 2  # unstack y from whatever + put-down
        # Cost to move x onto y
        cost += 2  # pick-up/unstack + stack
        
        return cost

    def _compute(self, state, task_network):
        if not task_network:
            return 0
        
        # Term 1: Sum of estimated costs for remaining tasks in the network
        task_cost = 0
        primitive_count = 0
        
        for task in task_network:
            gid = task.global_id
            ttype = self.task_type_cache.get(gid, "")
            targs = self.task_args_cache.get(gid, [])
            
            if isinstance(task, OperatorClass):
                # Primitive task - costs 1 (unless nop)
                if gid in self.nop_op_ids:
                    pass  # 0 cost
                else:
                    primitive_count += 1
            elif isinstance(task, AbstractTaskClass):
                if ttype == "do_put_on" and len(targs) == 2:
                    task_cost += self._estimate_do_put_on_cost(targs[0], targs[1], state)
                elif ttype == "do_clear" and len(targs) == 1:
                    task_cost += self._estimate_do_clear_cost(targs[0], state)
                elif ttype == "do_on_table" and len(targs) == 1:
                    task_cost += self._estimate_do_on_table_cost(targs[0], state)
                elif ttype == "do_move" and len(targs) == 2:
                    task_cost += self._estimate_do_move_cost(targs[0], targs[1], state)
                else:
                    # Unknown task type, assume at least 1
                    task_cost += 1
        
        term1 = task_cost + primitive_count
        
        # Term 2: Goal distance - count unsatisfied goal on-facts
        # Each unsatisfied on(x,y) goal needs at minimum 2 actions
        unsatisfied_goals = 0
        for i, (x, y) in enumerate(self.goal_on_pairs):
            fid = self.goal_on_fact_ids[i]
            if not ((state >> fid) & 1):
                unsatisfied_goals += 1
        
        term2 = 2 * unsatisfied_goals
        
        # Term 3: Check for goals that are currently true but will be disrupted
        # If on(x,y) is a goal and is true, but x or y needs to be moved for another
        # task, that goal will be undone and redone (cost >= 4 extra actions)
        # This is the "false trivial" detection from P5
        disruption_cost = 0
        # Build set of blocks that need to be cleared by remaining tasks
        blocks_to_clear = set()
        blocks_to_move = set()  # blocks that will be unstacked/moved
        
        for task in task_network:
            gid = task.global_id
            ttype = self.task_type_cache.get(gid, "")
            targs = self.task_args_cache.get(gid, [])
            
            if isinstance(task, AbstractTaskClass):
                if ttype == "do_clear" and len(targs) >= 1:
                    blocks_to_clear.add(targs[0])
                elif ttype == "do_on_table" and len(targs) >= 1:
                    # This block needs to end up on table, so anything on it gets moved
                    blocks_to_clear.add(targs[0])
                elif ttype == "do_put_on" and len(targs) >= 2:
                    # x will be moved onto y; both need clearing
                    blocks_to_clear.add(targs[0])
                    blocks_to_clear.add(targs[1])
                    blocks_to_move.add(targs[0])
                elif ttype == "do_move" and len(targs) >= 2:
                    blocks_to_move.add(targs[0])
        
        # For each currently-satisfied goal on(x,y), check if it will be disrupted
        for i, (x, y) in enumerate(self.goal_on_pairs):
            fid = self.goal_on_fact_ids[i]
            if (state >> fid) & 1:  # Goal is currently true
                # Will it be disrupted? If x needs to be cleared (something put on x)
                # or x needs to be moved, or y needs to be cleared and x is on y
                if x in blocks_to_move:
                    # x will be moved away from y, then back - but wait,
                    # do_put_on checks if on(x,y) already holds first
                    # So this only matters if x moves BEFORE the goal task
                    pass
                # If something above x needs clearing, x itself might get unstacked
                # Check: is x clear? If not, clearing x means unstacking what's above
                # That doesn't move x. But if x needs to be moved (in blocks_to_move)
                # and on(x,y) is currently true, the do_put_on will see it's true (nop)
                # unless x gets moved away first by another task.
                # For a simple admissible bound, we check if x is explicitly moved
                # by looking at tasks that would unstack x from y
                pass
        
        # Return max of terms for admissible estimate
        h = max(term1, term2)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
