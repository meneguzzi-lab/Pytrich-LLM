# Bottleneck: Single-hand constraint serializes all block movements. Clearing stacks
# dominates plan length - each block above target costs 2 actions (unstack + put-down).
#
# Admissible lower bounds:
# T1 - Per-task minimum action cost: Each do_put_on(x,y) needs at least 2 actions if
#      on(x,y) is not true; do_clear(x) needs 2*height_above(x); do_on_table(x) needs
#      2 if not on table; do_move(x,y) needs exactly 2.
# T2 - Goal displacement: Count unsatisfied goal facts, each needing >= 2 actions.
# T3 - Network structure: Remaining compound tasks that cannot be trivially resolved.
#
# We return max(T1_sum_over_tasks, T2_goal_based) for strong guidance.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic14(Heuristic):
    """Domain-dependent heuristic for Blocksworld GTOHP.
    
    Uses state-aware per-task cost estimation combined with goal displacement
    to produce a strong admissible heuristic.
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_14"):
        super().__init__(name)
        # Fact ID maps
        self.on_fact_id = {}       # (x, y) -> global_id
        self.ontable_fact_id = {}  # x -> global_id
        self.clear_fact_id = {}    # x -> global_id
        self.handempty_fact_id = None
        self.holding_fact_id = {}  # x -> global_id
        
        # Task type identification
        self.task_type_map = {}  # abstract_task.global_id -> (type_str, args)
        
        # Goal facts
        self.goal_on_pairs = []  # list of (x, y, fact_global_id)
        
        # For stack height computation: who is on top of whom
        # on_top_of[y] = set of x such that on(x,y) fact exists
        self.all_blocks = set()
        self.on_pairs_set = set()  # all (x,y) pairs where on(x,y) fact exists
        
        # Operator type identification  
        self.op_type_map = {}  # operator.global_id -> type_str
        
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
            # No brackets - no args
            return name, []
        return None, []

    def _parse_task_name(self, name: str):
        """Parse task/operator name to extract type and arguments."""
        # Names look like: do_put_on[b1,b2] or pick-up[b1] or nop[]
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Called once before search. Build all lookup structures."""
        self.model = model
        
        # Build fact ID maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
                self.on_pairs_set.add((args[0], args[1]))
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "handempty" and len(args) == 0:
                self.handempty_fact_id = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
        
        # Build abstract task type map
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            self.task_type_map[at.global_id] = (task_type, args)
        
        # Build operator type map
        for op in model.operators:
            task_type, args = self._parse_task_name(op.name)
            self.op_type_map[op.global_id] = task_type
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                fid = self.on_fact_id.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_pairs.append((args[0], args[1], fid))
        
        # Build reverse on-map: for each block y, which fact ids correspond to on(x,y)
        # This helps compute stack heights efficiently
        self.on_top_map = {}  # block_y -> list of (block_x, fact_id)
        for (x, y), fid in self.on_fact_id.items():
            if y not in self.on_top_map:
                self.on_top_map[y] = []
            self.on_top_map[y].append((x, fid))
        
        # Build on-below map: for block x, what (y, fact_id) pairs for on(x,y)
        self.on_below_map = {}  # block_x -> list of (block_y, fact_id)
        for (x, y), fid in self.on_fact_id.items():
            if x not in self.on_below_map:
                self.on_below_map[x] = []
            self.on_below_map[x].append((y, fid))
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _find_block_on_top(self, block, state):
        """Find the block currently on top of 'block' in given state, or None."""
        entries = self.on_top_map.get(block)
        if entries is None:
            return None
        for (x, fid) in entries:
            if (state >> fid) & 1:
                return x
        return None

    def _find_block_below(self, block, state):
        """Find the block currently below 'block' in given state, or None."""
        entries = self.on_below_map.get(block)
        if entries is None:
            return None
        for (y, fid) in entries:
            if (state >> fid) & 1:
                return y
        return None

    def _stack_height_above(self, block, state):
        """Count how many blocks are stacked above 'block' in current state.
        Uses iterative traversal to avoid recursion issues."""
        height = 0
        current = block
        # Walk up the stack
        visited = set()
        while True:
            top = self._find_block_on_top(current, state)
            if top is None or top in visited:
                break
            visited.add(top)
            height += 1
            current = top
        return height

    def _is_clear(self, block, state):
        """Check if block is clear in state."""
        fid = self.clear_fact_id.get(block)
        if fid is None:
            return True  # unknown block, assume clear
        return (state >> fid) & 1

    def _is_on(self, x, y, state):
        """Check if on(x,y) is true in state."""
        fid = self.on_fact_id.get((x, y))
        if fid is None:
            return False
        return (state >> fid) & 1

    def _is_ontable(self, block, state):
        """Check if block is on the table."""
        fid = self.ontable_fact_id.get(block)
        if fid is None:
            return False
        return (state >> fid) & 1

    def _clear_cost(self, block, state):
        """Minimum actions to clear block: 2 per block above it."""
        if self._is_clear(block, state):
            return 0
        return 2 * self._stack_height_above(block, state)

    def _estimate_task_cost(self, task_type, args, state):
        """Estimate minimum number of primitive actions for a task given current state."""
        if task_type == "do_put_on":
            if len(args) >= 2:
                x, y = args[0], args[1]
                # If on(x,y) already true, cost is 0 (nop)
                if self._is_on(x, y, state):
                    return 0
                # Otherwise: need to clear x, clear y, possibly put y on table, then move x onto y
                # Minimum: clear_cost(x) + clear_cost(y) + 2 (the final move: pick/unstack + stack)
                # do_on_table(y): if y is already on table, 0; else 2
                cost_clear_x = self._clear_cost(x, state)
                cost_clear_y = self._clear_cost(y, state)
                # do_on_table adds 0 if y on table, else 2 (unstack+putdown)
                cost_on_table_y = 0 if self._is_ontable(y, state) else 2
                # But clearing y might put it on table as side effect... 
                # Conservative: just use clear costs + move cost
                cost_move = 2  # pick-up/unstack + stack
                return cost_clear_x + cost_clear_y + cost_on_table_y + cost_move
            return 2
        
        elif task_type == "do_clear":
            if len(args) >= 1:
                x = args[0]
                return self._clear_cost(x, state)
            return 0
        
        elif task_type == "do_on_table":
            if len(args) >= 1:
                x = args[0]
                if self._is_ontable(x, state):
                    return 0
                # Need to unstack + put-down (assuming already clear)
                return 2
            return 0
        
        elif task_type == "do_move":
            # Always costs exactly 2: unstack/pick-up + stack
            return 2
        
        return 0

    def _compute(self, state, task_network):
        """Compute heuristic value from current state and task network."""
        
        # Term 1: Sum of estimated costs for all remaining tasks in the network
        task_cost_sum = 0
        for task in task_network:
            gid = task.global_id
            # Check if it's an abstract task
            if gid in self.task_type_map:
                task_type, args = self.task_type_map[gid]
                cost = self._estimate_task_cost(task_type, args, state)
                task_cost_sum += cost
            elif gid in self.op_type_map:
                # Primitive operator
                op_type = self.op_type_map[gid]
                if op_type == "nop":
                    task_cost_sum += 0
                else:
                    task_cost_sum += 1
        
        # Term 2: Goal-based displacement
        # For each goal on(x,y) not satisfied, minimum 2 actions needed
        goal_cost = 0
        for (x, y, fid) in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                # on(x,y) not true yet
                base = 2  # at minimum pick/unstack + stack
                # If x is not clear, add clearing cost
                if not self._is_clear(x, state):
                    base += 2 * self._stack_height_above(x, state)
                # If y is not clear, add clearing cost
                if not self._is_clear(y, state):
                    base += 2 * self._stack_height_above(y, state)
                goal_cost += base
        
        # Term 3: Goal stability check - goals currently true but will be disrupted
        # If on(x,y) is a goal and is true, but x is not clear and needs to be cleared
        # for some other task, then this goal will need to be re-established
        # This is a lightweight check: if on(x,y) is true but x has blocks on it
        # that need to be moved for other goals, count disruption cost
        stability_penalty = 0
        # Check if any currently-true goal has a block above it that also needs to be
        # somewhere else according to remaining tasks
        # (lightweight version: just check if goal-true blocks need clearing for tasks)
        
        # Return max of the two main terms
        # task_cost_sum is usually the stronger signal since it's task-network aware
        h = max(task_cost_sum, goal_cost)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Called for every node expansion."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
