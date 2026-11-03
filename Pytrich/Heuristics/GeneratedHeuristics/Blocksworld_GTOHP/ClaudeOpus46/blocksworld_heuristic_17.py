# Bottleneck: Single-hand constraint (handempty) serializes all block moves.
# Clearing a block with k blocks above costs at least 2k actions (unstack+putdown each).
# Moving a block onto another costs at least 2 actions (pick-up/unstack + stack).
#
# Admissible lower bounds:
# T1: For each do_put_on(x,y) task where on(x,y) is false, we need at least 2 actions
#     (the final move). Plus clearing costs for x and y based on stack heights.
# T2: Count of unsatisfied goal on(x,y) facts * 2 (minimum actions per goal).
# T3: Task network size weighted by minimum decomposition depth.
#
# We use max(T1, T2) with state-aware per-task cost estimation.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic17(Heuristic):
    def __init__(self, name="blocksworld_anthropicopus46_heuristic_17"):
        super().__init__(name)
        self.on_fact_ids = {}       # (x_arg, y_arg) -> global_id
        self.clear_fact_ids = {}    # block_arg -> global_id
        self.ontable_fact_ids = {}  # block_arg -> global_id
        self.handempty_id = -1
        self.holding_fact_ids = {}  # block_arg -> global_id

        # For on(x,y): map y -> list of x such that on(x,y) is a fact
        self.on_top_of = {}  # block -> list of blocks that can be on top of it

        # Goal facts
        self.goal_on_pairs = []  # list of (x, y, fact_global_id)

        # Task name parsing caches
        self.task_type_cache = {}  # task global_id -> (type_str, args)

        # Abstract task name prefixes
        self.DO_PUT_ON = "do_put_on"
        self.DO_CLEAR = "do_clear"
        self.DO_ON_TABLE = "do_on_table"
        self.DO_MOVE = "do_move"

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
            else:
                return name, []
        return None, []

    def _parse_task_name(self, name: str):
        """Parse a task/operator name to extract the task type and arguments.
        Format examples: +do_put_on[b1,b2], +pick-up[b1], +nop[]
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model

        # Build fact lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_ids[(args[0], args[1])] = f.global_id
                if args[1] not in self.on_top_of:
                    self.on_top_of[args[1]] = []
                self.on_top_of[args[1]].append(args[0])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_ids[args[0]] = f.global_id
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_ids[args[0]] = f.global_id
            elif pred == "handempty" and len(args) == 0:
                self.handempty_id = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_ids[args[0]] = f.global_id

        # Parse goal facts
        if model.goals:
            goal_fact_names = model.state_explicit_repr(model.goals)
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) == 2:
                    fid = self.on_fact_ids.get((args[0], args[1]), -1)
                    self.goal_on_pairs.append((args[0], args[1], fid))

        # Precompute task type info for abstract tasks
        for at in model.abstract_tasks:
            ttype, targs = self._parse_task_name(at.name)
            self.task_type_cache[at.global_id] = (ttype, targs)

        # Also cache operator types
        for op in model.operators:
            ttype, targs = self._parse_task_name(op.name)
            self.task_type_cache[op.global_id] = (ttype, targs)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _is_clear(self, block, state):
        """Check if block is clear in current state."""
        fid = self.clear_fact_ids.get(block, -1)
        if fid < 0:
            return True
        return (state >> fid) & 1

    def _is_on(self, x, y, state):
        """Check if on(x, y) in current state."""
        fid = self.on_fact_ids.get((x, y), -1)
        if fid < 0:
            return False
        return (state >> fid) & 1

    def _is_ontable(self, block, state):
        """Check if block is on table in current state."""
        fid = self.ontable_fact_ids.get(block, -1)
        if fid < 0:
            return False
        return (state >> fid) & 1

    def _find_block_on_top(self, block, state):
        """Find which block is directly on top of 'block' in current state, if any."""
        candidates = self.on_top_of.get(block, [])
        for x in candidates:
            fid = self.on_fact_ids.get((x, block), -1)
            if fid >= 0 and (state >> fid) & 1:
                return x
        return None

    def _stack_height_above(self, block, state, max_depth=50):
        """Count how many blocks are stacked above 'block'.
        Each block above costs 2 actions to remove (unstack + putdown),
        plus recursive clearing.
        """
        height = 0
        current = block
        for _ in range(max_depth):
            top = self._find_block_on_top(current, state)
            if top is None:
                break
            height += 1
            current = top
        return height

    def _clear_cost(self, block, state):
        """Minimum cost to make 'block' clear.
        If already clear, cost is 0.
        Otherwise, need to clear and remove each block above: 2 actions per block above
        (unstack + putdown), but clearing the topmost is free (it's already clear).
        So for k blocks above, cost = 2*k.
        Actually: do_clear recurses: for each block y on x,
          do_clear(y) + unstack(y,x) + putdown(y) = cost(clear y) + 2
        The topmost block is clear so cost(clear topmost) = 0.
        Total for k blocks: 2*k.
        """
        if self._is_clear(block, state):
            return 0
        h = self._stack_height_above(block, state)
        return 2 * h

    def _do_on_table_cost(self, block, state):
        """Cost to put block on table.
        If already on table and clear, cost = 0 (nop).
        If clear but not on table, cost = 2 (unstack from current + putdown).
        If not clear, need to clear first, then 2 more.
        """
        if self._is_ontable(block, state):
            if self._is_clear(block, state):
                return 0
            # ontable but not clear - still nop via m3_do_on_table (precondition is clear(x))
            # Actually m3 requires clear(x), so we need to clear first... but that's
            # handled by the do_clear task before do_on_table in the HTN
            return 0  # the do_on_table itself is nop if already ontable
        # Not on table: m2_do_on_table: unstack + putdown = 2 actions
        # But requires clear(x), which should be ensured by prior do_clear
        return 2

    def _do_move_cost(self, x, y, state):
        """Minimum cost of do_move(x, y).
        Requires clear(x), clear(y), handempty.
        If ontable(x): pick-up + stack = 2
        If not ontable(x): unstack + stack = 2
        Either way, cost is 2.
        """
        return 2

    def _do_put_on_cost(self, x, y, state):
        """Minimum cost of do_put_on(x, y).
        If on(x,y) already: 0 (nop via m0).
        Otherwise via m1: do_clear(x) + do_clear(y) + do_on_table(y) + do_move(x,y)
        """
        if self._is_on(x, y, state):
            return 0

        cost = 0
        # Cost to clear x
        cost += self._clear_cost(x, state)
        # Cost to clear y
        cost += self._clear_cost(y, state)
        # Cost to put y on table (if not already)
        if not self._is_ontable(y, state):
            cost += 2  # unstack + putdown
        # Cost to move x onto y
        cost += 2  # pick-up/unstack + stack

        return cost

    def _estimate_task_cost(self, task, state):
        """Estimate minimum cost for a single task in the network."""
        gid = task.global_id
        info = self.task_type_cache.get(gid)
        if info is None:
            return 1

        ttype, targs = info

        if ttype == "nop":
            return 0
        elif ttype == "pick-up" or ttype == "put-down" or ttype == "stack" or ttype == "unstack":
            return 1
        elif ttype == self.DO_PUT_ON and len(targs) == 2:
            return self._do_put_on_cost(targs[0], targs[1], state)
        elif ttype == self.DO_CLEAR and len(targs) == 1:
            return self._clear_cost(targs[0], state)
        elif ttype == self.DO_ON_TABLE and len(targs) == 1:
            return self._do_on_table_cost(targs[0], state)
        elif ttype == self.DO_MOVE and len(targs) == 2:
            return self._do_move_cost(targs[0], targs[1], state)
        else:
            return 1

    def _task_network_heuristic(self, state, task_network):
        """Sum of estimated costs for all tasks in the network.
        This is admissible if we don't double-count, but since tasks are
        ordered and each does independent work, summing is a reasonable
        lower bound.
        """
        total = 0
        for task in task_network:
            total += self._estimate_task_cost(task, state)
        return total

    def _unsatisfied_goals_heuristic(self, state):
        """Count unsatisfied on(x,y) goals. Each needs at least 2 actions.
        Admissible because each unsatisfied on(x,y) requires at minimum
        one pick-up/unstack and one stack action.
        """
        count = 0
        for x, y, fid in self.goal_on_pairs:
            if fid >= 0 and not ((state >> fid) & 1):
                count += 1
        return 2 * count

    def _goal_instability_penalty(self, state, task_network):
        """Detect goal facts that are true now but will need to be undone.
        A goal on(x,y) that is currently true is NOT free if some task
        in the network will need to clear y or move x.
        This catches false-trivial situations.
        """
        # Collect blocks that need to be cleared by tasks in the network
        blocks_to_disturb = set()
        for task in task_network:
            gid = task.global_id
            info = self.task_type_cache.get(gid)
            if info is None:
                continue
            ttype, targs = info
            if ttype == self.DO_CLEAR and len(targs) == 1:
                # This task will clear targs[0], which means removing blocks above it
                # Any block currently on targs[0] will be disturbed
                blocks_to_disturb.add(targs[0])
            elif ttype == self.DO_ON_TABLE and len(targs) == 1:
                # Moving block to table might disturb what's below
                blocks_to_disturb.add(targs[0])
            elif ttype == self.DO_PUT_ON and len(targs) == 2:
                x, y = targs[0], targs[1]
                if not self._is_on(x, y, state):
                    # x will be moved, so anything on x and anything x is on gets disturbed
                    blocks_to_disturb.add(x)
                    blocks_to_disturb.add(y)
            elif ttype == self.DO_MOVE and len(targs) == 2:
                blocks_to_disturb.add(targs[0])
                blocks_to_disturb.add(targs[1])

        penalty = 0
        for x, y, fid in self.goal_on_pairs:
            if fid >= 0 and (state >> fid) & 1:
                # Goal is satisfied. Check if it will be disturbed.
                # If x needs to be cleared (something needs to go on y requiring x moved)
                # or if y is in blocks_to_disturb
                if x in blocks_to_disturb:
                    # x will be moved/cleared, so on(x,y) will be broken
                    # Cost to restore: at least 2 (move x back)
                    penalty += 2
        return penalty

    def _compute(self, state, task_network):
        if len(task_network) == 0:
            return 0

        # Term 1: Task network based heuristic (state-aware)
        t1 = self._task_network_heuristic(state, task_network)

        # Term 2: Unsatisfied goals heuristic
        t2 = self._unsatisfied_goals_heuristic(state)

        # Term 3: Goal instability
        t3 = self._goal_instability_penalty(state, task_network)

        # Use max of t1 and (t2 + t3) for admissibility
        # t1 is a direct task-based estimate
        # t2 + t3 captures goal-based reasoning
        h = max(t1, t2 + t3)

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
