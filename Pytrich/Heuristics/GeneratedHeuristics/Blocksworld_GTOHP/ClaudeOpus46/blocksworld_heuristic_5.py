# Bottleneck: Single hand constraint - only one block held at a time.
# Clearing deeply stacked blocks dominates plan length.
#
# Admissible lower bounds:
# T1: For each abstract task in the task network, estimate minimum actions
#     based on current state (clearing costs, move costs). Admissible because
#     each task truly requires at least that many actions.
# T2: Count remaining unsatisfied goal on-facts; each needs >= 2 actions.
#     Admissible since each unsatisfied on-goal needs at least unstack+stack.
# T3: Stack height clearing - blocks that need clearing require 2 actions per
#     block above them. Admissible as each block removal is unstack+putdown.
#
# Return max(T1, T2, T3) which is admissible since max of admissible bounds
# is admissible.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic5(Heuristic):
    """Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Uses multi-component max of:
    - State-aware task network cost estimation
    - Unsatisfied goal counting with clearing costs
    - Stack-height based clearing bounds
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_5"):
        super().__init__(name)
        # Fact ID maps
        self.on_fact_id = {}       # (x, y) -> global_id
        self.ontable_fact_id = {}  # x -> global_id
        self.clear_fact_id = {}    # x -> global_id
        self.handempty_fact_id = None
        self.holding_fact_id = {}  # x -> global_id
        
        # Goal structures
        self.goal_on_pairs = []    # list of (x, y, fact_global_id)
        
        # Block info
        self.all_blocks = set()
        
        # Task type identification
        self.task_name_cache = {}  # abstract_task.global_id -> (task_type, args)
        self.op_name_cache = {}    # operator.global_id -> (op_type, args)
        
        # on_fact reverse: for block y, which (x,y) on-facts exist
        self.on_top_of = {}  # block -> list of (x, fact_id) where on(x, block)
        # on_fact by bottom: block x -> list of (y, fact_id) where on(x, y)
        self.block_on = {}   # block -> list of (y, fact_id) where on(block, y)

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
        """Parse task/operator name to extract type and arguments."""
        # Names look like: do_put_on[b1,b2] or pick-up[b1] or nop[]
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        # Try without brackets
        paren_idx = name.find("(")
        if paren_idx > 0:
            task_type = name[:paren_idx]
            args_str = name[paren_idx + 1:].rstrip(")")
            args = [a.strip() for a in args_str.split(",")] if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact ID maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
                # Build reverse maps
                if args[1] not in self.on_top_of:
                    self.on_top_of[args[1]] = []
                self.on_top_of[args[1]].append((args[0], f.global_id))
                if args[0] not in self.block_on:
                    self.block_on[args[0]] = []
                self.block_on[args[0]].append((args[1], f.global_id))
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "handempty":
                self.handempty_fact_id = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
        
        # Parse goal on-facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                fid = self.on_fact_id.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_pairs.append((args[0], args[1], fid))
        
        # Cache task type parsing for abstract tasks
        for at in model.abstract_tasks:
            self.task_name_cache[at.global_id] = self._parse_task_name(at.name)
        
        # Cache operator type parsing
        for op in model.operators:
            self.op_name_cache[op.global_id] = self._parse_task_name(op.name)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _is_clear(self, block, state):
        """Check if block is clear in current state."""
        fid = self.clear_fact_id.get(block)
        if fid is None:
            return True
        return (state >> fid) & 1

    def _is_on(self, x, y, state):
        """Check if on(x, y) in current state."""
        fid = self.on_fact_id.get((x, y))
        if fid is None:
            return False
        return (state >> fid) & 1

    def _is_ontable(self, block, state):
        """Check if block is on table in current state."""
        fid = self.ontable_fact_id.get(block)
        if fid is None:
            return False
        return (state >> fid) & 1

    def _is_holding(self, block, state):
        """Check if holding block in current state."""
        fid = self.holding_fact_id.get(block)
        if fid is None:
            return False
        return (state >> fid) & 1

    def _find_block_on_top(self, block, state):
        """Find which block is directly on top of 'block' in current state, if any."""
        if block not in self.on_top_of:
            return None
        for (x, fid) in self.on_top_of[block]:
            if (state >> fid) & 1:
                return x
        return None

    def _stack_height_above(self, block, state):
        """Count how many blocks are stacked above 'block' in current state."""
        count = 0
        current = block
        while True:
            top = self._find_block_on_top(current, state)
            if top is None:
                break
            count += 1
            current = top
            if count > 1000:  # safety
                break
        return count

    def _clearing_cost(self, block, state):
        """Minimum actions to clear block: 2 per block above it (unstack + putdown)."""
        if self._is_clear(block, state):
            return 0
        height = self._stack_height_above(block, state)
        return 2 * height

    def _estimate_task_cost(self, task, state):
        """Estimate minimum actions for an abstract task given current state."""
        info = self.task_name_cache.get(task.global_id)
        if info is None:
            return 1
        task_type, args = info
        
        if task_type == "do_put_on" and len(args) == 2:
            x, y = args[0], args[1]
            # If on(x,y) already true, cost is 0 (nop)
            if self._is_on(x, y, state):
                return 0
            # Need to: clear x, clear y, ensure y on table or somewhere, move x onto y
            # Minimum: clear x + clear y + move x (2 actions: pick/unstack + stack)
            cost = 0
            # Clearing x
            cost += self._clearing_cost(x, state)
            # Clearing y
            cost += self._clearing_cost(y, state)
            # Moving x onto y: at least 2 actions (pick-up/unstack + stack)
            cost += 2
            # If y is not on table, do_on_table needs to put y on table first
            # That's unstack + put-down = 2 more actions
            if not self._is_ontable(y, state) and not self._is_clear(y, state):
                # y might need to be put on table, but that's already in clearing cost
                pass
            elif not self._is_ontable(y, state):
                # y is clear but not on table: need unstack + putdown = 2
                cost += 2
            return cost
        
        elif task_type == "do_clear" and len(args) == 1:
            x = args[0]
            return self._clearing_cost(x, state)
        
        elif task_type == "do_on_table" and len(args) == 1:
            x = args[0]
            if self._is_ontable(x, state):
                return 0
            # Need unstack + put-down = 2
            return 2
        
        elif task_type == "do_move" and len(args) == 2:
            x, y = args[0], args[1]
            # pick-up/unstack + stack = 2
            return 2
        
        return 1

    def _estimate_operator_cost(self, op):
        """Estimate cost of a primitive operator."""
        info = self.op_name_cache.get(op.global_id)
        if info is None:
            return 1
        op_type, args = info
        if op_type == "nop":
            return 0
        return 1

    def _compute(self, state, task_network):
        if not task_network:
            return 0
        
        # Term 1: State-aware task network cost estimation
        t1 = 0
        for task in task_network:
            gid = task.global_id
            if gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                # Abstract task
                t1 += self._estimate_task_cost(task, state)
            elif gid >= self.model.iop_init and gid <= self.model.iop_end:
                # Primitive operator
                t1 += self._estimate_operator_cost(task)
        
        # Term 2: Unsatisfied goal on-facts with clearing costs
        t2 = 0
        for x, y, fid in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                # Goal on(x,y) not satisfied: need at least 2 actions
                t2 += 2
        
        # Term 3: Count blocks that need to be cleared for remaining tasks
        # Look at first few tasks to estimate clearing bottleneck
        t3 = 0
        blocks_to_clear = set()
        task_count = min(len(task_network), 20)  # limit scan
        for i in range(task_count):
            task = task_network[i]
            gid = task.global_id
            if gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                info = self.task_name_cache.get(gid)
                if info is not None:
                    task_type, args = info
                    if task_type == "do_clear" and len(args) == 1:
                        blocks_to_clear.add(args[0])
                    elif task_type == "do_put_on" and len(args) == 2:
                        blocks_to_clear.add(args[0])
                        blocks_to_clear.add(args[1])
        
        # For the max clearing cost block, use it as a bottleneck
        max_clear_cost = 0
        for b in blocks_to_clear:
            cc = self._clearing_cost(b, state)
            if cc > max_clear_cost:
                max_clear_cost = cc
        t3 = max_clear_cost
        
        return max(t1, t2, t3)
