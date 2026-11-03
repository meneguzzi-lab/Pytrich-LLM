"""
Blocksworld_GTOHP Domain-Dependent Heuristic

Bottleneck: Single-hand constraint - only one block held at a time.
Clearing a tower of height h costs 2h actions (unstack + put-down each).
Moving a block to its goal position costs 2 actions (pick-up/unstack + stack).

Admissible lower bounds:
1. Per-task state-aware cost: For each do_put_on(x,y) task, if on(x,y) not true,
   we need to clear x (2 * height above x), clear y (2 * height above y),
   and move x to y (2 actions). This is a lower bound because each clearing
   action is unavoidable.
2. Task network size lower bound: Each non-trivial abstract task requires at
   least 2 primitive actions in its minimum decomposition.
3. Max of per-task estimates: Taking max over individual task costs gives
   admissible bound since tasks are ordered (sequential).

We use additive costs for ordered subtasks (since they're sequential) and
state-aware estimation for each task.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic0(Heuristic):
    def __init__(self, name="blocksworld_anthropicopus46_heuristic_0"):
        super().__init__(name)
        # Fact ID maps
        self.on_fact_id = {}      # (x, y) -> global_id
        self.clear_fact_id = {}   # x -> global_id
        self.ontable_fact_id = {} # x -> global_id
        self.handempty_fact_id = None
        self.holding_fact_id = {} # x -> global_id
        
        # Goal tracking
        self.goal_on_pairs = []   # list of (x, y) for goal on(x,y)
        self.goal_on_fact_ids = [] # corresponding fact global_ids
        
        # Block info
        self.all_blocks = set()
        
        # Task name parsing cache
        self.task_name_cache = {}
        
        # Abstract task type identification
        self.do_put_on_tasks = set()   # global_ids of do_put_on abstract tasks
        self.do_clear_tasks = set()    # global_ids of do_clear abstract tasks
        self.do_on_table_tasks = set() # global_ids of do_on_table abstract tasks
        self.do_move_tasks = set()     # global_ids of do_move abstract tasks
        
        # Task argument maps
        self.task_args = {}  # global_id -> (arg1, arg2, ...)
        
        # Operator identification
        self.nop_ops = set()
        self.pick_up_ops = set()
        self.put_down_ops = set()
        self.stack_ops = set()
        self.unstack_ops = set()

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

    def _parse_task_name(self, name: str):
        """Parse task/operator name to get type and arguments."""
        if name in self.task_name_cache:
            return self.task_name_cache[name]
        
        original = name
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            result = (task_type, args)
        else:
            result = (name, [])
        
        self.task_name_cache[original] = result
        return result

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact ID maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "handempty":
                self.handempty_fact_id = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
        
        # Parse goal state
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                self.goal_on_pairs.append((args[0], args[1]))
                fid = self.on_fact_id.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_fact_ids.append(fid)
        
        # Build abstract task type maps
        for at in model.abstract_tasks:
            ttype, targs = self._parse_task_name(at.name)
            self.task_args[at.global_id] = targs
            if ttype == "do_put_on":
                self.do_put_on_tasks.add(at.global_id)
            elif ttype == "do_clear":
                self.do_clear_tasks.add(at.global_id)
            elif ttype == "do_on_table":
                self.do_on_table_tasks.add(at.global_id)
            elif ttype == "do_move":
                self.do_move_tasks.add(at.global_id)
        
        # Build operator type maps
        for op in model.operators:
            ttype, targs = self._parse_task_name(op.name)
            self.task_args[op.global_id] = targs
            if ttype == "nop":
                self.nop_ops.add(op.global_id)
            elif ttype == "pick-up":
                self.pick_up_ops.add(op.global_id)
            elif ttype == "put-down":
                self.put_down_ops.add(op.global_id)
            elif ttype == "stack":
                self.stack_ops.add(op.global_id)
            elif ttype == "unstack":
                self.unstack_ops.add(op.global_id)
        
        # Build reverse on-map for stack height computation
        # For each block, which on(?,block) facts exist
        self.on_top_of = {}  # block -> list of (other_block, fact_global_id)
        for (x, y), fid in self.on_fact_id.items():
            if y not in self.on_top_of:
                self.on_top_of[y] = []
            self.on_top_of[y].append((x, fid))
        
        # Also build: for each block, what's on top of it (on(?, block))
        self.block_on_top = {}  # block -> list of (other_block, fact_global_id)
        for (x, y), fid in self.on_fact_id.items():
            if y not in self.block_on_top:
                self.block_on_top[y] = []
            self.block_on_top[y].append((x, fid))
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _find_block_on_top(self, block, state):
        """Find which block is currently on top of 'block', or None if clear."""
        if block in self.block_on_top:
            for (x, fid) in self.block_on_top[block]:
                if (state >> fid) & 1:
                    return x
        return None

    def _stack_height_above(self, block, state, max_depth=50):
        """Count how many blocks are stacked above 'block' in current state."""
        height = 0
        current = block
        for _ in range(max_depth):
            top = self._find_block_on_top(current, state)
            if top is None:
                break
            height += 1
            current = top
        return height

    def _is_clear(self, block, state):
        """Check if block is clear in current state."""
        fid = self.clear_fact_id.get(block)
        if fid is None:
            return True
        return (state >> fid) & 1

    def _is_on(self, x, y, state):
        """Check if on(x, y) is true in current state."""
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

    def _cost_to_clear(self, block, state):
        """Estimate minimum actions to clear block. Each block above costs 2 actions."""
        if self._is_clear(block, state):
            return 0
        h = self._stack_height_above(block, state)
        return 2 * h  # unstack + put-down for each block above

    def _estimate_do_put_on(self, x, y, state):
        """Estimate cost for do_put_on(x, y).
        
        If on(x,y) already true -> 0 (nop)
        Otherwise: clear x + clear y + ensure y on table + move x to y
        Minimum: cost_clear_x + cost_clear_y + move_cost(2)
        """
        if self._is_on(x, y, state):
            return 0
        
        cost = 0
        # Cost to clear x
        cost += self._cost_to_clear(x, state)
        # Cost to clear y
        cost += self._cost_to_clear(y, state)
        # Cost to move x onto y: pick-up/unstack + stack = 2
        cost += 2
        # If y is not on table, we may need to put it on table first
        # do_on_table costs 2 if not already on table (unstack + put-down)
        # But the clearing of y already handles unstacking things above y
        # We still need to check if y itself needs to be moved to table
        if not self._is_ontable(y, state) and not self._is_clear(y, state):
            # y might need do_on_table which costs 2 more if y is not on table
            pass  # This is already somewhat captured; avoid double counting
        elif not self._is_ontable(y, state):
            # y needs to be put on table: unstack y from whatever + put-down = 2
            cost += 2
        
        return cost

    def _estimate_do_clear(self, x, state):
        """Estimate cost for do_clear(x)."""
        return self._cost_to_clear(x, state)

    def _estimate_do_on_table(self, x, state):
        """Estimate cost for do_on_table(x)."""
        if self._is_ontable(x, state):
            return 0
        # Need to unstack + put-down = 2 (assuming already clear)
        return 2

    def _estimate_do_move(self, x, y, state):
        """Estimate cost for do_move(x, y). Requires clear x, clear y, handempty."""
        # pick-up/unstack + stack = 2
        return 2

    def _estimate_task_cost(self, task, state):
        """Estimate the minimum primitive action cost for a task given current state."""
        gid = task.global_id
        args = self.task_args.get(gid, [])
        
        # Check if it's an operator (primitive)
        if gid in self.nop_ops:
            return 0
        if gid in self.pick_up_ops or gid in self.put_down_ops:
            return 1
        if gid in self.stack_ops or gid in self.unstack_ops:
            return 1
        
        # Abstract tasks
        if gid in self.do_put_on_tasks:
            if len(args) >= 2:
                return self._estimate_do_put_on(args[0], args[1], state)
            return 2
        elif gid in self.do_clear_tasks:
            if len(args) >= 1:
                return self._estimate_do_clear(args[0], state)
            return 0
        elif gid in self.do_on_table_tasks:
            if len(args) >= 1:
                return self._estimate_do_on_table(args[0], state)
            return 0
        elif gid in self.do_move_tasks:
            if len(args) >= 2:
                return self._estimate_do_move(args[0], args[1], state)
            return 2
        
        # Unknown task, return 1
        return 1

    def _unsatisfied_goals_cost(self, state):
        """Count cost based on unsatisfied goal on-predicates.
        Each unsatisfied on(x,y) goal needs at least 2 actions.
        """
        cost = 0
        for fid in self.goal_on_fact_ids:
            if not ((state >> fid) & 1):
                cost += 2
        return cost

    def _compute(self, state, task_network):
        """Compute heuristic value.
        
        Term 1: Sum of per-task state-aware cost estimates for tasks in network.
                This is admissible for ordered (sequential) task networks since
                each task's work is unavoidable and sequential.
        Term 2: Unsatisfied goals lower bound (each needs >= 2 actions).
        Term 3: Number of remaining tasks that are non-trivial (>= 1 action each).
        
        Return max(term1, term2) for admissible bound.
        """
        if len(task_network) == 0:
            return 0
        
        # Term 1: Additive per-task cost (state-aware)
        # For large task networks, only estimate first N tasks in detail
        # and use cheaper estimates for the rest
        task_cost_sum = 0
        detailed_limit = min(len(task_network), 200)  # Limit detailed analysis
        
        for i in range(detailed_limit):
            task = task_network[i]
            task_cost_sum += self._estimate_task_cost(task, state)
        
        # For remaining tasks, use quick estimate
        for i in range(detailed_limit, len(task_network)):
            task = task_network[i]
            gid = task.global_id
            if gid in self.nop_ops:
                pass  # 0 cost
            elif gid in self.do_put_on_tasks:
                # Quick: at least 2 if not already satisfied
                args = self.task_args.get(gid, [])
                if len(args) >= 2:
                    fid = self.on_fact_id.get((args[0], args[1]))
                    if fid is not None and (state >> fid) & 1:
                        pass  # already on, 0 cost
                    else:
                        task_cost_sum += 2
                else:
                    task_cost_sum += 2
            elif gid in self.do_clear_tasks:
                args = self.task_args.get(gid, [])
                if len(args) >= 1:
                    if not self._is_clear(args[0], state):
                        task_cost_sum += 2
            elif gid in self.do_on_table_tasks:
                args = self.task_args.get(gid, [])
                if len(args) >= 1:
                    if not self._is_ontable(args[0], state):
                        task_cost_sum += 2
            elif gid in self.do_move_tasks:
                task_cost_sum += 2
            elif gid in self.pick_up_ops or gid in self.put_down_ops or gid in self.stack_ops or gid in self.unstack_ops:
                task_cost_sum += 1
        
        # Term 2: Unsatisfied goals
        goals_cost = self._unsatisfied_goals_cost(state)
        
        return max(task_cost_sum, goals_cost)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
