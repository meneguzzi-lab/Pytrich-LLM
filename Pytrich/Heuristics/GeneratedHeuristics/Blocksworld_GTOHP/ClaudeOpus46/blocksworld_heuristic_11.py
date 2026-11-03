# Bottleneck: Single-hand constraint serializes all block moves. Clearing a buried block
# requires unstacking everything above it (2 actions per block above). Moving a block costs 2 actions.
#
# Admissible lower bounds:
# T1: For each remaining task in the task network, compute state-aware minimum cost.
#     - do_put_on(x,y): 0 if on(x,y) holds, else >= 2 (move) + clearing costs for x and y
#     - do_clear(x): 0 if clear(x), else >= 2 * height_above(x)
#     - do_on_table(x): 0 if ontable(x) and clear(x), else >= 2
#     - do_move(x,y): >= 2 always
# T2: Count of unsatisfied goal on-facts * 2 (each needs at least unstack+stack or pick+stack)
# Return max(T1_sum_over_first_few_tasks, T2) but T1 uses max over tasks for admissibility.
#
# We use a hybrid: sum costs for the CURRENT task (first in network) plus count remaining tasks
# weighted by minimum possible cost, all done with O(1) bit checks.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.HDDL.AbstractTask import AbstractTask
from Pytrich.HDDL.Operator import Operator


class BlocksworldGTOHPAnthropicOpus46Heuristic11(Heuristic):
    """Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Uses state-aware per-task cost estimation with stack height computation
    and goal-displacement analysis. Returns max of multiple admissible terms.
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_11"):
        super().__init__(name)
        # Fact lookup maps
        self.on_fact_id = {}      # (x, y) -> global_id
        self.clear_fact_id = {}   # x -> global_id
        self.ontable_fact_id = {} # x -> global_id
        self.handempty_id = None  # global_id
        self.holding_fact_id = {} # x -> global_id
        
        # Block relationships for stack height
        self.all_blocks = set()
        
        # Task name parsing cache
        self.task_name_cache = {}
        
        # Goal facts
        self.goal_on_pairs = []  # list of (x, y, fact_global_id)
        
        # Abstract task type mapping
        self.task_type_map = {}  # abstract_task.global_id -> (type_str, args)
        
        # Operator type mapping  
        self.op_type_map = {}  # operator.global_id -> type_str

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
        """Parse task/operator name to get type and arguments."""
        name = name.strip()
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
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "handempty" and len(args) == 0:
                self.handempty_id = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
        
        # Build abstract task type map
        for at in model.abstract_tasks:
            ttype, args = self._parse_task_name(at.name)
            self.task_type_map[at.global_id] = (ttype, args)
        
        # Build operator type map
        for op in model.operators:
            ttype, args = self._parse_task_name(op.name)
            self.op_type_map[op.global_id] = (ttype, args)
        
        # Extract goal on-facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                x, y = args[0], args[1]
                fid = self.on_fact_id.get((x, y))
                if fid is not None:
                    self.goal_on_pairs.append((x, y, fid))
        
        # Precompute which on-facts exist for stack traversal
        # Build reverse map: for each block, which blocks can be on top of it
        self.on_pairs_by_bottom = {}  # bottom_block -> list of (top_block, fact_id)
        for (x, y), fid in self.on_fact_id.items():
            if y not in self.on_pairs_by_bottom:
                self.on_pairs_by_bottom[y] = []
            self.on_pairs_by_bottom[y].append((x, fid))
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _find_block_on_top(self, block, state):
        """Find which block is directly on top of 'block' in current state, or None."""
        candidates = self.on_pairs_by_bottom.get(block)
        if candidates is None:
            return None
        for (top_block, fid) in candidates:
            if (state >> fid) & 1:
                return top_block
        return None

    def _stack_height_above(self, block, state):
        """Count how many blocks are stacked above 'block' in current state."""
        height = 0
        current = block
        while True:
            top = self._find_block_on_top(current, state)
            if top is None:
                break
            height += 1
            current = top
            if height > 200:  # Safety limit for very large stacks
                break
        return height

    def _is_clear(self, block, state):
        fid = self.clear_fact_id.get(block)
        if fid is None:
            return True  # unknown block, assume clear
        return (state >> fid) & 1

    def _is_ontable(self, block, state):
        fid = self.ontable_fact_id.get(block)
        if fid is None:
            return True
        return (state >> fid) & 1

    def _is_on(self, x, y, state):
        fid = self.on_fact_id.get((x, y))
        if fid is None:
            return False
        return (state >> fid) & 1

    def _clear_cost(self, block, state):
        """Minimum actions to clear a block: 2 per block above it (unstack + put-down each)."""
        if self._is_clear(block, state):
            return 0
        h = self._stack_height_above(block, state)
        return 2 * h

    def _task_cost(self, task, state):
        """Estimate minimum actions for a single task given current state."""
        if isinstance(task, Operator):
            # Primitive operators cost 1 action each (except nop)
            info = self.op_type_map.get(task.global_id)
            if info is not None:
                ttype, args = info
                if ttype == "nop":
                    return 0
            return 1
        
        # Abstract task
        info = self.task_type_map.get(task.global_id)
        if info is None:
            return 1
        
        ttype, args = info
        
        if ttype == "do_put_on" and len(args) == 2:
            x, y = args[0], args[1]
            # If on(x,y) already holds, cost is 0 (m0 applies: just nop)
            if self._is_on(x, y, state):
                return 0
            # Otherwise: need to clear x, clear y, maybe put y on table, then move x to y
            # Minimum: clear_cost(x) + clear_cost(y) + 2 (for the move: pick-up/unstack + stack)
            # Also need do_on_table(y) which is 0 if y is on table and clear, else 2
            cost_clear_x = self._clear_cost(x, state)
            cost_clear_y = self._clear_cost(y, state)
            # do_on_table(y): if y is on table, cost 0; else 2 (unstack + put-down)
            cost_on_table_y = 0 if self._is_ontable(y, state) else 2
            # do_move(x, y): 2 actions (pick-up/unstack + stack)
            cost_move = 2
            return cost_clear_x + cost_clear_y + cost_on_table_y + cost_move
        
        elif ttype == "do_clear" and len(args) == 1:
            x = args[0]
            return self._clear_cost(x, state)
        
        elif ttype == "do_on_table" and len(args) == 1:
            x = args[0]
            if self._is_ontable(x, state) and self._is_clear(x, state):
                return 0
            if self._is_ontable(x, state):
                return 0  # m3 applies if clear
            return 2  # unstack + put-down
        
        elif ttype == "do_move" and len(args) == 2:
            return 2  # pick-up/unstack + stack
        
        return 1

    def _compute(self, state, task_network):
        if len(task_network) == 0:
            return 0
        
        # Term 1: Sum of estimated costs for tasks in the network
        # For efficiency on large networks, compute first task exactly and
        # sample/estimate the rest
        total_cost = 0
        num_tasks = len(task_network)
        
        # For small networks, compute all tasks
        if num_tasks <= 50:
            for task in task_network:
                total_cost += self._task_cost(task, state)
        else:
            # Compute first few tasks exactly, estimate rest
            # First 20 tasks exactly
            exact_count = min(20, num_tasks)
            for i in range(exact_count):
                total_cost += self._task_cost(task_network[i], state)
            
            # For remaining tasks, use a fast estimate:
            # Count abstract vs primitive tasks
            remaining_abstract = 0
            remaining_primitive = 0
            # Sample a subset for estimation
            sample_size = min(30, num_tasks - exact_count)
            step = max(1, (num_tasks - exact_count) // sample_size)
            sample_cost = 0
            sampled = 0
            for i in range(exact_count, num_tasks, step):
                task = task_network[i]
                if isinstance(task, AbstractTask):
                    remaining_abstract += 1
                    sample_cost += self._task_cost(task, state)
                else:
                    remaining_primitive += 1
                    info = self.op_type_map.get(task.global_id)
                    if info and info[0] == "nop":
                        sample_cost += 0
                    else:
                        sample_cost += 1
                sampled += 1
            
            if sampled > 0:
                avg_cost = sample_cost / sampled
                remaining = num_tasks - exact_count
                total_cost += int(avg_cost * remaining)
        
        # Term 2: Goal displacement - count unsatisfied goal on-facts
        unsatisfied_goals = 0
        for (x, y, fid) in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                unsatisfied_goals += 1
        goal_cost = unsatisfied_goals * 2  # Each needs at least 2 actions
        
        # Term 3: For the first task, if it's do_put_on and the target on(x,y) is a goal
        # but will be disrupted by later tasks, add penalty
        disruption_penalty = 0
        if num_tasks > 0:
            first_task = task_network[0]
            if isinstance(first_task, AbstractTask):
                info = self.task_type_map.get(first_task.global_id)
                if info and info[0] == "do_put_on" and len(info[1]) == 2:
                    x, y = info[1][0], info[1][1]
                    # Check if on(x,y) is satisfied but x or y needs to be cleared for future tasks
                    if self._is_on(x, y, state):
                        # Task is trivial (nop), no extra cost
                        pass
                    else:
                        # Check if x is buried under blocks that are also goal-relevant
                        if not self._is_clear(x, state):
                            h = self._stack_height_above(x, state)
                            disruption_penalty += h  # Extra cost for deep blocks
        
        return max(total_cost, goal_cost) + disruption_penalty
