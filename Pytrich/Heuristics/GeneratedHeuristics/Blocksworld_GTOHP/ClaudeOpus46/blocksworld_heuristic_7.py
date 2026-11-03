# Bottleneck: Single hand (handempty) serializes all block moves. Clearing stacks
# dominates plan length - each block above a target costs 2 actions (unstack+put-down).
#
# Admissible lower bounds:
# T1: For each task in the task network, estimate minimum primitives needed given current state.
#     - do_put_on(x,y): 0 if on(x,y) already true, else 2 (move) + clearing costs for x and y
#     - do_clear(x): 0 if clear(x), else 2*height_above(x)
#     - do_on_table(x): 0 if ontable(x), else 2
#     - do_move(x,y): 2 (pick/unstack + stack)
#     Each is a lower bound since these are the minimum actions for that subtask.
# T2: Number of unsatisfied goal on-facts, each needing >= 2 actions.
# Return max(T1_sum_of_first_k_tasks, T2) for admissibility of the max.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic7(Heuristic):
    """Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Uses state-aware per-task cost estimation with stack height analysis
    and goal displacement counting.
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_7"):
        super().__init__(name)
        self.on_fact_id = {}       # (x_arg, y_arg) -> global_id
        self.clear_fact_id = {}    # x_arg -> global_id
        self.ontable_fact_id = {}  # x_arg -> global_id
        self.handempty_id = -1
        self.holding_fact_id = {}  # x_arg -> global_id
        
        # For stack height computation
        self.on_facts_by_bottom = {}  # y_arg -> list of (x_arg, global_id)
        
        # Task name -> task type mapping
        self.task_type_map = {}  # abstract_task.global_id -> (type_str, args)
        
        # Goal facts
        self.goal_on_pairs = []  # list of ((x,y), fact_global_id)
        
        # All block names for iteration
        self.all_blocks = set()
        
        # on_fact: for each block x, which blocks could be on top of x
        # We store: for block x, the fact_id of on(y, x) for each y
        self.on_top_of = {}  # bottom_block -> [(top_block, fact_global_id), ...]

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
        return None, []

    def _parse_task_name(self, name: str):
        """Parse task/operator name to extract type and arguments."""
        # Format like: do_put_on[b1,b2] or pick-up[b1] or nop[]
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
        
        # Build fact index maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
                # Index by bottom block for stack traversal
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
                self.handempty_id = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])

        # Build task type map for abstract tasks
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            self.task_type_map[at.global_id] = (task_type, args)

        # Build goal on-pairs
        if model.goals:
            goal_fact_names = model.state_explicit_repr(model.goals)
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) == 2:
                    fid = self.on_fact_id.get((args[0], args[1]), -1)
                    if fid >= 0:
                        self.goal_on_pairs.append(((args[0], args[1]), fid))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_stack_height_above(self, block, state):
        """Count how many blocks are stacked above 'block' in current state.
        Uses iterative traversal to avoid recursion issues."""
        if block not in self.on_top_of:
            return 0
        
        height = 0
        current = block
        while True:
            found_top = None
            if current in self.on_top_of:
                for top_block, fid in self.on_top_of[current]:
                    if (state >> fid) & 1:
                        found_top = top_block
                        height += 1
                        break
            if found_top is None:
                break
            current = found_top
        return height

    def _is_clear(self, block, state):
        if block in self.clear_fact_id:
            return (state >> self.clear_fact_id[block]) & 1
        return 1  # If no clear fact exists, assume clear

    def _is_ontable(self, block, state):
        if block in self.ontable_fact_id:
            return (state >> self.ontable_fact_id[block]) & 1
        return 0

    def _is_on(self, x, y, state):
        fid = self.on_fact_id.get((x, y), -1)
        if fid >= 0:
            return (state >> fid) & 1
        return 0

    def _estimate_clear_cost(self, block, state):
        """Estimate minimum actions to make block clear.
        Each block above needs unstack+put-down = 2 actions."""
        if self._is_clear(block, state):
            return 0
        height = self._get_stack_height_above(block, state)
        return 2 * height  # Each block: unstack + put-down

    def _estimate_on_table_cost(self, block, state):
        """Estimate cost to put block on table.
        If already on table and clear, cost=0.
        If clear but not on table, cost=2 (unstack+put-down).
        If not clear, need to clear first + 2."""
        if self._is_ontable(block, state):
            if self._is_clear(block, state):
                return 0
            else:
                return 0  # It's on table, just not clear - but do_on_table only cares about ontable
        # Not on table: need to clear it first, then unstack+put-down
        clear_cost = self._estimate_clear_cost(block, state)
        return clear_cost + 2  # unstack from current position + put-down

    def _estimate_task_cost(self, task, state):
        """Estimate minimum primitive actions for a task given current state."""
        from Pytrich.HDDL.AbstractTask import AbstractTask as AT
        from Pytrich.HDDL.Operator import Operator as OP
        
        if isinstance(task, OP):
            # Primitive task: cost 1 (unless nop)
            task_type, args = self._parse_task_name(task.name)
            if task_type == "nop":
                return 0
            return 1
        
        if not isinstance(task, AT):
            return 0
            
        if task.global_id not in self.task_type_map:
            return 1
        
        task_type, args = self.task_type_map[task.global_id]
        
        if task_type == "do_put_on" and len(args) == 2:
            x, y = args[0], args[1]
            # If on(x,y) already holds, cost = 0
            if self._is_on(x, y, state):
                return 0
            # Otherwise: need to clear x, clear y, possibly put y on table, then move x to y
            # Minimum: clear_x + clear_y + move(2) = clearing costs + 2
            # But if y is already clear and on table, and x is clear and on table: just 2
            clear_x = self._estimate_clear_cost(x, state)
            clear_y = self._estimate_clear_cost(y, state)
            # do_on_table for y: if y not on table, need extra work
            # do_move: 2 actions (pick-up/unstack + stack)
            on_table_y = 0
            if not self._is_ontable(y, state) and not self._is_on(x, y, state):
                # y needs to be freed from whatever it's on, but do_on_table
                # handles that. However the clear cost already accounts for
                # clearing above y. The on_table cost is for moving y itself.
                if not self._is_ontable(y, state):
                    on_table_y = 2  # unstack y from its support + put-down
            move_cost = 2  # pick-up/unstack x + stack x on y
            return clear_x + clear_y + on_table_y + move_cost
        
        elif task_type == "do_clear" and len(args) >= 1:
            x = args[0]
            return self._estimate_clear_cost(x, state)
        
        elif task_type == "do_on_table" and len(args) >= 1:
            x = args[0]
            if self._is_ontable(x, state):
                return 0
            return 2  # unstack + put-down
        
        elif task_type == "do_move" and len(args) == 2:
            return 2  # pick-up/unstack + stack
        
        return 1  # fallback

    def _compute(self, state, task_network):
        if not task_network:
            # Check if goals are met
            if self.model.goals and not (self.model.goals & state == self.model.goals):
                return float('inf')
            return 0
        
        # Term 1: Sum of estimated costs for tasks in the network
        # We process tasks and sum up their estimated costs
        # To keep it fast, we limit how many tasks we deeply analyze
        task_cost_sum = 0
        max_single_cost = 0
        
        for task in task_network:
            cost = self._estimate_task_cost(task, state)
            task_cost_sum += cost
            if cost > max_single_cost:
                max_single_cost = cost
        
        # Term 2: Goal-based displacement
        # Count unsatisfied goal on-facts
        goal_cost = 0
        for (x, y), fid in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                # This goal is not satisfied - needs at least 2 actions
                goal_cost += 2
                # Additional cost if x is not clear
                if not self._is_clear(x, state):
                    h_above = self._get_stack_height_above(x, state)
                    goal_cost += 2 * h_above
        
        # Term 3: Count of remaining tasks as baseline
        # Each non-nop task needs at least 1 action
        task_count = len(task_network)
        
        # Return max of the terms for tighter bound
        # task_cost_sum is usually the strongest since it's state-aware
        h = max(task_cost_sum, goal_cost, task_count)
        
        return h
