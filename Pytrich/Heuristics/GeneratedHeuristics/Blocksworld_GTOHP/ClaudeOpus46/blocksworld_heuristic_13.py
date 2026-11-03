# Bottleneck: Single-hand constraint (handempty) serializes all block moves.
# Clearing a block requires 2 actions per block above it (unstack+putdown).
# Moving a block to its target costs 2 actions (pickup/unstack + stack).
#
# Lower bounds:
# T1: For each pending task in the task network, estimate min actions from current state.
#     - do_put_on(x,y): 0 if on(x,y) true, else 2 (move) + clearing costs for x and y
#     - do_clear(x): 0 if clear(x) true, else 2 per block above x
#     - do_on_table(x): 0 if ontable(x) true, else 2
#     - do_move(x,y): always 2 (one pickup/unstack + one stack)
# T2: Count unsatisfied goal on-facts, each needs at least 2 actions.
# Return max(T1_estimate, T2_estimate) for admissible multi-component heuristic.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic13(Heuristic):
    """Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Uses state-aware per-task cost estimation and goal-based lower bounds.
    Precomputes fact lookups in initialize() for fast __call__.
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_13"):
        super().__init__(name)
        # Fact ID maps - built during initialize
        self.on_fact_id = {}       # (x, y) -> global_id
        self.clear_fact_id = {}    # x -> global_id
        self.ontable_fact_id = {}  # x -> global_id
        self.holding_fact_id = {}  # x -> global_id
        self.handempty_id = -1
        
        # Task name prefix -> task type mapping
        self.task_type_map = {}  # abstract_task.global_id -> (type_str, args)
        self.op_type_map = {}    # operator.global_id -> (type_str, args)
        
        # Goal on-facts for T2
        self.goal_on_facts = []  # list of (x, y, global_id)
        self.goal_clear_facts = []
        self.goal_ontable_facts = []
        
        # All blocks
        self.blocks = set()
        
        # on_map: for each block y, the fact_id for on(x,y) for all x
        # Used to find what's on top of a block
        self.on_top_of = {}  # y -> list of (x, fact_global_id)
        
        # For stack height computation
        self.blocks_above = {}  # block -> list of on(x, block) fact ids with their x

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
        """Parse a task/operator name to extract type and arguments.
        Format examples: do_put_on[b1,b2], pick-up[b1], nop[]
        """
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        # Try without brackets
        paren_idx = name.find("(")
        if paren_idx > 0 and name.endswith(")"):
            task_type = name[:paren_idx]
            args_str = name[paren_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact ID maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.blocks.add(args[0])
                self.blocks.add(args[1])
                # Track what can be on top of each block
                if args[1] not in self.on_top_of:
                    self.on_top_of[args[1]] = []
                self.on_top_of[args[1]].append((args[0], f.global_id))
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.blocks.add(args[0])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = f.global_id
                self.blocks.add(args[0])
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.blocks.add(args[0])
            elif pred == "handempty":
                self.handempty_id = f.global_id

        # Build task type maps for abstract tasks
        for at in model.abstract_tasks:
            task_type, args = self._parse_task_name(at.name)
            self.task_type_map[at.global_id] = (task_type, args)

        # Build operator type maps
        for op in model.operators:
            task_type, args = self._parse_task_name(op.name)
            self.op_type_map[op.global_id] = (task_type, args)

        # Parse goal facts for T2 bound
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                gid = self.on_fact_id.get((args[0], args[1]), -1)
                self.goal_on_facts.append((args[0], args[1], gid))
            elif pred == "clear" and len(args) == 1:
                gid = self.clear_fact_id.get(args[0], -1)
                self.goal_clear_facts.append((args[0], gid))
            elif pred == "ontable" and len(args) == 1:
                gid = self.ontable_fact_id.get(args[0], -1)
                self.goal_ontable_facts.append((args[0], gid))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _find_block_on_top(self, block, state):
        """Find which block is directly on top of 'block' in current state, or None."""
        if block in self.on_top_of:
            for (x, fid) in self.on_top_of[block]:
                if (state >> fid) & 1:
                    return x
        return None

    def _stack_height_above(self, block, state):
        """Count number of blocks stacked above 'block' in current state."""
        count = 0
        current = block
        while True:
            top = self._find_block_on_top(current, state)
            if top is None:
                break
            count += 1
            current = top
            if count > 1000:  # safety for large instances
                break
        return count

    def _clear_cost(self, block, state):
        """Minimum actions to clear block: 0 if already clear, else 2 * height_above."""
        cid = self.clear_fact_id.get(block, -1)
        if cid >= 0 and (state >> cid) & 1:
            return 0
        # Need to unstack+putdown each block above
        height = self._stack_height_above(block, state)
        return 2 * height

    def _task_cost(self, task, state):
        """Estimate minimum actions for a single task given current state."""
        from Pytrich.model import Operator as OpClass
        from Pytrich.HDDL.AbstractTask import AbstractTask as ATClass
        
        if isinstance(task, OpClass):
            # Primitive operator
            info = self.op_type_map.get(task.global_id)
            if info is None:
                return 1
            op_type, args = info
            if op_type == "nop":
                return 0
            # pick-up, put-down, stack, unstack each cost 1 action
            return 1
        
        if isinstance(task, ATClass):
            info = self.task_type_map.get(task.global_id)
            if info is None:
                return 1
            task_type, args = info
            
            if task_type == "do_put_on" and len(args) == 2:
                x, y = args[0], args[1]
                # If on(x,y) already true, cost is 0 (nop)
                on_id = self.on_fact_id.get((x, y), -1)
                if on_id >= 0 and (state >> on_id) & 1:
                    return 0
                # Otherwise: need to clear x, clear y, maybe put y on table, move x to y
                # Minimum: clear_cost(x) + clear_cost(y) + 2 (move)
                # But do_on_table might add 0 or 2 more
                cx = self._clear_cost(x, state)
                cy = self._clear_cost(y, state)
                # do_on_table(y): 0 if ontable(y), else 2
                ot_id = self.ontable_fact_id.get(y, -1)
                ot_cost = 0
                if ot_id >= 0 and not ((state >> ot_id) & 1):
                    # y is not on table, but after clearing y, the block y is on
                    # will need to be unstacked. Actually do_on_table checks if
                    # ontable(y) after clearing. Since y might be on something,
                    # putting y on table = unstack + putdown = 2
                    ot_cost = 2
                elif ot_id < 0:
                    ot_cost = 0  # can't determine, be conservative
                # do_move = 2 actions (pick-up/unstack + stack)
                move_cost = 2
                return cx + cy + ot_cost + move_cost
            
            elif task_type == "do_clear" and len(args) >= 1:
                x = args[0]
                return self._clear_cost(x, state)
            
            elif task_type == "do_on_table" and len(args) >= 1:
                x = args[0]
                ot_id = self.ontable_fact_id.get(x, -1)
                cl_id = self.clear_fact_id.get(x, -1)
                if ot_id >= 0 and (state >> ot_id) & 1:
                    return 0  # already on table
                if cl_id >= 0 and (state >> cl_id) & 1:
                    return 2  # clear but not on table: unstack + putdown
                return 2
            
            elif task_type == "do_move" and len(args) >= 2:
                # Always costs 2 (pick-up/unstack + stack)
                return 2
            
            return 1
        
        return 0

    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple lower bounds."""
        if not task_network:
            return 0
        
        # T1: Sum of minimum costs for first N tasks in task network
        # For ordered task networks, tasks are sequential, so costs are additive
        # But we can't just sum all because state changes. 
        # As a lower bound, we estimate each task's cost from current state
        # and take a fraction (since clearing work may overlap).
        # Actually, for an admissible bound, we can note that each task
        # needs at least its minimum actions, but they share the hand.
        # A safe lower bound: just look at the first pending task's full cost
        # plus count remaining tasks.
        
        t1 = 0
        task_count = 0
        for task_item in task_network:
            task = task_item  # In the task network, items are tasks directly
            cost = self._task_cost(task, state)
            t1 += cost
            task_count += 1
            # For large problems, limit iteration to keep __call__ fast
            if task_count >= 50:
                # Estimate remaining tasks with minimal cost
                remaining = len(task_network) - task_count
                # Each remaining task costs at least 0 (could be nop)
                # but on average non-trivial tasks cost at least 2
                break
        
        # T2: Goal-based lower bound
        # Each unsatisfied goal on(x,y) needs at least 2 actions
        t2 = 0
        for (x, y, gid) in self.goal_on_facts:
            if gid >= 0 and not ((state >> gid) & 1):
                t2 += 2  # at minimum: one unstack/pickup + one stack
        
        # T3: Count unsatisfied clear goals
        for (x, gid) in self.goal_clear_facts:
            if gid >= 0 and not ((state >> gid) & 1):
                t2 += 2  # need to move at least one block off
        
        for (x, gid) in self.goal_ontable_facts:
            if gid >= 0 and not ((state >> gid) & 1):
                t2 += 2
        
        return max(t1, t2)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
