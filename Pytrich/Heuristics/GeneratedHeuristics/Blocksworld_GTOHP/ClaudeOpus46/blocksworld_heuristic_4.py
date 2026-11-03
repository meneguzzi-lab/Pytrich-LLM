# Blocksworld_GTOHP Heuristic
# Bottleneck: Single-hand constraint (handempty) serializes all block moves.
# Clearing a block with height h above it costs 2h actions minimum.
# Each do_put_on(x,y) where on(x,y) is not true costs at least 2 actions (move x onto y).
#
# Admissible lower bounds:
# T1: Goal-based displacement - each unsatisfied on(x,y) goal needs >= 2 actions
# T2: Task-network aware cost - estimate per-task minimum cost using state
# T3: Blocks needing clearing - stack height based cost

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic4(Heuristic):
    """Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Uses multi-component max of:
    1. Goal displacement cost
    2. Task network estimated cost with state-aware per-task bounds
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_4"):
        super().__init__(name)
        self.on_fact_ids = {}       # (x_arg, y_arg) -> global_id
        self.ontable_fact_ids = {}  # x_arg -> global_id
        self.clear_fact_ids = {}    # x_arg -> global_id
        self.handempty_id = None
        self.holding_fact_ids = {}  # x_arg -> global_id
        
        # Goal related
        self.goal_on_pairs = []     # list of (x, y, fact_global_id)
        self.goal_ontable = []      # list of (x, fact_global_id)
        self.goal_clear = []        # list of (x, fact_global_id)
        
        # Task name indices for fast lookup
        self.task_name_to_type = {}  # abstract task global_id -> type string
        
        # For stack height computation: which block is on top of x?
        # on_fact for each (top, bottom) pair
        self.blocks = set()
        
        # Operator type detection
        self.nop_op_ids = set()
        
        # Abstract task type and args cache
        self.task_info_cache = {}  # global_id -> (type_str, args)
        
        # Precomputed: for each block, the fact id of on(?, block) - who is on top
        self.on_top_of = {}  # bottom_block -> list of (top_block, fact_id)

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
        """Parse a task/operator name to extract type and arguments."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def _get_stack_height_above(self, block, state):
        """Count how many blocks are stacked above 'block' in current state.
        Uses iterative approach following the chain upward."""
        height = 0
        current = block
        # Find what's on top of current
        while True:
            found_top = None
            if current in self.on_top_of:
                for top_block, fid in self.on_top_of[current]:
                    if (state >> fid) & 1:
                        found_top = top_block
                        break
            if found_top is None:
                break
            height += 1
            current = found_top
            if height > 1000:  # safety
                break
        return height

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts and build lookup maps
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
                self.blocks.add(args[0])
                self.blocks.add(args[1])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_ids[args[0]] = fact.global_id
                self.blocks.add(args[0])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_ids[args[0]] = fact.global_id
                self.blocks.add(args[0])
            elif pred == "handempty" and len(args) == 0:
                self.handempty_id = fact.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_ids[args[0]] = fact.global_id
                self.blocks.add(args[0])
        
        # Build on_top_of map: for each bottom block, which (top, fact_id) pairs exist
        for (top, bottom), fid in self.on_fact_ids.items():
            if bottom not in self.on_top_of:
                self.on_top_of[bottom] = []
            self.on_top_of[bottom].append((top, fid))
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                fid = self.on_fact_ids.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_pairs.append((args[0], args[1], fid))
            elif pred == "ontable" and len(args) == 1:
                fid = self.ontable_fact_ids.get(args[0])
                if fid is not None:
                    self.goal_ontable.append((args[0], fid))
            elif pred == "clear" and len(args) == 1:
                fid = self.clear_fact_ids.get(args[0])
                if fid is not None:
                    self.goal_clear.append((args[0], fid))
        
        # Cache abstract task info
        for at in model.abstract_tasks:
            ttype, targs = self._parse_task_name(at.name)
            self.task_info_cache[at.global_id] = (ttype, targs)
        
        # Cache operator info
        for op in model.operators:
            ttype, targs = self._parse_task_name(op.name)
            if ttype == "nop":
                self.nop_op_ids.add(op.global_id)
            self.task_info_cache[op.global_id] = (ttype, targs)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _estimate_task_cost(self, task_gid, state):
        """Estimate minimum cost for a single task given current state."""
        info = self.task_info_cache.get(task_gid)
        if info is None:
            return 1
        
        ttype, args = info
        
        if ttype == "nop":
            return 0
        
        if ttype == "pick-up":
            return 1
        if ttype == "put-down":
            return 1
        if ttype == "stack":
            return 1
        if ttype == "unstack":
            return 1
        
        if ttype == "do_clear":
            if len(args) >= 1:
                x = args[0]
                clear_fid = self.clear_fact_ids.get(x)
                if clear_fid is not None and (state >> clear_fid) & 1:
                    return 0  # already clear
                # Need to clear: at least 2 actions per block above (unstack + put-down)
                # plus recursively clearing the top block
                height = self._get_stack_height_above(x, state)
                return 2 * height
            return 1
        
        if ttype == "do_on_table":
            if len(args) >= 1:
                x = args[0]
                ontable_fid = self.ontable_fact_ids.get(x)
                if ontable_fid is not None and (state >> ontable_fid) & 1:
                    return 0  # already on table
                # Need unstack + put-down = 2 actions minimum
                return 2
            return 1
        
        if ttype == "do_move":
            if len(args) >= 2:
                x = args[0]
                y = args[1]
                # do_move requires clear(x), clear(y), handempty
                # then pick-up/unstack + stack = 2 actions
                return 2
            return 2
        
        if ttype == "do_put_on":
            if len(args) >= 2:
                x = args[0]
                y = args[1]
                on_fid = self.on_fact_ids.get((x, y))
                if on_fid is not None and (state >> on_fid) & 1:
                    return 0  # already on(x,y)
                
                # Need to: clear x, clear y, possibly put y on table, move x onto y
                # Minimum: clear_x_cost + clear_y_cost + move_cost(2)
                cost = 2  # move cost minimum
                
                # Cost to clear x
                clear_x_fid = self.clear_fact_ids.get(x)
                if clear_x_fid is not None and not ((state >> clear_x_fid) & 1):
                    hx = self._get_stack_height_above(x, state)
                    cost += 2 * hx
                
                # Cost to clear y
                clear_y_fid = self.clear_fact_ids.get(y)
                if clear_y_fid is not None and not ((state >> clear_y_fid) & 1):
                    hy = self._get_stack_height_above(y, state)
                    cost += 2 * hy
                
                # If y is not on table, do_on_table adds 2
                ontable_y_fid = self.ontable_fact_ids.get(y)
                if ontable_y_fid is not None and not ((state >> ontable_y_fid) & 1):
                    cost += 2
                
                return cost
            return 2
        
        return 1

    def _goal_displacement_cost(self, state):
        """Count unsatisfied goal on-facts. Each needs at least 2 actions."""
        cost = 0
        for x, y, fid in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                cost += 2
        for x, fid in self.goal_ontable:
            if not ((state >> fid) & 1):
                cost += 2
        for x, fid in self.goal_clear:
            if not ((state >> fid) & 1):
                cost += 2
        return cost

    def _task_network_cost(self, state, task_network):
        """Estimate cost from remaining task network using state-aware per-task bounds."""
        cost = 0
        for task in task_network:
            cost += self._estimate_task_cost(task.global_id, state)
        return cost

    def _compute(self, state, task_network):
        if len(task_network) == 0:
            return 0
        
        t1 = self._goal_displacement_cost(state)
        t2 = self._task_network_cost(state, task_network)
        
        return max(t1, t2)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
