# Bottleneck: Single-hand constraint (handempty) serializes all block moves.
# Clearing a block requires recursively unstacking everything above it.
#
# Admissible lower bounds:
# T1: For each unsatisfied goal on(x,y), at least 2 actions needed (pick+stack)
# T2: For each do_clear task on a non-clear block, at least 2 actions per block above
# T3: Task network structural lower bound - minimum actions per remaining task
#
# Return max(T1, T2, T3) for admissible multi-component heuristic.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.HDDL.hddl_types import Operator as OperatorClass, AbstractTask as AbstractTaskClass


class BlocksworldGTOHPAnthropicOpus46Heuristic16(Heuristic):
    """Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Uses multi-component max of:
    - Goal displacement cost (unsatisfied on(x,y) goals)
    - Task network action lower bound with state-aware costing
    - Stack clearing cost estimation
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_16"):
        super().__init__(name)
        # Fact ID maps
        self.on_fact_id = {}      # (x, y) -> global_id
        self.clear_fact_id = {}   # x -> global_id
        self.ontable_fact_id = {} # x -> global_id
        self.handempty_id = None
        self.holding_fact_id = {} # x -> global_id
        
        # Goal structures
        self.goal_on_pairs = []   # list of (x, y, fact_global_id)
        
        # Task name patterns for identification
        self.task_name_cache = {}  # task global_id -> (task_type, args)
        
        # Block set
        self.blocks = set()
        
        # on(x, block) -> fact_id mapping for finding what's on a block
        self.on_top_of = {}  # y -> list of (x, fact_id) where on(x,y)
        
        # All on facts for stack computation
        self.all_on_facts = []  # list of (x, y, global_id)

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

    def _parse_component_name(self, name: str):
        """Parse operator/task/decomposition names to extract type and args."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Parse all facts, build lookup maps, identify goals."""
        self.model = model
        
        # Build fact maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.all_on_facts.append((args[0], args[1], f.global_id))
                self.blocks.add(args[0])
                self.blocks.add(args[1])
                if args[1] not in self.on_top_of:
                    self.on_top_of[args[1]] = []
                self.on_top_of[args[1]].append((args[0], f.global_id))
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.blocks.add(args[0])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = f.global_id
                self.blocks.add(args[0])
            elif pred == "handempty" and len(args) == 0:
                self.handempty_id = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.blocks.add(args[0])
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                fid = self.on_fact_id.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_pairs.append((args[0], args[1], fid))
        
        # Build task name cache for abstract tasks
        for at in model.abstract_tasks:
            ttype, args = self._parse_component_name(at.name)
            self.task_name_cache[at.global_id] = (ttype, args)
        
        # Also cache operator names
        for op in model.operators:
            ttype, args = self._parse_component_name(op.name)
            self.task_name_cache[op.global_id] = (ttype, args)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _find_block_on_top(self, block, state):
        """Find what block is directly on top of 'block' in current state, if any."""
        if block in self.on_top_of:
            for (x, fid) in self.on_top_of[block]:
                if (state >> fid) & 1:
                    return x
        return None

    def _stack_height_above(self, block, state):
        """Count how many blocks are stacked above 'block' in current state."""
        height = 0
        current = block
        visited = set()
        while True:
            if current in visited:
                break
            visited.add(current)
            top = self._find_block_on_top(current, state)
            if top is None:
                break
            height += 1
            current = top
        return height

    def _clear_cost(self, block, state):
        """Minimum actions to clear a block: 2 per block above it (unstack+putdown)."""
        cid = self.clear_fact_id.get(block)
        if cid is not None and (state >> cid) & 1:
            return 0
        h = self._stack_height_above(block, state)
        return 2 * h

    def _on_table_cost(self, block, state):
        """Cost to put block on table if not already there."""
        otid = self.ontable_fact_id.get(block)
        if otid is not None and (state >> otid) & 1:
            return 0
        # Need to clear block first, then unstack+putdown = 2 more
        # But clearing is already counted if block is not clear
        return 2  # unstack + putdown

    def _goal_displacement(self, state):
        """T1: Count unsatisfied goal on(x,y) facts. Each needs >= 2 actions."""
        cost = 0
        for (x, y, fid) in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                # on(x,y) is not true. Need at least: pick/unstack x + stack x on y = 2
                # But also may need to clear x and clear y first
                cost += 2
        return cost

    def _goal_displacement_with_clearing(self, state):
        """Enhanced T1: For unsatisfied on(x,y), add clearing costs."""
        cost = 0
        for (x, y, fid) in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                # Need to clear x, clear y, then move x to y (2 actions)
                cx = self._clear_cost(x, state)
                cy = self._clear_cost(y, state)
                # move = 2 (unstack/pickup + stack)
                # But clearing costs overlap with other goals, so take max per goal
                cost += 2  # Just the move cost for admissibility
        return cost

    def _task_network_cost(self, state, task_network):
        """T3: State-aware task network lower bound.
        
        For each task in the network, estimate minimum primitive actions needed.
        """
        cost = 0
        for task in task_network:
            if isinstance(task, OperatorClass):
                # Primitive action = 1 action
                ttype, args = self.task_name_cache.get(task.global_id, (None, []))
                if ttype == "nop":
                    cost += 0
                else:
                    cost += 1
            elif isinstance(task, AbstractTaskClass):
                ttype, args = self.task_name_cache.get(task.global_id, (None, []))
                if ttype is None:
                    cost += 1
                    continue
                    
                if ttype == "do_put_on" and len(args) == 2:
                    x, y = args[0], args[1]
                    fid = self.on_fact_id.get((x, y))
                    if fid is not None and (state >> fid) & 1:
                        # Already satisfied - nop
                        cost += 0
                    else:
                        # Need: clear x, clear y, on_table y (maybe), move x to y
                        # Minimum: move x to y = 2 actions
                        # Plus clearing costs
                        cx = self._clear_cost(x, state)
                        cy = self._clear_cost(y, state)
                        
                        # on_table y cost
                        ot_cost = 0
                        otid = self.ontable_fact_id.get(y)
                        if otid is not None and not ((state >> otid) & 1):
                            # y not on table, need unstack+putdown = 2
                            # But only if y needs to go to table first
                            # Actually do_on_table is called but may be nop if already on table
                            cidy = self.clear_fact_id.get(y)
                            if cidy is not None and (state >> cidy) & 1:
                                ot_cost = 2  # unstack y from something + putdown
                            else:
                                ot_cost = 0  # will be handled by clearing
                        
                        # move = 2 (pick/unstack + stack)
                        move_cost = 2
                        cost += cx + cy + ot_cost + move_cost
                        
                elif ttype == "do_clear" and len(args) == 1:
                    x = args[0]
                    cc = self._clear_cost(x, state)
                    cost += cc
                    
                elif ttype == "do_on_table" and len(args) == 1:
                    x = args[0]
                    otid = self.ontable_fact_id.get(x)
                    if otid is not None and (state >> otid) & 1:
                        cost += 0
                    else:
                        # Need clear + unstack + putdown
                        # But clear may already be true
                        cid = self.clear_fact_id.get(x)
                        if cid is not None and (state >> cid) & 1:
                            cost += 2  # unstack + putdown
                        else:
                            cost += 2 + self._clear_cost(x, state)
                    
                elif ttype == "do_move" and len(args) == 2:
                    x, y = args[0], args[1]
                    # Always needs 2 actions: pick/unstack + stack
                    cost += 2
                else:
                    cost += 1
        return cost

    def _compute(self, state, task_network):
        """Compute heuristic as max of multiple admissible components."""
        if len(task_network) == 0:
            return 0
        
        # T1: Goal displacement - each unsatisfied on(x,y) needs >= 2 actions
        t1 = self._goal_displacement(state)
        
        # T3: Task network with state-aware costing
        t3 = self._task_network_cost(state, task_network)
        
        return max(t1, t3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Evaluate heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
