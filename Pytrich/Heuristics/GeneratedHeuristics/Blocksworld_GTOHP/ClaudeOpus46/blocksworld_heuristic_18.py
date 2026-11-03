# Bottleneck: Single-hand constraint serializes all block movements. Clearing blocks
# above targets dominates plan length. Each block above a target needs unstack+put-down (2 actions).
#
# Admissible lower bounds:
# T1: For each unsatisfied on(x,y) goal, at least 2 actions needed (pick/unstack + stack)
# T2: Stack-height clearing estimate - blocks above targets need 2 actions each to remove
# T3: Task network lower bound - each non-trivial compound task needs minimum actions

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator as OperatorClass
from Pytrich.model import AbstractTask as AbstractTaskClass


class BlocksworldGTOHPAnthropicOpus46Heuristic18(Heuristic):
    """Domain-dependent heuristic for Blocksworld GTOHP.
    
    Uses max of multiple admissible lower bounds:
    1. Unsatisfied goal facts (each needs >= 2 actions)
    2. Stack clearing costs based on current state
    3. Task network structural lower bound
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_18"):
        super().__init__(name)
        # Fact maps
        self.on_fact_id = {}      # (x, y) -> global_id
        self.clear_fact_id = {}   # x -> global_id
        self.ontable_fact_id = {} # x -> global_id
        self.holding_fact_id = {} # x -> global_id
        self.handempty_fact_id = None
        
        # Goal structures
        self.goal_on_pairs = []   # list of (x, y, fact_global_id)
        
        # Block info
        self.all_blocks = set()
        
        # Reverse map: for each block y, what block x has on(x,y) fact
        self.on_top_of = {}  # y -> list of (x, fact_id)
        
        # Task name parsing cache
        self.task_name_cache = {}
        
        # Operator/task type identification
        self.nop_ids = set()
        self.pickup_ids = set()
        self.putdown_ids = set()
        self.stack_ids = set()
        self.unstack_ids = set()
        
        # Abstract task type identification
        self.do_put_on_ids = set()
        self.do_clear_ids = set()
        self.do_on_table_ids = set()
        self.do_move_ids = set()
        
        # Task parameter extraction
        self.task_params = {}  # global_id -> (type_str, args)

    def _parse_fact_name(self, name):
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

    def _parse_component_name(self, name):
        """Parse operator or task name to get type and arguments."""
        name = name.strip()
        paren_idx = name.find("(")
        if paren_idx >= 0 and name.endswith(")"):
            task_type = name[:paren_idx].strip()
            args_str = name[paren_idx+1:-1].strip()
            args = [a.strip() for a in args_str.split(",")] if args_str else []
            return task_type, args
        # Try bracket format
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx].strip()
            args_str = name[bracket_idx+1:-1].strip()
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
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "handempty":
                self.handempty_fact_id = f.global_id
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                fid = self.on_fact_id.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_pairs.append((args[0], args[1], fid))
        
        # Classify operators
        for op in model.operators:
            ttype, args = self._parse_component_name(op.name)
            self.task_params[op.global_id] = (ttype, args)
            ttype_lower = ttype.lower().replace("-", "").replace("_", "")
            if "nop" in ttype_lower:
                self.nop_ids.add(op.global_id)
            elif "pickup" in ttype_lower:
                self.pickup_ids.add(op.global_id)
            elif "putdown" in ttype_lower:
                self.putdown_ids.add(op.global_id)
            elif "stack" in ttype_lower and "unstack" not in ttype_lower:
                self.stack_ids.add(op.global_id)
            elif "unstack" in ttype_lower:
                self.unstack_ids.add(op.global_id)
        
        # Classify abstract tasks
        for at in model.abstract_tasks:
            ttype, args = self._parse_component_name(at.name)
            self.task_params[at.global_id] = (ttype, args)
            ttype_lower = ttype.lower().replace("-", "").replace("_", "")
            if "doputon" in ttype_lower:
                self.do_put_on_ids.add(at.global_id)
            elif "doclear" in ttype_lower:
                self.do_clear_ids.add(at.global_id)
            elif "doontable" in ttype_lower:
                self.do_on_table_ids.add(at.global_id)
            elif "domove" in ttype_lower:
                self.do_move_ids.add(at.global_id)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_stack_height_above(self, block, state):
        """Count how many blocks are stacked above 'block' in current state.
        Uses on_top_of map and bit checks. Returns the count."""
        # Check if block is already clear
        if block in self.clear_fact_id:
            if (state >> self.clear_fact_id[block]) & 1:
                return 0
        
        # Find what's on top of this block
        if block not in self.on_top_of:
            return 0
        
        for top_block, fact_id in self.on_top_of[block]:
            if (state >> fact_id) & 1:
                # top_block is on block
                return 1 + self._get_stack_height_above(top_block, state)
        return 0

    def _estimate_clear_cost(self, block, state):
        """Estimate minimum actions to make block clear.
        Each block above needs unstack + put-down = 2 actions."""
        height = self._get_stack_height_above(block, state)
        return height * 2

    def _compute_goal_term(self, state):
        """T1: For each unsatisfied on(x,y) goal, estimate the minimum cost.
        - If on(x,y) already true: 0
        - Otherwise: need to clear x + clear y + move x onto y (2 actions for move)
        This is admissible because each unsatisfied goal needs at least these actions.
        We take max over individual goals to stay admissible (goals share clearing work)."""
        total_unsatisfied = 0
        max_single_goal_cost = 0
        
        for x, y, fid in self.goal_on_pairs:
            if (state >> fid) & 1:
                continue
            # Goal not satisfied - estimate cost
            total_unsatisfied += 1
            cost = 2  # minimum: pick/unstack x + stack x on y
            
            # Add clearing cost for x
            if x in self.clear_fact_id:
                if not ((state >> self.clear_fact_id[x]) & 1):
                    cost += self._estimate_clear_cost(x, state)
            
            # Add clearing cost for y  
            if y in self.clear_fact_id:
                if not ((state >> self.clear_fact_id[y]) & 1):
                    cost += self._estimate_clear_cost(y, state)
            
            max_single_goal_cost = max(max_single_goal_cost, cost)
        
        # Admissible: each unsatisfied goal needs at least 2 actions,
        # but they're serialized (one hand), so sum of minimum 2 per goal is admissible
        # Actually for admissibility with shared subgoals, use max per goal
        # But we can also use: total_unsatisfied * 2 is admissible because
        # each on(x,y) requires a unique stack action
        return total_unsatisfied * 2

    def _compute_task_network_term(self, state, task_network):
        """T3: Estimate cost from remaining task network.
        For each task, estimate its minimum decomposition cost based on state."""
        if not task_network:
            return 0
        
        total = 0
        for task in task_network:
            gid = task.global_id
            
            # Primitive operators
            if isinstance(task, OperatorClass):
                if gid not in self.nop_ids:
                    total += 1
                continue
            
            # Abstract tasks
            if not isinstance(task, AbstractTaskClass):
                continue
            
            params = self.task_params.get(gid)
            if params is None:
                total += 1
                continue
            
            ttype, args = params
            ttype_lower = ttype.lower().replace("-", "").replace("_", "")
            
            if "doputon" in ttype_lower and len(args) >= 2:
                x, y = args[0], args[1]
                # Check if on(x,y) is already true -> can use m0 (nop)
                fid = self.on_fact_id.get((x, y))
                if fid is not None and (state >> fid) & 1:
                    total += 0  # nop
                else:
                    # Need: do_clear(x) + do_clear(y) + do_on_table(y) + do_move(x,y)
                    # do_move needs 2 actions minimum (pick/unstack + stack)
                    cost = 2  # minimum for do_move
                    
                    # Clear x cost
                    if x in self.clear_fact_id:
                        if not ((state >> self.clear_fact_id[x]) & 1):
                            cost += self._estimate_clear_cost(x, state)
                    
                    # Clear y cost
                    if y in self.clear_fact_id:
                        if not ((state >> self.clear_fact_id[y]) & 1):
                            cost += self._estimate_clear_cost(y, state)
                    
                    # on_table y: if y not on table and not clear-then-ontable
                    if y in self.ontable_fact_id:
                        if not ((state >> self.ontable_fact_id[y]) & 1):
                            cost += 2  # unstack + put-down
                    
                    total += cost
                    
            elif "doclear" in ttype_lower and len(args) >= 1:
                x = args[0]
                if x in self.clear_fact_id:
                    if (state >> self.clear_fact_id[x]) & 1:
                        total += 0  # already clear, nop
                    else:
                        total += self._estimate_clear_cost(x, state)
                        
            elif "doontable" in ttype_lower and len(args) >= 1:
                x = args[0]
                if x in self.ontable_fact_id:
                    if (state >> self.ontable_fact_id[x]) & 1:
                        total += 0  # already on table, nop
                    else:
                        total += 2  # unstack + put-down
                        
            elif "domove" in ttype_lower and len(args) >= 2:
                total += 2  # pick-up/unstack + stack minimum
            else:
                total += 1
        
        return total

    def _compute_clearing_term(self, state, task_network):
        """T2: Estimate total clearing work needed.
        For blocks that need to be cleared (referenced in tasks), sum clearing costs.
        Admissible because clearing is serialized (one hand)."""
        blocks_to_clear = set()
        
        for task in task_network:
            gid = task.global_id
            if not isinstance(task, AbstractTaskClass):
                continue
            
            params = self.task_params.get(gid)
            if params is None:
                continue
            
            ttype, args = params
            ttype_lower = ttype.lower().replace("-", "").replace("_", "")
            
            if "doclear" in ttype_lower and len(args) >= 1:
                x = args[0]
                if x in self.clear_fact_id:
                    if not ((state >> self.clear_fact_id[x]) & 1):
                        blocks_to_clear.add(x)
            elif "doputon" in ttype_lower and len(args) >= 2:
                x, y = args[0], args[1]
                fid = self.on_fact_id.get((x, y))
                if fid is not None and (state >> fid) & 1:
                    continue  # already satisfied
                if x in self.clear_fact_id:
                    if not ((state >> self.clear_fact_id[x]) & 1):
                        blocks_to_clear.add(x)
                if y in self.clear_fact_id:
                    if not ((state >> self.clear_fact_id[y]) & 1):
                        blocks_to_clear.add(y)
        
        total_clearing = 0
        for block in blocks_to_clear:
            total_clearing += self._estimate_clear_cost(block, state)
        
        return total_clearing

    def _compute(self, state, task_network):
        if not task_network:
            return 0
        
        t1 = self._compute_goal_term(state)
        t2 = self._compute_task_network_term(state, task_network)
        t3 = self._compute_clearing_term(state, task_network)
        
        return max(t1, t2, t3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
