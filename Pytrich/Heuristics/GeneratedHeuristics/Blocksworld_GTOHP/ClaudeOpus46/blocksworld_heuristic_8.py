"""
Blocksworld GTOHP Domain-Dependent Heuristic

Bottleneck Analysis:
  The single constraint that most serializes work in Blocksworld is the requirement
  to clear blocks before moving them. Only one block can be held at a time (handempty),
  and to access a buried block, every block above it must be unstacked one-by-one.
  Each unstack+putdown costs 2 actions, and each move (pick-up/unstack + stack) costs 2.

Admissible Lower Bounds:
  Term 1 - Goal-distance lower bound: For each goal fact on(x,y) not currently true,
    we need at minimum 2 actions (pick-up/unstack x + stack x y). If x is not clear,
    we need additional 2 actions per block above x. If y is not clear, we need additional
    2 actions per block above y. We take the max over all unsatisfied goals since the
    hand is shared (serialized), but we can also sum disjoint costs.
    
  Term 2 - Task network lower bound: Each primitive action in the task network costs
    at least 1. Each abstract task that isn't trivially satisfied costs at least its
    minimum decomposition depth. We estimate based on state-aware analysis of remaining
    tasks.

  Term 3 - Unsatisfied goal count with clearing cost: Count of blocks that need to be
    moved, weighted by the clearing work needed. This captures the total volume of work.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic8(Heuristic):
    def __init__(self, name="blocksworld_anthropicopus46_heuristic_8"):
        super().__init__(name)
        # Maps to be built during initialization
        self.on_fact_id = {}       # (x, y) -> global_id for on(x,y)
        self.clear_fact_id = {}    # x -> global_id for clear(x)
        self.ontable_fact_id = {}  # x -> global_id for ontable(x)
        self.holding_fact_id = {}  # x -> global_id for holding(x)
        self.handempty_id = None   # global_id for handempty
        
        # Goal tracking
        self.goal_on = {}          # (x, y) -> global_id for goal on(x,y) facts
        self.goal_clear = {}       # x -> global_id
        self.goal_ontable = {}     # x -> global_id
        
        # Block list
        self.blocks = set()
        
        # Task name -> type mapping for fast task classification
        self.task_type_cache = {}
        
        # Operator type cache
        self.op_type_cache = {}
        
        # Abstract task type cache  
        self.abstract_task_cache = {}

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
        # No brackets - nullary predicate
        return name, []

    def _parse_task_name(self, name: str):
        """Parse operator/task name to extract type and arguments."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Build all lookup maps during initialization."""
        self.model = model
        
        # Build fact maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.blocks.add(args[0])
                self.blocks.add(args[1])
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
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                gid = self.on_fact_id.get((args[0], args[1]))
                if gid is not None:
                    self.goal_on[(args[0], args[1])] = gid
            elif pred == "clear" and len(args) == 1:
                gid = self.clear_fact_id.get(args[0])
                if gid is not None:
                    self.goal_clear[args[0]] = gid
            elif pred == "ontable" and len(args) == 1:
                gid = self.ontable_fact_id.get(args[0])
                if gid is not None:
                    self.goal_ontable[args[0]] = gid
        
        # Cache operator types
        for op in model.operators:
            ttype, args = self._parse_task_name(op.name)
            self.op_type_cache[op.global_id] = (ttype, args)
        
        # Cache abstract task types
        for at in model.abstract_tasks:
            ttype, args = self._parse_task_name(at.name)
            self.abstract_task_cache[at.global_id] = (ttype, args)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_block_above(self, block, state):
        """Find which block is on top of 'block' in current state, or None if clear."""
        clear_id = self.clear_fact_id.get(block)
        if clear_id is not None and (state >> clear_id) & 1:
            return None
        # Need to find which block is on top of this one
        # We check on(y, block) for all y
        for (x, y), fid in self.on_fact_id.items():
            if y == block and (state >> fid) & 1:
                return x
        return None

    def _stack_height_above(self, block, state, max_depth=50):
        """Count how many blocks are stacked above 'block'.
        Returns the count of blocks that need to be removed."""
        clear_id = self.clear_fact_id.get(block)
        if clear_id is not None and (state >> clear_id) & 1:
            return 0
        
        count = 0
        current = block
        depth = 0
        while depth < max_depth:
            above = self._get_block_above(current, state)
            if above is None:
                break
            count += 1
            current = above
            depth += 1
        return count

    def _cost_to_clear(self, block, state):
        """Estimate minimum actions to make 'block' clear.
        Each block above needs unstack + put-down = 2 actions."""
        height = self._stack_height_above(block, state)
        return 2 * height

    def _is_on(self, x, y, state):
        """Check if on(x,y) is true in state."""
        fid = self.on_fact_id.get((x, y))
        if fid is None:
            return False
        return (state >> fid) & 1

    def _is_clear(self, x, state):
        """Check if clear(x) is true in state."""
        fid = self.clear_fact_id.get(x)
        if fid is None:
            return False
        return (state >> fid) & 1

    def _is_ontable(self, x, state):
        """Check if ontable(x) is true in state."""
        fid = self.ontable_fact_id.get(x)
        if fid is None:
            return False
        return (state >> fid) & 1

    def _estimate_task_cost(self, task, state):
        """Estimate the minimum cost of accomplishing a single task given current state.
        Uses state-aware analysis for realistic lower bounds."""
        from Pytrich.HDDL.hddl_abstract_task import AbstractTask as ATClass
        from Pytrich.HDDL.hddl_operator import Operator as OpClass
        
        if isinstance(task, OpClass):
            # Primitive tasks cost 1 each (unless nop)
            ttype, args = self.op_type_cache.get(task.global_id, (None, []))
            if ttype == "nop":
                return 0
            return 1
        
        if isinstance(task, ATClass):
            ttype, args = self.abstract_task_cache.get(task.global_id, (None, []))
            
            if ttype == "do_put_on" and len(args) == 2:
                x, y = args[0], args[1]
                # If on(x,y) already true, cost is 0 (method m0 applies: nop)
                if self._is_on(x, y, state):
                    return 0
                # Otherwise: need to clear x, clear y, put y on table if needed, move x to y
                # Minimum: clear x + clear y + move x onto y
                # move = 2 actions (pick-up/unstack + stack)
                cost = 2  # base cost for the final move
                # Cost to clear x
                if not self._is_clear(x, state):
                    cost += self._cost_to_clear(x, state)
                # Cost to clear y
                if not self._is_clear(y, state):
                    cost += self._cost_to_clear(y, state)
                # If y is not on table and we need to put it on table first
                # (method m1 calls do_on_table y which may need unstack+putdown)
                # But do_on_table only matters if y isn't already on table
                # This adds 2 more if y needs to be moved to table
                if not self._is_ontable(y, state) and self._is_clear(y, state):
                    # y needs unstack + put-down = 2
                    cost += 2
                return cost
            
            elif ttype == "do_clear" and len(args) >= 1:
                x = args[0]
                if self._is_clear(x, state):
                    return 0
                return self._cost_to_clear(x, state)
            
            elif ttype == "do_on_table" and len(args) >= 1:
                x = args[0]
                if self._is_ontable(x, state):
                    return 0
                # Need clear x first, then unstack + put-down
                if self._is_clear(x, state):
                    return 2  # unstack + put-down
                return self._cost_to_clear(x, state) + 2
            
            elif ttype == "do_move" and len(args) >= 2:
                x, y = args[0], args[1]
                # Precondition requires clear x, clear y, handempty
                # Cost: 2 (unstack/pick-up + stack)
                return 2
            
            # Unknown abstract task - return minimum decomposition estimate
            # Check decompositions for minimum subtask count
            min_cost = 1
            if hasattr(task, 'decompositions') and task.decompositions:
                best = float('inf')
                for dec in task.decompositions:
                    if dec.applicable(state):
                        # Count non-nop tasks in decomposition
                        dec_cost = 0
                        for subtask in dec.task_network:
                            st_type, st_args = self._parse_task_name(subtask.name)
                            if st_type != "nop":
                                dec_cost += 1
                        best = min(best, max(dec_cost, 1))
                if best < float('inf'):
                    min_cost = best
            return min_cost
        
        return 1

    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple admissible terms."""
        if len(task_network) == 0:
            # Check if goals are satisfied
            if (state & self.model.goals) == self.model.goals:
                return 0
            return 1
        
        # Term 1: Task network cost estimation (state-aware)
        # Sum the estimated cost of each task, taking state into account
        task_cost = 0
        for task in task_network:
            task_cost += self._estimate_task_cost(task, state)
        
        # Term 2: Unsatisfied goal-based lower bound
        # For each goal on(x,y) not true, we need at least 2 actions
        # We count the total minimum actions needed
        goal_cost = 0
        for (x, y), fid in self.goal_on.items():
            if not ((state >> fid) & 1):
                # Goal on(x,y) not satisfied - minimum 2 actions
                base = 2
                # If x is not clear, add clearing cost
                if not self._is_clear(x, state):
                    base += self._cost_to_clear(x, state)
                # If y is not clear, add clearing cost  
                if not self._is_clear(y, state):
                    base += self._cost_to_clear(y, state)
                goal_cost = max(goal_cost, base)
        
        # Term 3: Count of unsatisfied goals as a volume measure
        unsat_count = 0
        for (x, y), fid in self.goal_on.items():
            if not ((state >> fid) & 1):
                unsat_count += 2  # Each unsatisfied on-goal needs at least 2 actions
        for x, fid in self.goal_clear.items():
            if not ((state >> fid) & 1):
                unsat_count += 2
        for x, fid in self.goal_ontable.items():
            if not ((state >> fid) & 1):
                unsat_count += 2
        
        # Term 4: Goal stability - goals currently true that will be disrupted
        # Check if any currently-true goal on(x,y) has x or y involved in
        # an unsatisfied goal, meaning it might need to be undone and redone
        disruption_cost = 0
        for (x, y), fid in self.goal_on.items():
            if (state >> fid) & 1:
                # This goal is currently true - but will it stay true?
                # Check if anything needs to go on top of x (another goal on(z,x))
                # or if y needs to be moved for another goal
                for (x2, y2), fid2 in self.goal_on.items():
                    if (x2, y2) != (x, y) and not ((state >> fid2) & 1):
                        # Check if clearing x2 or y2 would disturb on(x,y)
                        if x2 == x or y2 == x:
                            # Another unsatisfied goal involves x
                            disruption_cost += 1
                            break
        
        h = max(task_cost, goal_cost, unsat_count + disruption_cost)
        return h
