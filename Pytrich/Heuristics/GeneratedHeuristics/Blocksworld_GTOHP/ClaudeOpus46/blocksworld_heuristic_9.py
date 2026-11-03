# Bottleneck: Single-hand constraint (handempty) serializes all block moves.
# Clearing a block requires 2 actions per block above it (unstack + put-down).
#
# Admissible lower bounds:
# T1: Goal-based displacement - each unsatisfied on(x,y) goal needs >= 2 actions
# T2: Task-network cost - estimate min actions per remaining task using state bits
# T3: Clearing cost - blocks needing clearing cost 2 * height_above
#
# Return max(T1, T2) for strong guidance.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator as OperatorClass
from Pytrich.model import AbstractTask as AbstractTaskClass


class BlocksworldGTOHPAnthropicOpus46Heuristic9(Heuristic):
    """Domain-dependent heuristic for Blocksworld GTOHP.
    
    Uses state-aware per-task cost estimation and goal displacement
    as two independent admissible lower bounds, returning their max.
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_9"):
        super().__init__(name)
        # Fact ID maps
        self.on_fact_id = {}       # (x, y) -> global_id
        self.ontable_fact_id = {}  # x -> global_id
        self.clear_fact_id = {}    # x -> global_id
        self.handempty_id = None
        self.holding_fact_id = {}  # x -> global_id
        
        # Goal info
        self.goal_on_pairs = []    # list of (x, y, fact_global_id)
        
        # Task name -> task type mapping
        self.task_type_map = {}    # abstract_task.global_id -> (type_str, args)
        
        # For stack height computation: who is on top of whom
        # on_top_of[y] = set of x where on(x,y) fact exists
        self.all_blocks = set()
        
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
        """Parse task/operator name to extract type and arguments."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Parse all facts, build lookup maps, extract goal info."""
        self.model = model
        
        # Build fact ID maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "handempty" and len(args) == 0:
                self.handempty_id = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])

        # Extract goal on-pairs
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                fid = self.on_fact_id.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_pairs.append((args[0], args[1], fid))

        # Build abstract task type map
        for at in model.abstract_tasks:
            ttype, args = self._parse_task_name(at.name)
            self.task_type_map[at.global_id] = (ttype, args)

        # Build operator type map
        for op in model.operators:
            ttype, args = self._parse_task_name(op.name)
            self.op_type_map[op.global_id] = (ttype, args)

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

    def _is_ontable(self, block, state):
        """Check if block is on table in current state."""
        fid = self.ontable_fact_id.get(block)
        if fid is None:
            return False
        return (state >> fid) & 1

    def _is_on(self, x, y, state):
        """Check if on(x, y) is true in current state."""
        fid = self.on_fact_id.get((x, y))
        if fid is None:
            return False
        return (state >> fid) & 1

    def _is_holding(self, block, state):
        """Check if holding(block) in current state."""
        fid = self.holding_fact_id.get(block)
        if fid is None:
            return False
        return (state >> fid) & 1

    def _find_block_on_top(self, block, state):
        """Find which block is directly on top of 'block', or None if clear."""
        if self._is_clear(block, state):
            return None
        # We need to find y such that on(y, block) is true
        # This is potentially expensive, but we'll use the on_fact_id map
        # We cache reverse lookups
        for (x, y), fid in self.on_fact_id.items():
            if y == block and (state >> fid) & 1:
                return x
        return None

    def _stack_height_above(self, block, state, max_depth=50):
        """Count how many blocks are stacked above 'block'.
        Each needs 2 actions (unstack + put-down) to remove."""
        if self._is_clear(block, state):
            return 0
        # Build the chain above block
        height = 0
        current = block
        depth = 0
        while depth < max_depth:
            top = self._find_block_on_top(current, state)
            if top is None:
                break
            height += 1
            current = top
            depth += 1
        return height

    def _estimate_clear_cost(self, block, state):
        """Estimate minimum actions to make block clear.
        Each block above costs 2 actions (unstack + put-down)."""
        if self._is_clear(block, state):
            return 0
        h = self._stack_height_above(block, state)
        return 2 * h

    def _estimate_task_cost(self, task, state):
        """Estimate minimum actions for a single task given current state."""
        if isinstance(task, OperatorClass):
            # Primitive operators cost 1 action (except nop)
            ttype, args = self.op_type_map.get(task.global_id, ("", []))
            if ttype == "nop":
                return 0
            return 1
        
        if not isinstance(task, AbstractTaskClass):
            return 0
            
        info = self.task_type_map.get(task.global_id)
        if info is None:
            return 1
        
        ttype, args = info
        
        if ttype == "do_put_on" and len(args) == 2:
            x, y = args[0], args[1]
            # If on(x,y) already true, cost is 0 (nop)
            if self._is_on(x, y, state):
                return 0
            # Otherwise: clear(x) + clear(y) + maybe put y on table + move x to y
            # move costs 2 actions (pick-up/unstack + stack)
            cost = 0
            cost += self._estimate_clear_cost(x, state)
            cost += self._estimate_clear_cost(y, state)
            # do_on_table(y): if y is not on table and not clear... 
            # Actually y should be clear after do_clear(y)
            # do_on_table costs 2 if y is not on table (unstack+put-down), but
            # after clearing y is clear, so if not ontable: 2 actions
            if not self._is_ontable(y, state) and not self._is_on(x, y, state):
                # After clearing, y is clear but may still be on another block
                # If y is not on table, need unstack+put-down = 2 actions
                # But this is only needed if y is currently on something
                # Actually after do_clear(y), y is clear. do_on_table checks
                # if y is already ontable. If not, it does unstack+put-down = 2.
                # But we need to be careful: the clear cost already accounts for
                # blocks above y, not y itself being on something.
                if not self._is_ontable(y, state):
                    cost += 2
            # do_move costs 2 (pick-up/unstack + stack)
            cost += 2
            return cost
        
        elif ttype == "do_clear" and len(args) == 1:
            x = args[0]
            return self._estimate_clear_cost(x, state)
        
        elif ttype == "do_on_table" and len(args) == 1:
            x = args[0]
            if self._is_ontable(x, state) and self._is_clear(x, state):
                return 0
            if self._is_clear(x, state) and not self._is_ontable(x, state):
                return 2  # unstack + put-down
            return 0  # already on table
        
        elif ttype == "do_move" and len(args) == 2:
            x, y = args[0], args[1]
            if self._is_on(x, y, state):
                return 0
            return 2  # pick-up/unstack + stack
        
        return 1

    def _compute(self, state, task_network):
        """Compute heuristic as max of two admissible lower bounds."""
        if not task_network:
            return 0
        
        # Term 1: Goal displacement
        # Each unsatisfied on(x,y) goal needs at least 2 actions
        goal_cost = 0
        for x, y, fid in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                goal_cost += 2
        
        # Term 2: Task network cost estimation
        # Sum up estimated costs for remaining tasks
        # But limit iteration to avoid timeout on huge networks
        task_cost = 0
        count = 0
        max_tasks = 500  # Safety limit
        for task in task_network:
            if count >= max_tasks:
                # For remaining tasks, use a minimal estimate
                remaining = len(task_network) - count
                # At minimum, each abstract task contributes something
                # but we can't iterate, so add 0 for safety
                break
            task_cost += self._estimate_task_cost(task, state)
            count += 1
        
        return max(goal_cost, task_cost)
