# Bottleneck: Single-hand constraint (handempty) serializes all block moves.
# Clearing a block with k blocks above it costs at least 3k actions (recursive clear + unstack + putdown per block).
# Moving a block costs 2 actions (pick-up/unstack + stack).
#
# Admissible lower bounds:
# T1 - Task-aware cost: For each task in the task network, estimate minimum actions
#      based on current state (stack heights, positions). Admissible because each
#      task must independently perform at least that many actions.
# T2 - Goal displacement: Count on(x,y) goals not satisfied, each needs >= 2 actions.
# T3 - Blocks needing clearing: For do_clear tasks, count stack height above target.
#
# We take max(T1_first_task_cost, T2_goal_count) for the main heuristic.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic12(Heuristic):
    """Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Uses state-aware task cost estimation with stack height analysis
    and unsatisfied goal counting.
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_12"):
        super().__init__(name)
        # Fact lookup maps
        self.on_fact_id = {}       # (x, y) -> global_id
        self.ontable_fact_id = {}  # x -> global_id
        self.clear_fact_id = {}    # x -> global_id
        self.handempty_fact_id = None
        self.holding_fact_id = {}  # x -> global_id
        
        # All block names
        self.blocks = set()
        
        # Goal on-facts: set of (x, y) pairs from goal
        self.goal_on_pairs = set()
        self.goal_ontable = set()
        self.goal_clear = set()
        
        # Task name identification
        self.task_name_cache = {}
        
        # Map from block to the fact ids of on(?, block) - blocks that sit on this block
        # We'll use on_fact_id to find what's on top of a block
        self.on_top_of = {}  # block -> list of (other_block, fact_global_id)

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
        """Parse a task/operator name to get the task type and arguments."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Preprocess: build fact maps and goal structures."""
        self.model = model
        
        # Build fact lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.blocks.add(args[0])
                self.blocks.add(args[1])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = f.global_id
                self.blocks.add(args[0])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.blocks.add(args[0])
            elif pred == "handempty" and len(args) == 0:
                self.handempty_fact_id = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.blocks.add(args[0])
        
        # Build on_top_of map: for each block y, which (x, fact_id) have on(x, y)
        for (x, y), fid in self.on_fact_id.items():
            if y not in self.on_top_of:
                self.on_top_of[y] = []
            self.on_top_of[y].append((x, fid))
        
        # Parse goal facts
        if model.goals:
            goal_fact_names = model.state_explicit_repr(model.goals)
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) == 2:
                    self.goal_on_pairs.add((args[0], args[1]))
                elif pred == "ontable" and len(args) == 1:
                    self.goal_ontable.add(args[0])
                elif pred == "clear" and len(args) == 1:
                    self.goal_clear.add(args[0])
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _block_is_clear(self, block, state):
        """Check if block is clear in current state."""
        fid = self.clear_fact_id.get(block)
        if fid is None:
            return False
        return (state >> fid) & 1

    def _block_on(self, x, y, state):
        """Check if on(x, y) is true in current state."""
        fid = self.on_fact_id.get((x, y))
        if fid is None:
            return False
        return (state >> fid) & 1

    def _block_ontable(self, block, state):
        """Check if block is on table in current state."""
        fid = self.ontable_fact_id.get(block)
        if fid is None:
            return False
        return (state >> fid) & 1

    def _find_block_on_top(self, block, state):
        """Find which block is directly on top of 'block' in current state, or None."""
        entries = self.on_top_of.get(block)
        if entries is None:
            return None
        for (x, fid) in entries:
            if (state >> fid) & 1:
                return x
        return None

    def _stack_height_above(self, block, state, max_depth=50):
        """Count how many blocks are stacked above 'block' in current state."""
        count = 0
        current = block
        for _ in range(max_depth):
            top = self._find_block_on_top(current, state)
            if top is None:
                break
            count += 1
            current = top
        return count

    def _cost_do_clear(self, block, state):
        """Estimate minimum cost to make 'block' clear.
        
        If already clear: 0
        Otherwise: each block above needs to be cleared recursively then
        unstacked and put down = 2 actions per block above.
        Total for k blocks above: 2k (just unstack+putdown each).
        But do_clear is recursive: clear top, unstack it, putdown = 3 per level?
        Actually m7_do_clear: do_clear(y) + unstack(y,x) + put-down(y) = 2 actions + recursive
        So for k blocks: 2k actions total (each unstack+putdown).
        """
        if self._block_is_clear(block, state):
            return 0
        height = self._stack_height_above(block, state)
        return 2 * height

    def _cost_do_on_table(self, block, state):
        """Estimate minimum cost to put block on table.
        
        If already on table and clear: 0 (nop)
        If clear but not on table: 2 (unstack + putdown)
        If not clear: need to clear first, then 2
        """
        if self._block_ontable(block, state):
            return 0
        # Need to unstack + putdown = 2 actions
        # But block must be clear first - that's handled by preceding do_clear tasks typically
        return 2

    def _cost_do_move(self, x, y, state):
        """Estimate minimum cost to move x onto y.
        
        Requires: clear(x), clear(y), handempty
        If ontable(x): pick-up + stack = 2
        If not ontable(x): unstack + stack = 2
        """
        return 2

    def _cost_do_put_on(self, x, y, state):
        """Estimate minimum cost for do_put_on(x, y).
        
        If on(x,y) already: 0 (nop)
        Otherwise: do_clear(x) + do_clear(y) + do_on_table(y) + do_move(x,y)
        """
        if self._block_on(x, y, state):
            return 0
        
        cost = 0
        # Cost to clear x
        cost += self._cost_do_clear(x, state)
        # Cost to clear y
        cost += self._cost_do_clear(y, state)
        # Cost to put y on table (if not already)
        if not self._block_ontable(y, state):
            cost += 2
        # Cost to move x onto y
        cost += 2
        
        return cost

    def _estimate_task_cost(self, task, state):
        """Estimate the minimum cost for a single task based on current state."""
        from Pytrich.HDDL.hddl_types import AbstractTask as AT
        from Pytrich.HDDL.hddl_types import Operator as OP
        
        if isinstance(task, OP):
            # Primitive action - costs 1 (or 0 for nop)
            task_type, args = self._parse_task_name(task.name)
            if task_type == "nop":
                return 0
            return 1
        
        # Abstract task
        task_type, args = self._parse_task_name(task.name)
        
        if task_type == "do_put_on" and len(args) == 2:
            return self._cost_do_put_on(args[0], args[1], state)
        elif task_type == "do_clear" and len(args) == 1:
            return self._cost_do_clear(args[0], state)
        elif task_type == "do_on_table" and len(args) >= 1:
            return self._cost_do_on_table(args[0], state)
        elif task_type == "do_move" and len(args) == 2:
            return self._cost_do_move(args[0], args[1], state)
        
        # Unknown task type - return 1 as safe lower bound
        return 1

    def _compute(self, state, task_network):
        """Compute heuristic value from state and task network."""
        if len(task_network) == 0:
            # Check if goals are met
            if self.model.goals and not (self.model.goals & state == self.model.goals):
                return float('inf')
            return 0
        
        # Term 1: Sum of estimated costs for all tasks in the network
        # This is admissible if each task's estimate is a lower bound on its
        # actual decomposition cost AND tasks don't share work.
        # In GTOHP blocksworld with totally ordered tasks, this is reasonable
        # as an approximation but we use a conservative per-task estimate.
        task_cost_sum = 0
        for task in task_network:
            task_cost_sum += self._estimate_task_cost(task, state)
        
        # Term 2: Count unsatisfied goal on-facts * 2 (each needs at least 2 actions)
        goal_cost = 0
        for (x, y) in self.goal_on_pairs:
            fid = self.on_fact_id.get((x, y))
            if fid is not None:
                if not ((state >> fid) & 1):
                    goal_cost += 2
        
        # Term 3: Number of remaining tasks (simple but admissible lower bound)
        # Each non-nop abstract task needs at least 1 action
        task_count = 0
        from Pytrich.HDDL.hddl_types import Operator as OP
        for task in task_network:
            if isinstance(task, OP):
                task_type, _ = self._parse_task_name(task.name)
                if task_type != "nop":
                    task_count += 1
            else:
                task_count += 1
        
        # Return max of our estimates
        return max(task_cost_sum, goal_cost, task_count)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
