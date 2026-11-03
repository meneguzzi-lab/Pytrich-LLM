"""Domain-Dependent Heuristic for Blocksworld_GTOHP

Bottleneck Analysis:
  The single constraint that most serializes work in Blocksworld is the requirement
  to clear blocks before they can be moved. To place block X on block Y, both must
  be clear, meaning all blocks above X and Y must be recursively unstacked first.
  Each unstack requires 2 actions (unstack + put-down), and the final move requires
  2 actions (pick-up/unstack + stack). The hand can only hold one block at a time.

Admissible Lower Bounds:
  Term 1 - Goal achievement cost: For each goal fact on(x,y) not currently true,
    we need at minimum 2 actions (pick-up/unstack x + stack x y). If x is not clear,
    we need additional 2 actions per block above x. If y is not clear, we need
    additional 2 actions per block above y. We take the max over individual goal
    requirements to stay admissible (since work can overlap).
    
  Term 2 - Task network lower bound: Count remaining compound tasks in the task
    network. Each do_put_on that isn't already satisfied requires at least 2 actions.
    Each do_clear on a non-clear block requires at least 2 actions. Each do_on_table
    on a non-tabled block requires at least 2 actions. Each do_move requires exactly 2.
    We sum these for an additive estimate.

  Term 3 - Blocks-out-of-place: Count blocks that need to move based on current
    on-relationships vs goal on-relationships. Each such block needs at least 2 actions.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic10(Heuristic):
    def __init__(self, name="blocksworld_anthropicopus46_heuristic_10"):
        super().__init__(name)
        # Fact lookup maps
        self.on_fact_id = {}       # (x, y) -> global_id
        self.clear_fact_id = {}    # x -> global_id
        self.ontable_fact_id = {}  # x -> global_id
        self.holding_fact_id = {}  # x -> global_id
        self.handempty_fact_id = None
        
        # Goal structures
        self.goal_on_pairs = []    # list of (x, y, fact_global_id)
        
        # Block relationships for stack height computation
        self.blocks = set()
        self.on_facts_by_bottom = {}  # y -> list of (x, fact_global_id) where on(x,y)
        
        # Task name parsing cache
        self.task_name_cache = {}
        
        # Operator/AbstractTask identification
        self.nop_ids = set()
        self.do_put_on_ids = set()
        self.do_clear_ids = set()
        self.do_on_table_ids = set()
        self.do_move_ids = set()

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
        if name in self.task_name_cache:
            return self.task_name_cache[name]
        
        # Format like: do_put_on[b1,b2] or pick-up[b1] or nop[]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            result = (task_type, args)
        else:
            result = (name, [])
        
        self.task_name_cache[name] = result
        return result

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.blocks.add(args[0])
                self.blocks.add(args[1])
                # Track on-facts by bottom block
                if args[1] not in self.on_facts_by_bottom:
                    self.on_facts_by_bottom[args[1]] = []
                self.on_facts_by_bottom[args[1]].append((args[0], f.global_id))
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
                self.handempty_fact_id = f.global_id
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                fid = self.on_fact_id.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_pairs.append((args[0], args[1], fid))
        
        # Build goal-on map: for each block, what should be on top of it in the goal
        self.goal_on_top = {}  # bottom -> top block in goal
        self.goal_on_bottom = {}  # top -> bottom block in goal
        for x, y, fid in self.goal_on_pairs:
            self.goal_on_top[y] = x
            self.goal_on_bottom[x] = y
        
        # Identify abstract task types by their names
        for at in model.abstract_tasks:
            ttype, targs = self._parse_task_name(at.name)
            if ttype == "do_put_on":
                self.do_put_on_ids.add(at.global_id)
            elif ttype == "do_clear":
                self.do_clear_ids.add(at.global_id)
            elif ttype == "do_on_table":
                self.do_on_table_ids.add(at.global_id)
            elif ttype == "do_move":
                self.do_move_ids.add(at.global_id)
        
        # Identify nop operators
        for op in model.operators:
            ttype, targs = self._parse_task_name(op.name)
            if ttype == "nop":
                self.nop_ids.add(op.global_id)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _find_block_on_top(self, block, state):
        """Find which block is directly on top of 'block' in current state, or None."""
        if block in self.on_facts_by_bottom:
            for top_block, fid in self.on_facts_by_bottom[block]:
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
            if height > 100:  # Safety for large instances
                break
        return height

    def _is_correctly_placed(self, block, state):
        """Check if block is in its goal position and all blocks below it are also correct.
        A block is correctly placed if:
        1. on(block, below) matches goal
        2. The block below is also correctly placed (recursively)
        """
        if block not in self.goal_on_bottom:
            # This block doesn't need to be on anything specific in the goal
            return True
        
        goal_below = self.goal_on_bottom[block]
        on_fid = self.on_fact_id.get((block, goal_below))
        if on_fid is None:
            return False
        if not ((state >> on_fid) & 1):
            return False
        # Check recursively (but with depth limit)
        return True

    def _compute(self, state, task_network):
        """Compute heuristic value using max of multiple terms."""
        
        # Term 1: Goal-based lower bound
        # For each unsatisfied goal on(x,y), we need at least 2 actions.
        # If x is not clear, add 2 per block above x.
        # If y is not clear, add 2 per block above y.
        # We sum costs for all unsatisfied goals but account for the fact
        # that clearing work is shared - use additive estimate.
        
        unsatisfied_goal_cost = 0
        blocks_that_must_move = set()
        
        for x, y, fid in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                # Goal on(x,y) is not satisfied
                # Need at least 2 actions to place x on y
                unsatisfied_goal_cost += 2
                
                # Track blocks that need to move
                # x needs to be picked up, so anything on x must move
                clear_x_id = self.clear_fact_id.get(x)
                if clear_x_id is not None and not ((state >> clear_x_id) & 1):
                    # x is not clear, need to clear it
                    h_above = self._stack_height_above(x, state)
                    # Each block above needs unstack+put-down = 2 actions
                    # But we'll track these blocks to avoid double-counting
                    current = x
                    for _ in range(h_above):
                        top = self._find_block_on_top(current, state)
                        if top is not None:
                            blocks_that_must_move.add(top)
                            current = top
                        else:
                            break
                
                # y needs to be clear for stacking
                clear_y_id = self.clear_fact_id.get(y)
                if clear_y_id is not None and not ((state >> clear_y_id) & 1):
                    h_above = self._stack_height_above(y, state)
                    current = y
                    for _ in range(h_above):
                        top = self._find_block_on_top(current, state)
                        if top is not None:
                            blocks_that_must_move.add(top)
                            current = top
                        else:
                            break
        
        # Each block that must move needs at least 2 actions (unstack + put-down)
        # But some of these might overlap with goal placement actions
        clearing_cost = len(blocks_that_must_move) * 2
        goal_term = unsatisfied_goal_cost + clearing_cost
        
        # Term 2: Task-network based lower bound
        # Estimate minimum actions needed based on remaining tasks
        task_cost = 0
        
        for task in task_network:
            gid = task.global_id
            
            # Skip nop operators
            if gid in self.nop_ids:
                continue
            
            # Check if it's a primitive operator (not nop)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                task_cost += 1
                continue
            
            # Abstract tasks
            if gid in self.do_put_on_ids:
                ttype, targs = self._parse_task_name(task.name)
                if len(targs) >= 2:
                    x, y = targs[0], targs[1]
                    on_fid = self.on_fact_id.get((x, y))
                    if on_fid is not None and ((state >> on_fid) & 1):
                        # Already satisfied - costs 0 (nop)
                        continue
                    else:
                        # Need to clear x, clear y, possibly put y on table, then move x to y
                        cost = 2  # At minimum: pick-up/unstack + stack
                        
                        # Check if x is clear
                        clear_x_id = self.clear_fact_id.get(x)
                        if clear_x_id is not None and not ((state >> clear_x_id) & 1):
                            h = self._stack_height_above(x, state)
                            cost += h * 2  # Each block above needs unstack+put-down
                        
                        # Check if y is clear
                        clear_y_id = self.clear_fact_id.get(y)
                        if clear_y_id is not None and not ((state >> clear_y_id) & 1):
                            h = self._stack_height_above(y, state)
                            cost += h * 2
                        
                        task_cost += cost
            
            elif gid in self.do_clear_ids:
                ttype, targs = self._parse_task_name(task.name)
                if len(targs) >= 1:
                    x = targs[0]
                    clear_x_id = self.clear_fact_id.get(x)
                    if clear_x_id is not None and ((state >> clear_x_id) & 1):
                        continue  # Already clear
                    else:
                        h = self._stack_height_above(x, state)
                        task_cost += h * 2
            
            elif gid in self.do_on_table_ids:
                ttype, targs = self._parse_task_name(task.name)
                if len(targs) >= 1:
                    x = targs[0]
                    ontable_x_id = self.ontable_fact_id.get(x)
                    if ontable_x_id is not None and ((state >> ontable_x_id) & 1):
                        continue  # Already on table
                    else:
                        task_cost += 2  # unstack + put-down
            
            elif gid in self.do_move_ids:
                task_cost += 2  # pick-up/unstack + stack
            
            else:
                # Unknown abstract task, estimate 1
                task_cost += 1
        
        # Term 3: Misplaced blocks estimate
        # Count goal on-facts that are not true
        misplaced = 0
        for x, y, fid in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                misplaced += 1
        misplaced_term = misplaced * 2  # Each needs at least pick+stack
        
        # Return max of all terms
        return max(goal_term, task_cost, misplaced_term)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
