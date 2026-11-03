"""Domain-Dependent Heuristic for Blocksworld_GTOHP

Bottleneck Analysis:
  The single constraint that most serializes work in Blocksworld is the
  requirement to clear blocks before they can be moved. To place block X
  on block Y, both X and Y must be clear, and clearing a block requires
  recursively unstacking everything above it. Each unstack+putdown costs
  2 actions, and each pick-up+stack (or unstack+stack) costs 2 actions.

Admissible Lower Bounds:
  Term 1 - Goal Achievement Cost: For each goal fact on(x,y) not currently
    true in the state, we need at minimum 2 actions (pick-up/unstack + stack).
    Additionally, if x is not clear, we need 2 actions per block above x.
    If y is not clear, we need 2 actions per block above y. We take the
    maximum over individual goal costs (admissible since goals share resources
    like handempty, but max of individual lower bounds is still a lower bound).

  Term 2 - Task Network Lower Bound: For each compound task remaining in the
    task network, estimate a state-aware minimum decomposition cost. do_put_on
    tasks where on(x,y) already holds cost 0 (nop). Otherwise, do_put_on needs
    at least clearing x, clearing y, possibly putting y on table, and moving x
    to y. do_clear tasks where clear(x) holds cost 0. do_on_table where
    ontable(x) holds cost 0. We sum costs for the first task in the ordered
    network (since tasks are ordered, only the first matters for immediate
    progress, but we can sum all as a lower bound on total remaining work
    since each task's actions are distinct moves).

  Term 3 - Unsatisfied Goals Count: Number of goal on(x,y) facts not in
    current state, times 2 (minimum actions per unsatisfied goal). This is
    admissible because each missing on-relation requires at least one
    stack action (cost 1) plus one pick/unstack (cost 1).
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic2(Heuristic):
    def __init__(self, name="blocksworld_anthropicopus46_heuristic_2"):
        super().__init__(name)
        # Fact lookup maps
        self.on_fact_id = {}      # (x, y) -> global_id
        self.clear_fact_id = {}   # x -> global_id
        self.ontable_fact_id = {} # x -> global_id
        self.handempty_fact_id = None
        self.holding_fact_id = {} # x -> global_id

        # Goal info
        self.goal_on_pairs = []   # list of (x, y, fact_global_id)

        # Block relationships: which block is on top of x?
        # on_top_of[x] = list of (y, fact_id) where on(y, x) is a fact
        self.on_top_of = {}  # block -> list of (block_on_top, fact_global_id)
        # on_below[x] = list of (y, fact_id) where on(x, y) is a fact
        self.on_below = {}

        # Task name parsing cache
        self.task_name_cache = {}

        # Abstract task type identification
        self.do_put_on_ids = set()
        self.do_clear_ids = set()
        self.do_on_table_ids = set()
        self.do_move_ids = set()

        # Operator identification
        self.nop_ids = set()

        # All blocks
        self.all_blocks = set()

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
        """Parse a task/operator name to extract the task type and arguments."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Parse all facts and build lookup maps for O(1) state queries."""
        self.model = model

        # Build fact lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
                # Track which blocks can be on top of which
                if args[1] not in self.on_top_of:
                    self.on_top_of[args[1]] = []
                self.on_top_of[args[1]].append((args[0], f.global_id))
                if args[0] not in self.on_below:
                    self.on_below[args[0]] = []
                self.on_below[args[0]].append((args[1], f.global_id))
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "handempty" and len(args) == 0:
                self.handempty_fact_id = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = f.global_id
                self.all_blocks.add(args[0])

        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                fid = self.on_fact_id.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_pairs.append((args[0], args[1], fid))

        # Identify abstract task types
        for at in model.abstract_tasks:
            ttype, targs = self._parse_task_name(at.name)
            if "do_put_on" in ttype:
                self.do_put_on_ids.add(at.global_id)
            elif "do_clear" in ttype:
                self.do_clear_ids.add(at.global_id)
            elif "do_on_table" in ttype:
                self.do_on_table_ids.add(at.global_id)
            elif "do_move" in ttype:
                self.do_move_ids.add(at.global_id)

        # Identify nop operators
        for op in model.operators:
            ttype, targs = self._parse_task_name(op.name)
            if "nop" in ttype:
                self.nop_ids.add(op.global_id)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _stack_height_above(self, block, state):
        """Count how many blocks are stacked above 'block' in current state.
        This uses the on_top_of map and checks which on(y, block) facts are true.
        Returns the total height of the tower above block."""
        height = 0
        current = block
        max_depth = 50  # Safety limit
        while max_depth > 0:
            max_depth -= 1
            found = False
            if current in self.on_top_of:
                for (top_block, fid) in self.on_top_of[current]:
                    if (state >> fid) & 1:
                        height += 1
                        current = top_block
                        found = True
                        break
            if not found:
                break
        return height

    def _is_clear(self, block, state):
        fid = self.clear_fact_id.get(block)
        if fid is None:
            return True
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

    def _cost_to_clear(self, block, state):
        """Estimate minimum actions to make 'block' clear.
        Each block above requires unstack + putdown = 2 actions."""
        if self._is_clear(block, state):
            return 0
        height = self._stack_height_above(block, state)
        return 2 * height  # Each block above: unstack + put-down

    def _cost_do_put_on(self, x, y, state):
        """Estimate minimum actions for do_put_on(x, y).
        If on(x,y) already true, cost is 0 (nop method).
        Otherwise: clear x + clear y + possibly put y on table + move x to y.
        move costs 2 (pick-up/unstack + stack)."""
        if self._is_on(x, y, state):
            return 0

        cost = 0
        # Cost to clear x
        cost += self._cost_to_clear(x, state)
        # Cost to clear y
        cost += self._cost_to_clear(y, state)
        # Cost for do_on_table(y): if y is not on table and is clear (after clearing),
        # needs unstack + put-down = 2
        if not self._is_ontable(y, state):
            cost += 2
        # Cost to move x to y: pick-up/unstack + stack = 2
        cost += 2
        return cost

    def _cost_do_clear(self, x, state):
        """Estimate minimum actions for do_clear(x)."""
        return self._cost_to_clear(x, state)

    def _cost_do_on_table(self, x, state):
        """Estimate minimum actions for do_on_table(x)."""
        if self._is_ontable(x, state):
            return 0
        # Need to unstack + put-down = 2 (assuming x is already clear)
        return 2

    def _cost_do_move(self, x, y, state):
        """Estimate minimum actions for do_move(x, y)."""
        # pick-up/unstack + stack = 2
        return 2

    def _estimate_task_cost(self, task, state):
        """Estimate cost for a single task based on current state."""
        gid = task.global_id

        # Check if it's an operator
        if gid >= self.model.iop_init and gid <= self.model.iop_end:
            if gid in self.nop_ids:
                return 0
            return 1  # Each primitive action costs 1

        # It's an abstract task - parse its name for arguments
        if gid not in self.task_name_cache:
            ttype, targs = self._parse_task_name(task.name)
            self.task_name_cache[gid] = (ttype, targs)
        else:
            ttype, targs = self.task_name_cache[gid]

        if gid in self.do_put_on_ids and len(targs) >= 2:
            return self._cost_do_put_on(targs[0], targs[1], state)
        elif gid in self.do_clear_ids and len(targs) >= 1:
            return self._cost_do_clear(targs[0], state)
        elif gid in self.do_on_table_ids and len(targs) >= 1:
            return self._cost_do_on_table(targs[0], state)
        elif gid in self.do_move_ids and len(targs) >= 2:
            return self._cost_do_move(targs[0], targs[1], state)

        return 1  # Default fallback

    def _compute(self, state, task_network):
        """Compute the heuristic value using max of multiple terms."""
        if len(task_network) == 0:
            # Check if goals are met
            if (state & self.model.goals) == self.model.goals:
                return 0
            return 0

        # Term 1: Task network cost - sum of estimated costs for all tasks
        task_cost = 0
        for task in task_network:
            task_cost += self._estimate_task_cost(task, state)

        # Term 2: Unsatisfied goals lower bound
        # Each unsatisfied on(x,y) goal needs at least 2 actions
        unsatisfied_goal_cost = 0
        for (x, y, fid) in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                unsatisfied_goal_cost += 2

        # Term 3: Goal instability detection
        # A goal on(x,y) currently true but x is not clear and will need
        # to be disturbed adds cost. Check if any currently-true goal
        # has blocks on top that need to be moved for other goals.
        instability_cost = 0
        for (x, y, fid) in self.goal_on_pairs:
            if (state >> fid) & 1:
                # Goal is currently satisfied - but is it stable?
                # Check if something is on top of x that shouldn't be
                if not self._is_clear(x, state):
                    # Something is on x. Check if the block on x is a goal block
                    # that should be there
                    if x in self.on_top_of:
                        for (top_block, top_fid) in self.on_top_of[x]:
                            if (state >> top_fid) & 1:
                                # top_block is on x. Is on(top_block, x) a goal?
                                goal_match = False
                                for (gx, gy, gfid) in self.goal_on_pairs:
                                    if gx == top_block and gy == x:
                                        goal_match = True
                                        break
                                if not goal_match:
                                    # This block needs to be moved: unstack + putdown
                                    # Plus the current goal will be broken and needs
                                    # to be re-established
                                    instability_cost += 2
                                break

        # Return max of the terms for admissibility
        # Task cost is additive (each task needs its own actions)
        # Goal cost is a separate lower bound
        h = max(task_cost, unsatisfied_goal_cost + instability_cost)
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Called for every node expansion."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
