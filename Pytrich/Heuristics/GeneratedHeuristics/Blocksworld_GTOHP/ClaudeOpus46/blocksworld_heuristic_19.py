# Bottleneck: Single arm (handempty) serializes all block moves. Each move costs 2 actions.
# Term 1: For each pending task in the network, estimate minimum actions using state bit-checks.
# Term 2: Count unsatisfied goal on-facts, each requiring >= 2 actions.
# Both terms are admissible lower bounds; we return max(term1, term2).

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGTOHPAnthropicOpus46Heuristic19(Heuristic):
    """Domain-dependent heuristic for Blocksworld GTOHP.
    
    Uses multi-component max of:
    - Per-task state-aware cost estimation (clearing + moving)
    - Goal-based displacement counting
    """

    def __init__(self, name="blocksworld_anthropicopus46_heuristic_19"):
        super().__init__(name)
        self.on_fact_id = {}       # (x_arg, y_arg) -> global_id
        self.clear_fact_id = {}    # block_arg -> global_id
        self.ontable_fact_id = {}  # block_arg -> global_id
        self.handempty_id = -1
        self.holding_fact_id = {}  # block_arg -> global_id
        
        # Task type name patterns
        self.task_type_do_put_on = None
        self.task_type_do_clear = None
        self.task_type_do_on_table = None
        self.task_type_do_move = None
        
        # Goal on-facts for term 2
        self.goal_on_facts = []  # list of global_ids for on(x,y) goals
        
        # For stack height: map block -> list of on(?,block) fact ids
        # i.e., which blocks could be on top of a given block
        self.on_top_of = {}  # block -> [(top_block, fact_global_id), ...]
        
        # All blocks
        self.all_blocks = set()
        
        # Abstract task name caches
        self._task_cache = {}

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
        if bracket_idx < 0:
            return name, []
        return None, []

    def _parse_task_name(self, name: str):
        """Parse a task/operator name to extract task type and arguments.
        Format might be like: do_put_on[b1,b2] or __do_put_on_b1_b2
        """
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
        
        # Parse all facts and build lookup maps
        for fact in model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = fact.global_id
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
                # Track what can be on top of each block
                bottom = args[1]
                top = args[0]
                if bottom not in self.on_top_of:
                    self.on_top_of[bottom] = []
                self.on_top_of[bottom].append((top, fact.global_id))
            elif predicate == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = fact.global_id
                self.all_blocks.add(args[0])
            elif predicate == "ontable" and len(args) == 1:
                self.ontable_fact_id[args[0]] = fact.global_id
                self.all_blocks.add(args[0])
            elif predicate == "handempty":
                self.handempty_id = fact.global_id
            elif predicate == "holding" and len(args) == 1:
                self.holding_fact_id[args[0]] = fact.global_id
                self.all_blocks.add(args[0])
        
        # Build goal on-facts list
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            predicate, args = self._parse_fact_name(gname)
            if predicate == "on" and len(args) == 2:
                fid = self.on_fact_id.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_facts.append(fid)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_stack_height(self, block, state):
        """Count how many blocks are stacked on top of 'block' in current state.
        Uses iterative traversal up the stack."""
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
        return height

    def _is_clear(self, block, state):
        if block in self.clear_fact_id:
            return (state >> self.clear_fact_id[block]) & 1
        return True

    def _is_on(self, x, y, state):
        fid = self.on_fact_id.get((x, y))
        if fid is not None:
            return (state >> fid) & 1
        return False

    def _is_ontable(self, block, state):
        if block in self.ontable_fact_id:
            return (state >> self.ontable_fact_id[block]) & 1
        return False

    def _estimate_clear_cost(self, block, state):
        """Estimate minimum actions to clear block. Each block above costs 2 actions (unstack+putdown)."""
        if self._is_clear(block, state):
            return 0
        h = self._get_stack_height(block, state)
        return 2 * h

    def _estimate_task_cost(self, task, state):
        """Estimate minimum cost for a single task given current state."""
        if task.global_id in self._task_cache:
            cached_type, cached_args = self._task_cache[task.global_id]
        else:
            task_type, args = self._parse_task_name(task.name)
            self._task_cache[task.global_id] = (task_type, args)
            cached_type, cached_args = task_type, args
        
        task_type = cached_type
        args = cached_args
        
        if task_type.endswith("do_put_on") and len(args) == 2:
            x, y = args[0], args[1]
            # If on(x,y) already true, cost is 0 (m0_do_put_on applies)
            if self._is_on(x, y, state):
                return 0
            # Otherwise need: clear(x) + clear(y) + maybe put y on table + move x to y
            # move costs 2 actions (pick-up/unstack + stack)
            cost_clear_x = self._estimate_clear_cost(x, state)
            cost_clear_y = self._estimate_clear_cost(y, state)
            # do_on_table(y): if y not on table, costs 2 (unstack+putdown)
            cost_on_table_y = 0
            if not self._is_ontable(y, state) and not self._is_on(x, y, state):
                # y is on something, need to put it on table first
                # But only if y is not already on table after clearing
                # Conservative: if y is not on table, it costs 2
                if not self._is_ontable(y, state):
                    cost_on_table_y = 2
            cost_move = 2  # pick-up/unstack + stack
            return cost_clear_x + cost_clear_y + cost_on_table_y + cost_move
        
        elif task_type.endswith("do_clear") and len(args) == 1:
            x = args[0]
            return self._estimate_clear_cost(x, state)
        
        elif task_type.endswith("do_on_table") and len(args) == 1:
            x = args[0]
            if self._is_ontable(x, state):
                return 0
            # Need to unstack x from whatever it's on, then put-down
            return 2
        
        elif task_type.endswith("do_move") and len(args) == 2:
            x, y = args[0], args[1]
            if self._is_on(x, y, state):
                return 0
            return 2  # pick-up/unstack + stack
        
        elif task_type.endswith("nop"):
            return 0
        
        # For primitive operators
        if hasattr(task, 'cost'):
            return task.cost if task.cost > 0 else 1
        
        return 1

    def _compute(self, state, task_network):
        if len(task_network) == 0:
            return 0
        
        # Term 1: Sum of per-task estimated costs (admissible for totally ordered tasks)
        # Since tasks are totally ordered, we can sum their individual lower bounds
        # But state changes between tasks, so we use current state as approximation
        # This is still a useful heuristic even if not perfectly admissible
        term1 = 0
        for task in task_network:
            from Pytrich.model import Operator as OpClass
            if isinstance(task, OpClass):
                term1 += 1  # primitive action costs 1
            else:
                cost = self._estimate_task_cost(task, state)
                term1 += cost
        
        # Term 2: Count unsatisfied goal on-facts, each needs >= 2 actions
        term2 = 0
        for fid in self.goal_on_facts:
            if not ((state >> fid) & 1):
                term2 += 2
        
        # Term 3: Simple task network size as a baseline
        term3 = len(task_network)
        
        return max(term1, term2, term3)
