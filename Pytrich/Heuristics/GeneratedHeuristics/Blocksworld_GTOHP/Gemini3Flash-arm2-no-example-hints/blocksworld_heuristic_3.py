from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic3(Heuristic):
    """
    Domain-Dependent Heuristic for Blocksworld GTOHP
    
    Bottleneck Analysis:
    Stack clearing is the primary bottleneck. To move a block, all blocks above it in the current 
    stack must be unstacked and put down (costing 2 actions per block above). 
    Goal stability is the second critical factor: an 'on' relationship that is currently 
    true is only useful if it is 'stable' (i.e., the base it rests on does not need to move).
    
    Admissibility Sketches:
    1. Task Network Cost: Every abstract task in the network must eventually decompose into at least 
       one primitive action. A lower bound for 'do_put_on(x,y)' is 2 + clear_cost(x) + clear_cost(y). 
       Summing these costs is admissible because tasks are ordered.
    2. Clear Cost: For any block x, clear_cost(x) is 2 * (number of blocks currently above x), 
       representing the unavoidable unstack + put-down sequence.
    3. Goal Stability: A goal on(x,y) that is currently true but unstable MUST be broken and 
       re-stacked, adding at least 4 to the total cost.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_3="blocksworld_gemini3flash_heuristic_3"):
        super().__init__(blocksworld_gemini3flash_heuristic_3)
        self.on_facts = []
        self.ontable_facts = []
        self.holding_facts = []
        self.handempty_id = -1
        self.goal_on_pairs = set()
        self.goal_on_where_x_is_i = set()
        self.block_name_to_idx = {}
        self.task_data = {}
        self.num_blocks = 0
        self.all_block_indices = []

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
        return None, []

    def _parse_task_args(self, name):
        name = name.replace('(', '').replace(')', '').strip()
        parts = name.split()
        if not parts: return None, []
        return parts[0], parts[1:]

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map Blocks
        all_block_names = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            for a in args: all_block_names.add(a)
        sorted_blocks = sorted(list(all_block_names))
        self.block_name_to_idx = {name: i for i, name in enumerate(sorted_blocks)}
        self.num_blocks = len(sorted_blocks)
        self.all_block_indices = list(range(self.num_blocks))

        # 2. Map Facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not args: 
                if pred == "handempty": self.handempty_id = f.global_id
                continue
            arg_idxs = [self.block_name_to_idx[a] for a in args]
            if pred == "on": self.on_facts.append((arg_idxs[0], arg_idxs[1], f.global_id))
            elif pred == "ontable": self.ontable_facts.append((arg_idxs[0], f.global_id))
            elif pred == "holding": self.holding_facts.append((arg_idxs[0], f.global_id))

        # 3. Map Goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on":
                idxs = (self.block_name_to_idx[args[0]], self.block_name_to_idx[args[1]])
                self.goal_on_pairs.add(idxs)
                self.goal_on_where_x_is_i.add(idxs[0])

        # 4. Map Task Data
        all_tasks = list(model.abstract_tasks) + list(model.operators)
        for task in all_tasks:
            ttype, targs = self._parse_task_args(task.name)
            arg_idxs = [self.block_name_to_idx[a] for a in targs if a in self.block_name_to_idx]
            self.task_data[task.global_id] = (ttype, arg_idxs)

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _compute(self, state, task_network):
        current_on = {}
        blocks_above_map = {}
        current_ontable = [False] * self.num_blocks
        is_holding = None

        for x, y, gid in self.on_facts:
            if (state >> gid) & 1:
                current_on[x] = y
                blocks_above_map[y] = x

        for x, gid in self.ontable_facts:
            if (state >> gid) & 1: current_ontable[x] = True

        for x, gid in self.holding_facts:
            if (state >> gid) & 1: is_holding = x

        hand_empty = (state >> self.handempty_id) & 1 if self.handempty_id != -1 else False

        # Precompute clear costs
        clear_costs = [0] * self.num_blocks
        for i in range(self.num_blocks):
            count = 0
            curr = i
            while curr in blocks_above_map:
                count += 1
                curr = blocks_above_map[curr]
            clear_costs[i] = 2 * count

        # Precompute stability (O(N) check)
        is_stable = [False] * self.num_blocks
        for i in range(self.num_blocks):
            if current_ontable[i]:
                if i not in self.goal_on_where_x_is_i:
                    is_stable[i] = True
                    curr = i
                    while curr in blocks_above_map:
                        above = blocks_above_map[curr]
                        if (above, curr) in self.goal_on_pairs:
                            is_stable[above] = True
                            curr = above
                        else: break

        h = 0
        if not hand_empty and is_holding is not None: h += 1

        tasks_covering_goals = set()
        for task in task_network:
            info = self.task_data.get(task.global_id)
            if not info: continue
            ttype, targs = info
            if ttype == "do_put_on":
                x, y = targs
                tasks_covering_goals.add((x, y))
                if not (current_on.get(x) == y and is_stable[x]):
                    h += 2 + clear_costs[x] + clear_costs[y]
            elif ttype == "do_on_table":
                x = targs[0]
                if not (current_ontable[x] and is_stable[x]):
                    h += 2 + clear_costs[x]
            elif ttype == "do_move":
                x, y = targs
                h += 2 + clear_costs[x] + clear_costs[y]
            elif ttype == "do_clear":
                x = targs[0]
                h += clear_costs[x]
            elif ttype in ["pick-up", "put-down", "stack", "unstack"]: h += 1
            elif ttype == "nop": h += 0
            else: h += 1

        # Add safety term for goals not in task network
        for x, y in self.goal_on_pairs:
            if (x, y) not in tasks_covering_goals:
                if not (current_on.get(x) == y and is_stable[x]):
                    h += 2

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
