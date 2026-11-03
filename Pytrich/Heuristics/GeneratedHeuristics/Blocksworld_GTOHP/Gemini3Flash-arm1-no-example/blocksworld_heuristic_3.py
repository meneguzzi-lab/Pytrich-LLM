from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# ANALYSIS:
# 1. Bottleneck: Block movement is strictly sequential due to the handempty constraint. 
#    Unstacking a block requires clearing everything above it first, and stacking requires clearing the target.
# 2. Admissible Lower Bounds:
#    - h_tn: Sum of minimum required actions for each task in the remaining task network. 
#      Tasks like do_move or do_put_on (if not satisfied) cost at least 2 actions (pick/stack). 
#      do_clear costs 2 actions for every block currently above the target.
#    - h_goals: Minimum actions required to satisfy goal predicates. Each on(x, y) not currently true 
#      requires at least 2 actions (unless holding x, then 1).
# 3. Method: Preprocess fact IDs and bitmasks for O(1) state checks. Use a max(h_tn, h_goals) combination.

class BlocksworldGTOHPGemini3FlashHeuristic3(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_3="BlocksworldGTOHPGemini3FlashHeuristic3"):
        super().__init__(blocksworld_gemini3flash_heuristic_3)
        self.on_fact_ids = {} # (x, y) -> gid
        self.ontable_fact_ids = {} # x -> gid
        self.clear_fact_ids = {} # x -> gid
        self.holding_fact_ids = {} # x -> gid
        self.fact_to_top_block = {} # gid -> block_name (y in on[y, x])
        self.all_on_above_x_mask = {} # x -> int (mask of all on[y, x] gids)
        self.task_info_cache = {} # task_gid -> (name, args_list)
        self.on_goals = [] # List of (x, y)
        self.ontable_goals = [] # List of x

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Build Fact Maps
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
                self.fact_to_top_block[fact.global_id] = args[0] # args[0] is on top of args[1]
                self.all_on_above_x_mask[args[1]] = self.all_on_above_x_mask.get(args[1], 0) | (1 << fact.global_id)
            elif pred == "ontable":
                self.ontable_fact_ids[args[0]] = fact.global_id
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = fact.global_id
            elif pred == "holding":
                self.holding_fact_ids[args[0]] = fact.global_id

        # 2. Parse Goal State
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gf in goal_fact_names:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                self.on_goals.append((args[0], args[1]))
            elif pred == "ontable":
                self.ontable_goals.append(args[0])

        # 3. Precompute Task Info
        for task in model.abstract_tasks:
            name_only = task.name.split("[")[0]
            args = self._get_args_from_name(task.name)
            self.task_info_cache[task.global_id] = (name_only, args)
        for op in model.operators:
            name_only = op.name.split("[")[0]
            args = self._get_args_from_name(op.name)
            self.task_info_cache[op.global_id] = (name_only, args)

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, tn):
        # Term 1: Task Network Progress
        h_tn = 0
        for task in tn:
            h_tn += self._get_task_cost(task, state)
        
        # Term 2: Goal Satisfaction
        h_goals = 0
        for x, y in self.on_goals:
            if not self._is_on(x, y, state):
                h_goals += 1 if self._is_holding(x, state) else 2
        for x in self.ontable_goals:
            if not self._is_ontable(x, state):
                h_goals += 1 if self._is_holding(x, state) else 2
        
        return max(h_tn, h_goals)

    def _get_task_cost(self, task, state):
        info = self.task_info_cache.get(task.global_id)
        if not info: return 0
        name, args = info
        if name == "do_clear":
            return 2 * self._count_above(args[0], state)
        elif name == "do_put_on":
            if self._is_on(args[0], args[1], state): return 0
            return 1 if self._is_holding(args[0], state) else 2
        elif name == "do_move":
            return 1 if self._is_holding(args[0], state) else 2
        elif name == "do_on_table":
            if self._is_ontable(args[0], state): return 0
            return 1 if self._is_holding(args[0], state) else 2
        elif name == "nop":
            return 0
        elif hasattr(task, 'cost'):
            return task.cost
        return 0

    def _count_above(self, x, state):
        count = 0
        curr = x
        while True:
            mask = self.all_on_above_x_mask.get(curr, 0)
            matches = state & mask
            if matches == 0: break
            fid = (matches & -matches).bit_length() - 1
            curr = self.fact_to_top_block.get(fid)
            if not curr: break
            count += 1
        return count

    def _is_on(self, x, y, state):
        fid = self.on_fact_ids.get((x, y))
        return (state >> fid) & 1 if fid is not None else False

    def _is_ontable(self, x, state):
        fid = self.ontable_fact_ids.get(x)
        return (state >> fid) & 1 if fid is not None else False

    def _is_holding(self, x, state):
        fid = self.holding_fact_ids.get(x)
        return (state >> fid) & 1 if fid is not None else False

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return predicate, args
        return None, []

    def _get_args_from_name(self, name):
        if "[" not in name: return []
        return name[name.find("[")+1:-1].split(",")
