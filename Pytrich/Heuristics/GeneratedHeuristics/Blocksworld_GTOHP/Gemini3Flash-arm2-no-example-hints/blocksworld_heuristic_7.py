from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import math

class BlocksworldGTOHPGemini3FlashHeuristic7(Heuristic):
    """
    Domain-Dependent Heuristic for Blocksworld_GTOHP.
    
    Bottleneck Analysis:
    Stack clearing is the primary bottleneck. To move block X or place anything on block Y, 
    all blocks above them must be unstacked first. This creates a recursive cost structure.
    
    Admissibility Sketches:
    1. Network Cost (sum of task_costs): Each abstract task (do_put_on, do_clear, etc.) has a 
       minimum number of primitive actions (unstack, stack, etc.) required to decompose 
       based on the current state. Summing these is admissible as tasks are ordered.
    2. Stability Cost: A block is unstable if it is not at its goal position or sits on 
       an unstable base. Every unstable block must be moved at least twice (unstack + stack).
    3. Multi-component Max: Using max(network_cost, stability_cost) provides a robust 
       admissible lower bound by capturing both current task obligations and latent state goals.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_7="blocksworld_gemini3flash_heuristic_7"):
        super().__init__(blocksworld_gemini3flash_heuristic_7)
        self.on_ids = {}        # (x, y) -> global_id
        self.ontable_ids = {}   # x -> global_id
        self.clear_ids = {}     # x -> global_id
        self.holding_ids = {}   # x -> global_id
        self.handempty_id = -1
        self.goals_on = {}      # x -> y
        self.goals_ontable = set()
        self.all_blocks = []

    def _parse_fact_name(self, name: str):
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []

    def _parse_task_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        blocks = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
            elif pred == "ontable":
                self.ontable_ids[args[0]] = f.global_id
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "holding":
                self.holding_ids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id
            for arg in args:
                blocks.add(arg)
        
        self.all_blocks = list(blocks)
        
        goal_names = model.state_explicit_repr(model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on":
                self.goals_on[args[0]] = args[1]
            elif pred == "ontable":
                self.goals_ontable.add(args[0])

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # 1. Build world map
        who_is_on_top_of = {} # y -> x if on(x, y)
        who_is_under = {}     # x -> y if on(x, y)
        for (x, y), fid in self.on_ids.items():
            if (state >> fid) & 1:
                who_is_on_top_of[y] = x
                who_is_under[x] = y

        # 2. Precompute stack heights (above costs)
        above_cache = {}
        def get_above(b):
            if b in above_cache: return above_cache[b]
            if b not in who_is_on_top_of:
                above_cache[b] = 0
                return 0
            res = 1 + get_above(who_is_on_top_of[b])
            above_cache[b] = res
            return res

        for b in self.all_blocks:
            get_above(b)

        # 3. Calculate Network Cost (Admissible Sum)
        net_cost = 0
        for task in task_network:
            if hasattr(task, 'cost'):
                net_cost += task.cost
            else:
                tname, targs = self._parse_task_name(task.name)
                if tname == 'do_clear' and targs:
                    net_cost += 2 * above_cache.get(targs[0], 0)
                elif tname == 'do_on_table' and targs:
                    x = targs[0]
                    if not (state >> self.ontable_ids.get(x, 0)) & 1:
                        net_cost += 2 * above_cache.get(x, 0) + 2
                elif tname == 'do_move' and targs:
                    net_cost += 2
                elif tname == 'do_put_on' and len(targs) == 2:
                    x, y = targs[0], targs[1]
                    on_fid = self.on_ids.get((x, y))
                    if not (on_fid is not None and (state >> on_fid) & 1):
                        # Clear x, Clear y, potentially move y to table, move x to y
                        h_x = above_cache.get(x, 0)
                        h_y = above_cache.get(y, 0)
                        # Based on m1: clear x, clear y, do_on_table y, do_move x y
                        ontable_y = (state >> self.ontable_ids.get(y, 0)) & 1
                        net_cost += 2*h_x + 2*h_y + (2 if not ontable_y else 0) + 2

        # 4. Calculate Stability Cost (Bottleneck identification)
        stable_blocks = set()
        # Blocks on table stable if goal allows
        for b in self.all_blocks:
            if (state >> self.ontable_ids.get(b, 0)) & 1:
                if b in self.goals_ontable or b not in self.goals_on:
                    stable_blocks.add(b)
        # Propagate stability up
        changed = True
        while changed:
            changed = False
            for b in self.all_blocks:
                if b in stable_blocks: continue
                y = who_is_under.get(b)
                if y and y in stable_blocks and self.goals_on.get(b) == y:
                    stable_blocks.add(b)
                    changed = True
        
        stability_cost = 2 * (len(self.all_blocks) - len(stable_blocks))

        # 5. Hand Status
        hand_penalty = 0
        if self.handempty_id != -1 and not (state >> self.handempty_id) & 1:
            hand_penalty = 1

        # 6. Combine terms
        h = max(net_cost, stability_cost) + hand_penalty
        
        # Tie-breaker for task network progress
        h += 0.001 * len(task_network)
        
        return math.ceil(h)
