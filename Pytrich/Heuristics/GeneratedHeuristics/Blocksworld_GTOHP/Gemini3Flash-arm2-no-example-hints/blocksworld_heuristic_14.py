from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class BlocksworldGTOHPGemini3FlashHeuristic14(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld_GTOHP.
    
    BOTTLENECK: Stack clearing. To move a block, everything above it must be unstacked.
    ADMISSIBILITY SKETCH:
    1. h_tasks: Sum of minimum decomposition costs for each task in the network based on state.
       Each do_clear(x) costs at least 2*height(x). Each do_put_on(x,y) costs at least 2*hx + 2*hy + 4.
    2. h_goals: P5 goal-stability logic. A goal on(x,y) that is currently true but has an unstable base y
       costs at least 5 because it MUST be broken and rebuilt.
    3. max(h_tasks, h_goals) is an admissible lower bound because both are independent requirements.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_14="blocksworld_heuristic_14"):
        super().__init__(blocksworld_gemini3flash_heuristic_14)
        self.on_ids_by_bottom = {}  # bottom -> [(top, fid)]
        self.on_ids_by_top = {}     # top -> [(bottom, fid)]
        self.on_fact_id = {}        # (top, bottom) -> fid
        self.ontable_id = {}        # block -> fid
        self.clear_id = {}          # block -> fid
        self.handempty_id = None
        
        self.all_blocks = []
        self.task_info = {}         # global_id -> (type, args)
        self.goal_on_ids = []       # list of (x, y, fid)

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

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map facts
        blocks_set = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            for a in args: blocks_set.add(a)
            
            if pred == "on":
                top, bottom = args[0], args[1]
                if bottom not in self.on_ids_by_bottom: self.on_ids_by_bottom[bottom] = []
                if top not in self.on_ids_by_top: self.on_ids_by_top[top] = []
                self.on_ids_by_bottom[bottom].append((top, f.global_id))
                self.on_ids_by_top[top].append((bottom, f.global_id))
                self.on_fact_id[(top, bottom)] = f.global_id
            elif pred == "ontable":
                self.ontable_id[args[0]] = f.global_id
            elif pred == "clear":
                self.clear_id[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id
        
        self.all_blocks = list(blocks_set)

        # 2. Pre-cache task info
        for t in model.abstract_tasks:
            parts = t.name.split()
            self.task_info[t.global_id] = (parts[0], parts[1:])
        for o in model.operators:
            parts = o.name.split()
            self.task_info[o.global_id] = (parts[0], parts[1:])

        # 3. Identify goal structures
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                x, y = args[0], args[1]
                if (x, y) in self.on_fact_id:
                    self.goal_on_ids.append((x, y, self.on_fact_id[(x, y)]))

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _get_height(self, block, state, memo):
        if block in memo: return memo[block]
        h = 0
        curr = block
        while True:
            found = False
            for top_b, fid in self.on_ids_by_bottom.get(curr, []):
                if (state >> fid) & 1:
                    h += 1
                    curr = top_b
                    found = True
                    break
            if not found: break
        memo[block] = h
        return h

    def _is_stable(self, block, state, busy, memo_stable):
        if block in memo_stable: return memo_stable[block]
        if block in busy:
            memo_stable[block] = False
            return False
        
        # ontable check
        ot_id = self.ontable_id.get(block)
        if ot_id and (state >> ot_id) & 1:
            res = block not in busy
            memo_stable[block] = res
            return res

        # recursion on support
        bottom = None
        for low_b, fid in self.on_ids_by_top.get(block, []):
            if (state >> fid) & 1:
                bottom = low_b
                break
        
        if bottom is None:
            memo_stable[block] = False
            return False
            
        res = self._is_stable(bottom, state, busy, memo_stable)
        memo_stable[block] = res and (block not in busy)
        return memo_stable[block]

    def _compute(self, state, task_network):
        if not task_network: return 0

        # 1. Identify blocks that will be moved (busy)
        busy = set()
        for task_obj in task_network:
            info = self.task_info.get(task_obj.global_id)
            if info:
                t_type, t_args = info
                if t_type in ["do_put_on", "do_move", "do_on_table", "pick-up", "unstack", "put-down", "stack"] and t_args:
                    busy.add(t_args[0])

        memo_height = {}
        memo_stable = {}
        h_tasks = 0
        
        if self.handempty_id and not ((state >> self.handempty_id) & 1):
            h_tasks += 1

        # 2. Task-based lower bound (P2, P4)
        for task_obj in task_network:
            info = self.task_info.get(task_obj.global_id)
            if not info: continue
            t_type, t_args = info
            
            if t_type == "do_put_on":
                x, y = t_args[0], t_args[1]
                on_id = self.on_fact_id.get((x, y))
                is_satisfied = on_id and (state >> on_id) & 1 and self._is_stable(x, state, busy, memo_stable)
                if not is_satisfied:
                    hx = self._get_height(x, state, memo_height)
                    hy = self._get_height(y, state, memo_height)
                    h_tasks += 2 * hx + 2 * hy + 4
            elif t_type == "do_clear":
                x = t_args[0]
                hx = self._get_height(x, state, memo_height)
                h_tasks += 2 * hx
            elif t_type == "do_on_table":
                x = t_args[0]
                ot_id = self.ontable_id.get(x)
                is_ot = ot_id and (state >> ot_id) & 1
                if not is_ot:
                    hx = self._get_height(x, state, memo_height)
                    h_tasks += 2 * hx + 2
            elif t_type == "do_move":
                x, y = t_args[0], t_args[1]
                hx = self._get_height(x, state, memo_height)
                hy = self._get_height(y, state, memo_height)
                h_tasks += 2 * hx + 2 * hy + 2
            elif t_type in ["pick-up", "unstack", "stack", "put-down"]:
                h_tasks += 1

        # 3. Goal-based lower bound (P5)
        h_goals = 0
        for x, y, fid in self.goal_on_ids:
            currently_on = (state >> fid) & 1
            if not currently_on or not self._is_stable(y, state, busy, memo_stable):
                h_goals += 5

        # 4. Tie-breaking (P6)
        tie_breaker = 0.001 * len(task_network)

        return max(h_tasks, h_goals) + tie_breaker

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = int(self._compute(node.state, node.task_network))
        super().update_info(h_value)
        return h_value
