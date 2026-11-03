from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class BlocksworldGTOHPGemini3FlashHeuristic8(Heuristic):
    """
    Heuristic for Blocksworld_GTOHP.

    1. Bottleneck: Stack clearing — moving any block requires recursively unstacking every block above it first.
    2. Admissible Bounds:
       - Task-based action count: Each high-level task (do_put_on, do_move, etc.) is estimated by the minimum 
         primitive actions required (pick-up, unstack, stack, put-down) given current stack heights.
       - Clear-cost: To clear a block x, every block above it must be unstacked and put down, costing 2 actions per block.
       - Displacement cost: Currently true goals (e.g., on(x,y)) are only free if the entire stack below x is stable.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_8="BlocksworldGTOHPGemini3FlashHeuristic8"):
        super().__init__(blocksworld_gemini3flash_heuristic_8)
        self.on_fact_ids = {}
        self.ontable_fact_ids = {}
        self.clear_fact_ids = {}
        self.holding_fact_ids = {}
        self.block_names = set()
        self.on_above_map = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        for i in range(model.ifacts_init, model.ifacts_end + 1):
            f = model.facts[i]
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            for arg in args: self.block_names.add(arg)
            if pred == "on":
                self.on_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "ontable":
                self.ontable_fact_ids[args[0]] = f.global_id
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = f.global_id
            elif pred == "holding":
                self.holding_fact_ids[args[0]] = f.global_id

        self.on_above_map = {b: [] for b in self.block_names}
        for (top, bottom), fid in self.on_fact_ids.items():
            self.on_above_map[bottom].append((top, fid))

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        if not task_network: return 0
        
        heights = self._get_all_heights(state)
        under_map = self._get_under_map(state)
        
        targets = set()
        for task in task_network:
            tname, targs = self._parse_task_parts(task.name)
            if targs: targets.add(targs[0])

        memo_u = {}
        def is_under_moving(b):
            if b in memo_u: return memo_u[b]
            under = under_map.get(b)
            if under is None: res = False
            elif under in targets: res = True
            else: res = is_under_moving(under)
            memo_u[b] = res
            return res

        total_h = 0
        for task in task_network:
            tname, targs = self._parse_task_parts(task.name)
            if not tname: continue
            
            if tname == "do_put_on":
                x, y = targs[0], targs[1]
                fid = self.on_fact_ids.get((x, y))
                if fid is not None and (state >> fid) & 1 and not is_under_moving(x):
                    continue
                total_h += heights[x] * 2 + heights[y] * 2 + 3
            elif tname == "do_clear":
                total_h += heights[targs[0]] * 2
            elif tname == "do_on_table":
                x = targs[0]
                fid = self.ontable_fact_ids.get(x)
                if fid is not None and (state >> fid) & 1 and not is_under_moving(x):
                    continue
                total_h += heights[x] * 2 + 2
            elif tname == "do_move":
                x, y = targs[0], targs[1]
                total_h += heights[x] * 2 + heights[y] * 2 + 2
            elif tname in ["pick-up", "put-down", "stack", "unstack"]: 
                total_h += 1
        
        return int(total_h)

    def _get_all_heights(self, state):
        on_map = {}
        for (top, bottom), fid in self.on_fact_ids.items():
            if (state >> fid) & 1: on_map[bottom] = top
        memo_h = {}
        def get_h(b):
            if b in memo_h: return memo_h[b]
            top = on_map.get(b)
            res = (1 + get_h(top)) if top else 0
            memo_h[b] = res
            return res
        return {b: get_h(b) for b in self.block_names}

    def _get_under_map(self, state):
        under_map = {}
        for (top, bottom), fid in self.on_fact_ids.items():
            if (state >> fid) & 1: under_map[top] = bottom
        return under_map

    def _parse_task_parts(self, task_name):
        words = re.findall(r'[a-zA-Z0-9\-\_]+', task_name)
        if not words: return None, []
        if words[0].startswith('task') and len(words) > 1: return words[1], words[2:]
        return words[0], words[1:]

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
