from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The primary bottleneck in Blocksworld is the serial dependency of stacking. To move or stack a block,
# it must be 'clear', which requires unstacking all blocks currently above it. Each unstacked block 
# requires a sequence of unstack-putdown actions before the target block can be manipulated.
#
# Admissibility Sketches:
# 1. do_clear(x) cost: 2 * (number of blocks above x). Each block above x must be unstacked (1) and put down (1).
# 2. do_put_on(x, y) cost: 2 + (2*height_x) + (2*height_y). To put x on y, we must clear x (2*h), clear y (2*h), 
#    pick up x (1), and stack x on y (1). This is a lower bound on actions required.
# 3. do_on_table(x) cost: 2 + (2*height_x). To put x on the table, we must clear it and perform unstack+putdown.

class BlocksworldGTOHPGemini3FlashHeuristic6(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_6="BlocksworldGTOHPGemini3FlashHeuristic6"):
        super().__init__(blocksworld_gemini3flash_heuristic_6)
        self.on_fact_ids = {} # (top, bottom) -> global_id
        self.ontable_fact_ids = {} # block -> global_id
        self.on_by_bottom = {} # bottom -> list of (fact_id, top)
        self.goal_fact_ids = []
        self.fact_info = {} # global_id -> (pred, args)
        self.block_names = set()
        self._height_cache = {}

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

    def _parse_task_args(self, task):
        name = task.name
        if "[" in name:
            bracket_idx = name.find("[")
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",")
            return task_type, args
        else:
            parts = name.split()
            if not parts:
                return "", []
            return parts[0], parts[1:]

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            self.fact_info[f.global_id] = (pred, args)
            if args:
                self.block_names.update(args)
            if pred == "on":
                self.on_fact_ids[(args[0], args[1])] = f.global_id
                if args[1] not in self.on_by_bottom:
                    self.on_by_bottom[args[1]] = []
                self.on_by_bottom[args[1]].append((f.global_id, args[0]))
            elif pred == "ontable":
                self.ontable_fact_ids[args[0]] = f.global_id

        goal_bits = model.goals
        idx = 0
        while (1 << idx) <= goal_bits:
            if (goal_bits >> idx) & 1:
                self.goal_fact_ids.append(idx)
            idx += 1

        blocksworld_gemini3flash_heuristic_6 = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, blocksworld_gemini3flash_heuristic_6)

    def _get_height(self, block, state):
        if block in self._height_cache:
            return self._height_cache[block]
        h = 0
        curr = block
        while curr in self.on_by_bottom:
            found = False
            for fact_id, top_block in self.on_by_bottom[curr]:
                if (state >> fact_id) & 1:
                    h += 1
                    curr = top_block
                    found = True
                    break
            if not found:
                break
        self._height_cache[block] = h
        return h

    def _compute(self, state, task_network):
        self._height_cache = {}
        h_tasks = 0
        
        for task in task_network:
            task_type, args = self._parse_task_args(task)
            if not args:
                if hasattr(task, 'cost'): h_tasks += task.cost
                continue

            x = args[0]
            y = args[1] if len(args) > 1 else None

            if task_type == "do_put_on":
                if (x, y) in self.on_fact_ids and (state >> self.on_fact_ids[(x, y)]) & 1:
                    pass
                else:
                    h_tasks += 2 + 2 * self._get_height(x, state) + (2 * self._get_height(y, state) if y else 0)
            elif task_type == "do_on_table":
                if x in self.ontable_fact_ids and (state >> self.ontable_fact_ids[x]) & 1:
                    pass
                else:
                    h_tasks += 2 + 2 * self._get_height(x, state)
            elif task_type == "do_clear":
                h_tasks += 2 * self._get_height(x, state)
            elif task_type == "do_move":
                h_tasks += 2
            elif task_type == "nop":
                pass
            else:
                h_tasks += getattr(task, 'cost', 1)

        h_goals = 0
        if not task_network:
            for fact_id in self.goal_fact_ids:
                if not ((state >> fact_id) & 1):
                    pred, args = self.fact_info.get(fact_id, (None, []))
                    if pred == "on":
                        h_goals += 2 + 2 * self._get_height(args[0], state) + 2 * self._get_height(args[1], state)
                    elif pred == "ontable":
                        h_goals += 2 + 2 * self._get_height(args[0], state)
                    else:
                        h_goals += 1
        
        return max(h_tasks, h_goals)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        blocksworld_gemini3flash_heuristic_6 = self._compute(node.state, node.task_network)
        super().update_info(blocksworld_gemini3flash_heuristic_6)
        return blocksworld_gemini3flash_heuristic_6
