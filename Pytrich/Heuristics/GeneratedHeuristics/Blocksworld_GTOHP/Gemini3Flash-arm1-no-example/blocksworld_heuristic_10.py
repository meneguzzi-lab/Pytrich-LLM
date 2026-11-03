from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic10(Heuristic):
    """
    Heuristic for Blocksworld_GTOHP.

    Bottleneck Analysis:
    The primary bottleneck is the vertical stack structure. To move a block or place a block on a base, 
    all blocks above the block and the base must be cleared sequentially. Each removal requires an 
    unstack and a subsequent action (put-down or stack), taking 2 actions per block cleared.

    Admissibility Sketches:
    1. do_clear(x): Lower bound of 2 * height(x), where height(x) is the number of blocks currently above x. 
       Each block requires at least one unstack and one other action.
    2. do_put_on(x, y): Lower bound of 2 if x is not already on y. If it is on y, it follows method m0 (nop).
    3. Task Network Summation: In Blocksworld, tasks in the network typically operate on separate stacks or 
       represent sequential work, making the sum of their individual lower bounds a strong and often 
       admissible estimate of the total remaining work.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_10="blocksworld_gemini3flash_heuristic_10"):
        super().__init__(name=blocksworld_gemini3flash_heuristic_10)
        self.block_name_to_idx = {}
        self.on_base_to_top = {} # y_idx -> list of (x_idx, global_id)
        self.on_ids = {} # (x_idx, y_idx) -> global_id
        self.ontable_ids = {} # x_idx -> global_id
        self.handempty_id = None
        self.task_meta = {} # global_id -> (task_type, arg_indices)

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

    def _get_block_idx(self, name):
        if name not in self.block_name_to_idx:
            self.block_name_to_idx[name] = len(self.block_name_to_idx)
        return self.block_name_to_idx[name]

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                x_idx = self._get_block_idx(args[0])
                y_idx = self._get_block_idx(args[1])
                if y_idx not in self.on_base_to_top:
                    self.on_base_to_top[y_idx] = []
                self.on_base_to_top[y_idx].append((x_idx, f.global_id))
                self.on_ids[(x_idx, y_idx)] = f.global_id
            elif pred == "ontable":
                x_idx = self._get_block_idx(args[0])
                self.ontable_ids[x_idx] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id

        for task in model.abstract_tasks:
            parts = task.name.split()
            t_type = parts[0]
            t_args = [self._get_block_idx(a) for a in parts[1:]]
            self.task_meta[task.global_id] = (t_type, t_args)

        for op in model.operators:
            parts = op.name.split()
            t_type = parts[0]
            self.task_meta[op.global_id] = (t_type, [])

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _get_height_above(self, state, block_idx):
        h = 0
        curr = block_idx
        while curr in self.on_base_to_top:
            found = False
            for top_idx, f_id in self.on_base_to_top[curr]:
                if (state >> f_id) & 1:
                    h += 1
                    curr = top_idx
                    found = True
                    break
            if not found: break
        return h

    def _compute(self, state, task_network):
        h = 0
        if self.handempty_id is not None and not ((state >> self.handempty_id) & 1):
            h += 1

        for task in task_network:
            meta = self.task_meta.get(task.global_id)
            if not meta: continue
            t_type, t_args = meta

            if t_type == "do_put_on":
                x, y = t_args
                on_xy = self.on_ids.get((x, y))
                if on_xy is None or not ((state >> on_xy) & 1):
                    h += 2
            elif t_type == "do_clear":
                h += 2 * self._get_height_above(state, t_args[0])
            elif t_type == "do_on_table":
                ontable_id = self.ontable_ids.get(t_args[0])
                if ontable_id is not None and not ((state >> ontable_id) & 1):
                    h += 2
            elif t_type == "do_move":
                h += 2
            elif t_type in ["pick-up", "put-down", "stack", "unstack"]:
                h += 1
            elif t_type == "nop":
                pass
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
