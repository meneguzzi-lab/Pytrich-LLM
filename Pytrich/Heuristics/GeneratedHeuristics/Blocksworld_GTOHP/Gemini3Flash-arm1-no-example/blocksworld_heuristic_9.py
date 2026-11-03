from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class BlocksworldGTOHPGemini3FlashHeuristic9(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld GTOHP.
    
    1. Bottleneck Analysis:
    The primary bottleneck is stack serialization. To move a block or clear a target, 
    all blocks above them must be unstacked to the table, and the target block 
    must often be moved to the table itself according to the HTN methods.
    
    2. Admissible Lower Bounds:
    - Clearing Cost: For any block X, every block above it requires an unstack and a put-down.
      This contributes at least 2 actions per block above X.
    - Movement Cost: Moving block X to Y requires at least 2 actions (pick-up + stack or unstack + stack).
    - Task Network Summation: In this ordered domain, the tasks in the current network 
      represent future work that must be completed sequentially. Summing state-aware 
    min-costs for each task provides an admissible lower bound.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_9="blocksworld_gemini3flash_heuristic_9"):
        super().__init__(blocksworld_gemini3flash_heuristic_9)
        self.on_fact_to_pair = {}
        self.on_ids = {}
        self.clear_ids = {}
        self.ontable_fact_ids = {}
        self.holding_ids = {}
        self.handempty_id = None

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

    def initialize(self, model: Model, initial_node: HTNNode):
        for i in range(model.ifacts_init, model.ifacts_end + 1):
            f = model.get_component(i)
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_fact_to_pair[f.global_id] = (args[0], args[1])
                self.on_ids[(args[0], args[1])] = f.global_id
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "ontable":
                self.ontable_fact_ids[args[0]] = f.global_id
            elif pred == "holding":
                self.holding_ids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network) -> int:
        on_top = {}
        temp_state = state
        while temp_state:
            lsb = temp_state & -temp_state
            bit_pos = lsb.bit_length() - 1
            if bit_pos in self.on_fact_to_pair:
                u, v = self.on_fact_to_pair[bit_pos]
                on_top[v] = u
            temp_state ^= lsb

        memo_height = {}
        def get_height(block):
            if block in memo_height: return memo_height[block]
            if block not in on_top:
                memo_height[block] = 0
                return 0
            h = 1 + get_height(on_top[block])
            memo_height[block] = h
            return h

        total_h = 0
        is_handempty = (state >> self.handempty_id) & 1 if self.handempty_id is not None else True

        for task in task_network:
            if hasattr(task, 'cost'):
                total_h += task.cost
                continue
            
            name = task.name
            parts = re.split(r'[\s\(\),]+', name.strip())
            if not parts: continue
            t_type = parts[0]
            args = parts[1:]

            if t_type == "do_put_on":
                if len(args) < 2: continue
                x, y = args[0], args[1]
                on_id = self.on_ids.get((x, y))
                if on_id is not None and (state >> on_id) & 1:
                    continue
                
                cost = 0
                holding_x = (state >> self.holding_ids[x]) & 1 if x in self.holding_ids else 0
                if holding_x:
                    cost += 1 + 2 * get_height(y)
                else:
                    cost += 2 + 2 * get_height(x) + 2 * get_height(y)
                    if not is_handempty:
                        cost += 1
                total_h += cost

            elif t_type == "do_clear":
                if len(args) < 1: continue
                x = args[0]
                total_h += 2 * get_height(x)

            elif t_type == "do_move":
                if len(args) < 2: continue
                x, y = args[0], args[1]
                total_h += 2 + 2 * get_height(x) + 2 * get_height(y)

            elif t_type == "do_on_table":
                if len(args) < 1: continue
                x = args[0]
                ot_id = self.ontable_fact_ids.get(x)
                if ot_id is not None and (state >> ot_id) & 1:
                    continue
                total_h += 2 + 2 * get_height(x)

        return int(total_h + (0.01 * len(task_network)))
