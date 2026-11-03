from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import math

# Bottleneck Analysis:
# The primary bottleneck in the Blocksworld_GTOHP domain is the stack serialization. 
# To move any block, the stack above it must be cleared, and to place a block on a target, 
# the target must also be clear and potentially moved to the table first (per the m1 decomposition).

# Admissible Lower Bounds:
# 1. State-Aware Task Cost: Each task in the network has a minimum action cost based on the current state 
#    (e.g., do_clear(x) costs 2 * number of blocks currently above x). This is admissible as these actions are unavoidable.
# 2. Hand Status: If the hand is not empty, at least one action (stack or put-down) is required before a new pick/unstack.

class BlocksworldGTOHPGemini3FlashHeuristic13(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_13="blocksworld_gemini3flash_heuristic_13"):
        super().__init__(blocksworld_gemini3flash_heuristic_13)
        self.on_ids = {} # block_x -> block_y -> global_id
        self.on_id_to_args = {} # global_id -> (block_x, block_y)
        self.all_on_bits = 0
        self.ontable_ids = {}
        self.holding_ids = {}
        self.handempty_id = None
        self.task_info = {}

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

    def _parse_task_name(self, name):
        name = name.strip().lower()
        if name.startswith('(') and name.endswith(')'):
            name = name[1:-1]
        parts = name.split()
        if not parts: return None, []
        return parts[0], parts[1:]

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'on':
                x, y = args
                if x not in self.on_ids: self.on_ids[x] = {}
                self.on_ids[x][y] = f.global_id
                self.on_id_to_args[f.global_id] = (x, y)
                self.all_on_bits |= (1 << f.global_id)
            elif pred == 'ontable':
                self.ontable_ids[args[0]] = f.global_id
            elif pred == 'holding':
                self.holding_ids[args[0]] = f.global_id
            elif pred == 'handempty':
                self.handempty_id = f.global_id

        for abt in model.abstract_tasks:
            self.task_info[abt.global_id] = self._parse_task_name(abt.name)
        for op in model.operators:
            self.task_info[op.global_id] = self._parse_task_name(op.name)

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        current_on = {} # below -> above (who is on top of me?)
        holding_any = None

        temp_on = state & self.all_on_bits
        while temp_on:
            lsb = (temp_on & -temp_on).bit_length() - 1
            a, b = self.on_id_to_args[lsb]
            current_on[b] = a
            temp_on &= temp_on - 1

        for b, f_id in self.holding_ids.items():
            if (state >> f_id) & 1:
                holding_any = b
                break

        memo_clear = {}
        def get_clear_cost(block):
            if block in memo_clear: return memo_clear[block]
            cost = 0
            above = current_on.get(block)
            if above:
                cost = 2 + get_clear_cost(above)
            memo_clear[block] = cost
            return cost

        h = 0.0
        if not ((state >> self.handempty_id) & 1 if self.handempty_id is not None else True):
            h += 1.0

        for task in task_network:
            info = self.task_info.get(task.global_id)
            if not info: continue
            pred, args = info
            
            if pred == 'do_put_on':
                x, y = args[0], args[1]
                on_xy_id = self.on_ids.get(x, {}).get(y)
                if on_xy_id is not None and (state >> on_xy_id) & 1:
                    continue
                
                cost = 0
                if holding_any == x: cost = 1
                else: cost = 2 + get_clear_cost(x)
                cost += get_clear_cost(y)
                ontable_y_id = self.ontable_ids.get(y)
                if ontable_y_id is not None and not ((state >> ontable_y_id) & 1):
                    cost += 2
                h += cost
            elif pred == 'do_clear':
                h += get_clear_cost(args[0])
            elif pred == 'do_move':
                h += 1 if holding_any == args[0] else 2
            elif pred == 'do_on_table':
                x = args[0]
                ontable_x_id = self.ontable_ids.get(x)
                if ontable_x_id is not None and (state >> ontable_x_id) & 1:
                    continue
                h += 1 if holding_any == x else 2
            elif pred in ['pick-up', 'unstack', 'stack', 'put-down']:
                h += 1

        return int(h) + 0.001 * len(task_network)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
