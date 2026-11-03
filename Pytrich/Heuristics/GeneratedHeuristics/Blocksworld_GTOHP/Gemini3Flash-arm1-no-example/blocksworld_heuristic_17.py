from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic17(Heuristic):
    """
    Heuristic for Blocksworld_GTOHP.

    Bottleneck Analysis:
    In Blocksworld, the dominant cost is clearing obstacles to make a block or target location 'clear'. 
    Unstacking a block requires moving all blocks above it to the table (2 actions per block: unstack + put-down).

    Admissible Lower Bounds:
    1. State-Aware Task Cost: Each abstract task in the HTN network (do_put_on, do_clear, etc.) 
       is assigned a minimum cost based on the current stack heights in the state.
       - do_clear(x): Costs 2 * (number of blocks above x).
       - do_on_table(x): Costs 2 * (number of blocks above x) + 2 if x is not on the table.
       - do_put_on(x, y): Costs (clear x) + (clear y) + (move y to table) + (move x to y). 
    2. Primitive Action Cost: Each remaining primitive operator in the network (stack, unstack, pick-up, put-down) costs 1.

    Logic is admissible because it represents the absolute minimum number of primitive actions 
    required to decompose and execute the current task network.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_17="blocksworld_heuristic"):
        super().__init__(blocksworld_gemini3flash_heuristic_17)
        self.on_ids = {}
        self.on_fact_ids_inv = {}
        self.clear_ids = {}
        self.ontable_ids = {}
        self.holding_ids = {}
        self.handempty_id = -1
        self.all_blocks = set()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = [a.strip() for a in args_str.split(",")] if args_str else []
            return predicate, args
        return None, []

    def _parse_task_name(self, name: str):
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = [a.strip() for a in args_str.split(",")]
            return task_type, args
        else:
            parts = name.split()
            if len(parts) > 1: return parts[0], parts[1:]
            return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
                self.on_fact_ids_inv[f.global_id] = (args[0], args[1])
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "ontable":
                self.ontable_ids[args[0]] = f.global_id
            elif pred == "holding":
                self.holding_ids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id
            for a in args: self.all_blocks.add(a)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network) -> int:
        if not task_network:
            return 0

        current_on = {}
        held_block = None
        temp_state = state
        while temp_state:
            lsb = temp_state & -temp_state
            bit = lsb.bit_length() - 1
            if bit in self.on_fact_ids_inv:
                x, y = self.on_fact_ids_inv[bit]
                current_on[y] = x
            elif bit in self.holding_ids.values():
                for b, gid in self.holding_ids.items():
                    if gid == bit:
                        held_block = b
                        break
            temp_state ^= lsb

        def count_above(block):
            if block == held_block: return 0
            count = 0
            curr = block
            while curr in current_on:
                count += 1
                curr = current_on[curr]
            return count

        h = 0.0
        for task in task_network:
            name, args = self._parse_task_name(task.name)
            if not hasattr(task, "decompositions"): # Operator
                if "nop" not in name: h += 1.0
                continue
            
            # Abstract Task Analysis
            if name == "do_clear":
                if args: h += 2.0 * count_above(args[0])
            elif name == "do_on_table":
                if args:
                    if not ((state >> self.ontable_ids.get(args[0], -1)) & 1):
                        h += 2.0 * count_above(args[0]) + 2.0
            elif name == "do_put_on":
                if len(args) == 2:
                    if not ((state >> self.on_ids.get((args[0], args[1]), -1)) & 1):
                        cost_clear_x = 2.0 * count_above(args[0])
                        cost_clear_y = 2.0 * count_above(args[1])
                        cost_move = 1.0 if args[0] == held_block else 2.0
                        cost_on_table_y = 0.0 if (state >> self.ontable_ids.get(args[1], -1)) & 1 else 2.0
                        h += cost_clear_x + cost_clear_y + cost_on_table_y + cost_move
            elif name == "do_move":
                if len(args) >= 2:
                    h += 1.0 if args[0] == held_block else 2.0
            else:
                h += 1.0 # Default fallback

        return int(h + 0.001 * len(task_network))
