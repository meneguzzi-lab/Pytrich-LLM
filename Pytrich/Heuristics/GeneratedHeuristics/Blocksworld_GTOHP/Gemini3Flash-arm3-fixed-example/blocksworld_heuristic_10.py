"""
Domain-dependent heuristic for the Blocksworld_GTOHP domain.

Bottleneck Analysis:
In Blocksworld, the dominant serializing constraint is the stack hierarchy. To move a block 'X' 
or place a block on 'Y', both 'X' and 'Y' must be clear. Clearing a block requires removing 
all blocks stacked above it. Each removal (unstack + put-down) costs 2 actions. Moving 
a block from its current location to a new one (pick-up/unstack + stack) costs 2 actions.

Admissible Lower Bounds:
1. Task Cost (Movements): Every 'do_put_on(x, y)' where x is not on y and every 'do_move(x, y)' 
   requires at least 2 actions (costing 2). If the block is already held, it costs 1 (stack).
2. Clearing Obstructions: For every block 'X' that must be clear (as a moving block or 
   a target), every block currently above it must be removed (costing 2 per block).
3. Hand Management: If the hand is not empty and we need to move or clear other blocks, 
   we must first 'put-down' or 'stack' the current block (costing 1).

Admissibility Sketch:
This heuristic is a lower bound because it only counts the minimum required physical actions 
(pick-up, put-down, stack, unstack) to satisfy the preconditions of the task network tasks.
By using sets to track 'blocks_to_move' and 'all_above', we avoid double-counting actions 
that satisfy multiple clearing requirements simultaneously.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic10(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_10=None, name="BlocksworldGTOHPGemini3FlashHeuristic10"):
        super().__init__(name=name)
        self.on_fact_info = {}       # bit_pos -> (block, target)
        self.ontable_fact_ids = {}   # block -> global_id
        self.clear_fact_ids = {}     # block -> global_id
        self.holding_fact_ids = {}   # block -> global_id
        self.blocks = []
        self.task_data_cache = {}    # task_global_id -> (name, args)

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        blocks_found = set()

        for i, fact in enumerate(model.facts):
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_fact_info[i] = (args[0], args[1])
                blocks_found.add(args[0])
                blocks_found.add(args[1])
            elif pred == "ontable":
                self.ontable_fact_ids[args[0]] = i
                blocks_found.add(args[0])
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = i
                blocks_found.add(args[0])
            elif pred == "holding":
                self.holding_fact_ids[args[0]] = i
                blocks_found.add(args[0])

        self.blocks = list(blocks_found)
        
        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

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

    def _parse_task_name(self, name: str):
        s = name.strip("() ")
        parts = s.split()
        if not parts: return None, []
        return parts[0], parts[1:]

    def _get_state_info(self, state):
        on_map = {}  # target -> block_on_top
        holding = None
        temp_state = state
        while temp_state:
            lsb = temp_state & -temp_state
            bit_pos = lsb.bit_length() - 1
            if bit_pos in self.on_fact_info:
                block, target = self.on_fact_info[bit_pos]
                on_map[target] = block
            temp_state ^= lsb
        
        for b in self.blocks:
            if b in self.holding_fact_ids and (state >> self.holding_fact_ids[b]) & 1:
                holding = b
                break
        return on_map, holding

    def _get_all_above(self, block, on_map):
        above_set = set()
        curr = block
        while curr in on_map:
            above = on_map[curr]
            if above in above_set: break # safety
            above_set.add(above)
            curr = above
        return above_set

    def _compute(self, state, task_network):
        blocks_to_move = set()
        blocks_to_clear = set()
        h = 0

        for task in task_network:
            if task.global_id not in self.task_data_cache:
                self.task_data_cache[task.global_id] = self._parse_task_name(task.name)
            
            t_name, t_args = self.task_data_cache[task.global_id]
            
            if t_name == "do_put_on":
                x, y = t_args[0], t_args[1]
                # Check if on(x, y) is true
                is_on = False
                for bit_pos, (b, t) in self.on_fact_info.items():
                    if b == x and t == y:
                        if (state >> bit_pos) & 1:
                            is_on = True
                        break
                if not is_on:
                    blocks_to_move.add(x)
                    blocks_to_clear.add(x)
                    blocks_to_clear.add(y)
            elif t_name == "do_clear":
                blocks_to_clear.add(t_args[0])
            elif t_name == "do_move":
                blocks_to_move.add(t_args[0])
                blocks_to_clear.add(t_args[0])
                blocks_to_clear.add(t_args[1])
            elif hasattr(task, 'cost'):
                h += task.cost

        on_map, holding = self._get_state_info(state)
        all_above = set()
        for b in blocks_to_clear:
            all_above.update(self._get_all_above(b, on_map))

        h += 2 * len(blocks_to_move)
        h += 2 * len(all_above)

        if holding:
            if holding in blocks_to_move:
                h -= 1
            elif blocks_to_move or all_above:
                h += 1

        return max(0, h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
