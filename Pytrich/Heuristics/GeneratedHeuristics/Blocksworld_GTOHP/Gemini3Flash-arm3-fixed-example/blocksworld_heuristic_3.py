"""
Domain-dependent heuristic for the Blocksworld_GTOHP HTN planning domain.

Bottleneck analysis:
In this Blocksworld variant, the bottleneck is the sequential nature of stacks. 
A block can only be moved if it is clear, and in many methods, the destination 
block must also be placed on the table before stacking. To perform any move 
or clearing operation, the hand must also be empty.

Admissibility sketches:
1. do_clear(x): Each block above x requires at least one unstack and one 
   put-down action. Minimum cost is 2 * height(x).
2. do_on_table(x): If x is not already on the table, it must be cleared 
   (2 * height(x)) and then moved to the table (unstack + put-down = 2 actions).
3. do_put_on(x, y): If not already on(x, y), it requires clearing x (2*hx), 
   clearing y (2*hy), potentially moving y to the table (2), and moving x to y (2).
4. Hand status: If holding a block, an additional put-down or stack action is 
   mandatory to return to a state where standard tasks (which require handempty) 
   can proceed.

Total cost is estimated by summing these lower bounds for all tasks in the 
remaining task network.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Operator

class BlocksworldGTOHPGemini3FlashHeuristic3(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_3=None, name="BlocksworldGTOHPGemini3FlashHeuristic3"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Fact and mapping lookups
        self.on_id_to_args = {}    # fact_id -> (over, under)
        self.on_fact_to_id = {}    # (over, under) -> fact_id
        self.ontable_lookup = {}   # block -> fact_id
        self.handempty_id = -1
        self.all_blocks = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on" and len(args) == 2:
                self.on_id_to_args[fact.global_id] = (args[0], args[1])
                self.on_fact_to_id[(args[0], args[1])] = fact.global_id
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_lookup[args[0]] = fact.global_id
                self.all_blocks.add(args[0])
            elif pred == "handempty":
                self.handempty_id = fact.global_id
            elif pred in ["holding", "clear"]:
                for a in args: self.all_blocks.add(a)

        self.preprocessing_time = time.time() - start_time
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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
        name = name.strip()
        if name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            if not parts: return None, []
            return parts[0], parts[1:]
        parts = name.split()
        if not parts: return None, []
        return parts[0], parts[1:]

    def _compute(self, state: int, task_network) -> int:
        # 1. Build on-map to trace stacks efficiently
        on_map = {} # under -> over
        for f_id, (over, under) in self.on_id_to_args.items():
            if (state >> f_id) & 1:
                on_map[under] = over

        # 2. Pre-calculate heights (number of blocks above) with memoization
        heights = {}
        def get_h(b):
            if b not in heights:
                if b in on_map:
                    heights[b] = 1 + get_h(on_map[b])
                else:
                    heights[b] = 0
            return heights[b]
        
        for b in self.all_blocks: get_h(b)

        # 3. Sum costs of remaining tasks
        total_cost = 0
        
        # Penalty for holding a block
        if self.handempty_id != -1:
            if not ((state >> self.handempty_id) & 1):
                total_cost += 1

        for task in task_network:
            if isinstance(task, Operator):
                total_cost += task.cost
                continue
            
            pred, args = self._parse_task_name(task.name)
            if not pred or pred == "nop":
                continue

            if pred == "do_put_on" and len(args) == 2:
                x, y = args[0], args[1]
                f_id = self.on_fact_to_id.get((x, y))
                if f_id is not None and (state >> f_id) & 1:
                    pass # Task already satisfied
                else:
                    hx = heights.get(x, 0)
                    hy = heights.get(y, 0)
                    on_table_f = self.ontable_lookup.get(y)
                    y_on_table = on_table_f is not None and (state >> on_table_f) & 1
                    # Cost: Clear x (2*hx) + Clear y (2*hy) + Put y on table (2 if not on table) + Move x (2)
                    total_cost += 2*hx + 2*hy + (0 if y_on_table else 2) + 2
            
            elif pred == "do_clear" and len(args) == 1:
                total_cost += 2 * heights.get(args[0], 0)
            
            elif pred == "do_on_table" and len(args) == 1:
                x = args[0]
                on_table_f = self.ontable_lookup.get(x)
                if on_table_f is not None and (state >> on_table_f) & 1:
                    pass
                else:
                    total_cost += 2 * heights.get(x, 0) + 2
            
            elif pred == "do_move" and len(args) == 2:
                total_cost += 2 # Standard move actions (pick/unstack + stack)
            
            else:
                # Default for unknown abstract tasks
                total_cost += 1
                
        return total_cost

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
