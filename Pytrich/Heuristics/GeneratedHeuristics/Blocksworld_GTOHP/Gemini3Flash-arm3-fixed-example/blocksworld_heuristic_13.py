"""
Domain-dependent heuristic for the Blocksworld_GTOHP domain.

Bottleneck Analysis:
The single robotic arm is the primary bottleneck. Moving any block requires it to be clear
and the arm to be empty. Stack manipulation is serial and recursive; to move a block,
everything above it must be removed. This domain uses a specific HTN structure where 
top-level tasks like 'do_put_on' decompose into clearing and moving subtasks.

Admissibility Sketches:
1. Task Network Cost: Every task in the HTN network represents a set of future actions.
   By assigning a minimal cost to each task (e.g., 2 for do_on_table, 2 for do_move,
   and 2*N for do_clear), we create an admissible estimate of the remaining plan length.
2. State-aware Clearing: A 'do_clear(x)' task must move every block currently above x.
   Since each move requires at least 2 actions (unstack+put-down), the cost 2 * stack_height(x)
   is a tight lower bound.
3. Misplaced Block Penalty: Any block not at its goal position must be moved at least twice.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic13(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_13=None, name="BlocksworldGTOHPGemini3FlashHeuristic13"):
        super().__init__(name=name)
        self.on_above_map = {}      # block_idx -> list of (other_idx, fact_id)
        self.clear_fact_ids = {}    # block_idx -> fact_id
        self.on_fact_ids = {}       # (block_idx, target_idx) -> fact_id
        self.ontable_fact_ids = {}  # block_idx -> fact_id
        self.task_data = {}         # task_global_id -> (type_str, [arg_indices])
        self.block_to_idx = {}
        self.idx_to_block = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Identify all unique block objects
        blocks_found = set()
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred in ['on', 'ontable', 'clear', 'holding']:
                for arg in args:
                    blocks_found.add(arg)
        
        self.idx_to_block = sorted(list(blocks_found))
        self.block_to_idx = {name: i for i, name in enumerate(self.idx_to_block)}
        num_blocks = len(self.idx_to_block)

        # 2. Map facts to indices for fast bitwise checking
        self.on_above_map = {i: [] for i in range(num_blocks)}
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'on' and len(args) == 2:
                b_idx = self.block_to_idx.get(args[0])
                t_idx = self.block_to_idx.get(args[1])
                if b_idx is not None and t_idx is not None:
                    self.on_fact_ids[(b_idx, t_idx)] = fact.global_id
                    self.on_above_map[t_idx].append((b_idx, fact.global_id))
            elif pred == 'clear' and len(args) == 1:
                b_idx = self.block_to_idx.get(args[0])
                if b_idx is not None: self.clear_fact_ids[b_idx] = fact.global_id
            elif pred == 'ontable' and len(args) == 1:
                b_idx = self.block_to_idx.get(args[0])
                if b_idx is not None: self.ontable_fact_ids[b_idx] = fact.global_id

        # 3. Pre-parse task types and arguments
        for task in model.abstract_tasks:
            pred, args = self._parse_fact_name(task.name)
            arg_indices = [self.block_to_idx.get(a) for a in args if a in self.block_to_idx]
            self.task_data[task.global_id] = (pred, arg_indices)
        
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            arg_indices = [self.block_to_idx.get(a) for a in args if a in self.block_to_idx]
            self.task_data[op.global_id] = (pred, arg_indices)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",") if bracket_idx+1 < len(name)-1 else []
                return predicate, args
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            return parts[0], parts[1:]
        return None, []

    def _get_stack_height_above(self, state, block_idx):
        height = 0
        curr = block_idx
        while True:
            above = None
            for other_idx, f_id in self.on_above_map.get(curr, []):
                if (state >> f_id) & 1:
                    above = other_idx
                    break
            if above is not None:
                height += 1
                curr = above
            else:
                break
        return height

    def _compute(self, state, task_network):
        total_h = 0
        for task_obj in task_network:
            t_id = task_obj.global_id
            if t_id not in self.task_data:
                continue
            
            t_type, args = self.task_data[t_id]
            
            if t_type == 'do_put_on':
                x, y = args[0], args[1]
                f_id = self.on_fact_ids.get((x, y))
                if f_id is None or not ((state >> f_id) & 1):
                    # Needs clearing x, clearing y, moving x. 
                    # Method 1 subtasks: do_clear x, do_clear y, do_on_table y, do_move x y
                    # Estimating minimal actions:
                    h_x = 2 * self._get_stack_height_above(state, x)
                    h_y = 2 * self._get_stack_height_above(state, y)
                    total_h += (4 + h_x + h_y)
            
            elif t_type == 'do_clear':
                x = args[0]
                total_h += 2 * self._get_stack_height_above(state, x)
                
            elif t_type == 'do_on_table':
                x = args[0]
                f_id = self.ontable_fact_ids.get(x)
                if f_id is None or not ((state >> f_id) & 1):
                    total_h += 2
            
            elif t_type == 'do_move':
                total_h += 2
                
            elif t_type == 'nop':
                continue
            
            else:
                # Basic cost for unknown tasks or atomic operators
                total_h += 1
        
        return total_h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
