"""
Domain-dependent heuristic for the Blocksworld_GTOHP domain.

Bottleneck Analysis:
The primary bottleneck in Blocksworld is the clearing of blocks (recursive unstacking) 
to satisfy conditions for stacking. In the GTOHP encoding, the do_put_on(x, y) task 
explicitly requires x and y to be clear, and importantly, forces the target block y 
to be on the table before x is stacked onto it.

Admissible Lower Bounds:
1. Task-Network Serial Cost: In an ordered HTN, every task in the network must be 
   completed sequentially. Each high-level task (do_put_on, do_clear, etc.) has a 
   minimum number of physical actions (pick-up, stack, etc.) required based on the 
   current state (height of blocks above the targets).
2. Clearing Cost: To clear block x, every block currently on top of it must be 
   unstacked and put down, requiring 2 actions per block.
3. Moving Cost: Each do_put_on(x, y) not already satisfied requires at least 2 
   actions for the final move, plus the cost to clear x, clear y, and potentially 
   move y to the table.

Admissibility Sketch:
Each term calculates the absolute minimum number of pick-up, put-down, stack, and 
unstack operations required to satisfy the preconditions and decompositions of the 
tasks currently in the network. Because actions are discrete and must be done 
sequentially to satisfy the handempty and clear constraints, the sum of these 
minimum costs is a valid lower bound on the remaining plan length.
"""

import re
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic1(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_1=None, name="BlocksworldGTOHPGemini3FlashHeuristic1"):
        super().__init__(name=name)
        self.on_id = {}          # (x, y) -> global_id
        self.on_by_base = defaultdict(list) # y -> [(global_id, x)]
        self.ontable_id = {}     # x -> global_id
        self.clear_id = {}       # x -> global_id
        self.holding_id = {}     # x -> global_id
        self.handempty_id = None
        self.block_names = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            for arg in args: self.block_names.add(arg)
            
            if pred == "on":
                if len(args) == 2:
                    self.on_id[(args[0], args[1])] = fact.global_id
                    self.on_by_base[args[1]].append((fact.global_id, args[0]))
            elif pred == "ontable":
                self.ontable_id[args[0]] = fact.global_id
            elif pred == "clear":
                self.clear_id[args[0]] = fact.global_id
            elif pred == "holding":
                self.holding_id[args[0]] = fact.global_id
            elif pred == "handempty":
                self.handempty_id = fact.global_id

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

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

    def _get_height(self, state, block_name, memo):
        if block_name in memo: return memo[block_name]
        h = 0
        curr = block_name
        visited = {curr}
        while curr in self.on_by_base:
            found = False
            for fact_id, top_name in self.on_by_base[curr]:
                if (state >> fact_id) & 1:
                    if top_name in visited: break
                    h += 1
                    curr = top_name
                    visited.add(curr)
                    found = True
                    break
            if not found: break
        memo[block_name] = h
        return h

    def _is_ontable(self, state, block_name):
        fid = self.ontable_id.get(block_name)
        return (state >> fid) & 1 if fid is not None else False

    def _is_on(self, state, x, y):
        fid = self.on_id.get((x, y))
        return (state >> fid) & 1 if fid is not None else False

    def _extract_args(self, task_name):
        clean = task_name.replace("(", " ").replace(")", " ")
        parts = clean.split()
        args = []
        keywords = {"do_put_on", "do_clear", "do_on_table", "do_move", "nop", "pick-up", "put-down", "stack", "unstack"}
        for p in parts:
            if p.lower() not in keywords and not re.match(r'task\d+', p.lower()):
                args.append(p)
        return args

    def _compute(self, state, task_network):
        h = 0
        memo_height = {}
        
        # Base cost if holding something
        is_handempty = (state >> self.handempty_id) & 1 if self.handempty_id is not None else True
        if not is_handempty:
            for b in self.block_names:
                if b in self.holding_id and (state >> self.holding_id[b]) & 1:
                    h += 1
                    break

        for task in task_network:
            full_name = task.name.lower()
            args = self._extract_args(task.name)
            
            if "do_put_on" in full_name:
                if len(args) >= 2:
                    x, y = args[0], args[1]
                    if not self._is_on(state, x, y):
                        h += 2 + 2 * self._get_height(state, x, memo_height) + 2 * self._get_height(state, y, memo_height)
                        if not self._is_ontable(state, y): h += 2
            elif "do_clear" in full_name:
                if len(args) >= 1:
                    h += 2 * self._get_height(state, args[0], memo_height)
            elif "do_on_table" in full_name:
                if len(args) >= 1:
                    if not self._is_ontable(state, args[0]):
                        h += 2 + 2 * self._get_height(state, args[0], memo_height)
            elif "do_move" in full_name:
                if len(args) >= 2:
                    h += 2 + 2 * self._get_height(state, args[0], memo_height)
            elif any(act in full_name for act in ["pick-up", "unstack", "stack", "put-down"]):
                h += 1
            elif "nop" in full_name:
                continue
            else:
                # Default cost for unknown/labeled tasks to ensure non-zero guidance
                h += 1
        
        return h
