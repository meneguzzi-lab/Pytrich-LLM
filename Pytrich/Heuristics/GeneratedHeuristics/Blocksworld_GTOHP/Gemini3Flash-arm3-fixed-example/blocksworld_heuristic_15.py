import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic15(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Bottleneck Analysis:
    The bottleneck in Blocksworld is the serialized movement of blocks; only one block can be held 
    at a time, and a block must be clear before it can be moved. Stacks must be disassembled 
    to reach lower blocks.

    Admissible Lower Bounds:
    1. Task Network Cost: The sum of the minimum number of actions required to decompose or execute 
       every task currently in the task network. This is admissible as these tasks represent 
       disjoint work required by the HTN decomposition.
    2. Stack Clearing Cost: To clear a block, every block above it must be unstacked and 
       put down (2 actions per block). This is a strict lower bound on clearing work.
    3. Movement Cost: Each move or put-on task requires at least a pick/stack or unstack/stack 
       sequence (2 actions).
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_15=None, name="BlocksworldGTOHPGemini3FlashHeuristic15"):
        super().__init__(name=name)
        self.on_fact_ids = {}      # (block, target) -> global_id
        self.clear_fact_ids = {}   # block -> global_id
        self.ontable_fact_ids = {} # block -> global_id
        self.handempty_id = -1
        self.on_above_map = defaultdict(list) # block -> [(top_block, fact_id)]
        self.task_info_map = {}    # task_global_id -> {'name': str, 'args': list}

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Pre-parse all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            if pred == "on":
                self.on_fact_ids[(args[0], args[1])] = f.global_id
                self.on_above_map[args[1]].append((args[0], f.global_id))
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = f.global_id
            elif pred == "ontable":
                self.ontable_fact_ids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id

        # Pre-parse all abstract tasks and operators
        for task in model.abstract_tasks:
            name, args = self._parse_fact_name(task.name)
            self.task_info_map[task.global_id] = {'name': name, 'args': args}
        for op in model.operators:
            name, args = self._parse_fact_name(op.name)
            self.task_info_map[op.global_id] = {'name': name, 'args': args}

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        h = 0
        # Penalty for not having an empty hand
        if self.handempty_id != -1 and not ((state >> self.handempty_id) & 1):
            h += 1
            
        for task in task_network:
            h += self._estimate_task_cost(state, task.global_id)
        return h

    def _estimate_task_cost(self, state, task_id):
        info = self.task_info_map.get(task_id)
        if not info: return 0
        
        name = info['name']
        args = info['args']

        if name == "do_put_on":
            x, y = args[0], args[1]
            # m0: already on target
            on_id = self.on_fact_ids.get((x, y))
            if on_id is not None and (state >> on_id) & 1:
                return 0
            # m1: cost(clear x) + cost(clear y) + cost(on_table y) + cost(move x y)
            return self._estimate_clear_cost(state, x) + \
                   self._estimate_clear_cost(state, y) + \
                   self._estimate_on_table_cost(state, y) + 2
        
        elif name == "do_on_table":
            return self._estimate_on_table_cost(state, args[0])
            
        elif name == "do_clear":
            return self._estimate_clear_cost(state, args[0])
            
        elif name == "do_move":
            return 2 # pick/unstack + stack
            
        elif name in ["pick-up", "put-down", "stack", "unstack"]:
            return 1
            
        return 0

    def _estimate_clear_cost(self, state, block):
        cost = 0
        curr = block
        while True:
            clear_id = self.clear_fact_ids.get(curr)
            if clear_id is not None and (state >> clear_id) & 1:
                return cost
            
            found_top = False
            for top_block, f_id in self.on_above_map.get(curr, []):
                if (state >> f_id) & 1:
                    cost += 2 # unstack + put-down
                    curr = top_block
                    found_top = True
                    break
            if not found_top: break
        return cost

    def _estimate_on_table_cost(self, state, block):
        ontable_id = self.ontable_fact_ids.get(block)
        if ontable_id is not None and (state >> ontable_id) & 1:
            return 0
        return 2 + self._estimate_clear_cost(state, block)

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
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []
