from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The primary bottleneck in the Blocksworld_GTOHP domain is the serialization of unstacking/stacking actions 
# caused by the 'clear' and 'handempty' constraints. Specifically, the grounded task structure (m1_do_put_on) 
# requires blocks to be clear and the target block to be on the table before moving, making plan length 
# grow linearly with the number of blocks requiring repositioning.

# Admissible Lower Bounds:
# 1. Task Network Cost: Every non-trivial task (do_put_on, do_move, etc.) in the network requires at least 
#    two primitive actions (e.g., pick-up/stack or unstack/put-down) to complete if its preconditions are not met. 
#    Summing these minimum costs provides a lower bound on remaining actions.
# 2. Hand Status: If the hand is not empty, at least one action (put-down or stack) is required to free the 
#    hand to perform any further unstack/pick-up task.

class BlocksworldGTOHPGemini3FlashHeuristic0(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_0="blocksworld_gemini3flash_heuristic_0"):
        super().__init__(name=blocksworld_gemini3flash_heuristic_0)
        self.on_ids = {}      # (x, y) -> global_id
        self.clear_ids = {}   # x -> global_id
        self.ontable_ids = {} # x -> global_id
        self.handempty_id = -1
        self.task_metadata = {} # task_id -> (type, args)

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
        if "[" in name:
            bracket_idx = name.find("[")
            task_type = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return task_type, args
        elif " " in name:
            parts = name.split()
            return parts[0], parts[1:]
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "ontable":
                self.ontable_ids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id

        for t in model.abstract_tasks:
            self.task_metadata[t.global_id] = self._parse_task_name(t.name)
        
        for op in model.operators:
            self.task_metadata[op.global_id] = ("operator", [])

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        h = 0
        
        # Term 2: Hand status
        if self.handempty_id != -1:
            if not ((state >> self.handempty_id) & 1):
                h += 1

        # Term 1: Task decomposition lower bounds
        for task in task_network:
            t_id = task.global_id
            if t_id not in self.task_metadata:
                h += 1
                continue
            
            t_type, t_args = self.task_metadata[t_id]
            
            if t_type == "do_put_on":
                x, y = t_args[0], t_args[1]
                # If not on(x, y), we need at least 2 actions (move). 
                # m1 also requires clear x, clear y, and ontable y.
                on_id = self.on_ids.get((x, y))
                if on_id is not None and not ((state >> on_id) & 1):
                    h += 2
                    # Check subtask requirements from m1
                    if x in self.clear_ids and not ((state >> self.clear_ids[x]) & 1):
                        h += 2
                    if y in self.clear_ids and not ((state >> self.clear_ids[y]) & 1):
                        h += 2
                    if y in self.ontable_ids and not ((state >> self.ontable_ids[y]) & 1):
                        h += 2
            
            elif t_type == "do_clear":
                x = t_args[0]
                if x in self.clear_ids and not ((state >> self.clear_ids[x]) & 1):
                    h += 2
            
            elif t_type == "do_move":
                h += 2
                
            elif t_type == "do_on_table":
                x = t_args[0]
                if x in self.ontable_ids and not ((state >> self.ontable_ids[x]) & 1):
                    h += 2
            
            elif t_type == "operator":
                h += 1
            else:
                h += 1

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
