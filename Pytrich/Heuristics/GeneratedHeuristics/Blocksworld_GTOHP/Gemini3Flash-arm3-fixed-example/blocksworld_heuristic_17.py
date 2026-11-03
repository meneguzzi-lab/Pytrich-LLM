import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Blocksworld_GTOHP domain.

1. Bottleneck Analysis:
   The primary bottleneck is the robotic arm (serial constraint) and the stack structure 
   of the blocks (only the top block is 'clear'). Moving a block deep in a stack requires 
   first clearing everything above it, and each move requires a sequence of primitive actions.

2. Admissible Lower Bounds:
   Term 1 (State-Goal Cost): Every 'on' goal fact not currently true in the state requires at least 
   2 primitive actions (unstack/pick-up + stack). h_state = 2 * count(unsatisfied 'on' goals).
   Term 2 (Task Network Cost): The HTN task network is a sequential plan of promised work.
   Summing the minimum cost of each task in the network is admissible because they are ordered.
   Cost(do_put_on) = 2, Cost(do_move) = 2, Cost(do_clear) = 1, Cost(primitive) = 1.

3. Admissibility Sketch:
   Both terms are lower bounds. h_state counts the minimum actions to satisfy state goals.
   h_tn counts the minimum actions required by the current decomposition of the task network.
   The max(h_state, h_tn) is admissible because both represent unavoidable work.
"""

class BlocksworldGTOHPGemini3FlashHeuristic17(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_17=None, name="BlocksworldGTOHPGemini3FlashHeuristic17"):
        super().__init__(name=name)
        self.task_info = {}       # global_id -> (type, fact_id_to_check)
        self.on_goal_ids = []     # List of fact IDs for 'on' goals
        self.on_ids = defaultdict(dict) # [x][y] -> global_id
        self.ontable_ids = {}     # [x] -> global_id
        self.clear_ids = {}       # [x] -> global_id
        self.holding_ids = {}     # [x] -> global_id
        self.handempty_id = None

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map predicates to IDs for fast state checking
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_ids[args[0]][args[1]] = f.global_id
            elif pred == "ontable" and len(args) == 1:
                self.ontable_ids[args[0]] = f.global_id
            elif pred == "clear" and len(args) == 1:
                self.clear_ids[args[0]] = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_ids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id

        # 2. Precompute goal fact IDs
        if hasattr(self.model, 'goals') and self.model.goals:
            for bit_pos in range(self.model.goals.bit_length()):
                if (self.model.goals >> bit_pos) & 1:
                    f_name = self.model.facts[bit_pos].name
                    pred, args = self._parse_fact_name(f_name)
                    if pred == "on":
                        self.on_goal_ids.append(bit_pos)

        # 3. Precompute task costs and logic
        all_tasks = list(model.operators) + list(model.abstract_tasks)
        for t in all_tasks:
            name = t.name.strip("() ")
            parts = name.split()
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            if pred == "do_put_on" and len(args) == 2:
                f_id = self.on_ids.get(args[0], {}).get(args[1])
                self.task_info[t.global_id] = ('put_on', f_id)
            elif pred == "do_clear" and len(args) == 1:
                f_id = self.clear_ids.get(args[0])
                self.task_info[t.global_id] = ('clear', f_id)
            elif pred == "do_on_table" and len(args) == 1:
                f_id = self.ontable_ids.get(args[0])
                self.task_info[t.global_id] = ('ontable', f_id)
            elif pred == "do_move":
                self.task_info[t.global_id] = ('move', None)
            elif pred == "nop":
                self.task_info[t.global_id] = ('nop', None)
            else:
                # Primitive actions or other abstract tasks
                self.task_info[t.global_id] = ('prim', None)

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

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

    def _compute(self, state, task_network):
        # Term 1: State Goal Consistency
        h_state = 0
        for f_id in self.on_goal_ids:
            if not (state >> f_id) & 1:
                h_state += 2

        # Term 2: Task Network Promised Work
        h_tn = 0
        for t in task_network:
            info = self.task_info.get(t.global_id)
            if not info:
                continue
            t_type, f_id = info
            if t_type == 'put_on':
                if f_id is not None and not (state >> f_id) & 1: h_tn += 2
            elif t_type == 'clear':
                if f_id is not None and not (state >> f_id) & 1: h_tn += 1
            elif t_type == 'ontable':
                if f_id is not None and not (state >> f_id) & 1: h_tn += 1
            elif t_type == 'move':
                h_tn += 2
            elif t_type == 'prim':
                h_tn += 1
            # 'nop' costs 0

        # Use the maximum of the two admissible terms
        # Add a tiny tie-breaker for task network depth
        return max(h_state, h_tn) + 0.001 * len(task_network)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = int(self._compute(node.state, node.task_network))
        super().update_info(h_value)
        return h_value
