"""
Domain-dependent heuristic for the Blocksworld_GTOHP HDDL domain.

1. Bottleneck Analysis:
   The primary bottleneck in Blocksworld is the serialization of moves. Only one block can be held at a time, 
   and a block can only be moved if both it and its target are clear. This creates a strict dependency chain 
   where stacks must be dismantled and rebuilt block-by-block.

2. Admissible Lower Bounds:
   - Term 1: Unmet Goal State Cost. Each goal (on x y) or (ontable x) that is not currently satisfied 
     requires at least one move operation (pick-up/unstack and stack/put-down), costing at least 2 actions.
   - Term 2: Task Network Execution Cost. Every task in the HTN network represents a sequence of operations. 
     By assigning state-aware minimum costs to these tasks (e.g., do_put_on costs 2 if not satisfied), 
     we get a lower bound on the remaining work to fulfill the current decomposition.
   The heuristic returns max(Term 1, Term 2) to ensure admissibility while maximizing guidance.

3. Preprocessing:
   We parse the grounded abstract tasks and facts once to create a mapping from task IDs to cost functions. 
   These functions evaluate the cost of a task based on the current state bitwise.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic14(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_14=None, name="BlocksworldGTOHPGemini3FlashHeuristic14"):
        super().__init__(name=name)
        self.task_meta = {}  # global_id -> function(state)
        self.goal_fids = []
        self.fact_map = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        # 1. Map all facts for easy lookup
        for f in model.facts:
            self.fact_map[f.name] = f.global_id

        # 2. Pre-process Goal Facts
        if hasattr(model, 'goals') and model.goals:
            for bit_pos in range(model.goals.bit_length()):
                if (model.goals >> bit_pos) & 1:
                    self.goal_fids.append(bit_pos)

        # 3. Pre-process Task Costs
        # This handles Operators and AbstractTasks
        for t in model.operators + model.abstract_tasks:
            self.task_meta[t.global_id] = self._create_task_cost_func(t)

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

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

    def _create_task_cost_func(self, task):
        # Define minimum cost based on task type and current state
        name = task.name
        pred, args = self._parse_fact_name(name)

        # Primitive Operators
        if task.global_id <= self.model.iop_end:
            cost = getattr(task, 'cost', 1)
            return lambda state, c=cost: c

        # Abstract Tasks
        if pred == 'do_put_on' and len(args) == 2:
            x, y = args[0], args[1]
            on_fid = self.fact_map.get(f"+on[{x},{y}]")
            if on_fid is not None:
                return lambda state, fid=on_fid: 0 if (state >> fid) & 1 else 2
            return lambda state: 2

        if pred == 'do_on_table' and len(args) == 1:
            x = args[0]
            ontable_fid = self.fact_map.get(f"+ontable[{x}]")
            clear_fid = self.fact_map.get(f"+clear[{x}]")
            if ontable_fid is not None and clear_fid is not None:
                # m3 says 0 if clear, m2 says 2 if unstack needed
                return lambda state, ofid=ontable_fid, cfid=clear_fid: 0 if ((state >> ofid) & 1) or ((state >> cfid) & 1) else 2
            return lambda state: 2

        if pred == 'do_clear' and len(args) == 1:
            x = args[0]
            clear_fid = self.fact_map.get(f"+clear[{x}]")
            if clear_fid is not None:
                return lambda state, fid=clear_fid: 0 if (state >> fid) & 1 else 2
            return lambda state: 2

        if pred == 'do_move':
            return lambda state: 2

        # Default for other abstract tasks
        return lambda state: 0

    def _compute(self, state, task_network):
        # Term 1: Sum of costs of tasks in the network
        h_tasks = 0
        for task in task_network:
            if task.global_id in self.task_meta:
                h_tasks += self.task_meta[task.global_id](state)
            else:
                # Fallback to model cost if meta not found
                h_tasks += getattr(task, 'cost', 1)

        # Term 2: Misplaced blocks based on goal facts
        h_state = 0
        for fid in self.goal_fids:
            if not ((state >> fid) & 1):
                h_state += 2

        return max(h_tasks, h_state)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
