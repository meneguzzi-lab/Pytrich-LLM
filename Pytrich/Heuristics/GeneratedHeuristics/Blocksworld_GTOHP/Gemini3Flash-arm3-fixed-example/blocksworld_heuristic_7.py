"""
Domain-dependent heuristic for the Blocksworld_GTOHP domain.

Bottleneck Analysis:
Blocksworld is dominated by the serial constraint of the robot hand (holding at most one block) 
and the requirement to clear blocks before they can be moved. In GTOHP, tasks like do_put_on 
and do_move require the blocks involved to be clear and potentially on the table first.

Admissibility Sketches:
1. Task Network Cost (tn_cost): Every task in the task network (do_put_on, do_move, do_clear, etc.) 
   must eventually be decomposed into primitive actions. For example, do_move requires at least 
   2 actions (pick-up + stack). do_clear requires 2 actions (unstack + put-down) for every block 
   above the target. This is a lower bound on unavoidable action cost.
2. Misplaced Blocks Cost (goal_h): Every block not currently at its final goal destination (or 
   not sitting on a stable base) must be moved at least once. Each move involves at least 2 actions 
   (unstack/pick-up and stack/put-down). This is a standard admissible lower bound for Blocksworld.
3. Heuristic Value: max(tn_cost, goal_h). Both are admissible lower bounds, so their maximum is also admissible.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic7(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_7=None, name="BlocksworldGTOHPGemini3FlashHeuristic7"):
        super().__init__(name=name)
        self.on_fact_id_to_args = {}
        self.ontable_fact_ids = {}
        self.ontable_fact_ids_inv = {}
        self.holding_fact_id_to_block = {}
        self.handempty_id = -1
        self.task_info = {}
        self.goal_on = {}
        self.all_block_names = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_fact_id_to_args[f.global_id] = (args[0], args[1])
            elif pred == "ontable":
                self.ontable_fact_ids[args[0]] = f.global_id
                self.ontable_fact_ids_inv[f.global_id] = args[0]
            elif pred == "handempty":
                self.handempty_id = f.global_id
            elif pred == "holding":
                self.holding_fact_id_to_block[f.global_id] = args[0]
            for a in args:
                self.all_block_names.add(a)

        for abt in model.abstract_tasks:
            t_type, t_args = self._parse_task_name(abt.name)
            self.task_info[abt.global_id] = (t_type, t_args)

        if hasattr(self.model, "goals") and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                p, args = self._parse_fact_name(goal_name)
                if p == "on":
                    self.goal_on[args[0]] = args[1]
                elif p == "ontable":
                    self.goal_on[args[0]] = "TABLE"

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

    def _parse_task_name(self, name):
        name = name.strip()
        if name.startswith("(") and name.endswith(")"):
            name = name[1:-1]
        parts = name.split()
        if parts:
            return parts[0], parts[1:]
        return None, []

    def _clear_cost(self, x, on_child):
        count = 0
        curr = x
        while curr in on_child:
            count += 1
            curr = on_child[curr]
        return 2 * count

    def _compute(self, state, task_network):
        on_parent = {}
        on_child = {}
        hand_empty = (state >> self.handempty_id) & 1 if self.handempty_id != -1 else 1
        
        temp_state = state
        while temp_state:
            lsb = (temp_state & -temp_state).bit_length() - 1
            if lsb in self.on_fact_id_to_args:
                u, v = self.on_fact_id_to_args[lsb]
                on_parent[u] = v
                on_child[v] = u
            elif lsb in self.ontable_fact_ids_inv:
                u = self.ontable_fact_ids_inv[lsb]
                on_parent[u] = "TABLE"
            elif lsb in self.holding_fact_id_to_block:
                u = self.holding_fact_id_to_block[lsb]
                on_parent[u] = "HAND"
            temp_state &= temp_state - 1

        tn_cost = 0
        for task in task_network:
            if hasattr(task, "decompositions"):
                t_type, t_args = self.task_info.get(task.global_id, (None, []))
                if t_type == "do_put_on":
                    x, y = t_args[0], t_args[1]
                    if on_parent.get(x) == y:
                        continue
                    cost = 2 + self._clear_cost(x, on_child) + self._clear_cost(y, on_child)
                    if on_parent.get(y) != "TABLE":
                        cost += 2
                    tn_cost += cost
                elif t_type == "do_clear":
                    tn_cost += self._clear_cost(t_args[0], on_child)
                elif t_type == "do_move":
                    x, y = t_args[0], t_args[1]
                    tn_cost += 2 + self._clear_cost(x, on_child) + self._clear_cost(y, on_child)
                elif t_type == "do_on_table":
                    x = t_args[0]
                    if on_parent.get(x) != "TABLE":
                        tn_cost += 2 + self._clear_cost(x, on_child)
                else:
                    tn_cost += 1
            else:
                if not task.name.lower().startswith("nop"):
                    tn_cost += task.cost
        
        if not hand_empty:
            tn_cost += 1

        well_placed = set()
        queue = []
        for b in self.all_block_names:
            target = self.goal_on.get(b)
            if target == "TABLE" or b not in self.goal_on:
                if on_parent.get(b) == "TABLE":
                    well_placed.add(b)
                    queue.append(b)
        
        while queue:
            curr = queue.pop()
            if curr in on_child:
                b = on_child[curr]
                target = self.goal_on.get(b)
                if target == curr or b not in self.goal_on:
                    well_placed.add(b)
                    queue.append(b)
        
        goal_h = 2 * (len(self.all_block_names) - len(well_placed))
        return int(max(tn_cost, goal_h))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
