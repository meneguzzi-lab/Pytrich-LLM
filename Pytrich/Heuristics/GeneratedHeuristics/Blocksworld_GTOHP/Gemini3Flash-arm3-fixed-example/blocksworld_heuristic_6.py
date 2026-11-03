"""
Domain-dependent heuristic for the Blocksworld_GTOHP HDDL domain.

Bottleneck Analysis:
The primary bottleneck in Blocksworld is the single-hand constraint and stack dependency. 
To move any block x, all blocks currently on top of it must be moved to the table first. 
Each such removal requires 2 actions (unstack and put-down). Additionally, placing a block 
at its goal requires its destination to be clear and stable.

Admissible Lower Bounds:
1. Task Network Cost (h_tn):
   Each task in the HTN network represents a sequence of unavoidable actions.
   - do_clear(x): Costs 2 * depth(x) actions.
   - do_put_on(x, y): If not satisfied, costs clearing both plus 2-4 actions for placement.
   - do_move(x, y): Always costs 2 actions (pick/unstack + stack).
   Summing these is admissible as the HTN decompositions are largely sequential for goal-oriented tasks.

2. Goal-State Misplaced Blocks (h_goal):
   Trace stacks from the table up. A block is stable if it satisfies its goal and sits on a stable base.
   Every misplaced block (unstable) must be moved at least once to reached its goal position.
   h_goal = 2 * (number of misplaced blocks) - 1 (if currently holding a block).

Heuristic Value = max(h_tn, h_goal)
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic6(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_6=None, name="BlocksworldGTOHPGemini3FlashHeuristic6"):
        super().__init__(name=name)
        self.on_ids = {}            # (top, base) -> global_id
        self.ontable_ids = {}       # block -> global_id
        self.clear_ids = {}         # block -> global_id
        self.holding_ids = {}       # block -> global_id
        self.handempty_id = None
        
        self.fact_id_to_on = {}     # fid -> (top, base)
        self.fact_id_to_holding = {}# fid -> block
        self.fact_id_to_ontable = {}# fid -> block
        
        self.all_blocks = set()
        self.goal_on = {}           # block -> target_base
        self.goal_ontable = set()   # blocks that should be on table

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, [a.strip() for a in args]
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts: return parts[0], parts[1:]
        return None, []

    def _parse_task_name(self, name: str):
        name = name.strip().replace("(", "").replace(")", "")
        parts = name.split()
        if not parts: return None, []
        return parts[0], parts[1:]

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred or not args: continue
            self.all_blocks.update(args)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
                self.fact_id_to_on[f.global_id] = (args[0], args[1])
            elif pred == "ontable":
                self.ontable_ids[args[0]] = f.global_id
                self.fact_id_to_ontable[f.global_id] = args[0]
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "holding":
                self.holding_ids[args[0]] = f.global_id
                self.fact_id_to_holding[f.global_id] = args[0]
            elif pred == "handempty":
                self.handempty_id = f.global_id

        if hasattr(model, 'goals') and model.goals:
            goal_fact_names = model.state_explicit_repr(model.goals)
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "on": self.goal_on[args[0]] = args[1]
                elif pred == "ontable": self.goal_ontable.add(args[0])

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        current_on_top = {}
        current_ontable = set()
        current_holding = None
        
        temp_state = state
        while temp_state:
            fid = (temp_state & -temp_state).bit_length() - 1
            if fid in self.fact_id_to_on:
                t, b = self.fact_id_to_on[fid]
                current_on_top[b] = t
            elif fid in self.fact_id_to_holding:
                current_holding = self.fact_id_to_holding[fid]
            elif fid in self.fact_id_to_ontable:
                current_ontable.add(self.fact_id_to_ontable[fid])
            temp_state &= temp_state - 1

        def get_depth(block):
            d = 0
            curr = block
            while curr in current_on_top:
                d += 1
                curr = current_on_top[curr]
            return d

        tn_cost = 0
        for task in task_network:
            tname, targs = self._parse_task_name(task.name)
            if not tname: continue
            if "nop" in tname: continue
            
            if tname == "do_put_on":
                x, y = targs[0], targs[1]
                if current_on_top.get(y) == x: continue
                cost = get_depth(x) * 2 + get_depth(y) * 2 + 2
                if y not in current_ontable: cost += 2
                tn_cost += cost
            elif tname == "do_on_table":
                x = targs[0]
                if x in current_ontable: continue
                tn_cost += get_depth(x) * 2 + 2
            elif tname == "do_clear":
                tn_cost += get_depth(targs[0]) * 2
            elif tname == "do_move":
                tn_cost += 2
            else:
                tn_cost += 1

        stable = set()
        for b in current_ontable:
            if b in self.goal_on: continue
            stable.add(b)
            curr = b
            while True:
                top = current_on_top.get(curr)
                if not top or self.goal_on.get(top) != curr: break
                stable.add(top)
                curr = top
        
        h_goal = 2 * (len(self.all_blocks) - len(stable))
        if current_holding: h_goal = max(0, h_goal - 1)

        return max(tn_cost, h_goal)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value}
