"""
Domain-dependent heuristic for the Blocksworld_GTOHP domain.

Bottleneck Analysis:
The primary bottleneck in Blocksworld is the serial nature of block manipulation (only one block can be held at a time)
and the requirement that a block must be clear and its destination must be clear before a move can occur.

Admissibility Sketches:
1. Unplaced Blocks: Every block not in its goal position must be moved at least once (2 actions: unstack/pickup + stack/put-down).
2. Clearing Cost: Every block currently on top of an unstable block (one that must move) or on top of a goal target 
   must be moved at least once (2 actions) to clear the block below.
3. Admissibility: h = 2 * len(must_move_set) - (1 if holding a block else 0). This is a lower bound because each move 
   is exactly 2 actions (or 1 if already holding).
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic4(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_4=None, name="BlocksworldGTOHPGemini3FlashHeuristic4"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        # Precomputed fact mapping
        self.on_ids = {} # (x, y) -> global_id
        self.ontable_ids = {} # x -> global_id
        self.holding_ids = {} # x -> global_id
        self.fact_id_to_on_relation = {} # global_id -> (x, y)
        self.fact_id_to_ontable = {} # global_id -> x
        self.fact_id_to_holding = {} # global_id -> x
        self.blocks = []
        self.goal_on = {} # x -> target (y or 'table')

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        # 1. Parse all facts to build efficient lookups
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_ids[(args[0], args[1])] = f.global_id
                self.fact_id_to_on_relation[f.global_id] = (args[0], args[1])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_ids[args[0]] = f.global_id
                self.fact_id_to_ontable[f.global_id] = args[0]
            elif pred == "holding" and len(args) == 1:
                self.holding_ids[args[0]] = f.global_id
                self.fact_id_to_holding[f.global_id] = args[0]

        self.blocks = list(self.ontable_ids.keys())

        # 2. Extract Goal Configuration
        if hasattr(model, 'goals') and model.goals:
            goal_fact_names = model.state_explicit_repr(model.goals)
            for g_name in goal_fact_names:
                pred, args = self._parse_fact_name(g_name)
                if pred == "on" and len(args) == 2:
                    self.goal_on[args[0]] = args[1]
                elif pred == "ontable" and len(args) == 1:
                    self.goal_on[args[0]] = "table"
        
        # If goal_on is empty, extract from HTN root tasks (common in GTOHP)
        if not self.goal_on:
            for task in initial_node.task_network:
                if hasattr(task, 'name'):
                    parts = task.name.split()
                    if parts[0] == "do_put_on" and len(parts) == 3:
                        self.goal_on[parts[1]] = parts[2]
                    elif parts[0] == "do_on_table" and len(parts) == 2:
                        self.goal_on[parts[1]] = "table"

        self.preprocessing_time = time.time() - start_time
        h_val = self._calculate_h(initial_node.state)
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
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts: return parts[0], parts[1:]
        return None, []

    def _is_stable(self, x, current_on, current_ontable, stable_cache):
        if x in stable_cache: return stable_cache[x]
        if x not in self.goal_on:
            stable_cache[x] = True
            return True
        
        target = self.goal_on[x]
        if target == "table":
            res = (x in current_ontable)
        else:
            if current_on.get(x) == target:
                res = self._is_stable(target, current_on, current_ontable, stable_cache)
            else:
                res = False
        stable_cache[x] = res
        return res

    def _calculate_h(self, state: int) -> int:
        # Fast parse of current state
        current_on = {}
        current_ontable = set()
        current_above = {}
        is_holding = False

        temp_state = state
        while temp_state:
            fid = (temp_state & -temp_state).bit_length() - 1
            if fid in self.fact_id_to_on_relation:
                x, y = self.fact_id_to_on_relation[fid]
                current_on[x] = y
                current_above[y] = x
            elif fid in self.fact_id_to_ontable:
                current_ontable.add(self.fact_id_to_ontable[fid])
            elif fid in self.fact_id_to_holding:
                is_holding = True
            temp_state &= temp_state - 1

        stable_cache = {}
        must_move = set()

        # 1. Blocks not in stable goal position
        for x in self.blocks:
            if not self._is_stable(x, current_on, current_ontable, stable_cache):
                must_move.add(x)

        # 2. Expand must_move to include anything above them or above their target
        final_must_move = set(must_move)
        for x in must_move:
            # Add everything above x in current state
            curr = x
            while curr in current_above:
                curr = current_above[curr]
                if curr in final_must_move: break
                final_must_move.add(curr)
            
            # Add everything above target in current state (target must be clear)
            target = self.goal_on.get(x)
            if target and target != "table":
                curr = target
                while curr in current_above:
                    curr = current_above[curr]
                    if curr in final_must_move: break
                    final_must_move.add(curr)

        h = 2 * len(final_must_move)
        if is_holding: h -= 1
        return max(0, h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._calculate_h(node.state)
        super().update_info(h_val)
        return h_val
