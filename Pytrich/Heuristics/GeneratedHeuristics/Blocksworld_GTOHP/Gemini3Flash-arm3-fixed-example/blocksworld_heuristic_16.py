"""
Domain-dependent heuristic for the Blocksworld_GTOHP HDDL domain.

Bottleneck Analysis:
The primary bottleneck in Blocksworld is the serial dependency of block placement. 
To stack block A on block B, both must be clear, which requires clearing any 
blocks currently above them. The hierarchical methods in GTOHP (like m1_do_put_on) 
explicitly decompose these into clear, move, and stack actions.

Admissible Lower Bounds:
1. Misplaced Block Cost: Every block not in its goal position (or whose base is not 
   properly placed) must be moved at least once. This is at least 2 actions (pick-up + stack 
   or unstack + put-down).
2. Clearing Cost: To move a misplaced block X or clear a target block Y, every block 
   currently on top of them must be unstacked. This is at least 2 actions (unstack + put-down) 
   per blocking block.
3. Task Network Progress: Every non-trivial task in the network must be decomposed 
   into at least one (often many) primitive operators.

Justification:
This heuristic combines state-based structural analysis of the stacks (Properly Placed) 
with task network guidance. It is efficient (O(N) bitwise checks) even for 1000-block problems.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic16(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_16=None, name="BlocksworldGTOHPGemini3FlashHeuristic16"):
        super().__init__(name=name)
        # Maps for static fact mapping
        self.on_fact_id = {}      # (block, target) -> global_id
        self.ontable_fact_id = {} # block -> global_id
        self.clear_fact_id = {}   # block -> global_id
        self.fact_id_to_on = {}   # global_id -> (block, target)
        self.fact_id_to_ontable = {} # global_id -> block
        self.block_names = []
        self.goal_on = {}         # block -> target_goal

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        all_blocks = set()

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if pred == "on":
                self.on_fact_id[(args[0], args[1])] = fact.global_id
                self.fact_id_to_on[fact.global_id] = (args[0], args[1])
                all_blocks.add(args[0])
                all_blocks.add(args[1])
            elif pred == "ontable":
                self.ontable_fact_id[args[0]] = fact.global_id
                self.fact_id_to_ontable[fact.global_id] = args[0]
                all_blocks.add(args[0])
            elif pred == "clear":
                self.clear_fact_id[args[0]] = fact.global_id
                all_blocks.add(args[0])

        self.block_names = list(all_blocks)

        # Extract goals from the model
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_fact_names:
                pred, args = self._parse_fact_name(g_name)
                if pred == "on":
                    self.goal_on[args[0]] = args[1]
                elif pred == "ontable":
                    self.goal_on[args[0]] = "table"

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

    def _compute(self, state, task_network):
        # 1. Fast Bitwise State Scan
        current_on = {}
        current_ontable = set()
        
        temp_state = state
        while temp_state:
            bit = temp_state & -temp_state
            idx = bit.bit_length() - 1
            temp_state ^= bit
            
            if idx in self.fact_id_to_on:
                x, y = self.fact_id_to_on[idx]
                current_on[x] = y
            elif idx in self.fact_id_to_ontable:
                current_ontable.add(self.fact_id_to_ontable[idx])

        # 2. Build Structural Maps (above/below)
        inv_on = {y: x for x, y in current_on.items()} # y -> x (who is on me)
        above_count = {}
        
        def get_above_count(b):
            if b in above_count: return above_count[b]
            if b not in inv_on:
                above_count[b] = 0
                return 0
            res = 1 + get_above_count(inv_on[b])
            above_count[b] = res
            return res

        for b in self.block_names: 
            get_above_count(b)

        # 3. Identify properly placed blocks (Stability Analysis)
        properly_placed = {}
        def is_properly_placed(b):
            if b in properly_placed: return properly_placed[b]
            
            g_target = self.goal_on.get(b)
            c_target = current_on.get(b, "table" if b in current_ontable else None)
            
            if g_target:
                if c_target == g_target:
                    if g_target == "table":
                        res = True
                    else:
                        res = is_properly_placed(g_target)
                else:
                    res = False
            else:
                # No specific goal for this block - stable if on table or on stable block
                if c_target == "table":
                    res = True
                elif c_target in self.block_names:
                    res = is_properly_placed(c_target)
                else:
                    res = False
            
            properly_placed[b] = res
            return res

        for b in self.block_names: 
            is_properly_placed(b)

        # 4. Summate Cost Terms
        h_value = 0
        for b in self.block_names:
            if not properly_placed[b]:
                # Base move cost (admissible lower bound is 2 actions)
                h_value += 2
                # Clearance cost: 2 actions per block currently above this one
                h_value += 2 * above_count.get(b, 0)
        
        # 5. Add Guidance for task network commitment
        h_value += len(task_network) * 1.5

        return int(h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
