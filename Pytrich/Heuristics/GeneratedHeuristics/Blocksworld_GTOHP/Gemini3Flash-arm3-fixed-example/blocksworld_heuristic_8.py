"""
Domain-dependent heuristic for the Blocksworld_GTOHP domain.

Bottleneck Analysis:
The core bottleneck in Blocksworld is the requirement to clear a block by moving all blocks above it to the table before it can be moved to its destination.

Admissibility Sketches:
1. A block must be moved if its current position is not its goal position or if it is on top of a block that must be moved. Each such block contributes at least 1 action (move).
2. If a block to be moved is currently supported by another block (not the table), it requires an 'unstack' action (1 action).
3. If a block's goal is to be on top of another block, it requires a 'stack' action (1 action).
This heuristic computes the sum of these required actions for all badly-placed blocks.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic8(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_8=None, name="BlocksworldGTOHPGemini3FlashHeuristic8"):
        super().__init__(name=name)
        self.blocks = []
        self.goal_on = {}  # block -> target ('table' or block_name)
        self.on_fact_ids = {}      # block -> {target_block: global_id}
        self.ontable_fact_ids = {} # block -> global_id
        self.holding_fact_ids = {} # block -> global_id
        self.is_destination = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Identify all blocks and map facts
        block_names = set()
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred in ["on", "ontable", "clear", "holding"]:
                for arg in args: block_names.add(arg)
        
        self.blocks = sorted(list(block_names))
        
        for b in self.blocks:
            self.on_fact_ids[b] = {}
            
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_fact_ids[args[0]][args[1]] = fact.global_id
            elif pred == "ontable":
                self.ontable_fact_ids[args[0]] = fact.global_id
            elif pred == "holding":
                self.holding_fact_ids[args[0]] = fact.global_id

        # 2. Extract goals from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_fact_names:
                pred, args = self._parse_fact_name(g_name)
                if pred == "on":
                    self.goal_on[args[0]] = args[1]
                    self.is_destination.add(args[1])
                elif pred == "ontable":
                    self.goal_on[args[0]] = "table"
        
        # 3. Fallback: Parse initial task network for implied goals if model.goals is empty
        if not self.goal_on:
            for task in model.initial_tn:
                pred, args = self._parse_fact_name(task.name)
                if pred == "do_put_on":
                    self.goal_on[args[0]] = args[1]
                    self.is_destination.add(args[1])
                elif pred == "do_on_table":
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
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts: return parts[0], parts[1:]
        return None, []

    def _compute(self, state, task_network):
        # Fast reconstruction of current state
        current_on = {}  # block -> parent
        holding_block = None
        
        for b in self.blocks:
            if b in self.holding_fact_ids and (state >> self.holding_fact_ids[b]) & 1:
                holding_block = b
                continue
            if b in self.ontable_fact_ids and (state >> self.ontable_fact_ids[b]) & 1:
                current_on[b] = "table"
                continue
            for target, f_id in self.on_fact_ids[b].items():
                if (state >> f_id) & 1:
                    current_on[b] = target
                    break

        # Determine well-placed blocks (stable bottom-up)
        well_placed = {}
        def check_well_placed(b):
            if b in well_placed: return well_placed[b]
            if b == holding_block:
                well_placed[b] = False
                return False
            
            curr = current_on.get(b)
            goal = self.goal_on.get(b)
            
            if goal is not None:
                if curr != goal:
                    well_placed[b] = False
                elif curr == "table":
                    well_placed[b] = True
                else:
                    well_placed[b] = check_well_placed(curr)
            else:
                if curr == "table":
                    well_placed[b] = True
                else:
                    # Don't-care block is well-placed if its support is stable and not a target
                    if curr in self.is_destination:
                        well_placed[b] = False
                    else:
                        well_placed[b] = check_well_placed(curr)
            return well_placed[b]

        for b in self.blocks: check_well_placed(b)

        # Calculate heuristic sum
        h = 0
        for b in self.blocks:
            if b == holding_block:
                h += 1
                # If the goal for the held block is a block, it needs a stack action
                if self.goal_on.get(b, "table") != "table":
                    h += 1
            elif not well_placed.get(b, False):
                # A badly-placed block must be moved
                h += 1
                # If it's currently on a block, it needs unstacking
                if current_on.get(b) != "table" and current_on.get(b) is not None:
                    h += 1
                # If it must end up on a block, it needs stacking
                if self.goal_on.get(b, "table") != "table":
                    h += 1
        
        # Tie-breaker: guide search towards completing tasks
        h_tasks = len(task_network) * 0.01
        
        return h + h_tasks

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = int(self._compute(node.state, node.task_network))
        super().update_info(h_value)
        return h_value
