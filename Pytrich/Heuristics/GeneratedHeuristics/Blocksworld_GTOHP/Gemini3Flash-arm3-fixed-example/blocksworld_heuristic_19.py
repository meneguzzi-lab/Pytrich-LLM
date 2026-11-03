"""
Domain-dependent heuristic for the Blocksworld_GTOHP domain.

1. Bottleneck Analysis:
   The primary bottleneck in this domain is the serialization of moves. To move a block 'x' onto 'y', 
   both 'x' and 'y' must be cleared of any blocks currently on top of them. Clearing a single block 
   requires recursively unstacking every block above it and putting it down, which costs exactly 
   2 actions (unstack + put-down) per block. Thus, the cost is dominated by the height of the stacks 
   obstructing the required blocks.

2. Admissible Lower Bounds:
   - Term 1 (Task Network Cost): Since the task network represents an ordered sequence of goals/tasks, 
     the total cost is the sum of the minimum costs to satisfy each remaining task. 
   - Term 2 (Decomposition Cost): For any 'do_put_on(x, y)' task, a lower bound on the cost is 
     2 * (unique blocks above x or y) + 2 (the pick-up and stack of x itself). This is admissible 
     because these actions are strictly necessary to satisfy the GTOHP decomposition.
   - Term 3 (State Penalty): A sub-unit penalty is added for nodes where the hand is not empty, 
     as an additional action (stack or put-down) will be required to clear the hand.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

class BlocksworldGTOHPGemini3FlashHeuristic19(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_19=None, name="BlocksworldGTOHPGemini3FlashHeuristic19"):
        super().__init__(name=name)
        self.on_facts = []  # List of (fact_id, top_block, bottom_block)
        self.on_fact_id = {} # (x, y) -> global_id
        self.clear_fact_id = {} # x -> global_id
        self.ontable_fact_id = {} # x -> global_id
        self.handempty_id = -1
        self.blocks = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
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

    def _preprocess(self):
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_facts.append((f.global_id, args[0], args[1]))
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.blocks.add(args[0])
                self.blocks.add(args[1])
            elif pred == "clear":
                self.clear_fact_id[args[0]] = f.global_id
                self.blocks.add(args[0])
            elif pred == "ontable":
                self.ontable_fact_id[args[0]] = f.global_id
                self.blocks.add(args[0])
            elif pred == "handempty":
                self.handempty_id = f.global_id

    def _compute(self, state: int, task_network) -> int:
        # 1. Map current "on" relations to build stacks efficiently
        top_of = {}
        for fid, top, bottom in self.on_facts:
            if (state >> fid) & 1:
                top_of[bottom] = top

        # 2. Cache for stack contents to avoid O(N^2)
        memo_above = {}

        def get_above(block):
            if block in memo_above: return memo_above[block]
            above = set()
            curr = block
            while curr in top_of:
                curr = top_of[curr]
                above.add(curr)
            memo_above[block] = above
            return above

        total_h = 0.0
        
        # Admissible base penalty for hand state
        if self.handempty_id != -1 and not ((state >> self.handempty_id) & 1):
            total_h += 1.0

        for task in task_network:
            # Pytrich Operator or AbstractTask
            if hasattr(task, 'decompositions'): # AbstractTask
                name_parts = task.name.split()
                verb = name_parts[0].lower()
                args = name_parts[1:]

                if verb == "do_put_on":
                    x, y = args[0], args[1]
                    # Method m0 check
                    on_id = self.on_fact_id.get((x, y))
                    if on_id is not None and (state >> on_id) & 1:
                        continue
                    # Method m1 cost: clear x, clear y, then move x to y
                    # move costs 2, clearing each costs 2 * len(above)
                    # We union the stacks to avoid double-counting if x and y are in the same stack
                    above_x = get_above(x)
                    above_y = get_above(y)
                    # Blocks above x cannot include y if we are moving x to y (and vice versa)
                    # but the union handles this naturally.
                    total_h += 2 * len(above_x | above_y) + 2

                elif verb == "do_clear":
                    x = args[0]
                    clear_id = self.clear_fact_id.get(x)
                    if clear_id is not None and not ((state >> clear_id) & 1):
                        total_h += 2 * len(get_above(x))

                elif verb == "do_on_table":
                    x = args[0]
                    ontable_id = self.ontable_fact_id.get(x)
                    clear_id = self.clear_fact_id.get(x)
                    # m3: if clear, nop (0). m2: unstack + putdown (2)
                    if not ((state >> ontable_id) & 1 or (state >> clear_id) & 1):
                        total_h += 2

                elif verb == "do_move":
                    total_h += 2
            else: # Operator
                # Standard action costs
                name = task.name.lower()
                if "nop" not in name:
                    total_h += getattr(task, 'cost', 1)

        return int(total_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
