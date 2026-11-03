from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

# Bottleneck: The serial nature of stacks; only the top block is accessible, requiring $O(k)$ moves to reach a block at depth $k$.
# Admissibility Sketches:
# 1. Clearing Cost: Each block $z$ on top of a target $x$ must be moved at least once (unstack + putdown/stack). This is a lower bound of $2 * depth(x)$.
# 2. Movement Cost: Any task $do\_move(x, y)$ requires at least 2 actions (pick/unstack then stack). Any $do\_put\_on(x, y)$ not already satisfied requires at least these actions.
# 3. Holding Cost: If holding a block, at least 1 action is required to empty the hand.

class BlocksworldGTOHPGemini3FlashHeuristic8(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_8="blind"):
        super().__init__(name=blocksworld_gemini3flash_heuristic_8)
        self.on_ids = {}
        self.ontable_ids = {}
        self.clear_ids = {}
        self.holding_ids = {}
        self.handempty_id = None

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args, prefix
        return None, [], None

    def initialize(self, model: Model, initial_node: HTNNode):
        for fact in model.facts:
            pred, args, _ = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = fact.global_id
            elif pred == "ontable":
                self.ontable_ids[args[0]] = fact.global_id
            elif pred == "clear":
                self.clear_ids[args[0]] = fact.global_id
            elif pred == "holding":
                self.holding_ids[args[0]] = fact.global_id
            elif pred == "handempty":
                self.handempty_id = fact.global_id

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network) -> int:
        if not task_network:
            return 0

        total_h = 0
        
        # Pre-process state to build 'above' map for O(1) traversal
        above_map = {}
        for (top, bottom), f_id in self.on_ids.items():
            if (state >> f_id) & 1:
                above_map[bottom] = top

        def get_count_above(block):
            count = 0
            curr = block
            while curr in above_map:
                count += 1
                curr = above_map[curr]
            return count

        for task in task_network:
            t_name = task.name
            # Simplified parsing for grounded tasks like "do_put_on[b1,b2]"
            bracket_idx = t_name.find("[")
            if bracket_idx == -1:
                continue
            pred = t_name[:bracket_idx]
            args = t_name[bracket_idx+1:-1].split(",")

            if pred == "do_put_on":
                x, y = args[0], args[1]
                # Check if on(x, y) is already true
                on_id = self.on_ids.get((x, y))
                if on_id is not None and (state >> on_id) & 1:
                    continue
                
                # Base move cost
                total_h += 2
                # Clearing costs
                total_h += 2 * get_count_above(x)
                total_h += 2 * get_count_above(y)
                # Extra cost if y is not on table (m1 requirement)
                table_id = self.ontable_ids.get(y)
                if table_id is not None and not ((state >> table_id) & 1):
                    total_h += 2

            elif pred == "do_clear":
                total_h += 2 * get_count_above(args[0])

            elif pred == "do_on_table":
                table_id = self.ontable_ids.get(args[0])
                if table_id is not None and not ((state >> table_id) & 1):
                    total_h += 2

            elif pred == "do_move":
                total_h += 2
            
            elif pred in ["pick-up", "unstack", "stack", "put-down"]:
                total_h += 1

        # Hand status
        if self.handempty_id is not None and not ((state >> self.handempty_id) & 1):
            total_h += 1

        return total_h
