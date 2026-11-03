from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

# BOTTLENECK: Stack unstacking. To move or clear any block x, all blocks above it must be unstacked first.
# Admissible Term 1: Task Cost Summation. Since tasks in an HTN are ordered/serial in this domain, 
# we sum the minimum required primitive actions for each task in the network based on the current state.
# Admissible Term 2: Goal Stability. If on(x, y) is satisfied but y is unstable (must move for its own goal),
# x must be moved twice (unstack/putdown, then pickup/stack later), adding at least 4 actions.

class BlocksworldGTOHPGemini3FlashHeuristic4(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_4="blocksworld_heuristic_4"):
        super().__init__(blocksworld_gemini3flash_heuristic_4)
        self.on_ids_by_below = collections.defaultdict(dict)
        self.on_ids_by_above = collections.defaultdict(dict)
        self.ontable_ids = {}
        self.clear_ids = {}
        self.holding_ids = {}
        self.goal_on = {}

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

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates to global IDs for fast bitwise lookup
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                # args[0] is above, args[1] is below
                self.on_ids_by_below[args[1]][args[0]] = f.global_id
                self.on_ids_by_above[args[0]][args[1]] = f.global_id
            elif pred == "ontable":
                self.ontable_ids[args[0]] = f.global_id
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "holding":
                self.holding_ids[args[0]] = f.global_id

        # 2. Map goals
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                self.goal_on[args[0]] = args[1]
            elif pred == "ontable":
                self.goal_on[args[0]] = "table"

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _get_height(self, block, state):
        h = 0
        curr = block
        while True:
            found = False
            # Check if any block is on top of curr
            above_map = self.on_ids_by_below.get(curr, {})
            for above, fid in above_map.items():
                if (state >> fid) & 1:
                    h += 1
                    curr = above
                    found = True
                    break
            if not found: break
        return h

    def _is_unstable(self, block, state, memo):
        if block in memo: return memo[block]
        
        # Current base
        curr_below = None
        below_map = self.on_ids_by_above.get(block, {})
        for below, fid in below_map.items():
            if (state >> fid) & 1:
                curr_below = below
                break
        if curr_below is None:
            ontable_id = self.ontable_ids.get(block)
            if ontable_id is not None and (state >> ontable_id) & 1:
                curr_below = "table"

        # Goal base
        target_below = self.goal_on.get(block)

        # If wrong base, unstable
        if target_below is not None and curr_below != target_below:
            memo[block] = True
            return True
        
        # Recursive check
        if curr_below and curr_below != "table":
            if self._is_unstable(curr_below, state, memo):
                memo[block] = True
                return True

        memo[block] = False
        return False

    def _compute(self, state, task_network):
        h = 0
        memo_unstable = {}
        
        for task in task_network:
            if hasattr(task, 'decompositions'): # AbstractTask
                parts = task.name.split()
                t_name = parts[0]
                args = parts[1:]

                if t_name == "do_put_on":
                    x, y = args[0], args[1]
                    # Check if on(x,y) is true
                    on_id = self.on_ids_by_below.get(y, {}).get(x)
                    is_on = (on_id is not None and (state >> on_id) & 1)
                    
                    if not is_on:
                        # Base cost: 4 (move y to table, move x to y)
                        # Plus clear costs: 2 actions per block above
                        h += 4 + 2*self._get_height(x, state) + 2*self._get_height(y, state)
                    elif self._is_unstable(y, state, memo_unstable):
                        h += 4

                elif t_name == "do_clear":
                    x = args[0]
                    h += 2 * self._get_height(x, state)

                elif t_name == "do_on_table":
                    x = args[0]
                    ontable_id = self.ontable_ids.get(x)
                    if ontable_id is not None and not ((state >> ontable_id) & 1):
                        h += 2 + 2 * self._get_height(x, state)

                elif t_name == "do_move":
                    x, y = args[0], args[1]
                    h += 2 + 2 * self._get_height(x, state) + 2 * self._get_height(y, state)
            
            else: # Operator
                h += task.cost

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
