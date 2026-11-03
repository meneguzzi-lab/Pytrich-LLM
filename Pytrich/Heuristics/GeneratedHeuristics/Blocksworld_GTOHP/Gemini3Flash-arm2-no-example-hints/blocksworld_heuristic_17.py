from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import sys

class BlocksworldGTOHPGemini3FlashHeuristic17(Heuristic):
    """
    Domain-Dependent Heuristic for Blocksworld_GTOHP.
    
    Bottleneck Analysis:
    Stack clearing is the primary bottleneck. To move any block 'x', all blocks currently 
    above 'x' must be unstacked and put down, which is a serialized process.

    Admissibility Sketches:
    1. Recursive Task Cost: Every HTN task has a minimum decomposition cost in terms of 
       nop-count and primitive actions. Based on stack heights, we calculate the minimum 
       number of actions (unstack, put-down, pick-up, stack) plus nops required to 
       complete each task in the current network.
    2. Displacement Cost: Every block not in its goal-stable position must be moved at 
       least once (2 primitive actions: unstack/put-down or pick-up/stack). 
    3. Nop + Displacement: Since nop actions (HTN structure) and primitive actions 
       are disjoint in this domain, we can safely sum the minimum nops (one per 
       non-decomposed task) and the minimum displacement actions.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_17="BlocksworldGTOHPGemini3FlashHeuristic17"):
        super().__init__(blocksworld_gemini3flash_heuristic_17)
        self.fact_info = {}
        self.all_blocks = set()
        self.goal_on = {} # x -> y (x is on y)
        self.goal_ontable = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map fact IDs to predicates and arguments
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred in ['on', 'ontable', 'clear', 'handempty', 'holding']:
                self.fact_info[f.global_id] = (pred, args)
                for arg in args: self.all_blocks.add(arg)

        # 2. Extract Goals from model.goals bitset
        goal_names = model.state_explicit_repr(model.goals)
        for name in goal_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'on':
                self.goal_on[args[0]] = args[1]
            elif pred == 'ontable':
                self.goal_ontable.add(args[0])

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        if not task_network and not (self.model.goals & state == self.model.goals):
            return 1000000 # Unreachable
        
        # 1. Parse State into maps for fast lookup
        top_to_bottom = {} # x: y => on(x, y)
        bottom_to_top = {} # y: x => on(x, y)
        ontable_set = set()
        
        temp_state = state
        while temp_state:
            lsb = temp_state & -temp_state
            fid = lsb.bit_length() - 1
            if fid in self.fact_info:
                pred, args = self.fact_info[fid]
                if pred == 'on':
                    top, bottom = args[0], args[1]
                    top_to_bottom[top] = bottom
                    bottom_to_top[bottom] = top
                elif pred == 'ontable':
                    ontable_set.add(args[0])
            temp_state ^= lsb

        # 2. Precompute stack heights and stability
        memo_h = {}
        def get_h(block):
            if block in memo_h: return memo_h[block]
            if block not in bottom_to_top:
                res = 0
            else:
                res = 1 + get_h(bottom_to_top[block])
            memo_h[block] = res
            return res

        memo_stable = {}
        def is_stable(block):
            if block in memo_stable: return memo_stable[block]
            if block in self.goal_on:
                target = self.goal_on[block]
                res = (top_to_bottom.get(block) == target) and is_stable(target)
            elif block in self.goal_ontable:
                res = (block in ontable_set)
            else:
                # No goal for block, stable if on table
                res = (block in ontable_set)
            memo_stable[block] = res
            return res

        # 3. Cost logic based on stack heights
        def clear_cost(x): return 1 + 2 * get_h(x)
        def on_table_cost(x): return (clear_cost(x) - 1) + (1 if x in ontable_set else 2)
        
        # 4. Sum costs of tasks in current network
        h_recursive_sum = 0
        first_task_cost = 0
        for i, task in enumerate(task_network):
            cost = 1
            parts = task.name.split()
            pred = parts[0]
            args = parts[1:]
            
            if pred == 'do_put_on':
                x, y = args[0], args[1]
                if top_to_bottom.get(x) == y and is_stable(y):
                    cost = 1
                else:
                    cost = clear_cost(x) + clear_cost(y) + (1 if y in ontable_set else 2) + 2
            elif pred == 'do_clear':
                cost = clear_cost(args[0])
            elif pred == 'do_on_table':
                cost = on_table_cost(args[0])
            elif pred == 'do_move':
                x, y = args[0], args[1]
                cost = 2 * get_h(x) + 2 * get_h(y) + 2
            elif pred in ['pick-up', 'stack', 'unstack', 'put-down']:
                cost = 1
            elif pred == 'nop':
                cost = 1
                
            h_recursive_sum += cost
            if i == 0: first_task_cost = cost

        # 5. Calculate displacement bound
        num_displaced = sum(1 for b in self.all_blocks if not is_stable(b))
        h_nop_displacement = len(task_network) + 2 * num_displaced

        return max(h_recursive_sum, h_nop_displacement, first_task_cost)

    def _parse_fact_name(self, name: str):
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []
