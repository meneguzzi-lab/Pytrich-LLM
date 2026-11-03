from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import math

class BlocksworldGTOHPGemini3FlashHeuristic16(Heuristic):
    """
    Heuristic for Blocksworld GTOHP.
    
    Bottleneck Analysis:
    The primary bottleneck is the robotic arm (serial constraint) and the 'clear' requirement for unstacking/stacking. 
    A block deep in a stack requires a chain of unstack/put-down actions before it can be moved, which dominates plan cost.

    Admissible Lower Bounds:
    1. Task-Network State-Aware Cost: Each task in the network requires a minimum number of actions based on the current 
       block positions (e.g., do_clear(x) needs 2 actions for every block above x).
    2. Goal Stability Cost: Any goal on(A, B) that is not currently satisfied, or is satisfied but B is on a block that must 
       move, requires at least 2 actions (unstack/stack).
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_16="blind"):
        super().__init__(blocksworld_gemini3flash_heuristic_16)
        self.on_ids = {} # (x, y) -> id
        self.clear_ids = {} # x -> id
        self.ontable_ids = {} # x -> id
        self.holding_ids = {} # x -> id
        self.handempty_id = None
        self.blocks = []
        self.goal_on = [] # list of (x_id, y_id, fact_id)

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
        self.model = model
        self.blocks = [f.name.split('[')[1].split(']')[0] for f in model.facts if "ontable[" in f.name]
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "ontable":
                self.ontable_ids[args[0]] = f.global_id
            elif pred == "holding":
                self.holding_ids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                fact_id = self.on_ids.get((args[0], args[1]))
                if fact_id is not None:
                    self.goal_on.append((args[0], args[1], fact_id))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_stack_above(self, block, state):
        count = 0
        current = block
        found = True
        while found:
            found = False
            for other in self.blocks:
                if other == current: continue
                on_id = self.on_ids.get((other, current))
                if on_id is not None and (state >> on_id) & 1:
                    count += 1
                    current = other
                    found = True
                    break
        return count

    def _compute_task_cost(self, task, state):
        name = task.name.lower()
        if "nop" in name:
            return 0
        
        args = []
        if "[" in name:
            args = name.split("[")[1].split("]")[0].split(",")

        if "do_clear" in name and args:
            x = args[0]
            clear_id = self.clear_ids.get(x)
            if clear_id is not None and (state >> clear_id) & 1:
                return 0
            return 2 * (1 + self._get_stack_above(x, state))

        if ("do_put_on" in name or "do_move" in name) and len(args) >= 2:
            x, y = args[0], args[1]
            on_id = self.on_ids.get((x, y))
            if on_id is not None and (state >> on_id) & 1:
                return 0
            cost = 2
            clear_x = self.clear_ids.get(x)
            if clear_x is not None and not ((state >> clear_x) & 1):
                cost += 2 * self._get_stack_above(x, state)
            clear_y = self.clear_ids.get(y)
            if clear_y is not None and not ((state >> clear_y) & 1):
                cost += 2 * self._get_stack_above(y, state)
            return cost

        if "do_on_table" in name and args:
            x = args[0]
            ot_id = self.ontable_ids.get(x)
            if ot_id is not None and (state >> ot_id) & 1:
                return 0
            return 2
            
        return 1

    def _compute(self, state, task_network):
        tn_cost = 0
        for task in task_network:
            tn_cost += self._compute_task_cost(task, state)

        goal_cost = 0
        for x, y, fact_id in self.goal_on:
            if not ((state >> fact_id) & 1):
                goal_cost += 2
        
        return max(tn_cost, goal_cost)
