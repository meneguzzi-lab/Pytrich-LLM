from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

class BlocksworldGTOHPGemini3FlashHeuristic9(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld_GTOHP domain.
    
    Bottleneck analysis:
    The main bottleneck is block clearing (unstacking everything on top of a target block).
    In this HDDL variant, stacking a block onto another requires both blocks to be clear,
    and the bottom block to be either on the table or successfully cleared of its own obstacles.

    Admissibility Sketch:
    1. Task Network Cost: Each task in the network has a minimum cost based on the number 
       of blocks that must be unstacked (2 actions per block) and the movement actions themselves (2 actions).
    2. Goal Cost: Any goal fact (on x y) not currently handled by the task network adds 
       unavoidable costs to move x onto y.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_9=None, name="BlocksworldGTOHPGemini3FlashHeuristic9"):
        super().__init__(name=name)
        self.on_map_info = {}
        self.ontable_info = {}
        self.holding_info = {}
        self.goal_on = {}
        self.goal_ontable = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on" and len(args) == 2:
                self.on_map_info[fact.global_id] = (args[0], args[1])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_info[fact.global_id] = args[0]
            elif pred == "holding" and len(args) == 1:
                self.holding_info[fact.global_id] = args[0]

        goal_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2:
                self.goal_on[args[0]] = args[1]
            elif pred == "ontable" and len(args) == 1:
                self.goal_ontable.add(args[0])

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

    def _parse_task_name(self, name: str):
        name = name.strip("() ")
        parts = name.split()
        if not parts:
            return None, []
        return parts[0], parts[1:]

    def _get_state_info(self, state):
        on_map = {}
        under_map = {}
        holding_block = None
        ontable_blocks = set()
        temp_state = state
        while temp_state:
            lsb = temp_state & -temp_state
            fid = lsb.bit_length() - 1
            if fid in self.on_map_info:
                x, y = self.on_map_info[fid]
                on_map[x] = y
                under_map[y] = x
            elif fid in self.ontable_info:
                ontable_blocks.add(self.ontable_info[fid])
            elif fid in self.holding_info:
                holding_block = self.holding_info[fid]
            temp_state ^= lsb
        return on_map, under_map, holding_block, ontable_blocks

    def _compute_above(self, block, under_map):
        count = 0
        curr = block
        while curr in under_map:
            curr = under_map[curr]
            count += 1
            if count > 2000: break
        return count

    def _compute(self, state, task_network):
        on_map, under_map, holding_block, ontable_blocks = self._get_state_info(state)
        tn_cost = 0
        tasks_doing_put_on = set()
        tasks_doing_on_table = set()

        for task in task_network:
            tname, targs = self._parse_task_name(task.name)
            if not tname: continue
            
            if tname == "do_put_on" and len(targs) >= 2:
                x, y = targs[0], targs[1]
                tasks_doing_put_on.add((x, y))
                if on_map.get(x) != y:
                    tn_cost += 2 + 2 * self._compute_above(x, under_map) + 2 * self._compute_above(y, under_map)
            elif tname == "do_clear" and len(targs) >= 1:
                tn_cost += 2 * self._compute_above(targs[0], under_map)
            elif tname == "do_on_table" and len(targs) >= 1:
                x = targs[0]
                tasks_doing_on_table.add(x)
                if x not in ontable_blocks:
                    tn_cost += 2 + 2 * self._compute_above(x, under_map)
            elif tname == "do_move":
                tn_cost += 2
            elif tname in ["pick-up", "put-down", "stack", "unstack"]:
                tn_cost += 1

        goal_cost = 0
        for x, y in self.goal_on.items():
            if (x, y) not in tasks_doing_put_on:
                if on_map.get(x) != y:
                    goal_cost += 2 + 2 * self._compute_above(x, under_map) + 2 * self._compute_above(y, under_map)
        
        for x in self.goal_ontable:
            if x not in tasks_doing_on_table:
                if x not in ontable_blocks:
                    goal_cost += 2 + 2 * self._compute_above(x, under_map)

        return tn_cost + goal_cost

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
