import re
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic12(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Bottleneck Analysis:
    The core bottleneck in this Blocksworld variant is the requirement to clear stacks of blocks 
    and potentially move target blocks to the table (due to do_on_table y in m1_do_put_on) 
    before performing move/stack operations. Each block above a target requires at least two 
    actions (unstack and put-down/stack) to clear.
    
    Admissibility Sketches:
    1. Task Cost Estimation: For each task in the network, we estimate the minimum number 
       of actions to fulfill it. For do_put_on, it requires clearing both blocks (2*height) 
       and a move action (2), plus an on-table action for the target. Since these actions 
       are mandatory for the HTN structure, the sum is a valid lower bound for that task.
    2. Disjoint Tasks: Since tasks in Blocksworld HTN usually operate on different sub-stacks 
       or in a sequence of specific block movements, summing costs is a powerful and 
       mostly admissible guide in this specific domain.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_12=None, name="BlocksworldGTOHPGemini3FlashHeuristic12"):
        super().__init__(name=name)
        self.on_fact_ids = []
        self.ontable_fact_ids = []
        self.holding_fact_ids = []
        self.handempty_id = -1
        self.blocks = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self.on_fact_ids = []
        self.ontable_fact_ids = {}
        self.holding_fact_ids = {}
        self.handempty_id = -1
        self.blocks = set()

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if pred == "on" and len(args) == 2:
                self.on_fact_ids.append((args[0], args[1], fact.global_id))
                self.blocks.add(args[0])
                self.blocks.add(args[1])
            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact_ids[args[0]] = fact.global_id
                self.blocks.add(args[0])
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_ids[args[0]] = fact.global_id
                self.blocks.add(args[0])
            elif pred == "handempty":
                self.handempty_id = fact.global_id

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
        if name.startswith("+"):
            name = name[1:]
        if "[" in name and name.endswith("]"):
            parts = name.split("[")
            task_type = parts[0]
            args = parts[1][:-1].split(",")
            return task_type, args
        return name, []

    def _get_state_info(self, state_bitwise):
        current_on = {}
        supported_block = {}
        on_table = set()
        holding = None

        for b, s, fid in self.on_fact_ids:
            if (state_bitwise >> fid) & 1:
                current_on[b] = s
                supported_block[s] = b

        for b, fid in self.ontable_fact_ids.items():
            if (state_bitwise >> fid) & 1:
                on_table.add(b)

        for b, fid in self.holding_fact_ids.items():
            if (state_bitwise >> fid) & 1:
                holding = b

        return {
            'current_on': current_on,
            'supported_block': supported_block,
            'on_table': on_table,
            'holding': holding
        }

    def _count_above(self, x, supported_block):
        if x is None or x not in supported_block:
            return 0
        cnt = 0
        curr = x
        visited = {curr}
        while curr in supported_block:
            curr = supported_block[curr]
            if curr in visited:
                break
            visited.add(curr)
            cnt += 1
        return cnt

    def _get_task_cost(self, task_name, state_info):
        t_type, args = self._parse_task_name(task_name)
        if not args or args[0] == '':
            return 0

        x = args[0]
        y = args[1] if len(args) > 1 else None
        
        current_on = state_info['current_on']
        supported_block = state_info['supported_block']
        on_table = state_info['on_table']

        if t_type == "do_put_on":
            if y is None or current_on.get(x) == y:
                return 0
            cost = 2 * self._count_above(x, supported_block) + 2 * self._count_above(y, supported_block)
            cost += 2 # do_move
            if y not in on_table:
                cost += 2 # do_on_table y (unstack + put-down)
            return cost
        
        elif t_type == "do_clear":
            return 2 * self._count_above(x, supported_block)
            
        elif t_type == "do_on_table":
            base = 2 if x not in on_table else 0
            return base + 2 * self._count_above(x, supported_block)
            
        elif t_type == "do_move":
            return 2
            
        elif t_type in ["pick-up", "unstack", "stack", "put-down"]:
            return 1
            
        return 0

    def _compute(self, state_bitwise, task_network):
        if not task_network:
            return 0
        
        state_info = self._get_state_info(state_bitwise)
        h = 0
        for task in task_network:
            h += self._get_task_cost(task.name, state_info)

        # Add sub-unit tie-breaker to prefer shorter task networks
        h += 0.01 * len(task_network)
        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
