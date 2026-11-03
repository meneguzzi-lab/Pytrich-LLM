from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# BOTTLENECK: Stack clearing and reconstruction. To move a block, all blocks above it must be unstacked first.
# ADMISSIBILITY SKETCH:
# 1. Network Task Cost: For each abstract task, we compute the minimum number of primitive actions required 
#    (unstack, put-down, pick-up, stack) based on the current state's block heights. Since block moves 
#    are largely sequential in this domain, these costs are additive and provide a lower bound.
# 2. Holding Penalty: If the hand is not empty, at least one additional action (put-down or stack) is required.
# 3. Nop count: This domain uses nop actions which cost 0, so we count only primitive actions for cost.

class BlocksworldGTOHPGemini3FlashHeuristic5(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_5="blocksworld_gemini3flash_heuristic_5"):
        super().__init__(blocksworld_gemini3flash_heuristic_5)
        self.on_ids = {}
        self.ontable_ids = {}
        self.holding_ids = {}
        self.fact_info = {}
        self.blocks = set()
        self.goal_on = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        for fact in model.facts:
            gid = fact.global_id
            pred, args = self._parse_fact_name(fact.name)
            self.fact_info[gid] = (pred, args)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = gid
                self.blocks.update(args)
            elif pred == "ontable":
                self.ontable_ids[args[0]] = gid
                self.blocks.add(args[0])
            elif pred == "holding":
                self.holding_ids[args[0]] = gid
                self.blocks.add(args[0])
        
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

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

    def _get_height(self, x, under_map):
        h = 0
        curr = x
        while curr in under_map:
            h += 1
            curr = under_map[curr]
        return h

    def _compute(self, state, task_network):
        on_map = {}
        under_map = {}
        ontable_set = set()
        holding_block = None
        
        temp_state = state
        while temp_state:
            lsb = temp_state & -temp_state
            bit = lsb.bit_length() - 1
            if bit in self.fact_info:
                pred, args = self.fact_info[bit]
                if pred == "on":
                    on_map[args[0]] = args[1]
                    under_map[args[1]] = args[0]
                elif pred == "ontable":
                    ontable_set.add(args[0])
                elif pred == "holding":
                    holding_block = args[0]
            temp_state ^= lsb
            
        total_h = 0
        for task in task_network:
            if hasattr(task, "cost"): # Primitive Operator
                total_h += task.cost
            else: # Abstract Task
                parts = task.name.split()
                t_type = parts[0]
                args = parts[1:]
                if not args: continue
                
                if t_type == "do_put_on":
                    x, y = args[0], args[1]
                    oid = self.on_ids.get((x, y))
                    if not (oid is not None and (state >> oid) & 1):
                        total_h += 2 * self._get_height(x, under_map) + 2 * self._get_height(y, under_map) + 2
                elif t_type == "do_clear":
                    total_h += 2 * self._get_height(args[0], under_map)
                elif t_type == "do_move":
                    total_h += 2
                elif t_type == "do_on_table":
                    # m3 method makes this 0 if clear
                    pass
        
        if holding_block:
            total_h += 1
            
        # Tie-breaker for decomposition progress
        total_h += 0.001 * len(task_network)
        return total_h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
