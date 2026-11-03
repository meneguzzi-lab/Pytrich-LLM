from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class BlocksworldGTOHPGemini3FlashHeuristic1(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Bottleneck Analysis:
    Stack clearing is the primary bottleneck. To move a block 'x', all blocks currently 
    above it must be unstacked and put down, each requiring 2 primitive actions. 
    The cost to clear a block with 'h' blocks above it is thus at least 2*h.
    
    Admissibility Sketches:
    1. Task Decomposition Cost (h_tasks): Summing the minimum recursive decomposition 
       costs for each task in the network is admissible because each task in an 
       ordered HTN task network must be completed sequentially using a disjoint 
       set of primitive actions in this domain.
    2. Clear Cost: 2 * (number of blocks above) is a lower bound because each block 
       above must be unstacked and put down to reach the one below.
    3. Displacement Cost: A do_put_on(x, y) where on(x, y) is not true requires 
       at least a pick-up/unstack and a stack, costing 2 actions.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_1="blocksworld_gemini3flash_heuristic_1"):
        super().__init__(name=blocksworld_gemini3flash_heuristic_1)
        self.on_fact_ids = {}  # (x, y) -> global_id
        self.ontable_fact_ids = {}  # x -> global_id
        self.clear_fact_ids = {}  # x -> global_id
        self.on_fact_to_args = {}  # global_id -> (x, y)
        self.all_on_fact_ids = set()
        self.all_blocks = set()
        self.task_name_cache = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # Build fact maps
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "on":
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
                self.on_fact_to_args[fact.global_id] = (args[0], args[1])
                self.all_on_fact_ids.add(fact.global_id)
                self.all_blocks.update(args)
            elif pred == "ontable":
                self.ontable_fact_ids[args[0]] = fact.global_id
                self.all_blocks.update(args)
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = fact.global_id
                self.all_blocks.update(args)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        if not task_network: return 0

        # 1. Build structural maps of current state using bitwise efficiency
        # We only care about set bits corresponding to 'on' facts
        on_above = {}  # block_below -> block_above
        temp_state = state
        bit_idx = 0
        # Iterating over set bits is much faster for sparse states (like 1000 blocks)
        while temp_state:
            if temp_state & 1:
                if bit_idx in self.all_on_fact_ids:
                    x, y = self.on_fact_to_args[bit_idx]
                    on_above[y] = x
            temp_state >>= 1
            bit_idx += 1

        # 2. Identify blocks that MUST move later in the network (unstable)
        mover_tasks = set()
        for task in task_network:
            t_type, t_args = self._get_task_info(task)
            if t_type in ("do_put_on", "do_on_table", "do_move") and t_args:
                mover_tasks.add(t_args[0])

        # 3. Sum estimated decomposition costs
        h_tasks = 0
        for task in task_network:
            if hasattr(task, 'cost') and not hasattr(task, 'decompositions'): # Operator
                h_tasks += task.cost
            else: # AbstractTask
                h_tasks += self._estimate_abstract_task(task, state, on_above, mover_tasks)

        # 4. Tie-breaking to prefer progress
        h_total = h_tasks + (0.001 * len(task_network))
        return int(h_total) if h_total > 0 else 0

    def _estimate_abstract_task(self, task, state, on_above, mover_tasks):
        t_type, t_args = self._get_task_info(task)
        if not t_args: return 0
        
        if t_type == "do_put_on":
            x, y = t_args[0], t_args[1]
            # Goal stability: if already satisfied and base is not moving
            if self._is_on(x, y, state) and y not in mover_tasks:
                return 0
            return self._get_clear_cost(x, on_above) + self._get_clear_cost(y, on_above) + 2
        elif t_type == "do_clear":
            return self._get_clear_cost(t_args[0], on_above)
        elif t_type == "do_on_table":
            x = t_args[0]
            if self._is_ontable(x, state): return 0
            return self._get_clear_cost(x, on_above) + 2
        elif t_type == "do_move":
            return 2
        return 0

    def _get_clear_cost(self, block, on_above):
        cost = 0
        curr = block
        # Traverse the stack up from the block
        while curr in on_above:
            cost += 2
            curr = on_above[curr]
        return cost

    def _is_on(self, x, y, state):
        fid = self.on_fact_ids.get((x, y))
        return (state >> fid) & 1 if fid is not None else False

    def _is_ontable(self, x, state):
        fid = self.ontable_fact_ids.get(x)
        return (state >> fid) & 1 if fid is not None else False

    def _get_task_info(self, task):
        if task.name in self.task_name_cache:
            return self.task_name_cache[task.name]
        
        # Grounded task format: name[arg1,arg2]
        match = re.search(r'([^\s\[]+)\[([^\]]+)\]', task.name)
        if match:
            t_type = match.group(1)
            t_args = match.group(2).split(',')
            res = (t_type, t_args)
        else:
            res = (task.name, [])
        
        self.task_name_cache[task.name] = res
        return res

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
