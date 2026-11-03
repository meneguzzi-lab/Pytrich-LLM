from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# BOTTLENECK ANALYSIS:
# The Blocksworld_GTOHP domain's serial constraint is the recursive requirement to clear blocks. 
# To move any block 'x' or to put any block 'y' on the table, every block above them in a stack 
# must be unstacked and put down, each step requiring at least two actions (unstack + put-down).
# 
# ADMISSIBLE LOWER BOUNDS:
# 1. Task Network Cost (h_tn): Every remaining task in the network has a minimum action cost. 
#    - Primitives (pick-up, stack, etc.) cost 1.
#    - do_move(x, y) costs at least 2 (one pick-up/unstack + one stack).
#    - do_clear(x) and do_on_table(x) cost at least 2 per block currently above them in the state.
#    - do_put_on(x, y) costs clear(x) + clear(y) + move(x,y) + (table(y) if needed).
#    Since these represent unavoidable future actions to decompose the current network, their sum is admissible.
# 2. State Goal Cost (h_goal): Every goal 'on(x, y)' or 'ontable(x)' not currently satisfied in the state 
#    requires at least 2 actions to achieve (pick-up/unstack + stack/put-down). 
#    This is admissible as these goals are independent action targets.
# 
# The final heuristic is max(h_tn, h_goal) to maintain admissibility while capturing both 
# procedural progress (TN) and state-based progress (Goals).

class BlocksworldGTOHPGemini3FlashHeuristic4(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_4="blind"):
        super().__init__(blocksworld_gemini3flash_heuristic_4)
        self.on_ids = {} # (x, y) -> global_id
        self.on_lookup = {} # global_id -> (x, y)
        self.clear_ids = {} # x -> global_id
        self.ontable_ids = {} # x -> global_id
        self.handempty_id = None
        self.holding_ids = {} # x -> global_id
        self.goal_fact_ids = []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Build Fact Maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
                self.on_lookup[f.global_id] = (args[0], args[1])
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "ontable":
                self.ontable_ids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id
            elif pred == "holding":
                self.holding_ids[args[0]] = f.global_id
        
        # 2. Precompute Goal IDs
        goal_bits = model.goals
        if goal_bits:
            for bit_pos in range(goal_bits.bit_length()):
                if (goal_bits >> bit_pos) & 1:
                    self.goal_fact_ids.append(bit_pos)
                    
        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        # Build up_map (y -> x where x is on y) to calculate clear_cost efficiently
        up_map = {}
        temp_state = state
        while temp_state:
            bit = temp_state & -temp_state
            idx = bit.bit_length() - 1
            if idx in self.on_lookup:
                x, y = self.on_lookup[idx]
                up_map[y] = x
            temp_state ^= bit

        def get_clear_cost(block):
            cost = 0
            curr = block
            while curr in up_map:
                cost += 2 # unstack + put-down for the block on top
                curr = up_map[curr]
            return cost

        # Term 1: Task Network Progress
        h_tn = 0
        for task in task_network:
            clean_name = task.name.strip("()")
            parts = clean_name.split()
            if not parts: continue
            t_name = parts[0]
            args = parts[1:]

            if t_name == 'do_put_on' and len(args) >= 2:
                x, y = args[0], args[1]
                if not self._is_true(state, self.on_ids.get((x, y))):
                    h_tn += 2 # move cost
                    h_tn += get_clear_cost(x)
                    h_tn += get_clear_cost(y)
                    if not self._is_true(state, self.ontable_ids.get(y)): h_tn += 2
            elif t_name == 'do_move' and len(args) >= 2:
                x, y = args[0], args[1]
                h_tn += 2 + get_clear_cost(x) + get_clear_cost(y)
            elif t_name == 'do_clear' and len(args) >= 1:
                h_tn += get_clear_cost(args[0])
            elif t_name == 'do_on_table' and len(args) >= 1:
                x = args[0]
                if not self._is_true(state, self.ontable_ids.get(x)):
                    h_tn += 2 + get_clear_cost(x)
            elif t_name == 'nop':
                continue
            else:
                h_tn += 1 # Primitives (pick-up, stack, etc.)
        
        if not self._is_true(state, self.handempty_id): h_tn += 1

        # Term 2: Missing State Goals
        h_goal = 0
        for g_id in self.goal_fact_ids:
            if not ((state >> g_id) & 1): h_goal += 2
            
        return max(h_tn, h_goal)

    def _is_true(self, state, fact_id):
        if fact_id is None: return False
        return (state >> fact_id) & 1

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []
