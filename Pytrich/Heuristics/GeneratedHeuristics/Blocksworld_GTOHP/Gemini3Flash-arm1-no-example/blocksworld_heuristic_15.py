from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic15(Heuristic):
    """
    Heuristic for Blocksworld GTOHP domain.

    1. Bottleneck Analysis:
       The serial nature of the robotic hand (holding one block at a time) and the structural 
       dependency of the blocks (blocks must be clear to be moved) are the primary bottlenecks. 
       This forces a linear execution of tasks and significant clearing overhead.

    2. Admissible Lower Bounds:
       - Term 1: Task Network Work. Sum of costs for the task network. For the immediate task (head of the queue),
         we compute a realistic state-aware cost including clearing arguments. For future tasks, we use a
         minimum possible cost (e.g., 2 actions for a move/put_on).
       - Term 2: Hand-Emptying Cost. If the hand is holding a block and the next task requires it to be empty
         (like pick-up or unstack), a 'put-down' action is mandatory (+1).
       - Term 3: Goal Distance. Number of goal 'on' atoms not yet satisfied, providing a baseline if the TN is empty.

    3. Pattern Implementation:
       - Preprocesses fact mappings and goal IDs in initialize() (Pattern P3).
       - Uses fast bitwise operations in __call__() (Pattern P2, P5).
       - Sums costs for the ordered task network sequence (Pattern P4).
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_15="blind"):
        super().__init__(blocksworld_gemini3flash_heuristic_15)
        self.on_ids = {}
        self.ontable_ids = {}
        self.holding_ids = {}
        self.handempty_id = -1
        self.goal_ids = []
        self.fact_info = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            self.fact_info[f.global_id] = (pred, args)
            if pred == "on" and len(args) == 2:
                self.on_ids[(args[0], args[1])] = f.global_id
            elif pred == "ontable" and len(args) == 1:
                self.ontable_ids[args[0]] = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_ids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id

        # Process goals
        if model.goals:
            for bit_pos in range(model.goals.bit_length()):
                if (model.goals >> bit_pos) & 1:
                    self.goal_ids.append(bit_pos)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network) -> int:
        if not task_network:
            # Admissible bound for goals if TN is empty
            g_cost = 0
            for gid in self.goal_ids:
                if not ((state >> gid) & 1):
                    g_cost += 2
            return g_cost

        # Build a map of current block structure (what is on top of what)
        above_map = {}
        holding_block = None
        hand_empty = (state >> self.handempty_id) & 1 if self.handempty_id != -1 else True

        temp_state = state & ((1 << (self.model.ifacts_end + 1)) - 1)
        while temp_state:
            lsb = temp_state & -temp_state
            idx = lsb.bit_length() - 1
            if idx in self.fact_info:
                pred, args = self.fact_info[idx]
                if pred == "on":
                    # args[0] is on args[1]
                    above_map[args[1]] = args[0]
                elif pred == "holding":
                    holding_block = args[0]
            temp_state ^= lsb

        # Term 1: Task Network Cost
        # First task gets state-aware estimate, future tasks get minimal possible costs
        tn_cost = 0
        for i, task in enumerate(task_network):
            if i == 0:
                tn_cost += self._estimate_task_cost(task.name, state, above_map, hand_empty, holding_block)
            else:
                # Admissible lower bound for future work
                tn_cost += self._min_possible_cost(task.name)

        # Term 2: Small tie-breaker for network depth
        tn_cost += 0.01 * len(task_network)

        return int(tn_cost)

    def _estimate_task_cost(self, task_name, state, above_map, hand_empty, holding_block):
        name = task_name.lower()
        parts = name.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
        if not parts: return 0
        t_type = parts[0]
        args = parts[1:]

        if t_type in ["do_put_on", "do_move"]:
            if len(args) < 2: return 2
            x, y = args[0], args[1]
            # Check if already satisfied
            if t_type == "do_put_on":
                fid = self.on_ids.get((x, y))
                if fid is not None and (state >> fid) & 1: return 0
            
            cost = 2 + self._clear_cost(x, above_map) + self._clear_cost(y, above_map)
            if holding_block == x:
                cost = 1 + self._clear_cost(y, above_map)
            elif not hand_empty:
                cost += 1 # Must put down current block
            return cost

        elif t_type == "do_on_table":
            if not args: return 0
            x = args[0]
            fid = self.ontable_ids.get(x)
            if fid is not None and (state >> fid) & 1: return 0
            cost = 2 + self._clear_cost(x, above_map)
            if holding_block == x: cost = 1
            elif not hand_empty: cost += 1
            return cost

        elif t_type == "do_clear":
            if not args: return 0
            return self._clear_cost(args[0], above_map)

        elif t_type in ["pick-up", "unstack", "stack", "put-down"]:
            return 1
        
        elif t_type == "nop":
            return 0

        return 1 # Default guidance for recognized grounded actions

    def _min_possible_cost(self, task_name):
        name = task_name.lower()
        if "nop" in name: return 0
        if any(x in name for x in ["pick-up", "unstack", "stack", "put-down"]): return 1
        return 0 # Abstract tasks could potentially be satisfied (cost 0)

    def _clear_cost(self, block, above_map):
        cost = 0
        curr = block
        while curr in above_map:
            cost += 2
            curr = above_map[curr]
        return cost

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
