from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic0(Heuristic):
    """
    Domain-Dependent Heuristic for Blocksworld_GTOHP.
    
    Bottleneck Analysis:
    Stack clearing is the primary bottleneck. To move or stack a block, all blocks above it 
    must be unstacked first, each requiring a pair of primitive actions (unstack and put-down).
    
    Admissibility Sketches:
    1. clear_cost(x): The number of blocks above x determines the minimum number of unstack/put-down actions.
       Each block above x costs at least 2 actions. We use the hint's formula (1 + 2*h) for h > 0.
    2. do_put_on(x, y): If not satisfied, this task requires clearing x, clearing y, and the final move.
       The cost (clear_cost(x) + clear_cost(y) + 3) is a safe lower bound.
    3. Goal Stability: A goal 'on(x,y)' that is currently true but has an 'unstable' support (y or blocks below y
       must be moved for later tasks) is not free and must be re-stacked.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_0="blocksworld_gemini3flash_heuristic_0"):
        super().__init__(blocksworld_gemini3flash_heuristic_0)
        self.on_fact_map = {}  # (x, y) -> global_id
        self.ontable_ids = {}  # x -> global_id
        self.handempty_id = -1
        self.on_ids = []       # list of (x, y, global_id)

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
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_fact_map[(args[0], args[1])] = f.global_id
                self.on_ids.append((args[0], args[1], f.global_id))
            elif pred == "ontable":
                self.ontable_ids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_height(self, block, above_map):
        h = 0
        curr = block
        while curr in above_map:
            h += 1
            curr = above_map[curr]
        return h

    def _get_clear_cost(self, block, above_map):
        h = self._get_height(block, above_map)
        if h == 0: return 0
        return 1 + 2 * h

    def _is_unstable(self, block, moving_subjects, below_map):
        curr = block
        while curr:
            if curr in moving_subjects:
                return True
            curr = below_map.get(curr)
        return False

    def _compute(self, state: int, task_network) -> int:
        if not task_network:
            return 0

        # 1. Build current state adjacency maps
        above_map = {} # y -> x (x is on y)
        below_map = {} # x -> y (x is on y)
        for x, y, fid in self.on_ids:
            if (state >> fid) & 1:
                above_map[y] = x
                below_map[x] = y

        # 2. Identify blocks that will be moved by tasks in the network
        moving_subjects = set()
        for task in task_network:
            if hasattr(task, 'name'):
                parts = task.name.split()
                if parts[0] in ["do_put_on", "do_move", "do_on_table", "do_clear"] and len(parts) > 1:
                    moving_subjects.add(parts[1])

        # 3. Sum state-aware costs for tasks
        total_h = 0
        # Add a baseline cost if hand is not empty (must put down/stack before starting abstract tasks)
        if self.handempty_id != -1 and not ((state >> self.handempty_id) & 1):
            total_h += 1

        for task in task_network:
            if hasattr(task, 'cost'): # Operator
                total_h += task.cost
                continue
            
            parts = task.name.split()
            cmd = parts[0]
            args = parts[1:]
            
            if cmd == "do_put_on" and len(args) == 2:
                x, y = args[0], args[1]
                on_fid = self.on_fact_map.get((x, y))
                if on_fid is not None and (state >> on_fid) & 1 and not self._is_unstable(y, moving_subjects, below_map):
                    continue # m0_do_put_on -> nop
                total_h += self._get_clear_cost(x, above_map) + self._get_clear_cost(y, above_map) + 3
            
            elif cmd == "do_move" and len(args) == 2:
                x, y = args[0], args[1]
                total_h += self._get_clear_cost(x, above_map) + self._get_clear_cost(y, above_map) + 2
            
            elif cmd == "do_on_table" and len(args) == 1:
                x = args[0]
                table_fid = self.ontable_ids.get(x)
                if table_fid is not None and (state >> table_fid) & 1:
                    continue # m3_do_on_table -> nop
                total_h += self._get_clear_cost(x, above_map) + 2
            
            elif cmd == "do_clear" and len(args) == 1:
                x = args[0]
                total_h += self._get_clear_cost(x, above_map)
            
            elif cmd == "nop":
                pass

        return total_h
