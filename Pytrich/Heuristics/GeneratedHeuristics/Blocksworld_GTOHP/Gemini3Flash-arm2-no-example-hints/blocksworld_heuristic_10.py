import re
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic10(Heuristic):
    """
    Domain-Dependent Heuristic for Blocksworld GTOHP.
    
    BOTTLENECK: Stack clearing. To move or clear a block, all blocks currently on top of it 
    must be unstacked and put down, each requiring 2 primitive actions (unstack + put-down).
    
    ADMISSIBLE LOWER BOUNDS:
    1. Task Network Cost: Each task in the network has a minimum number of primitive actions 
       required to reach a state where it can be finished. For example, do_clear(x) requires 
       at least 2 * (number of blocks above x) actions.
    2. Goal Displacement: Every goal fact on(x, y) not currently satisfied will eventually 
       require at least a stack(x, y) and potentially a pick-up(x), plus any necessary 
       clearing operations.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_10="blocksworld_gemini3flash_heuristic_10"):
        super().__init__(name=blocksworld_gemini3flash_heuristic_10)
        self.on_fact_ids = []
        self.ontable_fids = {}
        self.fact_id_to_args = {}
        self.on_fids_by_x = {}
        self.on_fids_by_y = {}
        self.blocks = set()
        self.goal_on = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates to IDs for fast state checking
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if pred == "on":
                x, y = args[0], args[1]
                self.on_fact_ids.append(fact.global_id)
                self.fact_id_to_args[fact.global_id] = (x, y)
                if x not in self.on_fids_by_x: self.on_fids_by_x[x] = {}
                self.on_fids_by_x[x][y] = fact.global_id
                if y not in self.on_fids_by_y: self.on_fids_by_y[y] = {}
                self.on_fids_by_y[y][x] = fact.global_id
                self.blocks.add(x)
                self.blocks.add(y)
            elif pred == "ontable":
                self.ontable_fids[args[0]] = fact.global_id
                self.blocks.add(args[0])

        # 2. Extract explicit goal requirements
        goal_names = model.state_explicit_repr(model.goals)
        for name in goal_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        # A. Build current state stack info
        current_on = {}
        current_under = {}
        for fid in self.on_fact_ids:
            if (state >> fid) & 1:
                x, y = self.fact_id_to_args[fid]
                current_on[x] = y
                current_under[y] = x

        # B. Efficient Height Calculation
        memo_height = {}
        def get_height(b):
            if b in memo_height: return memo_height[b]
            if b not in current_under:
                memo_height[b] = 0
                return 0
            # Number of blocks on top of b
            res = 1 + get_height(current_under[b])
            memo_height[b] = res
            return res

        # C. Sum Task Network Estimates (Serial HTN focus)
        h_tasks = 0
        for task in task_network:
            name = task.name
            t_pred, t_args = self._parse_grounded_string(name)
            
            if t_pred == 'do_put_on':
                x, y = t_args[0], t_args[1]
                fid = self.on_fids_by_x.get(x, {}).get(y, -1)
                if not (fid != -1 and (state >> fid) & 1):
                    # Needs cleaning x, y, moving. m1 cost profile
                    h_tasks += 2 * get_height(x) + 2 * get_height(y) + 4
            elif t_pred == 'do_clear':
                x = t_args[0]
                h_tasks += 2 * get_height(x)
            elif t_pred == 'do_on_table':
                x = t_args[0]
                fid = self.ontable_fids.get(x, -1)
                if not (fid != -1 and (state >> fid) & 1):
                    h_tasks += 2 + 2 * get_height(x)
            elif t_pred == 'do_move':
                x, y = t_args[0], t_args[1]
                h_tasks += 2 + 2 * get_height(x) + 2 * get_height(y)
            elif t_pred == 'nop':
                pass
            else:
                # Primitive actions (unstack, stack, pick-up, put-down)
                h_tasks += 1

        # D. Goal Displacement (Secondary Bound)
        h_goals = 0
        for x, y in self.goal_on.items():
            fid = self.on_fids_by_x.get(x, {}).get(y, -1)
            if not (fid != -1 and (state >> fid) & 1):
                h_goals += 2

        # Use max to maintain admissibility while capturing different constraints
        # Add a tiny tie-breaker for fewer tasks to encourage task reduction
        return max(h_tasks, h_goals) + (0.01 * len(task_network))

    def _parse_fact_name(self, name: str):
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return predicate, args
        return None, []

    def _parse_grounded_string(self, name: str):
        # Handles formats like 'do_put_on b1 b2' or 'do_put_on[b1,b2]'
        parts = re.split(r'[ \[\]\,]+', name.strip())
        return parts[0], parts[1:]
