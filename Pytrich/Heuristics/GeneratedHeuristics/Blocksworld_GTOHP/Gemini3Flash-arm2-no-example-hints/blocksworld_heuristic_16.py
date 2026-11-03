from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic16(Heuristic):
    """
    Domain-Dependent Heuristic for Blocksworld GTOHP.
    
    Bottleneck Analysis:
    The primary bottleneck is stack clearing. To move or clear any block x, all blocks currently on top 
    of it in the current state must be removed via 'unstack' and 'put-down' operations.
    
    Admissibility Sketches:
    1. Task Recursive Cost (Term 1): Each task in the network has a minimum number of primitive actions 
       required for its completion, which depends on the current stack heights of its arguments. 
       Summing these is admissible as they represent distinct pieces of work in an ordered task network.
    2. Goal Stability / Displacement (Term 2): Any block not in its final goal position (or on an unstable base) 
       must be moved at least once (unstack + stack/putdown = 2 actions). 
    3. Nop-Disjointness: Since 'nop' and primitive actions are disjoint, we can sum an admissible bound 
       on 'nop' count with the maximum of our primitive action lower bounds.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_16="blocksworld_gemini3flash_heuristic_16"):
        super().__init__(name=blocksworld_gemini3flash_heuristic_16)
        self.goal_on = {}  # block -> target_block
        self.goal_ontable = set()
        self.on_fids = []  # list of (fid, top, bottom)
        self.on_fids_dict = {}  # (top, bottom) -> fid
        self.ontable_fids = {}  # block -> fid
        self.handempty_fid = None
        self.blocks = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        # Parse Goal State
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                self.goal_on[args[0]] = args[1]
            elif pred == "ontable":
                self.goal_ontable.add(args[0])

        # Map Facts to Global IDs
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            for arg in args:
                self.blocks.add(arg)
            
            if pred == "on":
                self.on_fids.append((f.global_id, args[0], args[1]))
                self.on_fids_dict[(args[0], args[1])] = f.global_id
            elif pred == "ontable":
                self.ontable_fids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_fid = f.global_id

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network: list) -> int:
        # 1. Build current state stack structure O(N)
        on_above = {} # bottom -> top
        on_below = {} # top -> bottom
        for fid, top, bottom in self.on_fids:
            if (state >> fid) & 1:
                on_above[bottom] = top
                on_below[top] = bottom

        # 2. Stability Analysis with memoization O(N)
        stable_cache = {}
        def is_stable(x):
            if x in stable_cache: return stable_cache[x]
            stable_cache[x] = False # detect cycles
            if (state >> self.ontable_fids.get(x, 0)) & 1:
                res = (x not in self.goal_on)
            elif x in on_below:
                y = on_below[x]
                if x in self.goal_on:
                    res = (self.goal_on[x] == y) and is_stable(y)
                elif x in self.goal_ontable:
                    res = False
                else:
                    res = is_stable(y)
            else:
                res = False
            stable_cache[x] = res
            return res

        # Populate stability cache for all blocks
        for b in self.blocks: 
            is_stable(b)

        # 3. Term 1 & 3: Task Network Costs O(TN * StackHeight)
        task_sum = 0
        nop_count = 0
        for task in task_network:
            parts = task.name.split()
            t_name = parts[0]
            args = parts[1:]
            if t_name == "nop":
                nop_count += 1
                continue
            
            # Recursive task cost logic
            h_x = self._get_height(args[0], on_above)
            clear_x = 2 * h_x
            
            if t_name == "do_clear":
                task_sum += clear_x
            elif t_name == "do_on_table":
                if not (state >> self.ontable_fids.get(args[0], 0)) & 1:
                    task_sum += clear_x + 2
            elif t_name == "do_move":
                h_y = self._get_height(args[1], on_above)
                task_sum += clear_x + 2 * h_y + 2
            elif t_name == "do_put_on":
                x, y = args[0], args[1]
                on_fid = self.on_fids_dict.get((x, y))
                if on_fid and (state >> on_fid) & 1:
                    if not stable_cache.get(x): task_sum += 4
                else:
                    h_y = self._get_height(y, on_above)
                    clear_y = 2 * h_y
                    task_sum += 2 * clear_x + 3 * clear_y + 4

        # 4. Term 2: Displacement Lower Bound
        displacement_cost = 0
        for b in self.blocks:
            if not stable_cache.get(b):
                displacement_cost += 2

        # Disjoint Hand cost
        hand_empty = (state >> self.handempty_fid) & 1 if self.handempty_fid is not None else True
        h_hand = 0 if hand_empty else 1

        return max(task_sum, displacement_cost) + nop_count + h_hand

    def _get_height(self, x, on_above):
        h = 0
        curr = x
        while curr in on_above:
            h += 1
            curr = on_above[curr]
        return h

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
