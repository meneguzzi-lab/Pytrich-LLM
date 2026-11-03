from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic15(Heuristic):
    """
    Domain-Dependent Heuristic for Blocksworld GTOHP.
    
    Bottleneck Analysis:
    Stack clearing is the primary bottleneck. To move a block 'x', all blocks above it must be unstacked and put down. 
    Each block above adds a cost of 2 actions (unstack + put-down).

    Admissibility Sketches:
    1. Task Network Cost: The sum of the minimum actions required to decompose and satisfy each task in the network. 
       Since tasks are serial and the effects of earlier tasks usually don't eliminate the need for actions in later tasks 
       (except for the 'nop' case which is handled), this is a solid lower bound.
    2. Stack Height Cost: For a 'do_put_on(x, y)' task, if x is not already well-placed on y, we MUST move x (at least 2 actions: 
       pick-up/unstack + stack) and we MUST clear x and clear y. This involves 2 actions for every block currently above x or y.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_15="blocksworld_gemini3flash_heuristic_15"):
        super().__init__(blocksworld_gemini3flash_heuristic_15)
        self.task_cache = {}
        self.fact_info = {}
        self.goal_on = {}
        self.handempty_id = -1

    def initialize(self, model: Model, initial_node: HTNNode):
        # Map fact IDs to useful data
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            self.fact_info[f.global_id] = (pred, args)
            if pred == "handempty":
                self.handempty_id = f.global_id

        # Map task IDs to names and args
        all_tasks = model.abstract_tasks + model.operators
        for t in all_tasks:
            self.task_cache[t.global_id] = self._parse_task_name(t.name)

        # Parse goal state
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        if not task_network and (state & self.model.goals) == self.model.goals:
            return 0

        # 1. Parse current state into fast-access maps
        on_map = {} # bottom -> top
        on_map_rev = {} # top -> bottom
        clear_set = set()
        handempty = False

        temp_state = state
        while temp_state:
            bit = (temp_state & -temp_state).bit_length() - 1
            info = self.fact_info.get(bit)
            if info:
                pred, args = info
                if pred == "on":
                    on_map[args[1]] = args[0]
                    on_map_rev[args[0]] = args[1]
                elif pred == "clear":
                    clear_set.add(args[0])
                elif pred == "handempty":
                    handempty = True
            temp_state &= temp_state - 1

        # 2. Sum costs of current tasks in network
        h = 0 if handempty else 1
        for task in task_network:
            tid = task.global_id
            info = self.task_cache.get(tid)
            if not info: continue
            name, args = info

            if name == "do_put_on":
                x, y = args[0], args[1]
                # If block x is already on y and is stable, cost is 0 (m0)
                if on_map_rev.get(x) == y and self._is_well_placed(x, on_map_rev):
                    pass
                else:
                    # Clear x, clear y, then pick and stack (m1)
                    h += 2 + 2 * self._get_height(x, on_map) + 2 * self._get_height(y, on_map)
            elif name == "do_clear":
                # Clear x via m6 (nop) or m7 (recurse)
                h += 2 * self._get_height(args[0], on_map)
            elif name == "do_move":
                # m4/m5: 2 actions (pick/unstack + stack)
                h += 2
            elif name == "do_on_table":
                # m2 (2 actions) or m3 (0 actions if clear)
                if args[0] not in clear_set:
                    h += 2 * self._get_height(args[0], on_map) + 2
                else:
                    h += 0
            elif name == "nop":
                h += 0
            else:
                # Primitive actions (unstack, stack, pick-up, put-down)
                h += 1

        # 3. Tie-breaker to favor deeper networks or progress
        h += 0.001 * len(task_network)
        return int(h)

    def _get_height(self, block, on_map):
        """Returns number of blocks stacked above 'block'."""
        count = 0
        curr = block
        while curr in on_map:
            count += 1
            curr = on_map[curr]
        return count

    def _is_well_placed(self, x, on_map_rev):
        """Returns True if x and all blocks beneath it match the goal positions."""
        curr = x
        while curr:
            actual_under = on_map_rev.get(curr)
            goal_under = self.goal_on.get(curr)
            if actual_under != goal_under:
                return False
            curr = actual_under
        return True

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
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_name = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_name, args
        return name, []
