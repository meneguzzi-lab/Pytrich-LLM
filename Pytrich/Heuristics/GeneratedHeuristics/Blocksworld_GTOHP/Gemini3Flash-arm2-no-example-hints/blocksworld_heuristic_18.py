import re
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic18(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld_GTOHP.

    BOTTLENECK: Stack clearing — to move a buried block, all blocks above must be unstacked first.

    ADMISSIBILITY SKETCHES:
    1. Task Network Cost: Sum of minimum actions to decompose each task in the current network.
       For do_put_on(x,y), at least 2*height(x) + 2*height(y) + 2 actions are needed (clearance + stacking).
    2. Goal-Stability Cost: Count of goal facts not yet satisfied or currently true but on an unstable base.
       If on(x,y) is a goal but y is not stable, x must be moved (at least 5 actions).
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_18="blocksworld_heuristic"):
        super().__init__(blocksworld_gemini3flash_heuristic_18)
        self.on_ids = {}            # [top][bottom] -> global_id
        self.ontable_ids = {}       # [block] -> global_id
        self.clear_ids = {}         # [block] -> global_id
        self.on_id_to_pair = {}     # global_id -> (top, bottom)
        self.ontable_id_to_block = {} # global_id -> block
        self.all_blocks = set()
        self.goals_bitwise = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self.goals_bitwise = model.goals

        for fact in model.facts:
            self.all_blocks.update(self._parse_fact_name(fact.name)[1])
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                top, bottom = args[0], args[1]
                if top not in self.on_ids: self.on_ids[top] = {}
                self.on_ids[top][bottom] = fact.global_id
                self.on_id_to_pair[fact.global_id] = (top, bottom)
            elif pred == "ontable":
                self.ontable_ids[args[0]] = fact.global_id
                self.ontable_id_to_block[fact.global_id] = args[0]
            elif pred == "clear":
                self.clear_ids[args[0]] = fact.global_id

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        on_map_bottom_to_top = {} # y -> x if on(x,y)
        on_map_top_to_bottom = {} # x -> y if on(x,y)
        on_table_set = set()
        
        temp_state = state
        while temp_state:
            fid = (temp_state & -temp_state).bit_length() - 1
            if fid in self.on_id_to_pair:
                top, bottom = self.on_id_to_pair[fid]
                on_map_bottom_to_top[bottom] = top
                on_map_top_to_bottom[top] = bottom
            elif fid in self.ontable_id_to_block:
                on_table_set.add(self.ontable_id_to_block[fid])
            temp_state &= temp_state - 1

        memo_h = {}
        def get_height(block):
            if block in memo_h: return memo_h[block]
            h, curr, visited = 0, block, {block}
            while curr in on_map_bottom_to_top:
                curr = on_map_bottom_to_top[curr]
                if curr in visited: break
                visited.add(curr)
                h += 1
            memo_h[block] = h
            return h

        def get_stability():
            stable = set(on_table_set)
            changed = True
            while changed:
                changed = False
                for top, bottom in on_map_top_to_bottom.items():
                    if top not in stable and bottom in stable:
                        fid = self.on_ids.get(top, {}).get(bottom)
                        if fid is not None and (self.goals_bitwise >> fid) & 1:
                            stable.add(top)
                            changed = True
            return stable

        stable_set = get_stability()
        
        network_h = 0
        for task in task_network:
            t_type, args = self._parse_task(task.name)
            if not t_type or t_type == 'nop': continue
            
            if t_type == 'do_put_on' and len(args) == 2:
                x, y = args[0], args[1]
                fid = self.on_ids.get(x, {}).get(y)
                if fid is not None and (state >> fid) & 1:
                    if x not in stable_set: network_h += 5
                else:
                    network_h += 4 + 2 * get_height(x) + 2 * get_height(y)
            elif t_type == 'do_clear' and len(args) == 1:
                network_h += 2 * get_height(args[0])
            elif t_type == 'do_on_table' and len(args) == 1:
                fid = self.ontable_ids.get(args[0])
                if fid is None or not (state >> fid) & 1: network_h += 2
            elif t_type == 'do_move':
                network_h += 2
            elif hasattr(task, 'cost'):
                network_h += task.cost
            else:
                network_h += 1

        goal_h = 0
        temp_goals = self.goals_bitwise
        while temp_goals:
            fid = (temp_goals & -temp_goals).bit_length() - 1
            if fid in self.on_id_to_pair:
                top, bottom = self.on_id_to_pair[fid]
                is_satisfied = (state >> fid) & 1
                if not is_satisfied:
                    goal_h += 3
                elif top not in stable_set:
                    goal_h += 5
            temp_goals &= temp_goals - 1

        return max(network_h, goal_h)

    def _parse_fact_name(self, name: str):
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return None, []

    def _parse_task(self, name):
        parts = [p for p in re.split(r'[ \(\),]+', name) if p]
        if not parts: return None, []
        return parts[0].lower(), parts[1:]
