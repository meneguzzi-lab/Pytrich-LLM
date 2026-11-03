from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic18(Heuristic):
    """
    Domain-Dependent Heuristic for Blocksworld_GTOHP.

    Bottleneck Analysis:
    The central bottleneck is the 'clear' and 'stack-stability' constraint. Blocks can only be moved if 
    they are clear, and the destination must also be clear. The HTN structure decomposes high-level 
    movements into multiple clear and unstack operations.

    Admissibility Sketches:
    1. h_tn (Task Network Cost): Each abstract task represents a minimum sequence of actions 
       (e.g., do_move requires 2 actions: pick-up and stack). We sum the minimum costs of all tasks 
       remaining in the network.
    2. h_state (Goal Distance): Every goal on(A, B) not currently satisfied (or unstable) requires 
       at least 2 actions (a move). Summing 2 actions per unsatisfied goal provides a lower bound.
    3. Composite Heuristic: Returning max(h_tn, h_state) ensures the value is both admissible 
       and informed by both the HTN progress and the desired state goals.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_18="BlocksworldGTOHP_Gemini3Flash_Heuristic_18"):
        super().__init__(blocksworld_gemini3flash_heuristic_18)
        self.on_map = {}  # (x, y) -> global_id
        self.on_by_y = {} # y -> list of (x, global_id)
        self.ontable_map = {} # x -> global_id
        self.holding_map = {} # x -> global_id
        self.handempty_id = -1
        self.blocks = []
        self.goal_on = []
        self.goal_ontable = []

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args, prefix
        return None, [], ""

    def initialize(self, model: Model, initial_node: HTNNode):
        block_set = set()
        for fact in model.facts:
            pred, args, _ = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_map[(args[0], args[1])] = fact.global_id
                if args[1] not in self.on_by_y: self.on_by_y[args[1]] = []
                self.on_by_y[args[1]].append((args[0], fact.global_id))
                block_set.update(args)
            elif pred == "ontable":
                self.ontable_map[args[0]] = fact.global_id
                block_set.update(args)
            elif pred == "holding":
                self.holding_map[args[0]] = fact.global_id
                block_set.update(args)
            elif pred == "handempty":
                self.handempty_id = fact.global_id
        
        self.blocks = list(block_set)
        
        # Parse state goals
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_names:
            pred, args, _ = self._parse_fact_name(gname)
            if pred == "on":
                self.goal_on.append((args[0], args[1]))
            elif pred == "ontable":
                self.goal_ontable.append(args[0])

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _count_above(self, state, block):
        count = 0
        curr = block
        while True:
            found_next = False
            for next_block, fid in self.on_by_y.get(curr, []):
                if (state >> fid) & 1:
                    count += 1
                    curr = next_block
                    found_next = True
                    break
            if not found_next:
                break
        return count

    def _compute(self, state, task_network):
        # 1. Task Network Heuristic
        h_tn = 0
        for task in task_network:
            name = task.name
            bracket_idx = name.find('[')
            t_pred = name[:bracket_idx] if bracket_idx != -1 else name
            t_args = name[bracket_idx+1:-1].split(',') if bracket_idx != -1 else []
            
            if t_pred == "do_put_on":
                x, y = t_args[0], t_args[1]
                fid = self.on_map.get((x, y))
                if not (fid is not None and (state >> fid) & 1):
                    # Cost: move(2) + clear_x(2*above) + clear_y(2*above)
                    h_tn += 2 + self._count_above(state, x) * 2 + self._count_above(state, y) * 2
            elif t_pred == "do_on_table":
                x = t_args[0]
                fid = self.ontable_map.get(x)
                if not (fid is not None and (state >> fid) & 1):
                    h_tn += 2
            elif t_pred == "do_move":
                h_tn += 2
            elif t_pred == "do_clear":
                h_tn += self._count_above(state, t_args[0]) * 2
            elif t_pred in ["pick-up", "put-down", "stack", "unstack"]:
                h_tn += 1

        # 2. State Goal Heuristic (Admissible Move Count)
        h_state = 0
        unstable = set()
        for x, y in self.goal_on:
            fid = self.on_map.get((x, y))
            if not (fid is not None and (state >> fid) & 1):
                unstable.add(x)
        for x in self.goal_ontable:
            fid = self.ontable_map.get(x)
            if not (fid is not None and (state >> fid) & 1):
                unstable.add(x)
        
        # If a block x is on y and y must move, x must move
        # Optimization: Simplified unstable count as an admissible lower bound
        h_state = len(unstable) * 2
        
        # Hand check
        if self.handempty_id != -1 and not ((state >> self.handempty_id) & 1):
            h_state = max(0, h_state - 1)
            
        return max(h_tn, h_state)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
