from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The primary bottleneck in the Blocksworld_GTOHP domain is the serial nature of block manipulation.
# To move or stack a block, both the block itself and its target destination must be clear.
# Clearing a block requires a recursive chain of unstacking actions, which dominates the plan length.
#
# Admissibility Sketches:
# 1. h_tasks: For each task in the network, we compute a minimum number of required actions.
#    - do_clear(x) requires 2 actions (unstack and put-down) for every block currently above x.
#    - do_put_on(x, y), do_move(x, y), and do_on_table(x) each require at least 2 actions if not satisfied.
#    - Operators (pick-up, stack, etc.) cost 1.
#    This is admissible because it represents the absolute minimum actions to decompose and execute the tasks.
# 2. h_goals: For any goal (on x y) or (ontable x) that is not currently satisfied and not currently
#    the target of an active task in the network, we add 2 actions. This is admissible as moving
#    a block into place requires at least two steps (e.g., pick-up + stack).

class BlocksworldGTOHPGemini3FlashHeuristic1(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_1="blocksworld_gemini3flash_heuristic_1"):
        super().__init__(blocksworld_gemini3flash_heuristic_1)
        self.on_ids = {} # (child, parent) -> global_id
        self.ontable_ids = {} # block -> global_id
        self.fid_to_on = {} # global_id -> (child, parent)
        self.task_lookup = {} # global_id -> (task_type, args)
        self.goal_on = []
        self.goal_ontable = []

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
        self.model = model
        # Precompute Fact Maps
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'on':
                u, v = args[0], args[1]
                self.on_ids[(u, v)] = fact.global_id
                self.fid_to_on[fact.global_id] = (u, v)
            elif pred == 'ontable':
                self.ontable_ids[args[0]] = fact.global_id

        # Precompute Task Parsing
        for t in model.operators:
            self.task_lookup[t.global_id] = ('op', [])
        for t in model.abstract_tasks:
            name_parts = t.name.split()
            t_type = name_parts[0]
            t_args = name_parts[1:] if len(name_parts) > 1 else []
            self.task_lookup[t.global_id] = (t_type, t_args)

        # Precompute Goals
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            p, args = self._parse_fact_name(gn)
            if p == 'on':
                self.goal_on.append((args[0], args[1]))
            elif p == 'ontable':
                self.goal_ontable.append(args[0])

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        # Identify blocks in the current state for height calculations
        # Using bit manipulation to find set bits efficiently
        active_on = {} # parent -> child (the one on top of it)
        temp_state = state
        while temp_state:
            # Get the index of the lowest set bit
            fid = (temp_state & -temp_state).bit_length() - 1
            if fid in self.fid_to_on:
                child, parent = self.fid_to_on[fid]
                active_on[parent] = child
            # Clear the lowest set bit
            temp_state &= temp_state - 1

        def count_above(block):
            count = 0
            curr = block
            while curr in active_on:
                count += 1
                curr = active_on[curr]
            return count

        h = 0
        tasks_covering_on = set()

        # Cost tasks currently in the network
        for task in task_network:
            info = self.task_lookup.get(task.global_id)
            if not info:
                h += 1
                continue
            
            t_type, t_args = info
            if t_type == 'op':
                h += 1
            elif t_type == 'do_clear' and t_args:
                h += 2 * count_above(t_args[0])
            elif t_type == 'do_put_on' and len(t_args) >= 2:
                u, v = t_args[0], t_args[1]
                on_fid = self.on_ids.get((u, v))
                if not (on_fid is not None and (state >> on_fid) & 1):
                    h += 2
                tasks_covering_on.add((u, v))
            elif t_type == 'do_move' and len(t_args) >= 2:
                h += 2
                tasks_covering_on.add((t_args[0], t_args[1]))
            elif t_type == 'do_on_table' and t_args:
                ot_fid = self.ontable_ids.get(t_args[0])
                if not (ot_fid is not None and (state >> ot_fid) & 1):
                    h += 2
            else:
                h += 1

        # Cost missing goals not covered by the task network
        for u, v in self.goal_on:
            on_fid = self.on_ids.get((u, v))
            if not (on_fid is not None and (state >> on_fid) & 1):
                if (u, v) not in tasks_covering_on:
                    h += 2
        
        for u in self.goal_ontable:
            ot_fid = self.ontable_ids.get(u)
            if not (ot_fid is not None and (state >> ot_fid) & 1):
                # Minimal assumption: if a goal ontable(u) is missing, it needs a task
                h += 1

        # Symmetry breaking / guidance: favor smaller task networks
        return int(h) + (0.001 * len(task_network))
