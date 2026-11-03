from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# BOTTLENECK: Stack clearing — to move a buried block, all blocks above must be unstacked first.
# ADMISSIBILITY SKETCHES:
# 1. Task Network Recursive Cost: For each task like do_put_on(x, y), it must clear x, clear y, and then move x.
#    Clearing a stack of height h takes at least 2*h actions (unstack + put-down). This is a safe lower bound.
# 2. Goal Stability: If on(x, y) is true but y is unstable (must be moved), x must be moved first (at least 5 actions).
# 3. Nop + Real Actions: Nop and primitive actions are disjoint, so summing their counts is safe.

class BlocksworldGTOHPGemini3FlashHeuristic12(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_12="blocksworld_gemini3flash_heuristic_12"):
        super().__init__(name=blocksworld_gemini3flash_heuristic_12)
        self.on_ids = {} # (x, y) -> global_id
        self.ontable_ids = {} # x -> global_id
        self.holding_ids = {} # x -> global_id
        self.handempty_id = -1
        self.id_to_fact = {} # global_id -> (pred, args)
        self.goal_on_map = {} # x -> y (if on(x, y) is a goal)
        self.all_blocks = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            self.id_to_fact[fact.global_id] = (pred, args)
            for arg in args:
                self.all_blocks.add(arg)
            
            if pred == "on":
                self.on_ids[(args[0], args[1])] = fact.global_id
            elif pred == "ontable":
                self.ontable_ids[args[0]] = fact.global_id
            elif pred == "holding":
                self.holding_ids[args[0]] = fact.global_id
            elif pred == "handempty":
                self.handempty_id = fact.global_id

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                self.goal_on_map[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, tn):
        # 1. Parse current state efficiently
        on_below = {} # x -> y (if on(x, y))
        on_above = {} # y -> x (if on(x, y))
        ontable_state = set()
        holding = None
        
        temp_state = state
        while temp_state:
            bit = (temp_state & -temp_state).bit_length() - 1
            if bit in self.id_to_fact:
                pred, args = self.id_to_fact[bit]
                if pred == "on":
                    on_below[args[0]] = args[1]
                    on_above[args[1]] = args[0]
                elif pred == "ontable":
                    ontable_state.add(args[0])
                elif pred == "holding":
                    holding = args[0]
            temp_state &= temp_state - 1

        # 2. Determine stability
        stable = set()
        for b in ontable_state:
            if b not in self.goal_on_map:
                stable.add(b)
                curr = b
                while curr in on_above:
                    top = on_above[curr]
                    if self.goal_on_map.get(top) == curr:
                        stable.add(top)
                        curr = top
                    else:
                        break

        # 3. Sum costs of remaining tasks
        h = 0.0
        for task in tn:
            if hasattr(task, 'cost'):
                h += task.cost
                continue
            
            parts = task.name.split()
            if not parts: continue
            t_name, t_args = parts[0], parts[1:]

            if t_name == "do_put_on":
                x, y = t_args[0], t_args[1]
                if holding == x:
                    h += self._get_height(y, on_above) * 2 + 1
                elif on_below.get(x) == y and y in stable:
                    pass
                elif on_below.get(x) == y and y not in stable:
                    h += 5
                else:
                    h += self._get_height(x, on_above) * 2 + self._get_height(y, on_above) * 2 + 2
            elif t_name == "do_clear":
                x = t_args[0]
                if holding != x:
                    h += self._get_height(x, on_above) * 2
            elif t_name == "do_on_table":
                x = t_args[0]
                if x not in ontable_state:
                    if holding == x:
                        h += 1
                    else:
                        h += self._get_height(x, on_above) * 2 + 2
            elif t_name == "do_move":
                x, y = t_args[0], t_args[1]
                if holding == x:
                    h += self._get_height(y, on_above) * 2 + 1
                else:
                    h += self._get_height(x, on_above) * 2 + self._get_height(y, on_above) * 2 + 2
            elif t_name == "nop":
                pass
            else:
                h += 1

        # Add tie-breaker to favor progression
        h += 0.001 * len(tn)
        return int(h)

    def _get_height(self, x, on_above):
        h = 0
        curr = x
        visited = {curr}
        while curr in on_above:
            curr = on_above[curr]
            if curr in visited: break
            visited.add(curr)
            h += 1
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
