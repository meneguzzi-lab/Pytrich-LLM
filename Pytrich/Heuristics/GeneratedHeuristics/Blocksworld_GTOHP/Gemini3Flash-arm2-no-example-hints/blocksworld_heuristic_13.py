from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic13(Heuristic):
    # BOTTLENECK: Stack clearing - moving a buried block requires unstacking all blocks above it.
    # ADMISSIBILITY SKETCHES:
    # 1. ClearCost(x): 2 actions (unstack + put-down) for every block in the stack above x. Admissible because each must be moved.
    # 2. MoveCost(x, y): If x is not on y, it requires at least a pick-up/unstack and a stack (2 actions).
    # 3. Stability: If on(x, y) is true but y must be moved to satisfy its own goal, x must also be moved (cost 2).

    def __init__(self, blocksworld_gemini3flash_heuristic_13="BlocksworldGTOHPGemini3FlashHeuristic13"):
        super().__init__(blocksworld_gemini3flash_heuristic_13)
        self.on_facts = []
        self.ontable_facts = []
        self.holding_ids = {}
        self.handempty_id = None
        self.goal_on = {}
        self.goal_ontable = set()
        self.blocks = set()

    def _parse_fact_name(self, name):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find(" [") if " [" in name else name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, [a.strip() for a in args]
        return None, []

    def _parse_task(self, name):
        parts = name.replace("[", " ").replace("]", " ").replace(",", " ").split()
        if not parts: return "", []
        return parts[0], parts[1:]

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_facts.append((f.global_id, args[0], args[1]))
                self.blocks.update(args)
            elif pred == "ontable":
                self.ontable_facts.append((f.global_id, args[0]))
                self.blocks.update(args)
            elif pred == "holding":
                self.holding_ids[args[0]] = f.global_id
                self.blocks.update(args)
            elif pred == "handempty":
                self.handempty_id = f.global_id

        goal_names = model.state_explicit_repr(model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on": self.goal_on[args[0]] = args[1]
            elif pred == "ontable": self.goal_ontable.add(args[0])

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _is_stable(self, b, under_map, memo):
        if b in memo: return memo[b]
        if b is None: return True
        curr_under = under_map.get(b)
        if b in self.goal_on:
            target_under = self.goal_on[b]
            res = (curr_under == target_under) and self._is_stable(target_under, under_map, memo)
        elif b in self.goal_ontable:
            res = (curr_under is None)
        else:
            res = self._is_stable(curr_under, under_map, memo)
        memo[b] = res
        return res

    def _compute(self, state, task_network):
        under_map = {}
        above_map = {}
        is_holding = None
        for gid, x, y in self.on_facts:
            if (state >> gid) & 1: under_map[x], above_map[y] = y, x
        for gid, x in self.ontable_facts:
            if (state >> gid) & 1: under_map[x] = None
        for x, gid in self.holding_ids.items():
            if (state >> gid) & 1: is_holding = x

        height_above = {}
        for b in self.blocks:
            h, curr = 0, b
            while curr in above_map:
                h += 1
                curr = above_map[curr]
            height_above[b] = h

        memo, h_total = {}, 0
        for task in task_network:
            tname, args = self._parse_task(task.name)
            if tname == "nop": continue
            if tname in ["pick-up", "put-down", "stack", "unstack"]:
                h_total += 1
            elif tname == "do_clear":
                h_total += 2 * height_above.get(args[0], 0)
            elif tname == "do_on_table":
                if under_map.get(args[0]) is not None: h_total += 2 + 2 * height_above.get(args[0], 0)
            elif tname in ["do_put_on", "do_move"]:
                x, y = args[0], args[1]
                if tname == "do_put_on" and under_map.get(x) == y and self._is_stable(x, under_map, memo): continue
                cost = 1 if is_holding == x else 2
                h_total += cost + 2 * height_above.get(x, 0) + 2 * height_above.get(y, 0)
        return h_total

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
