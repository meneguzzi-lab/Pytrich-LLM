from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.operators import Operator

# Bottleneck: The primary bottleneck is stack manipulation, where moving a block 
# requires clearing all blocks on top of it. Each move or clear operation 
# directly serializes the plan, and the hand-empty constraint enforces sequential actions.

# Admissibility Sketches:
# 1. Task Cost: Each 'do_put_on', 'do_move', 'do_on_table', and 'do_clear' task 
#    in the task network requires at least one primitive action (if not already 
#    satisfied). Specifically, a 'do_move' or unsatisfied 'do_put_on' 
#    requires at least 2 actions (pick/unstack + stack/putdown). 
# 2. Goal Cost: Every 'on(x, y)' goal not currently satisfied in the state 
#    requires at least 2 primitive actions to achieve. Since these actions 
#    cannot be performed simultaneously (single hand), the sum is admissible 
#    if we only count unique goals required by the current decomposition.

class BlocksworldGTOHPGemini3FlashHeuristic5(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_5="BlocksworldGTOHPGemini3FlashHeuristic5"):
        super().__init__(blocksworld_gemini3flash_heuristic_5)
        self.on_id = {}
        self.clear_id = {}
        self.ontable_id = {}
        self.handempty_id = None
        self.goal_ons = []
        self.task_data = {}

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = [a.strip() for a in args_str.split(",")] if args_str.strip() else []
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_id[(args[0], args[1])] = f.global_id
            elif pred == "clear":
                self.clear_id[args[0]] = f.global_id
            elif pred == "ontable":
                self.ontable_id[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.goal_ons.append((args[0], args[1]))

        for i in range(model.iabt_init, model.iabt_end + 1):
            task = model.get_component(i)
            parts = task.name.split()
            if len(parts) > 0:
                self.task_data[task.global_id] = (parts[0], parts[1:])

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _is_true(self, state, f_id):
        return (state >> f_id) & 1 if f_id is not None else False

    def _compute(self, state, task_network):
        h_tasks = 0
        if not self._is_true(state, self.handempty_id):
            h_tasks += 1

        for task in task_network:
            if task.global_id <= self.model.iop_end:
                h_tasks += getattr(task, 'cost', 1)
            else:
                data = self.task_data.get(task.global_id)
                if data:
                    t_type, args = data
                    if t_type == "do_put_on":
                        if not self._is_on(state, args[0], args[1]):
                            h_tasks += 2
                    elif t_type == "do_clear":
                        if not self._is_clear(state, args[0]):
                            h_tasks += 2
                    elif t_type == "do_move":
                        h_tasks += 2
                    elif t_type == "do_on_table":
                        if not self._is_ontable(state, args[0]):
                            h_tasks += 2
                else:
                    h_tasks += 1

        h_goals = 0
        for x, y in self.goal_ons:
            if not self._is_on(state, x, y):
                h_goals += 2

        return max(h_tasks, h_goals)

    def _is_on(self, state, x, y):
        f_id = self.on_id.get((x, y))
        return (state >> f_id) & 1 if f_id is not None else False

    def _is_clear(self, state, x):
        f_id = self.clear_id.get(x)
        return (state >> f_id) & 1 if f_id is not None else False

    def _is_ontable(self, state, x):
        f_id = self.ontable_id.get(x)
        return (state >> f_id) & 1 if f_id is not None else False

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
