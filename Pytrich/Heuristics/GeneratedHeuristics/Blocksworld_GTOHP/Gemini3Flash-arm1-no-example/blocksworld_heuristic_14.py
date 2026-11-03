from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The Blocksworld_GTOHP domain is a hierarchical version of the classic Blocksworld.
# The primary bottleneck is the single-hand constraint (handempty/holding) and the stack structure,
# which requires serial disassembly (clearing) before blocks can be moved.
# Admissibility Sketch:
# 1. Task Network Cost: Every task (do_put_on, do_clear, etc.) in the network requires a minimum
#    number of actions to reach a 'nop' or terminal state. We assign a conservative lower bound
#    (e.g., 2 actions for a move or a clearance) to each pending task.
# 2. Hand State: If the hand is not empty, at least one action (put-down or stack) is required
#    to return to a state where new operations can begin.
# 3. Disjoint Costs: Since the task network in ordered HTN represents a sequence of work,
#    summing the minimum costs of its elements is a valid admissible lower bound.

class BlocksworldGTOHPGemini3FlashHeuristic14(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_14="BlocksworldGTOHPGemini3FlashHeuristic14"):
        super().__init__(blocksworld_gemini3flash_heuristic_14)
        self.on_ids = {}
        self.clear_ids = {}
        self.ontable_ids = {}
        self.handempty_id = None
        self.task_info = {}
        self.op_info = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # Preprocess fact IDs
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_ids[tuple(args)] = fact.global_id
            elif pred == "clear":
                self.clear_ids[args[0]] = fact.global_id
            elif pred == "ontable":
                self.ontable_ids[args[0]] = fact.global_id
            elif pred == "handempty":
                self.handempty_id = fact.global_id

        # Preprocess Task and Operator metadata
        for task in model.abstract_tasks:
            name_parts = task.name.replace("(", "").replace(")", "").split()
            if name_parts:
                self.task_info[task.global_id] = (name_parts[0], name_parts[1:])

        for op in model.operators:
            name_parts = op.name.replace("(", "").replace(")", "").split()
            if name_parts:
                # Use the operator's defined cost if available
                self.op_info[op.global_id] = (name_parts[0], op.cost)

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        h = 0
        
        # 1. Sum costs of tasks in the network
        for task in task_network:
            if task.global_id in self.op_info:
                name, cost = self.op_info[task.global_id]
                if name != "nop":
                    h += cost
            elif task.global_id in self.task_info:
                t_type, args = self.task_info[task.global_id]
                if t_type == "do_put_on":
                    # Admissible: if not on(x,y), needs at least pick-up + stack = 2
                    if not self._is_true(state, self.on_ids.get(tuple(args))):
                        h += 2
                elif t_type == "do_clear":
                    # Admissible: if not clear, needs at least unstack + put-down = 2
                    if args and not self._is_true(state, self.clear_ids.get(args[0])):
                        h += 2
                elif t_type == "do_on_table":
                    # Admissible: if not on table, needs at least unstack + put-down = 2
                    if args and not self._is_true(state, self.ontable_ids.get(args[0])):
                        h += 2
                elif t_type == "do_move":
                    # do_move always results in 2 actions
                    h += 2

        # 2. Hand check
        if self.handempty_id is not None:
            if not ((state >> self.handempty_id) & 1):
                h += 1

        return h

    def _is_true(self, state, fact_id):
        if fact_id is None:
            return False
        return (state >> fact_id) & 1

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
            else:
                return name, []
        return None, []
