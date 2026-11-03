from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic19(Heuristic):
    """
    Heuristic for Blocksworld_GTOHP domain.

    Bottleneck Analysis:
    The primary bottleneck in Blocksworld is the linear dependency of blocks in a stack. 
    A block cannot be moved until all blocks above it are cleared and moved to the table or another stack. 
    This serialization of 'clear' and 'move' operations dictates the minimal plan length.

    Admissible Lower Bounds:
    1. Network Cost: Sum of minimal action costs for all tasks remaining in the task network. 
       Each do_put_on, do_move, or do_clear task requires at least 2 actions if its target state is not met.
    2. Goal Cost: Every unsatisfied goal fact (e.g., 'on ?x ?y') requires at least one 'stack' action 
       (which typically implies a preceding pick-up/unstack, total 2 actions).
    3. Hand Status: If the hand is not empty, at least one action (put-down or stack) is required to clear it.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_19="blocksworld_gemini3flash_heuristic_19"):
        super().__init__(blocksworld_gemini3flash_heuristic_19)
        self.on_bits = {}           # (x, y) -> global_id
        self.clear_bits = {}        # x -> global_id
        self.ontable_bits = {}      # x -> global_id
        self.handempty_bit = None
        self.goal_bits = []
        self.task_info = {}

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
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_bits[(args[0], args[1])] = f.global_id
            elif pred == "clear":
                self.clear_bits[args[0]] = f.global_id
            elif pred == "ontable":
                self.ontable_bits[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_bit = f.global_id

        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on":
                bit = self.on_bits.get((args[0], args[1]))
                if bit is not None: self.goal_bits.append(bit)
            elif pred == "ontable":
                bit = self.ontable_bits.get(args[0])
                if bit is not None: self.goal_bits.append(bit)

        for abt in model.abstract_tasks:
            self.task_info[abt.global_id] = self._parse_task_name(abt.name)
        for op in model.operators:
            self.task_info[op.global_id] = self._parse_task_name(op.name)

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _compute(self, state, task_network):
        network_h = 0
        for task in task_network:
            info = self.task_info.get(task.global_id)
            if not info: continue
            t_type, t_args = info
            if t_type == "do_clear":
                bit = self.clear_bits.get(t_args[0])
                if bit is not None and not ((state >> bit) & 1):
                    network_h += 2
            elif t_type == "do_put_on":
                bit = self.on_bits.get((t_args[0], t_args[1]))
                if bit is not None and not ((state >> bit) & 1):
                    network_h += 2
            elif t_type == "do_on_table":
                bit = self.ontable_bits.get(t_args[0])
                if bit is not None and not ((state >> bit) & 1):
                    network_h += 2
            elif t_type == "do_move":
                network_h += 2
            elif t_type in ["pick-up", "put-down", "stack", "unstack"]:
                network_h += 1

        goal_h = 0
        for bit in self.goal_bits:
            if not ((state >> bit) & 1):
                goal_h += 2

        hand_h = 0
        if self.handempty_bit is not None and not ((state >> self.handempty_bit) & 1):
            hand_h = 1

        return max(network_h, goal_h, hand_h) + 0.001 * len(task_network)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
