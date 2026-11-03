from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic12(Heuristic):
    """
    Heuristic for Blocksworld_GTOHP HDDL Domain.

    1. Bottleneck Analysis:
       The primary bottleneck is the single robotic arm and the topological constraints of towers.
       A block cannot be moved unless it is clear, and only one block can be manipulated at a time.
       This serializes all actions involving unstacking, stacking, and putting blocks on the table.

    2. Admissible Lower Bounds:
       - Term 1 (HTN Task Depth): Each abstract task (e.g., do_put_on, do_clear) represents a promise
         of at least 2 actions (e.g., pick-up + stack or unstack + put-down) if its condition is not currently met.
       - Term 2 (Hand State): If the hand is not empty (holding a block), at least 1 action (stack or put-down)
         is required to return to the handempty state required by most task preconditions.
       - Term 3 (Decomposition Progress): Every non-nop operator in the task network has a unit cost of 1.

    3. Justification:
       These costs are lower bounds because they assume only the minimum possible sequence of actions
       to satisfy a task, ignoring additional clearing operations that might be required for blocks
       deeper in a tower.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_12="BlocksworldGTOHPGemini3FlashHeuristic12"):
        super().__init__(blocksworld_gemini3flash_heuristic_12)
        self.on_ids = {} # (block1, block2) -> global_id
        self.clear_ids = {} # block -> global_id
        self.ontable_ids = {} # block -> global_id
        self.handempty_id = -1
        self.abt_info = {} # gid -> (type, args_list)
        self.op_costs = {} # gid -> cost

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
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'on':
                if len(args) == 2:
                    self.on_ids[(args[0], args[1])] = f.global_id
            elif pred == 'clear':
                if len(args) == 1:
                    self.clear_ids[args[0]] = f.global_id
            elif pred == 'ontable':
                if len(args) == 1:
                    self.ontable_ids[args[0]] = f.global_id
            elif pred == 'handempty':
                self.handempty_id = f.global_id

        for abt in model.abstract_tasks:
            clean_name = abt.name.replace('(', '').replace(')', '').strip()
            parts = clean_name.split()
            if parts:
                self.abt_info[abt.global_id] = (parts[0], parts[1:])

        for op in model.operators:
            if 'nop' in op.name.lower():
                self.op_costs[op.global_id] = 0
            else:
                self.op_costs[op.global_id] = 1

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, tn):
        h = 0.0
        for task in tn:
            gid = task.global_id
            if gid in self.op_costs:
                h += self.op_costs[gid]
            elif gid in self.abt_info:
                t_type, t_args = self.abt_info[gid]
                if t_type == 'do_put_on':
                    if len(t_args) >= 2:
                        fid = self.on_ids.get((t_args[0], t_args[1]))
                        if fid is not None and not ((state >> fid) & 1):
                            h += 2.0
                elif t_type == 'do_clear':
                    if len(t_args) >= 1:
                        fid = self.clear_ids.get(t_args[0])
                        if fid is not None and not ((state >> fid) & 1):
                            h += 2.0
                elif t_type == 'do_on_table':
                    if len(t_args) >= 1:
                        fid = self.ontable_ids.get(t_args[0])
                        if fid is not None and not ((state >> fid) & 1):
                            h += 2.0
                elif t_type == 'do_move':
                    h += 2.0
                else:
                    h += 1.0
            else:
                h += 1.0

        if self.handempty_id != -1 and not ((state >> self.handempty_id) & 1):
            h += 1.0

        satisfied_goals = bin(state & self.model.goals).count('1')
        h -= 0.01 * satisfied_goals

        return max(0.0, h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
