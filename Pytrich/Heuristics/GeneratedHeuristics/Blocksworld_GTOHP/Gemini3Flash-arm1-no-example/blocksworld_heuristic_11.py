from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic11(Heuristic):
    """
    Heuristic for Blocksworld_GTOHP domain.

    1. Bottleneck: The primary bottleneck is the sequential requirement to clear blocks (removing all obstacles above) 
       before they can be moved, and the specific domain method requirement to move target blocks to the table 
       before stacking them.

    2. Admissible Bounds:
       - h_tasks: Sum of minimal action counts for each task in the network. For example, a 'do_put_on' task 
         requires at least 2 actions (move/stack) plus clearing costs (2 actions each) if preconditions aren't met.
       - h_hand: A lower bound of 1 action if the hand is currently holding a block, as it must be emptied 
         before most high-level methods can be applied.

    Admissibility: Each term represents an unavoidable set of actions (decompositions or operations) 
    required to reach the goal state from the current task network and state.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_11="blocksworld_gemini3flash_heuristic_11"):
        super().__init__(name=blocksworld_gemini3flash_heuristic_11)
        self.fact_map = {}
        self.task_info_cache = {}
        self.handempty_id = -1

    def initialize(self, model: Model, initial_node: HTNNode):
        # Precompute fact mappings for O(1) lookups
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_map[(pred, tuple(args))] = fact.global_id
                if pred == "handempty":
                    self.handempty_id = fact.global_id

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network) -> int:
        h = 0

        # Term 1: Sum of minimal costs for tasks remaining in the network
        for task in task_network:
            if hasattr(task, 'cost'):  # Operator
                h += task.cost
            else:  # AbstractTask
                t_type, t_args = self._get_task_info(task)
                
                if t_type == 'do_put_on':
                    # x is moved to y. If already there, cost is 0.
                    if not self._is_true(state, 'on', t_args):
                        h += 2  # At least pick-up/unstack + stack
                        # Minimal clearing costs if clear/ontable not met
                        if not self._is_true(state, 'clear', (t_args[0],)): h += 2
                        if len(t_args) > 1:
                            if not self._is_true(state, 'clear', (t_args[1],)): h += 2
                            if not self._is_true(state, 'ontable', (t_args[1],)): h += 2
                
                elif t_type == 'do_on_table':
                    # x is moved to table. If already there, cost is 0.
                    if not self._is_true(state, 'ontable', (t_args[0],)):
                        h += 2  # At least unstack + put-down
                        if not self._is_true(state, 'clear', (t_args[0],)): h += 2

                elif t_type == 'do_clear':
                    # x must be clear. If not, at least one block above must be moved.
                    if not self._is_true(state, 'clear', (t_args[0],)):
                        h += 2 # unstack + put-down of obstacle
                
                elif t_type == 'do_move':
                    # Move x to y always takes 2 actions (grounded operators)
                    h += 2

        # Term 2: Hand status. If holding, must empty hand (+1 action)
        if self.handempty_id != -1:
            if not ((state >> self.handempty_id) & 1):
                h += 1

        return h

    def _is_true(self, state: int, pred: str, args: tuple) -> bool:
        fid = self.fact_map.get((pred, args))
        if fid is not None:
            return (state >> fid) & 1
        return False

    def _get_task_info(self, task):
        if task.global_id in self.task_info_cache:
            return self.task_info_cache[task.global_id]
        
        # Robust parsing of grounded task names like "do_put_on b1 b2" or "do_put_on[b1,b2]"
        name = task.name.strip().replace('(', '').replace(')', '').replace(',', ' ').replace('[', ' ').replace(']', ' ')
        parts = name.split()
        if not parts:
            res = (None, ())
        else:
            res = (parts[0], tuple(parts[1:]))
        
        self.task_info_cache[task.global_id] = res
        return res

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, [a.strip() for a in args]
        return None, []
