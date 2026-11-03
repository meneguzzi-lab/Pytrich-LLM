from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic7(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Bottleneck Analysis:
    The primary bottleneck in this domain is the serial constraint of the robotic arm. 
    Only one block can be moved at a time, and a block can only be moved if it is clear 
    and its destination is clear. This serializes all work and requires stacking/unstacking 
    chains of blocks.

    Admissibility Sketches:
    - Term 1 (Task Network Cost): Each high-level task (do_put_on, do_clear, etc.) must eventually 
      decompose into a sequence of primitive actions (pick-up, stack, unstack, put-down). 
      By estimating the minimum number of primitive actions required to satisfy each remaining 
      task in the ordered network based on the current state, we provide an admissible estimate 
      of the remaining work.
    - Term 2 (Hand Capacity): The goal state always requires the hand to be empty. If holding(x), 
      at least one more action (put-down or stack) is mandatory.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_7="blocksworld_gemini3flash_heuristic_7"):
        super().__init__(blocksworld_gemini3flash_heuristic_7)
        self.on_fids = {}  # (block_x, block_y) -> global_id
        self.ontable_fids = {}  # block -> global_id
        self.holding_fids = {}  # block -> global_id
        self.handempty_fid = None
        self.on_id_to_args = {} # global_id -> (block_x, block_y)

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates to global IDs for fast bitwise checking
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_fids[(args[0], args[1])] = fact.global_id
                self.on_id_to_args[fact.global_id] = (args[0], args[1])
            elif pred == "ontable":
                self.ontable_fids[args[0]] = fact.global_id
            elif pred == "holding":
                self.holding_fids[args[0]] = fact.global_id
            elif pred == "handempty":
                self.handempty_fid = fact.global_id

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network) -> int:
        if not task_network:
            return 0

        # 1. Build current stacking status (who is on top of whom)
        who_on = {} # y -> x (where x is on y)
        temp_state = state
        while temp_state:
            # Get lowest set bit
            fid = (temp_state & -temp_state).bit_length() - 1
            if fid in self.on_id_to_args:
                x, y = self.on_id_to_args[fid]
                who_on[y] = x
            temp_state &= temp_state - 1

        # 2. Hand status: if not handempty, we need at least 1 move to clear it
        hand_empty = (state >> self.handempty_fid) & 1 if self.handempty_fid is not None else True
        h = 0 if hand_empty else 1

        # 3. Process task network costs
        for task in task_network:
            name, args = self._parse_task_name(task.name)
            
            if name == "do_put_on":
                x, y = args[0], args[1]
                # If x is already on y, cost is 0 (m0)
                fid = self.on_fids.get((x, y), -1)
                if fid != -1 and (state >> fid) & 1:
                    continue
                else:
                    # m1: clear x, clear y, y on table, move x to y
                    h += self._get_clear_cost(x, who_on) 
                    h += self._get_clear_cost(y, who_on)
                    h += self._get_on_table_cost(y, state, who_on)
                    h += 2 # cost of do_move (2 actions)

            elif name == "do_clear":
                h += self._get_clear_cost(args[0], who_on)

            elif name == "do_on_table":
                h += self._get_on_table_cost(args[0], state, who_on)

            elif name == "do_move":
                h += 2 # move involves pick/unstack + stack

            elif name == "nop":
                continue
            
            else:
                # Primitive actions or unknown tasks
                if name in ["pick-up", "put-down", "stack", "unstack"]:
                    h += 1
                # else ignore (cost 0)
        
        return h

    def _get_clear_cost(self, block, who_on):
        """Estimate cost to clear block (2 actions per block above it)."""
        cost = 0
        cursor = block
        while cursor in who_on:
            cost += 2 # unstack + put-down
            cursor = who_on[cursor]
        return cost

    def _get_on_table_cost(self, block, state, who_on):
        """Estimate cost to move block to table."""
        fid = self.ontable_fids.get(block, -1)
        if fid != -1 and (state >> fid) & 1:
            return 0
        # Need to clear it, then move it to the table
        return self._get_clear_cost(block, who_on) + 2

    def _parse_task_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx == -1:
            return name, []
        task_type = name[:bracket_idx]
        args = name[bracket_idx+1:-1].split(",")
        return task_type, args

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
