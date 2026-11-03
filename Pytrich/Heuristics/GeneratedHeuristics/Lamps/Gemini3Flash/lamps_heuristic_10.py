import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class LampsGemini3FlashHeuristic10(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.

    1. Bottleneck Analysis:
       The primary cost in this domain comes from 'flip' actions. Each lamp that needs to change
       status (from Y to N or vice versa) must be the subject of at least one primitive 'flip'
       action, regardless of whether it was triggered by a manual 'turn' or by propagation.

    2. Admissible Lower Bounds:
       - Hamming Distance: Each coordinate (x, y) whose status in the current state does not match
         the goal status must be flipped. Since each 'flip' costs 1, the number of such coordinates
         is a strict lower bound on the remaining cost. Each flip changes only one lamp.
       - Admissibility Sketch: Every status change requires at least one primitive action. If
         H lamps differ from the goal, we need at least H flip actions. HTN-specific costs like
         precondition checks and no-ops only increase the total cost beyond H.

    3. Implementation:
       - initialize(): Maps (x, y, status) to global bit IDs and identifies goal bits.
       - __call__(): Efficiently counts goal bits that are false in the current state bitset.
    """
    def __init__(self, lamps_gemini3flash_heuristic_10=None, name="LampsGemini3FlashHeuristic10"):
        super().__init__(name=name)
        self.goal_bits = []
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        
        on_map = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 3:
                # (x, y, status) -> bit
                on_map[(args[0], args[1], args[2])] = f.global_id
        
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_fact_names:
                pred, args = self._parse_fact_name(g_name)
                if pred == "on" and len(args) == 3:
                    bit = on_map.get((args[0], args[1], args[2]))
                    if bit is not None:
                        self.goal_bits.append(bit)
                else:
                    # Fallback for any other goal predicates that might exist
                    for f in self.model.facts:
                        if f.name == g_name:
                            self.goal_bits.append(f.global_id)
                            break

        self.preprocessing_time = time.time() - start_time
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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

    def _compute(self, state_bitwise: int, task_network):
        # Primary admissible term: Hamming distance for goal facts
        h = 0
        for bit in self.goal_bits:
            # If the bit representing the goal status is not set, we need at least one flip
            if not ((state_bitwise >> bit) & 1):
                h += 1
        
        # Secondary tie-breaker: task network size
        # This provides a small incentive to reduce complexity in the decomposition stack
        h += 0.001 * len(task_network)
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsGemini3FlashHeuristic10(goals={len(self.goal_bits)})"
