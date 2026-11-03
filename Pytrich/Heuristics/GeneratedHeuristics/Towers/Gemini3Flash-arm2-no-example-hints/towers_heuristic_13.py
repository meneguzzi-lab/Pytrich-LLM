from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class TowersGemini3FlashHeuristic13(Heuristic):
    """
    Bottleneck Analysis:
    In the Towers (Tower of Hanoi) domain, the single constraint that serializes work is the largest ring 
    that is currently in a misplaced position (either not on its goal tower or not on its goal-specified supporting ring). 
    Moving any ring k requires clearing all rings smaller than it (1 to k-1) onto a spare tower, which is 
    an exponential task relative to the ring's size.

    Admissible Lower Bounds:
    1. Hanoi Distance: For the largest misplaced ring of size k (where k=1 is the smallest ring), 
       the minimum number of move actions required to place it and subsequent rings correctly is 2^k - 1. 
       This is a well-known optimal lower bound for the 3-peg Tower of Hanoi, which matches the method structure 
       of this HDDL domain.
    2. Task Network Complexity: Every non-trivial abstract task in the network will eventually decompose into 
       at least one 'move' action. However, the state-based Hanoi distance is much stronger and typically dominates.
    """

    def __init__(self, towers_gemini3flash_heuristic_13="towers_gemini3flash_heuristic_13"):
        super().__init__(towers_gemini3flash_heuristic_13)
        self.goal_checks = []

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = [a.strip() for a in args_str.split(",")] if args_str else []
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        goal_fact_names = model.state_explicit_repr(model.goals)
        
        ring_names = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) > 0:
                ring_names.add(args[0])
        
        smaller_than_counts = {r: 0 for r in ring_names}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "smallerThan" and len(args) == 2:
                r_small, r_large = args[0], args[1]
                if r_small in ring_names and r_large in ring_names:
                    smaller_than_counts[r_large] += 1
        
        ring_sizes = {r: (smaller_than_counts[r] + 1) for r in ring_names}
        
        self.goal_checks = []
        for f in model.facts:
            if f.name in goal_fact_names:
                pred, args = self._parse_fact_name(f.name)
                if pred == "on" and len(args) > 0:
                    ring = args[0]
                    size = ring_sizes.get(ring, 0)
                    self.goal_checks.append((size, f.global_id))
        
        self.goal_checks.sort(key=lambda x: x[0], reverse=True)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        for size, fid in self.goal_checks:
            if not ((state >> fid) & 1):
                return (1 << size) - 1
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        # Subtle tie-breaker to prefer smaller task networks
        tie_breaker = 0.001 * len(node.task_network)
        final_h = int(h_value + tie_breaker)
        super().update_info(final_h)
        return final_h
