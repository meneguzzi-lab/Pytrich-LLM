import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic15(Heuristic):
    """
    Heuristic for the Towers of Hanoi domain.

    1. Bottleneck Analysis:
    The primary constraint is the recursive dependency of ring movements. A ring 'k' cannot move 
    until all rings smaller than it are cleared from its current tower and its destination tower top.
    This structure results in an exponential relationship between the number of rings and moves.

    2. Admissible Lower Bounds:
    - Term 1: Exponential Misplacement Cost. In the optimal Towers of Hanoi solution, if the k-th 
      smallest ring must be moved, it implies at least 2^(k-1) moves remain. We define a ring as 
      'misplaced' if it is not currently on its goal object or if the object it is on is misplaced.
    - Term 2: Task Network Length. Each task in the HTN network represents at least one primitive 
      'move' action eventually. The current number of tasks is a simple lower bound on remaining effort.

    3. Implementation:
    - Precompute ring sizes and goal configurations in initialize().
    - Use bitwise operations for fast state checks in __call__().
    """
    def __init__(self, towers_gemini3flash_heuristic_15="towers_gemini3flash_heuristic_15"):
        super().__init__(name=towers_gemini3flash_heuristic_15)
        self.on_fact_ids = {}      # (ring, object) -> global_id
        self.ring_goal_on = {}     # ring -> object (from goals)
        self.ring_sizes = {}       # ring -> rank (1 is smallest)
        self.rings = set()

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
        self.model = model
        
        # 1. Parse facts to identify rings and 'on' mappings
        smaller_than_rels = set()
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
                self.rings.add(args[0])
            elif pred == "smallerThan":
                smaller_than_rels.add((args[0], args[1]))
        
        # 2. Rank rings by size using smallerThan relationship
        for r in self.rings:
            # size = 1 + number of other rings smaller than this one
            count_smaller = 0
            for x in self.rings:
                if (x, r) in smaller_than_rels:
                    count_smaller += 1
            self.ring_sizes[r] = count_smaller + 1

        # 3. Determine goal configuration
        # model.goals bitwise integer transformed to explicit names
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on" and len(args) == 2:
                self.ring_goal_on[args[0]] = args[1]

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _is_misplaced(self, ring, state):
        """
        A ring is misplaced if it's not on its target or if the object below it 
        is misplaced. This recursively validates the stack bottom-up.
        """
        curr = ring
        # Iterate down the stack to check if the path to the tower base matches goals
        visited = set()
        while curr in self.rings and curr not in visited:
            visited.add(curr)
            target = self.ring_goal_on.get(curr)
            if not target:
                return False # No specified goal for this ring
            
            fid = self.on_fact_ids.get((curr, target))
            if fid is None or not ((state >> fid) & 1):
                return True # Not on goal target
            
            # If it is on the goal target, and that target is a tower, we're done
            if target not in self.rings:
                return False
            
            # If the target is another ring, move down and check if that ring is correct
            curr = target
        return False

    def _compute(self, state, task_network):
        # Term 1: Admissible Hanoi Bound (2^(k-1) where k is size of largest misplaced ring)
        max_misplaced_size = 0
        for r in self.rings:
            if self._is_misplaced(r, state):
                size = self.ring_sizes.get(r, 0)
                if size > max_misplaced_size:
                    max_misplaced_size = size
        
        h_hanoi = (1 << (max_misplaced_size - 1)) if max_misplaced_size > 0 else 0

        # Term 2: Task Network Admissible bound
        h_tn = len(task_network)

        return max(h_hanoi, h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
