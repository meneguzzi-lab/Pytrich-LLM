from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Bottleneck Analysis:
The Towers domain is a classic recursive stack-based problem (Tower of Hanoi). The primary bottleneck is the strict linear sequence of moves required to relocate a stack: a ring cannot be moved until all rings smaller than it (which are currently above it) and all rings smaller than it on the destination are cleared. This serializes the work into a $2^n-1$ complexity structure.

Admissibility Sketches:
1. Misplaced Rings Bound: A ring r that is not currently on its target goal object (or sits on an object that is itself misplaced) MUST be moved at least once. Counting these provides a robust linear admissible lower bound.
2. Task Network Bound: Every task currently in the HTN task network (primitive or abstract) must eventually be decomposed into at least one action (except for specific base cases like exchangeClear). Counting remaining tasks provides a state-independent admissible lower bound on remaining work.
"""

class TowersGemini3FlashHeuristic4(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_4="towers_gemini3flash_heuristic_4"):
        super().__init__(name=towers_gemini3flash_heuristic_4)
        self.on_fact_ids = {}  # (ring, target) -> global_id
        self.ring_goals = {}   # ring -> goal_target_obj
        self.rings = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all available grounded facts
        rings_found = set()
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                # Format: on[ring, obj]
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
                rings_found.add(args[0])
        
        self.rings = list(rings_found)
        
        # 1. Primary goal source: model.goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.ring_goals[args[0]] = args[1]
        
        # 2. Secondary goal source: goal_on helper predicate in initial state
        init_state_facts = model.state_explicit_repr(initial_node.state)
        for f_name in init_state_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == "goal_on":
                if args[0] not in self.ring_goals:
                    self.ring_goals[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # Term 1: State-based Misplaced Rings (Linear Hanoi Bound)
        h_misplaced = 0
        memo = {}
        for r in self.rings:
            if self._is_misplaced(r, state, memo):
                h_misplaced += 1
        
        # Term 2: Task-based Bound
        # Count non-nop tasks. Each task in TN must resolve to at least one action.
        h_tasks = 0
        tie_breaker = 0.0
        for task in task_network:
            name = task.name.lower()
            # exchangeClear is the only no-op method base case
            if "exchangeclear" in name:
                continue
            h_tasks += 1
            # Symmetry/Progress tie-breaker: prefer Operators over AbstractTasks
            if self.model.iop_init <= task.global_id <= self.model.iop_end:
                tie_breaker += 0.001
            else:
                tie_breaker += 0.002

        return max(float(h_misplaced), float(h_tasks)) + tie_breaker

    def _is_misplaced(self, ring, state, memo):
        if ring in memo: return memo[ring]
        
        target = self.ring_goals.get(ring)
        if not target:
            memo[ring] = False
            return False
            
        # Check if the ring is on its goal target
        on_target_id = self.on_fact_ids.get((ring, target))
        is_on_target = (state >> on_target_id) & 1 if on_target_id is not None else False
        
        if not is_on_target:
            memo[ring] = True
            return True
            
        # If ring is on target, but target is a misplaced ring, then this ring must move too
        if target in self.ring_goals:
            res = self._is_misplaced(target, state, memo)
            memo[ring] = res
            return res
            
        memo[ring] = False
        return False

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
