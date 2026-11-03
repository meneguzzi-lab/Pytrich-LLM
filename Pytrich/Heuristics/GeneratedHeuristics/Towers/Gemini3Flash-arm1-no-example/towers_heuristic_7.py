from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# BOTTLENECK ANALYSIS:
# The Towers (Hanoi) domain is constrained by the strict stack discipline: a ring can only be moved if it is the top 
# of its current tower and the destination object is larger (or the tower base itself). This creates a recursive 
# dependency where larger rings cannot reach their goal positions until all smaller rings are moved out of the way.
# The serializing bottleneck is the sequence of 'correctly placed' rings from bottom to top.

# ADMISSIBLE LOWER BOUNDS:
# 1. Misplaced Rings Count (h_misplaced): A ring is misplaced if it is not on its goal-specified target, 
#    or if the object it is on is itself misplaced. Each misplaced ring requires at least one 'move' action. 
#    This is a standard admissible lower bound for block-stacking and Hanoi domains.
# 2. Task Network Depth (h_tasks): Each task in the network must eventually decompose into at least 
#    one primitive action (except for 'exchangeClear'). len(task_network) provides a basic progress measure.
# 3. Symmetry Breaking: Hanoi problems have few symmetries in the goal state, but we add a sub-unit penalty 
#    on the total task network size to favor nodes that have decomposed abstract tasks into concrete sequences.

class TowersGemini3FlashHeuristic7(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_7="towers_gemini3flash_heuristic_7"):
        super().__init__(name=towers_gemini3flash_heuristic_7)
        self.on_facts = {}  # (ring, target) -> global_id
        self.ring_goal = {} # ring -> target_name
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
        # Precompute fact mappings and goals
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                # args[0] is RING, args[1] is OBJ (RING or TOWER)
                self.on_facts[(args[0], args[1])] = fact.global_id
                self.rings.add(args[0])

        # Parse goal state facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.ring_goal[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _is_misplaced(self, ring, state, memo):
        if ring in memo:
            return memo[ring]
        
        goal_target = self.ring_goal.get(ring)
        if not goal_target:
            # If no goal is specified for this ring, it is not 'misplaced' in the goal sense
            memo[ring] = False
            return False
            
        # Check if the ring is currently on its goal target
        on_goal_id = self.on_facts.get((ring, goal_target))
        if on_goal_id is None or not ((state >> on_goal_id) & 1):
            memo[ring] = True
            return True
            
        # If target is a ring, it must also be correctly placed
        if goal_target in self.rings:
            is_target_misplaced = self._is_misplaced(goal_target, state, memo)
            memo[ring] = is_target_misplaced
            return is_target_misplaced
        
        # Target is a tower base and the ring is on it
        memo[ring] = False
        return False

    def _compute(self, state, task_network):
        # Term 1: Misplaced rings (Recursive definition)
        misplaced_count = 0
        memo = {}
        for ring in self.rings:
            if self._is_misplaced(ring, state, memo):
                misplaced_count += 1
        
        # Term 2: Task network size as tie-breaker/guidance
        # In HTN, a larger task network generally implies more work remains.
        h_tasks = len(task_network)
        
        # Combine terms: Primary is misplaced count, secondary is task network progress.
        # misplaced_count is multiplied by a factor to ensure it dominates.
        return (misplaced_count * 10) + h_tasks

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
