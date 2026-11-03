from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The core constraint in the Towers of Hanoi is the recursive serializing requirement: 
# to move a ring r, all rings smaller than r must be cleared from the source tower 
# and cannot be on the target tower. This creates a deep stack dependency.
#
# Admissibility Sketches:
# 1. Misplaced Rings: Any ring r that is not currently on its goal support, or whose support 
#    is itself misplaced, must move at least once. Counting these unstable rings 
#    provides an admissible lower bound because each must undergo at least one 'move' action.
# 2. Task Network: Every non-trivial task (like 'move_abstract' or 'shiftTower') in the network 
#    represents at least one future action. Counting the length of the task network is a standard 
#    admissible (or near-admissible) heuristic term in HTN planning.

class TowersGemini3FlashHeuristic14(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_14="towers_gemini3flash_heuristic_14"):
        super().__init__(name=towers_gemini3flash_heuristic_14)
        self.on_ids = {} # ring -> [(support, global_id)]
        self.goal_on = {} # ring -> target_support
        self.rings = set()
        self.towers = set()

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
        all_objs = set()
        
        # 1. Map grounded 'on' facts and collect objects
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                ring, sup = args[0], args[1]
                self.rings.add(ring)
                all_objs.add(ring)
                all_objs.add(sup)
                if ring not in self.on_ids: self.on_ids[ring] = []
                self.on_ids[ring].append((sup, fact.global_id))
        
        # 2. Identify towers (objects that are never moved/placed on top of something else)
        self.towers = all_objs - self.rings

        # 3. Determine goal configuration
        # First check explicit model goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            p, a = self._parse_fact_name(g_name)
            if p == "on":
                self.goal_on[a[0]] = a[1]
        
        # If no goal 'on' facts, check initial state for 'goal_on' helper predicates
        if not self.goal_on:
            init_fact_names = self.model.state_explicit_repr(initial_node.state)
            for f_name in init_fact_names:
                p, a = self._parse_fact_name(f_name)
                if p == "goal_on":
                    self.goal_on[a[0]] = a[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _is_stable(self, obj, state, memo):
        if obj in memo: return memo[obj]
        if obj in self.towers:
            memo[obj] = True
            return True
        if obj not in self.rings:
            memo[obj] = True
            return True

        # Find current support in state
        current_sup = None
        for sup, fid in self.on_ids.get(obj, []):
            if (state >> fid) & 1:
                current_sup = sup
                break
        
        if current_sup is None:
            memo[obj] = False
            return False

        # Stability check: must be on correct support AND support must be stable
        goal_sup = self.goal_on.get(obj)
        if goal_sup is None:
            # If no specific goal, stability is inherited from support
            res = self._is_stable(current_sup, state, memo)
        else:
            res = (current_sup == goal_sup) and self._is_stable(current_sup, state, memo)
        
        memo[obj] = res
        return res

    def _compute(self, state, task_network):
        memo = {}
        misplaced_count = 0
        for r in self.rings:
            if not self._is_stable(r, state, memo):
                misplaced_count += 1
        
        # Term 2: Task network complexity
        task_cost = len(task_network)
        
        return misplaced_count + task_cost

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
