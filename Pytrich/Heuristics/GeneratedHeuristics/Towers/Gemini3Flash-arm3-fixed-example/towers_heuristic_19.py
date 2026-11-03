from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic19(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.
    
    1. Bottleneck Analysis:
    The bottleneck in this domain is the serial dependency of ring movements. To move any ring 'r', 
    all rings currently on top of it must be cleared, and its target destination (another ring or a tower) 
    must also be clear and larger than 'r'. This creates a deep hierarchical structure where solving 
    a tower of size N requires solving towers of size N-1 twice.

    2. Admissible Lower Bounds:
    - Term 1 (Stability): A ring is 'stable' if it is on its target destination AND that destination is also stable 
      (towers are always stable). Any ring not stable MUST move at least once. This is a lower bound because 
      moving an unstable ring to its correct spot takes at least 1 move.
    - Term 2 (Task Network): Every non-trivial task in the HTN network represents pending work that must 
      eventually be decomposed into at least one action. In this domain, task network length is a reliable 
      conservative estimate of remaining steps.

    The heuristic returns the sum of these costs as an estimate for guidance.
    """
    def __init__(self, towers_gemini3flash_heuristic_19="towers_gemini3flash_heuristic_19"):
        super().__init__(name=towers_gemini3flash_heuristic_19)
        self.on_facts_info = []
        self.goal_map = {}
        self.rings = []
        self.towers = set()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args, prefix
        return None, [], ""

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        # 1. Parse goals to identify target positions
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args, _ = self._parse_fact_name(gname)
            if pred == "on":
                self.goal_map[args[0]] = args[1]
        
        # 2. Fallback if goals are empty in the model (common in HTN tasks-only problems)
        if not self.goal_map:
            for fact in self.model.facts:
                pred, args, _ = self._parse_fact_name(fact.name)
                if pred == "goal_on" and (initial_node.state >> fact.global_id) & 1:
                    self.goal_map[args[0]] = args[1]

        self.rings = list(self.goal_map.keys())

        # 3. Identify all 'on' fact IDs for bitwise checking and determine towers
        for fact in self.model.facts:
            pred, args, _ = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_facts_info.append((args[0], args[1], fact.global_id))
            elif pred == "towerTop":
                # The second argument of towerTop is always a tower
                if len(args) > 1:
                    self.towers.add(args[1])
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        # 1. Determine current ring-to-object mapping
        current_on = {}
        for r, o, fid in self.on_facts_info:
            if (state >> fid) & 1:
                current_on[r] = o
        
        # 2. Iteratively identify stable rings (bottom-up from towers)
        stable = set(self.towers)
        progress = True
        while progress:
            progress = False
            for r in self.rings:
                if r not in stable:
                    target = self.goal_map.get(r)
                    if target in stable and current_on.get(r) == target:
                        stable.add(r)
                        progress = True
        
        # 3. Calculate heuristic terms
        # Term 1: Number of unstable rings
        unstable_count = 0
        for r in self.rings:
            if r not in stable:
                unstable_count += 1
        
        # Term 2: Complexity of the remaining task network
        task_count = len(task_network)

        # Combining terms for guidance. Using addition captures the necessity of moving 
        # rings and decomposing high-level tasks into actions.
        return unstable_count + task_count

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
