from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic1(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.
    
    Bottleneck Analysis:
    The central bottleneck is the recursive requirement to move all rings on top of a ring r 
    to a buffer tower before r itself can be moved. This results in an exponential number 
    of actions (2^n - 1) for a tower of size n.

    Admissible Lower Bounds:
    1. Misplaced Rings: Any ring that is not 'stable' (meaning it and everything below it 
       are in their final goal configuration) must be moved at least once. This is a 
       conservative lower bound on the remaining actions.
    2. Task Network Size: Each task in the current task network (whether an abstract task 
       or an operator) represents at least one future action in the plan.

    The heuristic returns the sum of unstable rings and the current task network size,
    providing a clear guidance signal that decreases as rings are locked into place 
    and tasks are completed.
    """

    def __init__(self, towers_gemini3flash_heuristic_1="TowersGemini3FlashHeuristic1"):
        super().__init__(towers_gemini3flash_heuristic_1)
        self.goal_on = {}  # ring_name -> target_obj_name
        self.on_ids_by_ring = {} # ring -> {obj: global_id}
        self.all_on_fact_ids = [] # list of (ring, obj, global_id)
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
        # 1. Parse all facts to map the 'on' relations and identify rings/towers
        for i in range(model.ifacts_init, model.ifacts_end + 1):
            fact = model.get_component(i)
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                ring, obj = args[0], args[1]
                self.rings.add(ring)
                if ring not in self.on_ids_by_ring:
                    self.on_ids_by_ring[ring] = {}
                self.on_ids_by_ring[ring][obj] = fact.global_id
                self.all_on_fact_ids.append((ring, obj, fact.global_id))
            elif pred == "towerTop":
                self.towers.add(args[1])

        # 2. Identify goal states from model.goals bitwise representation
        goal_fact_names = model.state_explicit_repr(model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                # args: [ring, target_obj]
                self.goal_on[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state_bitwise, task_network) -> int:
        # Determine the current object each ring is sitting on
        current_on = {}
        for ring, obj, fact_id in self.all_on_fact_ids:
            if (state_bitwise >> fact_id) & 1:
                current_on[ring] = obj

        # Identify stable rings (bottom-up stack check)
        stable = set()
        changed = True
        while changed:
            changed = False
            for r in self.rings:
                if r not in stable:
                    # A ring is stable if it's on its goal target AND that target is stable (or is a tower)
                    target = self.goal_on.get(r)
                    if target and current_on.get(r) == target:
                        if target in self.towers or target in stable:
                            stable.add(r)
                            changed = True

        # Term 1: Rings that are not yet stable must be moved.
        # Only count rings that actually have a goal specified to ensure admissibility.
        unstable_count = 0
        for r in self.rings:
            if r in self.goal_on and r not in stable:
                unstable_count += 1

        # Term 2: Count of remaining tasks in the network.
        task_count = len(task_network)

        return unstable_count + task_count
