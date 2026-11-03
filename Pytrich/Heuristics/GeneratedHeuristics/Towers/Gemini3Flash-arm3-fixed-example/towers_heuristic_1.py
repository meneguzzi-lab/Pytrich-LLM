import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic1(Heuristic):
    """
    Domain-Dependent Heuristic for the Towers (Tower of Hanoi) domain.

    Bottleneck Analysis:
    The central bottleneck in the Towers domain is the recursive serial dependency of moving rings.
    A ring can only be moved if it is the 'towerTop' (no rings above it) and the destination 
    is also a 'towerTop' and satisfies the 'smallerThan' constraint. This means to move a 
    misplaced ring, all rings above it must first be moved to the auxiliary tower.

    Admissible Lower Bounds:
    1. Unstable Rings: A ring is 'stable' if it is currently on its goal-on target and that target 
       is also stable (a tower is always stable). Any ring that is not stable MUST be moved at least once. 
       The number of unstable rings is a solid admissible lower bound on the remaining number of move actions.
    2. Misplaced Support: If a ring R is on its goal target O, but O itself is not stable, R will eventually
       have to be moved to allow O to be moved. This is captured by the recursive definition of stability.
    """

    def __init__(self, towers_gemini3flash_heuristic_1="towers_gemini3flash_heuristic_1"):
        super().__init__(name=towers_gemini3flash_heuristic_1)
        self.rings = []
        self.towers = []
        self.goal_on_map = {}  # ring_name -> target_object_name
        self.on_fact_ids = {}  # (ring, obj) -> global_id
        self.ring_order = []   # Rings sorted from largest to smallest
        self.total_rings = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Identify all objects
        all_rings = set()
        all_towers = set()
        smaller_than_counts = {} # ring -> count of things it is smaller than

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                # args[0] is ring, args[1] is obj (ring or tower)
                all_rings.add(args[0])
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "towerTop":
                # args[0] can be ring or tower
                if "t" in args[0]: # Heuristic check for tower based on standard naming
                    all_towers.add(args[0])
            elif pred == "smallerThan":
                r = args[0]
                all_rings.add(r)
                smaller_than_counts[r] = smaller_than_counts.get(r, 0) + 1
        
        self.rings = list(all_rings)
        self.towers = list(all_towers)
        self.total_rings = len(self.rings)

        # 2. Sort rings from Largest to Smallest
        # In Hanoi, the largest ring is smallerThan the fewest number of other RINGS.
        # More specifically, if we count smallerThan(r, other_ring), the largest has count 0.
        ring_to_ring_smaller_counts = {}
        for r in self.rings:
            ring_to_ring_smaller_counts[r] = 0

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "smallerThan":
                r1, r2 = args[0], args[1]
                if r1 in all_rings and r2 in all_rings:
                    ring_to_ring_smaller_counts[r1] += 1
        
        # Sort: largest rings have smallest counts of being "smallerThan" other rings
        self.ring_order = sorted(self.rings, key=lambda r: ring_to_ring_smaller_counts[r])

        # 3. Parse Goals
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                self.goal_on_map[args[0]] = args[1]

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

    def _compute(self, state, task_network):
        # 1. Calculate Unstable Rings
        # A ring is stable if it's on its goal target AND its target is stable.
        # We iterate from largest to smallest because the stability of a ring
        # depends on the stability of the larger object beneath it.
        stable_objects = set(self.towers)
        stable_count = 0

        for r in self.ring_order:
            target = self.goal_on_map.get(r)
            if not target:
                continue # Should not happen in standard Towers
            
            on_id = self.on_fact_ids.get((r, target))
            if on_id is not None:
                is_on_target = (state >> on_id) & 1
                if is_on_target and target in stable_objects:
                    stable_objects.add(r)
                    stable_count += 1

        h_unstable = self.total_rings - stable_count

        # 2. HTN Guidance
        # The number of tasks in the network provides a fine-grained tie-breaker.
        # Every task must eventually decompose into actions.
        h_tasks = len(task_network) * 0.1

        return h_unstable + h_tasks

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return int(h_value) if h_value > 1000 else h_value
