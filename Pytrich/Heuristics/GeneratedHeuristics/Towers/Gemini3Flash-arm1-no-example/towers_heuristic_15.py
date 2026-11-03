from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic15(Heuristic):
    """
    Bottleneck Analysis:
    The fundamental bottleneck in the Towers domain is the recursive dependency of ring movements. 
    To move a specific ring to its goal, all rings currently on top of it must be cleared, and 
    the target destination must also be cleared of any smaller rings. This is the classic 
    Towers of Hanoi constraint, where the minimum number of moves required is 2^N - 1 for N rings.

    Admissible Lower Bounds:
    1. Hanoi Distance: For each ring i (from 0 to N-1, where 0 is the smallest), if it is not 
       'stable' (meaning it's not on its goal or its goal is not stable), it requires at 
       least 2^i moves to be placed correctly. This is a known admissible lower bound for 
       the number of moves in the Hanoi problem.
    2. Task Network Length: Each task in the network represents a unit of work that must 
       eventually be decomposed into actions. While less informative than the state-based 
       Hanoi distance, it provides tie-breaking guidance when the state remains the same 
       across decompositions.

    Admissibility Sketch:
    The sum of 2^i for all unstable rings is admissible because it represents the minimum number 
    of move actions required to achieve the goal state from the current configuration, assuming 
    a standard 3-peg Hanoi structure which this domain follows.
    """

    def __init__(self, towers_gemini3flash_heuristic_15="towers_gemini3flash_heuristic_15"):
        super().__init__(towers_gemini3flash_heuristic_15)
        self.ring_goal_on = {}  # ring -> target_obj
        self.on_fact_ids = {}   # ring -> {obj: fact_id}
        self.ring_powers = {}   # ring -> power (0 for smallest)
        self.rings = []
        self.towers = []

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
        smaller_than_pairs = set()

        # 1. Map 'on' fact IDs for fast state checking
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                ring, obj = args[0], args[1]
                if ring not in self.on_fact_ids:
                    self.on_fact_ids[ring] = {}
                self.on_fact_ids[ring][obj] = fact.global_id
                all_objs.add(ring)
                all_objs.add(obj)
            elif pred == "towerTop":
                self.towers.append(args[1]) if args[1] not in self.towers else None
            elif pred == "smallerThan":
                smaller_than_pairs.add((args[0], args[1]))

        self.towers = list(set(self.towers))
        self.rings = [obj for obj in all_objs if obj not in self.towers]

        # 2. Parse Goal State and Hints
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on":
                self.ring_goal_on[args[0]] = args[1]

        init_names = self.model.state_explicit_repr(self.model.initial_state)
        for iname in init_names:
            pred, args = self._parse_fact_name(iname)
            if pred == "goal_on":
                self.ring_goal_on[args[0]] = args[1]

        # 3. Precompute Ring Sizes (Powers for Hanoi Heuristic)
        for r in self.rings:
            # Power is the number of rings r_inner such that r_inner is smaller than r
            p = 0
            for r_inner in self.rings:
                if (r_inner, r) in smaller_than_pairs:
                    p += 1
            self.ring_powers[r] = p

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _compute(self, state, task_network):
        # Hanoi Distance: Sum of 2^power for all unstable rings
        h_state = 0
        stable = set(self.towers)

        # Process rings from largest to smallest to check stability bottom-up
        sorted_rings = sorted(self.rings, key=lambda r: self.ring_powers.get(r, 0), reverse=True)
        
        for r in sorted_rings:
            target = self.ring_goal_on.get(r)
            if target in stable:
                fact_id = self.on_fact_ids.get(r, {}).get(target)
                # Check if ring r is physically on the correct target in current state
                if fact_id is not None and (state >> fact_id) & 1:
                    stable.add(r)
                else:
                    h_state += (1 << self.ring_powers.get(r, 0))
            else:
                # Target is a ring that isn't stable, so this ring can't be stable
                h_state += (1 << self.ring_powers.get(r, 0))

        # Task Network guidance (tie-breaker magnitude)
        tn_guidance = 0.001 * len(task_network)

        return int(h_state + tn_guidance)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
