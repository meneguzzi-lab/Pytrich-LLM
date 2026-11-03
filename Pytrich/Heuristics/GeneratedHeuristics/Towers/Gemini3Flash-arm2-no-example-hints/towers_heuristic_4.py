from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic4(Heuristic):
    """
    Hanoi-specific heuristic for the Towers domain.

    Bottleneck Analysis: 
    The movement of the largest displaced ring. In the classic 3-peg Tower of Hanoi, 
    moving the k-th ring requires all k-1 smaller rings to be positioned on the 
    auxiliary peg. This domain follows this strict serialization.

    Admissibility Sketches:
    1. Hanoi Distance: The optimal number of moves to reach the goal configuration 
       from any valid 3-peg state is computed using a recursive O(N) formula. 
       Since the problem is a 3-peg Tower of Hanoi, this provides an exact 
       cost for the action sequences, making it perfectly admissible.
    2. Task network size: A small weight (0.1) per task in the network acts as 
       a tie-breaker to guide the search toward shallower task trees without 
       overestimating the total cost.
    """

    def __init__(self, towers_gemini3flash_heuristic_4="towers_gemini3flash_heuristic_4"):
        super().__init__(towers_gemini3flash_heuristic_4)
        self.tower_names = []
        self.rings_large_to_small = []
        self.on_fact_ids = {} # ring -> {obj: fact_id}
        self.goal_pegs = {}

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
        towers = set()
        rings = set()
        smaller_than_counts = {}

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "towerTop":
                towers.add(args[1])
            elif pred == "smallerThan":
                smaller, larger = args[0], args[1]
                smaller_than_counts[larger] = smaller_than_counts.get(larger, 0) + 1
                if smaller not in smaller_than_counts:
                    smaller_than_counts[smaller] = 0
                rings.add(smaller)

        self.tower_names = sorted(list(towers))
        actual_rings = [r for r in rings if r not in towers]
        self.rings_large_to_small = sorted(actual_rings, key=lambda r: smaller_than_counts.get(r, 0), reverse=True)

        for r in self.rings_large_to_small:
            self.on_fact_ids[r] = {}
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                r, obj = args[0], args[1]
                if r in self.on_fact_ids:
                    self.on_fact_ids[r][obj] = fact.global_id

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        goal_on = {}
        for f_name in goal_fact_names:
            pred, args = self._parse_fact_name(f_name)
            if pred == "on":
                goal_on[args[0]] = args[1]

        for r in self.rings_large_to_small:
            curr = goal_on.get(r)
            while curr in self.on_fact_ids:
                curr = goal_on.get(curr)
            if curr in self.tower_names:
                self.goal_pegs[r] = curr

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        current_pegs = {}
        for r in self.rings_large_to_small:
            for obj, fact_id in self.on_fact_ids[r].items():
                if (state >> fact_id) & 1:
                    if obj in self.tower_names:
                        current_pegs[r] = obj
                    else:
                        current_pegs[r] = current_pegs.get(obj)
                    break
        
        active_rings = [r for r in self.rings_large_to_small if r in self.goal_pegs and r in current_pegs]
        h = self.dist_config_to_config(active_rings, current_pegs, self.goal_pegs)
        return int(h + 0.1 * len(task_network))

    def get_aux_peg(self, p1, p2):
        for p in self.tower_names:
            if p != p1 and p != p2:
                return p
        return self.tower_names[0]

    def dist_config_to_config(self, rings, current_pegs, target_pegs):
        if not rings:
            return 0
        r = rings[0]
        if current_pegs[r] == target_pegs[r]:
            return self.dist_config_to_config(rings[1:], current_pegs, target_pegs)
        else:
            aux = self.get_aux_peg(current_pegs[r], target_pegs[r])
            return self.dist_to_peg(rings[1:], current_pegs, aux) + 1 + self.dist_from_peg(rings[1:], aux, target_pegs)

    def dist_to_peg(self, rings, current_pegs, target_peg):
        if not rings:
            return 0
        r = rings[0]
        if current_pegs[r] == target_peg:
            return self.dist_to_peg(rings[1:], current_pegs, target_peg)
        else:
            aux = self.get_aux_peg(current_pegs[r], target_peg)
            return self.dist_to_peg(rings[1:], current_pegs, aux) + (1 << (len(rings) - 1))

    def dist_from_peg(self, rings, src_peg, target_pegs):
        if not rings:
            return 0
        r = rings[0]
        if src_peg == target_pegs[r]:
            return self.dist_from_peg(rings[1:], src_peg, target_pegs)
        else:
            aux = self.get_aux_peg(src_peg, target_pegs[r])
            return (1 << (len(rings) - 1)) + self.dist_from_peg(rings[1:], aux, target_pegs)
