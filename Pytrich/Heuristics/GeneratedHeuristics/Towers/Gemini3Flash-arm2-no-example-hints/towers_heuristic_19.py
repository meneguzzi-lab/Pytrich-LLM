from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic19(Heuristic):
    """
    Bottleneck Analysis:
    The core bottleneck in the Towers (Tower of Hanoi) domain is the recursive constraint: 
    to move a ring r from one tower to another, all smaller rings currently above it 
    must be moved to a third tower. This structure results in an exponential state-space 
    distance relative to the number of displaced rings.

    Admissible Lower Bound Sketch:
    1. Hanoi Distance: The exact optimal number of moves to reach the goal configuration
       from the current state in a 3-peg Tower of Hanoi problem. Because the 'move' 
       action is the only primitive with cost 1, the Hanoi distance calculation 
       provides a perfect admissible lower bound for the primitive move count.
    """

    def __init__(self, towers_gemini3flash_heuristic_19="towers_gemini3flash_heuristic_19"):
        super().__init__(towers_gemini3flash_heuristic_19)
        self.ring_to_idx = {}
        self.tower_to_idx = {}
        self.rings = []
        self.on_ids = {}
        self.possible_on = {}
        self.target_pegs = []
        self.powers_of_2 = []

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
        all_rings = set()
        all_towers = set()

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'smallerThan':
                all_rings.add(args[0])
                if args[1].startswith('t'):
                    all_towers.add(args[1])
                else:
                    all_rings.add(args[1])
            if pred == 'on':
                self.on_ids[(args[0], args[1])] = f.global_id
                if args[0] not in self.possible_on:
                    self.possible_on[args[0]] = []
                self.possible_on[args[0]].append(args[1])

        ring_to_smaller_count = {r: 0 for r in all_rings}
        for f in model.facts:
            p, args = self._parse_fact_name(f.name)
            if p == 'smallerThan' and args[0] in all_rings and args[1] in all_rings:
                ring_to_smaller_count[args[1]] += 1
        
        self.rings = sorted(list(all_rings), key=lambda r: ring_to_smaller_count[r])
        self.ring_to_idx = {r: i for i, r in enumerate(self.rings)}
        self.tower_to_idx = {t: i+1 for i, t in enumerate(sorted(list(all_towers)))}

        init_on_links = {}
        for (r, o), fid in self.on_ids.items():
            if (initial_node.state >> fid) & 1:
                init_on_links[r] = o

        init_pegs = [0] * len(self.rings)
        for i in range(len(self.rings) - 1, -1, -1):
            r = self.rings[i]
            obj = init_on_links.get(r)
            if obj in self.tower_to_idx:
                init_pegs[i] = self.tower_to_idx[obj]
            elif obj in self.ring_to_idx:
                init_pegs[i] = init_pegs[self.ring_to_idx[obj]]
            else:
                init_pegs[i] = 1

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        goal_on_links = {}
        for g in goal_fact_names:
            p, args = self._parse_fact_name(g)
            if p == 'on':
                goal_on_links[args[0]] = args[1]

        self.target_pegs = [0] * len(self.rings)
        for i in range(len(self.rings) - 1, -1, -1):
            r = self.rings[i]
            curr = goal_on_links.get(r)
            if curr is None:
                self.target_pegs[i] = init_pegs[i]
            elif curr in self.tower_to_idx:
                self.target_pegs[i] = self.tower_to_idx[curr]
            elif curr in self.ring_to_idx:
                self.target_pegs[i] = self.target_pegs[self.ring_to_idx[curr]]
            else:
                self.target_pegs[i] = 1

        self.powers_of_2 = [2**i for i in range(len(self.rings) + 1)]
        h_val = self._compute(initial_node.state)
        return super().initialize(model, h_val)

    def state_to_peg_dist(self, pegs, k, annexed_dest):
        dist = 0
        curr_dest = annexed_dest
        for i in range(k - 1, -1, -1):
            if pegs[i] != curr_dest:
                dist += self.powers_of_2[i]
                curr_dest = 6 - pegs[i] - curr_dest
        return dist

    def peg_to_state_dist(self, target_pegs, k, annexed_src):
        dist = 0
        curr_src = annexed_src
        for i in range(k - 1, -1, -1):
            if target_pegs[i] != curr_src:
                dist += self.powers_of_2[i]
                curr_src = 6 - target_pegs[i] - curr_src
        return dist

    def _compute(self, state):
        if not self.rings:
            return 0
        on_links = {}
        for r in self.rings:
            for obj in self.possible_on.get(r, []):
                fid = self.on_ids.get((r, obj))
                if fid is not None and (state >> fid) & 1:
                    on_links[r] = obj
                    break
        current_pegs = [0] * len(self.rings)
        for i in range(len(self.rings) - 1, -1, -1):
            r = self.rings[i]
            obj = on_links.get(r)
            if obj in self.tower_to_idx:
                current_pegs[i] = self.tower_to_idx[obj]
            elif obj in self.ring_to_idx:
                current_pegs[i] = current_pegs[self.ring_to_idx[obj]]
            else:
                current_pegs[i] = 1
        k = -1
        for i in range(len(self.rings) - 1, -1, -1):
            if current_pegs[i] != self.target_pegs[i]:
                k = i
                break
        if k == -1:
            return 0
        other_peg = 6 - current_pegs[k] - self.target_pegs[k]
        return self.state_to_peg_dist(current_pegs, k, other_peg) + 1 + self.peg_to_state_dist(self.target_pegs, k, other_peg)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state)
        super().update_info(h_val)
        return h_val
