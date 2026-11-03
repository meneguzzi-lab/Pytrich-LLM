from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic12(Heuristic):
    """
    Domain-Dependent Heuristic for the Towers (Hanoi) domain.
    
    Bottleneck Analysis:
    The core bottleneck in the Towers domain is the serial constraint of the stack.
    To move a ring r, all rings currently above it must be moved to an auxiliary tower,
    and the target destination (if it's a ring) must also be clear and 'stable' (already well-placed).
    
    Admissible Lower Bounds:
    1. Misplaced Ring Cost (State-based): Any ring r not on its goal target must move.
       Moreover, a ring is 'unstable' if the object beneath it is not its goal target 
       or is itself unstable. For each unstable ring, we assign a cost proportional 
       to its size (a lower bound on the moves needed to clear it and place it).
    2. Task Network Depth: Each task or operator remaining in the task network represents 
       at least one future action (move), or a decomposition that will eventually produce actions.
    
    The heuristic returns the maximum of these terms to ensure admissibility while providing guidance.
    """

    def __init__(self, towers_gemini3flash_heuristic_12="TowersGemini3FlashHeuristic12"):
        super().__init__(towers_gemini3flash_heuristic_12)
        self.on_fact_ids = {} # (ring, target) -> global_id
        self.goal_on_map = {} # ring -> target_obj
        self.ring_sizes = {}  # ring -> int
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
        # 1. Map all possible 'on' facts to their global IDs
        # 2. Identify towers and rings
        # 3. Build ring size map (how many smaller rings exist for each ring)
        smaller_than_rels = []

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
                self.rings.add(args[0])
            elif pred == "towerTop":
                self.towers.add(args[1])
            elif pred == "smallerThan":
                # smallerThan(r_small, r_large)
                smaller_than_rels.append((args[0], args[1]))

        # Infer ring sizes from smallerThan relations
        for r in self.rings:
            # size = 1 + count of rings smaller than r
            smaller_count = 0
            for r_small, r_large in smaller_than_rels:
                if r_large == r and r_small in self.rings:
                    smaller_count += 1
            self.ring_sizes[r] = smaller_count + 1

        # 4. Extract goal targets from model.goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.goal_on_map[args[0]] = args[1]

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state_bitwise, task_network):
        memo_wp = {}

        def is_well_placed(r):
            if r in memo_wp: return memo_wp[r]
            # A ring is well-placed if it is on its goal target AND the target is stable
            target = self.goal_on_map.get(r)
            if not target:
                memo_wp[r] = True
                return True
            
            fid = self.on_fact_ids.get((r, target))
            if fid is None or not ((state_bitwise >> fid) & 1):
                memo_wp[r] = False
                return False
            
            # If target is a tower, the base is stable
            if target in self.towers:
                memo_wp[r] = True
                return True
            
            # Target is another ring; it must also be well-placed
            res = is_well_placed(target)
            memo_wp[r] = res
            return res

        h_state = 0
        for r in self.rings:
            if r in self.goal_on_map and not is_well_placed(r):
                # Every unstable ring must move at least once.
                # The size of the ring is a conservative lower bound on the 
                # number of actions involving it and its subordinates.
                h_state += self.ring_sizes.get(r, 1)
        
        # The task network represents remaining abstract work.
        h_tn = len(task_network)

        # Both represent unavoidable costs. We take the max for admissibility.
        return max(h_state, h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
