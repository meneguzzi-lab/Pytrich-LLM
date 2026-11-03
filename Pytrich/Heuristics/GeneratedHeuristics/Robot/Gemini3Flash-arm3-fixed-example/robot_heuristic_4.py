import time
from collections import deque, defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic4(Heuristic):
    """
    Domain-dependent heuristic for the Robot planning domain.

    1. Bottleneck Analysis:
       The primary bottleneck is the single robot arm. The robot can only carry one package at a time.
       This means all package transportation tasks (pickup -> move -> putdown) must be performed
       sequentially, making costs additive across all misplaced packages.

    2. Admissible Lower Bounds:
       - Interaction Cost: Each misplaced package requires exactly 1 pickup and 1 putdown (2 actions).
         If a package is already being held, it only requires 1 putdown.
       - Travel Cost: The robot must travel the distance between the package's current location
         and its goal location for every misplaced package. These move actions are unavoidable.
       - Reach Cost: If the robot is not holding a package, it must travel to the nearest 
         misplaced package's starting location. If it is holding a package, it must travel to its goal.

    The heuristic returns the sum of these costs, which is admissible because the single-arm 
    constraint prevents any parallel progress on multiple packages.
    """

    def __init__(self, robot_gemini3flash_heuristic_4=None, name="RobotGemini3FlashHeuristic4"):
        super().__init__(name=name)
        self.package_targets = {}  # pkg -> target_room
        self.room_distances = {}   # r1 -> r2 -> min_moves
        self.pkg_in_masks = {}     # pkg -> bitmask of all 'in' facts for this pkg
        self.pkg_in_fact_map = {}  # fact_id -> (pkg, room)
        self.holding_fact_map = {} # pkg -> fact_id
        self.rloc_fact_map = {}    # room -> fact_id
        self.all_pkgs = []
        self.all_rooms = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Identify Goals (Target locations for packages)
        # Check model.goals (explicit state goals)
        goal_facts = self.model.state_explicit_repr(self.model.goals)
        for fact_name in goal_facts:
            pred, args = self._parse_fact_name(fact_name)
            if pred == "in" and len(args) == 2:
                self.package_targets[args[0]] = args[1]
        
        # Fallback: Check goal_in predicates in initial state if no goal in found
        if not self.package_targets:
            init_facts = self.model.state_explicit_repr(initial_node.state)
            for fact_name in init_facts:
                pred, args = self._parse_fact_name(fact_name)
                if pred == "goal_in" and len(args) == 2:
                    self.package_targets[args[0]] = args[1]

        # 2. Build adjacency and compute all-pairs shortest paths
        adj = defaultdict(set)
        rooms = set()
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "door" and len(args) == 3:
                r1, r2, d = args
                adj[r1].add(r2)
                adj[r2].add(r1)
                rooms.add(r1)
                rooms.add(r2)
            elif pred == "rloc":
                rooms.add(args[0])
        
        self.all_rooms = list(rooms)
        for start_node in self.all_rooms:
            self.room_distances[start_node] = {r: 1000 for r in self.all_rooms}
            self.room_distances[start_node][start_node] = 0
            queue = deque([(start_node, 0)])
            while queue:
                curr, d = queue.popleft()
                for neighbor in adj[curr]:
                    if self.room_distances[start_node][neighbor] == 1000:
                        self.room_distances[start_node][neighbor] = d + 1
                        queue.append((neighbor, d + 1))

        # 3. Map facts for fast bitwise lookup
        pkgs = set(self.package_targets.keys())
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if pred == "in" and len(args) == 2:
                p, r = args[0], args[1]
                pkgs.add(p)
                if p not in self.pkg_in_masks: self.pkg_in_masks[p] = 0
                self.pkg_in_masks[p] |= (1 << fact.global_id)
                self.pkg_in_fact_map[fact.global_id] = (p, r)
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_map[args[0]] = fact.global_id
            elif pred == "rloc" and len(args) == 1:
                self.rloc_fact_map[args[0]] = fact.global_id
        
        self.all_pkgs = list(pkgs)
        h_val = self._compute_h(initial_node.state)
        return super().initialize(model, h_val)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        return None, []

    def _compute_h(self, state: int) -> int:
        # Find robot location
        robot_room = None
        for r in self.all_rooms:
            if (state >> self.rloc_fact_map[r]) & 1:
                robot_room = r
                break
        
        h = 0
        displaced_starts = []
        held_pkg = None

        for p in self.all_pkgs:
            target = self.package_targets.get(p)
            if not target: continue
            
            # Check if held
            is_held = (state >> self.holding_fact_map[p]) & 1 if p in self.holding_fact_map else 0
            if is_held:
                held_pkg = p
                h += 1 # putdown cost
                h += self.room_distances.get(robot_room, {}).get(target, 0)
                continue
            
            # Check where it is
            mask = self.pkg_in_masks.get(p, 0)
            current_bits = state & mask
            if current_bits:
                # Get the first bit set (the room fact)
                fact_id = (current_bits & -current_bits).bit_length() - 1
                _, current_room = self.pkg_in_fact_map[fact_id]
                
                if current_room != target:
                    h += 2 # pickup + putdown cost
                    h += self.room_distances.get(current_room, {}).get(target, 0)
                    displaced_starts.append(current_room)
        
        # Add reach cost to the first action
        if held_pkg:
            # Already counted distance to target above
            pass
        elif displaced_starts:
            reach_costs = [self.room_distances.get(robot_room, {}).get(start, 0) for start in displaced_starts]
            h += min(reach_costs)
            
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute_h(node.state)
        super().update_info(h_val)
        return h_val
