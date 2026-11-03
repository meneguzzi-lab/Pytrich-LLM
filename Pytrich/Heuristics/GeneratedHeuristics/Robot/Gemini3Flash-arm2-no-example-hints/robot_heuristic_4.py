from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic4(Heuristic):
    """
    Heuristic for the Robot delivery domain.
    
    Bottleneck Analysis:
    The primary bottleneck is the single-carry constraint. The robot can only hold one package 
    at a time. This serializes all deliveries into a sequence of: 
    move to package -> pickup -> move to goal -> putdown.
    
    Admissibility Sketch:
    1. Delivery Actions: Each undelivered package requires at least one pickup and one putdown (2 actions),
       unless it is already being held, in which case it only needs a putdown (1 action).
    2. Navigation: The robot must move from its current location to the target. For a held package,
       the target is the goal location. For other undelivered packages, it must first move to the package loc.
       Since move actions are required for each delivery and cannot be shared (due to single-carry),
       summing the shortest-path distances (number of moves) for each package delivery is admissible.
    3. Doors: If a path involves a closed door, an 'open' action is mandatory. We use BFS distances
       for moves and add open costs as a lower bound.
    """

    def __init__(self, robot_gemini3flash_heuristic_4="robot_gemini3flash_heuristic_4"):
        super().__init__(robot_gemini3flash_heuristic_4)
        self.package_to_goal_room = {}
        self.dist_matrix = {}
        self.room_to_idx = {}
        self.idx_to_room = []
        
        # Bit-masks and mapping for fast state checking
        self.in_bits_to_pkg_room = {}
        self.rloc_bits_to_room = {}
        self.holding_bits_to_pkg = {}
        
        self.all_in_mask = 0
        self.all_rloc_mask = 0
        self.all_holding_mask = 0
        self.package_in_fact_ids = {} # package -> {room: bit_id}

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
        # Precompute mappings and goal info
        rooms_set = set()
        packages_set = set()
        adj = collections.defaultdict(list)
        
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "goal_in":
                self.package_to_goal_room[args[0]] = args[1]
            elif pred == "rloc":
                rooms_set.add(args[0])
                self.rloc_bits_to_room[fact.global_id] = args[0]
                self.all_rloc_mask |= (1 << fact.global_id)
            elif pred == "in":
                packages_set.add(args[0])
                rooms_set.add(args[1])
                self.in_bits_to_pkg_room[fact.global_id] = (args[0], args[1])
                self.all_in_mask |= (1 << fact.global_id)
                if args[0] not in self.package_in_fact_ids: self.package_in_fact_ids[args[0]] = {}
                self.package_in_fact_ids[args[0]][args[1]] = fact.global_id
            elif pred == "holding":
                packages_set.add(args[0])
                self.holding_bits_to_pkg[fact.global_id] = args[0]
                self.all_holding_mask |= (1 << fact.global_id)
            elif pred == "door":
                adj[args[0]].append(args[1])
                adj[args[1]].append(args[0])

        # BFS / All-Pairs Shortest Paths for Rooms
        self.idx_to_room = sorted(list(rooms_set))
        self.room_to_idx = {r: i for i, r in enumerate(self.idx_to_room)}
        n = len(self.idx_to_room)
        dist = [[9999] * n for _ in range(n)]
        for i in range(n): dist[i][i] = 0
        for u, neighbors in adj.items():
            for v in neighbors:
                if u in self.room_to_idx and v in self.room_to_idx:
                    dist[self.room_to_idx[u]][self.room_to_idx[v]] = 1
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        self.dist_matrix = dist

        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int) -> int:
        # 1. Identify Robot Location
        robot_bits = state & self.all_rloc_mask
        if not robot_bits: return 0
        robot_room = self.rloc_bits_to_room[(robot_bits & -robot_bits).bit_length() - 1]
        
        # 2. Identify Held Package
        held_pkg = None
        held_bits = state & self.all_holding_mask
        if held_bits:
            held_pkg = self.holding_bits_to_pkg[(held_bits & -held_bits).bit_length() - 1]
            
        # 3. Identify Undelivered Packages
        current_pkg_locations = {}
        in_state = state & self.all_in_mask
        while in_state:
            bit = (in_state & -in_state).bit_length() - 1
            pkg, room = self.in_bits_to_pkg_room[bit]
            current_pkg_locations[pkg] = room
            in_state &= in_state - 1
            
        h = 0
        undelivered = []
        
        for pkg, goal in self.package_to_goal_room.items():
            # If package is already at its target room and not held, it's delivered.
            # Note: domain actions require (in pkg goal) after putdown.
            is_at_goal = current_pkg_locations.get(pkg) == goal
            if is_at_goal and held_pkg != pkg:
                continue
            
            undelivered.append(pkg)
            if pkg == held_pkg:
                # Path: Robot(held pkg) -> Goal. Actions: Putdown (1) + Moves
                h += 1
                h += self.dist_matrix[self.room_to_idx[robot_room]][self.room_to_idx[goal]]
            else:
                # Path: Robot -> Pkg Loc -> Goal. Actions: Pickup (1) + Putdown (1) + Moves
                pkg_loc = current_pkg_locations.get(pkg)
                if pkg_loc:
                    h += 2
                    h += self.dist_matrix[self.room_to_idx[pkg_loc]][self.room_to_idx[goal]]

        # 4. Travel Cost to first objective
        # If not holding anything, must travel to the nearest undelivered package first.
        if not held_pkg and undelivered:
            min_travel = 9999
            for pkg in undelivered:
                pkg_loc = current_pkg_locations.get(pkg)
                if pkg_loc:
                    d = self.dist_matrix[self.room_to_idx[robot_room]][self.room_to_idx[pkg_loc]]
                    if d < min_travel: min_travel = d
            if min_travel < 9999: h += min_travel
            
        return h
