from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini3FlashHeuristic6(Heuristic):
    """
    Robot Domain Heuristic

    Bottleneck Analysis:
    The primary bottleneck is the single-arm robot. The robot can only carry one package at a time.
    This means every package not at its goal must be picked up, transported, and put down.
    The total cost is dominated by the sequential transport of these packages and any necessary door operations.

    Admissibility Sketches:
    1. Transport Cost (h_transport): Each unplaced package p requires at least 1 pickup, 1 putdown, and 
       dist(loc_p, goal_p) moves. Since the robot can only carry one package, these costs are additive
       across all packages. (Sum of (dist + 2) is admissible).
    2. Robot-to-Package Cost (h_robot): The robot must move to the location of the first package or 
       the goal of the currently held package. This distance is an additional unavoidable cost.
    3. Door Cost (h_doors): While opening doors can be shared, we use a simple lower bound where each 
       necessary move transition is at least 1 action. (Already captured in dist_matrix BFS).
    """

    def __init__(self, robot_gemini3flash_heuristic_6="robot_gemini3flash_heuristic_6"):
        super().__init__(robot_gemini3flash_heuristic_6)
        self.goal_room_idx = {}  # pkg_idx -> room_idx
        self.in_bits = {}        # pkg_idx -> {room_idx: bit_id}
        self.rloc_bits = {}      # room_idx -> bit_id
        self.holding_bits = {}   # pkg_idx -> bit_id
        self.dists = []          # room_idx -> [room_idx -> distance]
        self.rooms = []          # list of room indices
        self.pkgs = []           # list of package indices
        self.in_mask = 0
        self.bit_to_pkg_room = {} # bit_id -> (pkg_idx, room_idx)

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
        # 1. Map entities to internal indices
        room_names = sorted(list(set([args[0] for f in model.facts for pred, args in [self._parse_fact_name(f.name)] if pred == "rloc"] or [])))
        pkg_names = sorted(list(set([args[0] for f in model.facts for pred, args in [self._parse_fact_name(f.name)] if pred == "in"] or [])))
        
        room_to_idx = {name: i for i, name in enumerate(room_names)}
        pkg_to_idx = {name: i for i, name in enumerate(pkg_names)}
        self.rooms = list(range(len(room_names)))
        self.pkgs = list(range(len(pkg_names)))

        # 2. Build adjacency for BFS
        adj = {i: [] for i in self.rooms}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "door":
                r1, r2 = args[0], args[1]
                if r1 in room_to_idx and r2 in room_to_idx:
                    adj[room_to_idx[r1]].append(room_to_idx[r2])

        # 3. All-pairs shortest paths (unweighted move distance)
        num_rooms = len(self.rooms)
        self.dists = [[float('inf')] * num_rooms for _ in range(num_rooms)]
        for start_node in self.rooms:
            self.dists[start_node][start_node] = 0
            queue = deque([start_node])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if self.dists[start_node][v] == float('inf'):
                        self.dists[start_node][v] = self.dists[start_node][u] + 1
                        queue.append(v)

        # 4. Map goals and bit IDs
        self.in_bits = {p: {} for p in self.pkgs}
        self.rloc_bits = {}
        self.holding_bits = {}
        self.in_mask = 0
        
        # Check initial state for goal_in locations
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "goal_in":
                p_idx = pkg_to_idx.get(args[0])
                r_idx = room_to_idx.get(args[1])
                if p_idx is not None and r_idx is not None:
                    self.goal_room_idx[p_idx] = r_idx
            elif pred == "in":
                p_idx = pkg_to_idx.get(args[0])
                r_idx = room_to_idx.get(args[1])
                if p_idx is not None and r_idx is not None:
                    self.in_bits[p_idx][r_idx] = f.global_id
                    self.in_mask |= (1 << f.global_id)
                    self.bit_to_pkg_room[f.global_id] = (p_idx, r_idx)
            elif pred == "rloc":
                r_idx = room_to_idx.get(args[0])
                if r_idx is not None:
                    self.rloc_bits[r_idx] = f.global_id
            elif pred == "holding":
                p_idx = pkg_to_idx.get(args[0])
                if p_idx is not None:
                    self.holding_bits[p_idx] = f.global_id

        h_val = self._compute(initial_node.state)
        return super().initialize(model, h_val)

    def _compute(self, state):
        robot_loc = -1
        for r_idx, bit_id in self.rloc_bits.items():
            if (state >> bit_id) & 1:
                robot_loc = r_idx
                break
        
        if robot_loc == -1: return 0

        held_pkg_idx = -1
        for p_idx, bit_id in self.holding_bits.items():
            if (state >> bit_id) & 1:
                held_pkg_idx = p_idx
                break

        h = 0
        unplaced_pkgs = []
        
        # Efficiently find where all packages are currently located
        temp_bits = state & self.in_mask
        while temp_bits:
            lsb = temp_bits & -temp_bits
            bit_id = lsb.bit_length() - 1
            p_idx, r_idx = self.bit_to_pkg_room[bit_id]
            
            goal_r = self.goal_room_idx.get(p_idx)
            if goal_r is not None and r_idx != goal_r:
                unplaced_pkgs.append((p_idx, r_idx))
            temp_bits &= ~lsb

        # 1. Base transport cost for packages on the floor
        min_dist_to_pkg = float('inf')
        for p_idx, r_idx in unplaced_pkgs:
            g_idx = self.goal_room_idx[p_idx]
            h += 2 + self.dists[r_idx][g_idx]
            if self.dists[robot_loc][r_idx] < min_dist_to_pkg:
                min_dist_to_pkg = self.dists[robot_loc][r_idx]

        # 2. Cost for the package the robot is currently holding
        if held_pkg_idx != -1:
            g_idx = self.goal_room_idx.get(held_pkg_idx)
            if g_idx is not None:
                h += 1 + self.dists[robot_loc][g_idx]
        elif unplaced_pkgs:
            # Robot must move to the first package
            h += min_dist_to_pkg

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state)
        super().update_info(h_val)
        return h_val
