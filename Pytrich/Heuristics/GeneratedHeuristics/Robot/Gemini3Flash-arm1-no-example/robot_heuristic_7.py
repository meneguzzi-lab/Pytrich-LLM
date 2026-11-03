from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

# BOTTLENECK ANALYSIS:
# The Robot domain is constrained by a single robot arm (single-carry capacity).
# Each package 'p' at a wrong location requires a sequence: move to source, pickup, move to goal, and putdown.
# Because the robot can only hold one package at a time, these sequences must be performed serially.
# This makes the total cost the sum of individual delivery costs plus the travel distance between deliveries.

# ADMISSIBLE LOWER BOUNDS:
# 1. Action Count (h_actions): Each package not at its goal requires at least one pickup and one putdown (+2 actions).
# 2. Minimum Travel (h_moves): Each package must be moved from its current location to its goal_in location.
#    The sum of shortest-path distances between these locations is a lower bound on total 'move' actions.
# 3. Robot Travel: The robot must move from its current location to the first package's location.

class RobotGemini3FlashHeuristic7(Heuristic):
    def __init__(self, robot_gemini3flash_heuristic_7="robot_gemini3flash_heuristic_7"):
        super().__init__(name=robot_gemini3flash_heuristic_7)
        self.goal_room = {}  # pkg_idx -> room_idx
        self.dist_matrix = []
        self.pkg_ids = []
        self.room_ids = []
        self.in_fact_ids = {}      # pkg_idx -> {room_idx: global_id}
        self.holding_fact_ids = {} # pkg_idx -> global_id
        self.rloc_fact_ids = {}    # room_idx -> global_id
        self.armempty_id = -1

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
        room_names = []
        pkg_names = []
        adj = collections.defaultdict(list)
        
        # 1. Parse objects and static predicates
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "door":
                adj[args[0]].append(args[1])
            elif pred == "goal_in":
                # goal_in[package, room]
                if args[0] not in pkg_names: pkg_names.append(args[0])
                if args[1] not in room_names: room_names.append(args[1])
            elif pred == "rloc":
                if args[0] not in room_names: room_names.append(args[0])
            elif pred == "in":
                if args[0] not in pkg_names: pkg_names.append(args[0])
                if args[1] not in room_names: room_names.append(args[1])

        self.room_ids = list(range(len(room_names)))
        self.pkg_ids = list(range(len(pkg_names)))
        room_to_idx = {name: i for i, name in enumerate(room_names)}
        pkg_to_idx = {name: i for i, name in enumerate(pkg_names)}

        # 2. Build Goal Map
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "goal_in":
                self.goal_room[pkg_to_idx[args[0]]] = room_to_idx[args[1]]

        # 3. Precompute Distance Matrix (BFS for all pairs)
        num_rooms = len(room_names)
        self.dist_matrix = [[999 for _ in range(num_rooms)] for _ in range(num_rooms)]
        for start_node, neighbors in adj.items():
            u = room_to_idx[start_node]
            self.dist_matrix[u][u] = 0
            for neighbor in neighbors:
                v = room_to_idx[neighbor]
                self.dist_matrix[u][v] = 1
        
        for k in range(num_rooms):
            for i in range(num_rooms):
                for j in range(num_rooms):
                    if self.dist_matrix[i][j] > self.dist_matrix[i][k] + self.dist_matrix[k][j]:
                        self.dist_matrix[i][j] = self.dist_matrix[i][k] + self.dist_matrix[k][j]

        # 4. Map dynamic fact IDs
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in":
                p_idx, r_idx = pkg_to_idx[args[0]], room_to_idx[args[1]]
                if p_idx not in self.in_fact_ids: self.in_fact_ids[p_idx] = {}
                self.in_fact_ids[p_idx][r_idx] = f.global_id
            elif pred == "holding":
                self.holding_fact_ids[pkg_to_idx[args[0]]] = f.global_id
            elif pred == "rloc":
                self.rloc_fact_ids[room_to_idx[args[0]]] = f.global_id
            elif pred == "armempty":
                self.armempty_id = f.global_id

        return super().initialize(model, self._compute(initial_node.state, initial_node.task_network))

    def _compute(self, state, task_network):
        # 1. Identify robot location
        robot_room = -1
        for r_idx in self.room_ids:
            if (state >> self.rloc_fact_ids[r_idx]) & 1:
                robot_room = r_idx
                break
        
        if robot_room == -1: return 1000

        # 2. Identify held package and packages at wrong locations
        held_pkg = -1
        wrong_pkgs = [] # list of (pkg_idx, current_room_idx)
        
        for p_idx in self.pkg_ids:
            if (state >> self.holding_fact_ids[p_idx]) & 1:
                held_pkg = p_idx
            else:
                # Check where it is
                for r_idx in self.room_ids:
                    if (state >> self.in_fact_ids[p_idx][r_idx]) & 1:
                        if r_idx != self.goal_room.get(p_idx, -1):
                            wrong_pkgs.append((p_idx, r_idx))
                        break

        # 3. Summation of costs
        h = 0
        current_loc = robot_room

        # If holding something, we must deliver it first
        if held_pkg != -1:
            g_idx = self.goal_room.get(held_pkg, robot_room)
            h += self.dist_matrix[current_loc][g_idx] + 1 # move + putdown
            current_loc = g_idx

        # Greedy approach for remaining packages
        # To keep it O(N^2) in call, we process the list
        pkgs_to_process = wrong_pkgs[:]
        while pkgs_to_process:
            best_idx = -1
            min_d = 1000
            for i, (p_idx, r_idx) in enumerate(pkgs_to_process):
                d = self.dist_matrix[current_loc][r_idx]
                if d < min_d:
                    min_d = d
                    best_idx = i
            
            p_idx, r_idx = pkgs_to_process.pop(best_idx)
            g_idx = self.goal_room.get(p_idx, r_idx)
            # cost = move to pkg + pickup + move to goal + putdown
            h += min_d + 1 + self.dist_matrix[r_idx][g_idx] + 1
            current_loc = g_idx

        # Tie-breaker: prefer smaller task networks to encourage decomposition progress
        h += 0.001 * len(task_network)
        return int(h) if h > 0 else 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
