from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini3FlashHeuristic0(Heuristic):
    """
    Robot Domain Heuristic Analysis:
    
    1. Bottleneck: Single-carry constraint. The robot can only carry one package at a time (armempty requirement). 
       This forces a serial delivery pattern: robot must move to a package, pickup, move to goal, putdown, and then 
       move to the next package.

    2. Admissible Lower Bounds:
       - Action Count Term: Each undelivered package requires exactly one pickup and one putdown action. 
         A package is undelivered if it is not at its goal_in room. If held, it needs 1 putdown; if not held, 2 actions.
       - Travel Term: For every package p, the robot must traverse the shortest path between start_p and goal_p.
         Additionally, the robot must move from its current location to the first package's location (or its goal if holding).
       - Admissibility: These terms are additive because the robot is a single agent and cannot overlap 
         pickups, putdowns, or the moves between start and goal locations for different packages.

    3. Preprocessing: BFS computes all-pairs shortest paths for rooms. Fact indices are mapped to enable O(1) bitwise lookups.
    4. Complexity: __call__ is O(P) where P is the number of packages, satisfying the constraint for large instances.
    """

    def __init__(self, robot_gemini3flash_heuristic_0="RobotGemini3FlashHeuristic0"):
        super().__init__(name=robot_gemini3flash_heuristic_0)
        self.rooms = {}
        self.pkgs = {}
        self.num_rooms = 0
        self.num_pkgs = 0
        
        self.dist = []
        self.pkg_goal_room = []
        self.pkg_goal_mask = []
        self.in_masks = []
        self.holding_masks = []
        self.rloc_masks = []
        
        self.all_rloc_mask = 0
        self.all_holding_mask = 0
        self.all_in_mask = []
        
        self.global_to_rid = {}
        self.global_to_pid = {}

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
        room_set = set()
        pkg_set = set()
        adj = {}

        # First pass: Identify objects
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'rloc': room_set.add(args[0])
            elif pred == 'in': 
                pkg_set.add(args[0])
                room_set.add(args[1])
            elif pred == 'holding': pkg_set.add(args[0])
            elif pred == 'goal_in':
                pkg_set.add(args[0])
                room_set.add(args[1])
            elif pred == 'door':
                room_set.add(args[0])
                room_set.add(args[1])

        self.rooms = {name: i for i, name in enumerate(sorted(list(room_set)))}
        self.pkgs = {name: i for i, name in enumerate(sorted(list(pkg_set)))}
        self.num_rooms = len(self.rooms)
        self.num_pkgs = len(self.pkgs)

        # BFS for All-Pairs Shortest Paths
        adj = {i: [] for i in range(self.num_rooms)}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'door':
                u, v = self.rooms[args[0]], self.rooms[args[1]]
                adj[u].append(v)

        self.dist = [[1000000] * self.num_rooms for _ in range(self.num_rooms)]
        for i in range(self.num_rooms):
            self.dist[i][i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if self.dist[i][v] == 1000000:
                        self.dist[i][v] = self.dist[i][u] + 1
                        queue.append(v)

        # Parse Goal Locations
        self.pkg_goal_room = [-1] * self.num_pkgs
        goal_facts = model.state_explicit_repr(model.goals)
        for g_name in goal_facts:
            pred, args = self._parse_fact_name(g_name)
            if pred == 'in':
                self.pkg_goal_room[self.pkgs[args[0]]] = self.rooms[args[1]]

        # Map Facts to bitmasks and IDs
        self.rloc_masks = [0] * self.num_rooms
        self.pkg_goal_mask = [0] * self.num_pkgs
        self.holding_masks = [0] * self.num_pkgs
        self.in_masks = [[0] * self.num_rooms for _ in range(self.num_pkgs)]
        self.all_in_mask = [0] * self.num_pkgs

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'rloc':
                rid = self.rooms[args[0]]
                self.rloc_masks[rid] = 1 << f.global_id
                self.all_rloc_mask |= (1 << f.global_id)
                self.global_to_rid[f.global_id] = rid
            elif pred == 'in':
                pid, rid = self.pkgs[args[0]], self.rooms[args[1]]
                self.in_masks[pid][rid] = 1 << f.global_id
                self.all_in_mask[pid] |= (1 << f.global_id)
                self.global_to_rid[f.global_id] = rid
                if self.pkg_goal_room[pid] == rid:
                    self.pkg_goal_mask[pid] = 1 << f.global_id
            elif pred == 'holding':
                pid = self.pkgs[args[0]]
                self.holding_masks[pid] = 1 << f.global_id
                self.all_holding_mask |= (1 << f.global_id)
                self.global_to_pid[f.global_id] = pid

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        if not task_network:
            return 0

        # 1. Identify Robot Location
        robot_room = -1
        r_bits = state & self.all_rloc_mask
        if r_bits:
            robot_room = self.global_to_rid[r_bits.bit_length() - 1]
        
        if robot_room == -1: return 0

        # 2. Identify Held Package
        held_pkg = -1
        h_bits = state & self.all_holding_mask
        if h_bits:
            held_pkg = self.global_to_pid[h_bits.bit_length() - 1]

        h = 0
        undelivered_pkgs = []
        
        # 3. Calculate delivery costs
        if held_pkg != -1:
            goal_r = self.pkg_goal_room[held_pkg]
            h += 1 + self.dist[robot_room][goal_r]

        for p_id in range(self.num_pkgs):
            if p_id == held_pkg: continue
            
            # If not at goal (bit is 0 in goal-in fact)
            if not (state & self.pkg_goal_mask[p_id]):
                undelivered_pkgs.append(p_id)
                # Find where it is
                p_bits = state & self.all_in_mask[p_id]
                if p_bits:
                    p_start = self.global_to_rid[p_bits.bit_length() - 1]
                    goal_r = self.pkg_goal_room[p_id]
                    h += 2 + self.dist[p_start][goal_r]

        # 4. Connection Cost (Robot to first package)
        if held_pkg == -1 and undelivered_pkgs:
            min_dist = 1000000
            for p_id in undelivered_pkgs:
                p_bits = state & self.all_in_mask[p_id]
                if p_bits:
                    p_start = self.global_to_rid[p_bits.bit_length() - 1]
                    d = self.dist[robot_room][p_start]
                    if d < min_dist: min_dist = d
            if min_dist < 1000000: h += min_dist

        return int(h)
