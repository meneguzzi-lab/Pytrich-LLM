import sys
import heapq
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic

class RobotGemini3ProHeuristic3(Heuristic):
    def __init__(self, robot_gemini3pro_heuristic_0, model):
        super().__init__(robot_gemini3pro_heuristic_0, model)
        self.model = model
        
        # Domain mappings
        self.rooms = {} # room_name -> int_id
        self.packages = {} # pkg_name -> int_id
        self.adj = [] # adjacency list: room_id -> [(neighbor_id, door_name)]
        self.package_goals = {} # pkg_id -> room_id
        
        # Bitmask indices for fast state parsing
        self.rloc_bits = []      # list of (room_id, global_id)
        self.in_bits = []        # list of (pkg_id, room_id, global_id)
        self.holding_bits = []   # list of (pkg_id, global_id)
        self.closed_bits = {}    # door_name -> global_id
        
        self._parse_domain()
        self._precompute_static_distances()

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

    def _parse_domain(self):
        room_set = set()
        pkg_set = set()
        
        # 1. Identify objects from static facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == 'rloc':
                room_set.add(args[0])
            elif pred == 'in':
                pkg_set.add(args[0])
                room_set.add(args[1])
            elif pred == 'goal_in':
                pkg_set.add(args[0])
                room_set.add(args[1])
            elif pred == 'door':
                room_set.add(args[0])
                room_set.add(args[1])
        
        self.rooms = {name: i for i, name in enumerate(sorted(list(room_set)))}
        self.packages = {name: i for i, name in enumerate(sorted(list(pkg_set)))}
        num_rooms = len(self.rooms)
        self.adj = [[] for _ in range(num_rooms)]
        
        # 2. Map facts to indices and build graph
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            gid = fact.global_id
            
            if pred == 'rloc':
                if args[0] in self.rooms:
                    self.rloc_bits.append((self.rooms[args[0]], gid))
            
            elif pred == 'in':
                if args[0] in self.packages and args[1] in self.rooms:
                    self.in_bits.append((self.packages[args[0]], self.rooms[args[1]], gid))
                    
            elif pred == 'holding':
                if args[0] in self.packages:
                    self.holding_bits.append((self.packages[args[0]], gid))
            
            elif pred == 'closed':
                self.closed_bits[args[0]] = gid
            
            elif pred == 'goal_in':
                if args[0] in self.packages and args[1] in self.rooms:
                    self.package_goals[self.packages[args[0]]] = self.rooms[args[1]]
            
            elif pred == 'door':
                if args[0] in self.rooms and args[1] in self.rooms:
                    u, v = self.rooms[args[0]], self.rooms[args[1]]
                    d = args[2]
                    self.adj[u].append((v, d))

    def _precompute_static_distances(self):
        # All-Pairs Shortest Path assuming all doors are open (weight 1)
        # This provides an admissible estimate for package transport cost
        n = len(self.rooms)
        dist = [[999999] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            
        for u in range(n):
            for v, _ in self.adj[u]:
                dist[u][v] = 1
        
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        self.static_dist = dist

    def h(self, node):
        state = node.state
        
        # 1. Determine Robot Location
        robot_idx = -1
        for r_idx, gid in self.rloc_bits:
            if (state >> gid) & 1:
                robot_idx = r_idx
                break
        
        if robot_idx == -1:
            return 9999
            
        # 2. Identify Closed Doors
        closed_doors = set()
        for dname, gid in self.closed_bits.items():
            if (state >> gid) & 1:
                closed_doors.add(dname)
        
        # 3. Dynamic BFS/Dijkstra from Robot
        # Calculates cost to reach any room considering current door states
        # Cost: 1 if open, 2 if closed (open + move)
        n_rooms = len(self.rooms)
        rob_dist = [999999] * n_rooms
        rob_dist[robot_idx] = 0
        pq = [(0, robot_idx)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > rob_dist[u]: continue
            
            for v, dname in self.adj[u]:
                weight = 2 if dname in closed_doors else 1
                new_dist = d + weight
                if new_dist < rob_dist[v]:
                    rob_dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
        
        # 4. Calculate Heuristic Cost
        h_val = 0
        
        # Track packages
        held_goals = []
        pending = []
        
        held_ids = set()
        for pid, gid in self.holding_bits:
            if (state >> gid) & 1:
                held_ids.add(pid)
                if pid in self.package_goals:
                    held_goals.append(self.package_goals[pid])
        
        for pid, rid, gid in self.in_bits:
            if pid in held_ids: continue
            if (state >> gid) & 1:
                if pid in self.package_goals:
                    g = self.package_goals[pid]
                    if rid != g:
                        pending.append((rid, g))
        
        # 5. Sum Costs
        min_start_dist = 999999
        
        # Cost for held packages
        # Robot needs to travel to goal and putdown
        for g in held_goals:
            d = rob_dist[g]
            if d >= 999999: d = 1000
            h_val += d + 1 # +1 for putdown
            
        # Cost for pending packages
        for u, g in pending:
            # Transport: pickup(1) + move(u->g) + putdown(1)
            # Use static_dist for move (admissible)
            travel = self.static_dist[u][g]
            if travel >= 999999: travel = 1000
            
            h_val += 1 + travel + 1
            
            # Distance from robot to start location 'u'
            d_start = rob_dist[u]
            if d_start < min_start_dist:
                min_start_dist = d_start
        
        # Connection cost: Robot must get to the first package
        # If holding, robot handles that first (included in held_goals cost)
        # If not holding and tasks pending, add distance to nearest start
        if not held_goals and pending:
            if min_start_dist < 999999:
                h_val += min_start_dist
        
        return h_val
