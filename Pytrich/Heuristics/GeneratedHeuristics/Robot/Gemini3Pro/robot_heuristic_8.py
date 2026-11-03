import sys
import math
from Pytrich.Heuristics.heuristic import Heuristic

class RobotGemini3ProHeuristic8(Heuristic):
    def __init__(self, robot_gemini3pro_heuristic_5):
        super().__init__(robot_gemini3pro_heuristic_5)
        self.model = robot_gemini3pro_heuristic_5.model
        
        # Mappings for domain objects
        self.room_to_id = {}
        self.id_to_room = {}
        self.pkg_to_id = {}
        self.id_to_pkg = {}
        
        # Cached facts for fast state lookup
        self.rloc_facts = [] # List of (room_idx, global_id)
        self.holding_facts = {} # pkg_idx -> global_id
        self.in_facts = {} # pkg_idx -> list of (room_idx, global_id)
        self.goal_in_facts = {} # pkg_idx -> room_idx (static goal destinations)
        
        # Distance matrix [room_i][room_j]
        self.dist_matrix = []
        
        self._initialize_domain()

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

    def _initialize_domain(self):
        rooms = set()
        packages = set()
        adj = {} # Adjacency list for connectivity graph
        pkg_goals = {} # Map package name to goal room name
        
        # 1. Scan facts to identify objects and build static graph
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
                
            if pred == 'rloc':
                rooms.add(args[0])
            elif pred == 'in':
                packages.add(args[0])
                rooms.add(args[1])
            elif pred == 'holding':
                packages.add(args[0])
            elif pred == 'goal_in':
                packages.add(args[0])
                rooms.add(args[1])
                pkg_goals[args[0]] = args[1]
            elif pred == 'door':
                # door loc1 loc2 d
                r1, r2 = args[0], args[1]
                rooms.add(r1)
                rooms.add(r2)
                if r1 not in adj: adj[r1] = []
                if r2 not in adj: adj[r2] = []
                adj[r1].append(r2)
                adj[r2].append(r1)

        # 2. Create indices for objects
        sorted_rooms = sorted(list(rooms))
        for idx, r in enumerate(sorted_rooms):
            self.room_to_id[r] = idx
            self.id_to_room[idx] = r
            
        sorted_pkgs = sorted(list(packages))
        for idx, p in enumerate(sorted_pkgs):
            self.pkg_to_id[p] = idx
            self.id_to_pkg[idx] = p
            
        n_rooms = len(rooms)
        
        # 3. Precompute All-Pairs Shortest Paths (Floyd-Warshall)
        # We assume doors are open (lower bound distance). 
        # Traversing a closed door would cost at least the same + open action.
        INF = 9999
        self.dist_matrix = [[INF] * n_rooms for _ in range(n_rooms)]
        
        for i in range(n_rooms):
            self.dist_matrix[i][i] = 0
            
        for r_name, neighbors in adj.items():
            u = self.room_to_id[r_name]
            for neighbor in neighbors:
                v = self.room_to_id[neighbor]
                self.dist_matrix[u][v] = 1 # Movement cost is 1
                
        for k in range(n_rooms):
            for i in range(n_rooms):
                for j in range(n_rooms):
                    if self.dist_matrix[i][k] + self.dist_matrix[k][j] < self.dist_matrix[i][j]:
                        self.dist_matrix[i][j] = self.dist_matrix[i][k] + self.dist_matrix[k][j]

        # 4. Map Facts to Indices for State Lookup
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
                
            if pred == 'rloc':
                rid = self.room_to_id.get(args[0])
                if rid is not None:
                    self.rloc_facts.append((rid, fact.global_id))
                    
            elif pred == 'in':
                pid = self.pkg_to_id.get(args[0])
                rid = self.room_to_id.get(args[1])
                if pid is not None and rid is not None:
                    if pid not in self.in_facts: self.in_facts[pid] = []
                    self.in_facts[pid].append((rid, fact.global_id))
            
            elif pred == 'holding':
                pid = self.pkg_to_id.get(args[0])
                if pid is not None:
                    self.holding_facts[pid] = fact.global_id
        
        # Map static goals
        for p_name, r_name in pkg_goals.items():
            pid = self.pkg_to_id.get(p_name)
            rid = self.room_to_id.get(r_name)
            if pid is not None and rid is not None:
                self.goal_in_facts[pid] = rid

    def h(self, node):
        state = node.state
        
        # 1. Determine current Robot Location
        robot_loc = -1
        for rid, gid in self.rloc_facts:
            if (state >> gid) & 1:
                robot_loc = rid
                break
        
        if robot_loc == -1:
            # If robot location is unknown in this state, heuristic cannot effectively estimate movement.
            return 0
            
        total_cost = 0
        
        # 2. Identify Pending Tasks
        pending_pickup_locs = []
        is_holding_something = False
        holding_goal_loc = -1
        
        for pid, goal_rid in self.goal_in_facts.items():
            # Check if package is already at goal
            is_satisfied = False
            if pid in self.in_facts:
                for rid, gid in self.in_facts[pid]:
                    if (state >> gid) & 1:
                        if rid == goal_rid:
                            is_satisfied = True
                        break
            
            if is_satisfied:
                continue
                
            # Check if robot is holding this package
            holding = False
            if pid in self.holding_facts:
                gid = self.holding_facts[pid]
                if (state >> gid) & 1:
                    holding = True
            
            if holding:
                # Robot is holding: Cost = Dist to Goal + Putdown(1)
                dist = self.dist_matrix[robot_loc][goal_rid]
                if dist >= 9999: dist = 0 # Fallback for disconnected components
                total_cost += dist + 1
                is_holding_something = True
                holding_goal_loc = goal_rid # Robot effectively ends up here
            else:
                # Robot not holding, package needs transport
                # Find current package location
                pkg_loc = -1
                if pid in self.in_facts:
                    for rid, gid in self.in_facts[pid]:
                        if (state >> gid) & 1:
                            pkg_loc = rid
                            break
                
                if pkg_loc != -1:
                    # Transport Cost: Dist(Pkg -> Goal) + Pickup(1) + Putdown(1)
                    dist_transport = self.dist_matrix[pkg_loc][goal_rid]
                    if dist_transport >= 9999: dist_transport = 0
                    total_cost += dist_transport + 2
                    pending_pickup_locs.append(pkg_loc)
        
        # 3. Add Approach Cost
        # The robot needs to travel to the start of the tasks.
        # This accounts for "fetch" distance for the first task.
        if pending_pickup_locs:
            min_approach = 9999
            
            if is_holding_something:
                # If holding, robot delivers first, then moves from delivery loc to next pickup
                for loc in pending_pickup_locs:
                    d = self.dist_matrix[holding_goal_loc][loc]
                    if d < min_approach:
                        min_approach = d
            else:
                # If not holding, robot moves from current loc to nearest pickup
                for loc in pending_pickup_locs:
                    d = self.dist_matrix[robot_loc][loc]
                    if d < min_approach:
                        min_approach = d
                        
            if min_approach < 9999:
                total_cost += min_approach
                    
        return total_cost
