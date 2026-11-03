from Pytrich.Heuristics.heuristic import Heuristic
from collections import defaultdict
import sys

class RobotGemini3ProHeuristic13(Heuristic):
    def __init__(self, model, robot_gemini3pro_heuristic_10=None):
        super().__init__(model)
        self.model = model
        self.INF = float('inf')
        
        # Mappings for fast state lookup
        # global_id -> value
        self.rloc_map = {} 
        self.holding_map = {} 
        self.in_map = defaultdict(list) # pkg -> [(global_id, room)]
        self.goal_locs = {} # pkg -> room
        
        # Topology
        self.room_indices = {} # room_name -> int_index
        self.distances = [] # Matrix [i][j] -> distance
        
        self._initialize_domain()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []

    def _initialize_domain(self):
        rooms = set()
        doors = defaultdict(list)
        
        # 1. Parse Facts to build maps and topology
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            fid = fact.global_id
            
            if pred == 'rloc':
                self.rloc_map[fid] = args[0]
                rooms.add(args[0])
            elif pred == 'holding':
                self.holding_map[fid] = args[0]
            elif pred == 'in':
                self.in_map[args[0]].append((fid, args[1]))
                rooms.add(args[1])
            elif pred == 'door':
                # Identify topology. Doors are static in this domain.
                # We check if they exist in the initial state to be safe, or just trust the facts.
                # Given pfile structure, door facts define the graph.
                u, v = args[0], args[1]
                doors[u].append(v)
                rooms.add(u)
                rooms.add(v)

        # 2. Parse Goals
        # model.goals is a bitwise integer. We iterate facts to find which are set.
        goal_mask = self.model.goals
        for fact in self.model.facts:
            if (goal_mask >> fact.global_id) & 1:
                pred, args = self._parse_fact_name(fact.name)
                if pred == 'in':
                    self.goal_locs[args[0]] = args[1]

        # 3. Compute All-Pairs Shortest Paths (Floyd-Warshall)
        sorted_rooms = sorted(list(rooms))
        self.room_indices = {r: i for i, r in enumerate(sorted_rooms)}
        n = len(rooms)
        
        # Initialize matrix
        self.distances = [[self.INF] * n for _ in range(n)]
        for i in range(n):
            self.distances[i][i] = 0
            
        # Fill edges
        for u_name, neighbors in doors.items():
            u = self.room_indices.get(u_name)
            if u is None: continue
            for v_name in neighbors:
                v = self.room_indices.get(v_name)
                if v is not None:
                    self.distances[u][v] = 1 # Distance is 1 (move action)
                    
        # Run FW
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.distances[i][j] > self.distances[i][k] + self.distances[k][j]:
                        self.distances[i][j] = self.distances[i][k] + self.distances[k][j]

    def compute(self, state, task_network=None):
        # 1. Determine Robot Location
        robot_loc = None
        for fid, loc in self.rloc_map.items():
            if (state >> fid) & 1:
                robot_loc = loc
                break
        
        if robot_loc is None:
            return self.INF # Should not happen in valid states
            
        # 2. Determine Held Package
        held_pkg = None
        for fid, pkg in self.holding_map.items():
            if (state >> fid) & 1:
                held_pkg = pkg
                break
        
        total_cost = 0
        current_u = self.room_indices.get(robot_loc)
        if current_u is None: return self.INF
        
        pending_pickup_locs = []
        
        # 3. Evaluate each goal package
        for pkg, goal_loc in self.goal_locs.items():
            goal_v = self.room_indices.get(goal_loc)
            if goal_v is None: continue

            if pkg == held_pkg:
                # Case: Carrying this package
                # Cost: Move to goal + Putdown (1)
                dist = self.distances[current_u][goal_v]
                if dist == self.INF: return self.INF
                total_cost += dist + 1
                
                # Heuristic optimization: After dropping this, the robot is at goal_loc.
                # We update current_u to assume the robot starts next transit from here.
                # This provides a tighter lower bound for the 'next' task.
                current_u = goal_v
            else:
                # Case: Package is somewhere on the map (or already at goal)
                # Find where it is
                pkg_loc = None
                for fid, loc in self.in_map[pkg]:
                    if (state >> fid) & 1:
                        pkg_loc = loc
                        break
                
                if pkg_loc == goal_loc:
                    continue # Already satisfied
                
                if pkg_loc:
                    start_v = self.room_indices.get(pkg_loc)
                    if start_v is None: continue
                    
                    # Cost: Move pkg to goal + Actions (Pickup=1, Putdown=1)
                    # Note: We do NOT include Robot->Pkg transit yet.
                    dist = self.distances[start_v][goal_v]
                    if dist == self.INF: return self.INF
                    total_cost += dist + 2
                    
                    pending_pickup_locs.append(start_v)
        
        # 4. Add transit cost for pending pickups
        # The robot must travel from its 'current' location (after dropping held item) 
        # to the nearest pending package.
        if pending_pickup_locs:
            min_transit = self.INF
            for start_v in pending_pickup_locs:
                dist = self.distances[current_u][start_v]
                if dist < min_transit:
                    min_transit = dist
            
            if min_transit < self.INF:
                total_cost += min_transit

        return total_cost
