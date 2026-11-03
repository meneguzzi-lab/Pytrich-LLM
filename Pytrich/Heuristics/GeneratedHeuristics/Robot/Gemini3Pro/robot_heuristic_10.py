import sys
from pytrich.heuristics.Heuristic import Heuristic

class RobotGemini3ProHeuristic10(Heuristic):
    def __init__(self, robot_gemini3pro_heuristic_7):
        super().__init__(robot_gemini3pro_heuristic_7)
        self.model = robot_gemini3pro_heuristic_7.model
        self.inf = 999999

        # Data structures for fast lookup
        self.rloc_facts = [] # list of (room_name, global_id)
        self.in_facts = {}   # obj -> list of (room_name, global_id)
        self.holding_facts = {} # obj -> global_id
        self.closed_facts = {}  # door -> global_id
        self.goals_list = [] # list of (obj, target_room, fact_id)
        
        self.rooms = set()
        self.room_to_idx = {}
        self.adj = {} # room -> [(neighbor, door_name)]
        
        # Parse domain and precompute
        self._parse_domain()
        self._precompute_static_distances()

    def _parse_fact_name(self, name):
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

    def _is_initially_true(self, gid):
        # Helper to check if a fact is in initial state (handling set or bitwise)
        if isinstance(self.model.initial_state, int):
            return (self.model.initial_state >> gid) & 1
        else:
            return gid in self.model.initial_state

    def _parse_domain(self):
        # 1. Parse Facts for Maps and Static Graph
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            gid = fact.global_id
            
            if pred == 'rloc':
                self.rooms.add(args[0])
                self.rloc_facts.append((args[0], gid))
            
            elif pred == 'in':
                obj, loc = args
                if obj not in self.in_facts: self.in_facts[obj] = []
                self.in_facts[obj].append((loc, gid))
                
            elif pred == 'holding':
                self.holding_facts[args[0]] = gid
                
            elif pred == 'closed':
                self.closed_facts[args[0]] = gid
                
            elif pred == 'door':
                # Valid door if present in initial state
                if self._is_initially_true(gid):
                    r1, r2, d = args
                    self.rooms.add(r1)
                    self.rooms.add(r2)
                    if r1 not in self.adj: self.adj[r1] = []
                    if r2 not in self.adj: self.adj[r2] = []
                    self.adj[r1].append((r2, d))
                    self.adj[r2].append((r1, d))

        # 2. Identify Goals from Model Bitmask
        goal_mask = self.model.goals
        for fact in self.model.facts:
            if (goal_mask >> fact.global_id) & 1:
                pred, args = self._parse_fact_name(fact.name)
                if pred == 'in':
                    # Goal: Object in Room
                    self.goals_list.append((args[0], args[1], fact.global_id))

    def _precompute_static_distances(self):
        # APSP using Floyd-Warshall on graph with open doors (weight 1)
        # Also stores the path to check for closed doors dynamically
        
        rooms_list = list(self.rooms)
        self.room_to_idx = {r: i for i, r in enumerate(rooms_list)}
        n = len(rooms_list)
        
        dist = [[self.inf] * n for _ in range(n)]
        nxt = [[None] * n for _ in range(n)]
        self.edge_doors = {} # (u, v) -> door_name

        # Init distances
        for i in range(n):
            dist[i][i] = 0
            
        for r_name, neighbors in self.adj.items():
            u = self.room_to_idx[r_name]
            for neighbor_name, door_name in neighbors:
                if neighbor_name in self.room_to_idx:
                    v = self.room_to_idx[neighbor_name]
                    if dist[u][v] > 1:
                        dist[u][v] = 1
                        nxt[u][v] = v
                        self.edge_doors[(u, v)] = door_name

        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        nxt[i][j] = nxt[i][k]

        self.static_dist = dist
        self.nxt = nxt

    def h(self, node, task, local_facts=None, global_facts=None):
        state = node.state
        
        # 1. Identify Pending Goals
        pending_goals = []
        for obj, target, gid in self.goals_list:
            if not ((state >> gid) & 1):
                pending_goals.append((obj, target))
        
        if not pending_goals:
            return 0

        # 2. Get Robot Location
        r_loc_name = None
        for r, gid in self.rloc_facts:
            if (state >> gid) & 1:
                r_loc_name = r
                break
        if r_loc_name is None or r_loc_name not in self.room_to_idx:
            return self.inf
        
        r_idx = self.room_to_idx[r_loc_name]

        # 3. Identify Holding Status
        holding_obj = None
        for obj, gid in self.holding_facts.items():
            if (state >> gid) & 1:
                holding_obj = obj
                break

        # Helper: Calculate path cost considering closed doors
        # Cost = Dist + Sum(1 for closed_door in path)
        def get_path_cost(u, v):
            if u == v: return 0
            base_d = self.static_dist[u][v]
            if base_d >= self.inf: return self.inf
            
            penalty = 0
            curr = u
            while curr != v:
                next_node = self.nxt[curr][v]
                if next_node is None: return self.inf
                
                # Check door status
                if (curr, next_node) in self.edge_doors:
                    d_name = self.edge_doors[(curr, next_node)]
                    if d_name in self.closed_facts:
                        d_gid = self.closed_facts[d_name]
                        if (state >> d_gid) & 1:
                            penalty += 1 # 'open' action cost
                curr = next_node
            return base_d + penalty

        total_cost = 0
        min_approach = self.inf

        # 4. Calculate Cost for Pending Goals
        for obj, target in pending_goals:
            t_idx = self.room_to_idx.get(target)
            if t_idx is None: 
                total_cost += self.inf
                continue

            if obj == holding_obj:
                # Already holding: Cost = Path(Robot, Target) + Putdown
                d = get_path_cost(r_idx, t_idx)
                total_cost += d + 1
            else:
                # Not holding this object: Cost = Path(Obj, Target) + Pickup + Putdown
                obj_loc_idx = None
                if obj in self.in_facts:
                    for r, gid in self.in_facts[obj]:
                        if (state >> gid) & 1:
                            if r in self.room_to_idx:
                                obj_loc_idx = self.room_to_idx[r]
                                break
                
                if obj_loc_idx is not None:
                    d_transit = get_path_cost(obj_loc_idx, t_idx)
                    total_cost += d_transit + 2
                    
                    # If robot is empty, check approach cost to this object
                    if holding_obj is None:
                        d_approach = get_path_cost(r_idx, obj_loc_idx)
                        if d_approach < min_approach:
                            min_approach = d_approach
                else:
                    # Object location unknown (should not happen)
                    total_cost += self.inf

        # 5. Add approach cost (Robot -> Nearest Pending Pickup)
        if holding_obj is None and min_approach < self.inf:
            total_cost += min_approach

        return total_cost
