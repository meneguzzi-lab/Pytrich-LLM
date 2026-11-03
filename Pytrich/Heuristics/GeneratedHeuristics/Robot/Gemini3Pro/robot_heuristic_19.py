from Pytrich.Heuristics.heuristic import Heuristic

class RobotGemini3ProHeuristic19(Heuristic):
    def __init__(self, model, robot_gemini3pro_heuristic_16=None):
        super().__init__(model)
        self.robot_gemini3pro_heuristic_16 = robot_gemini3pro_heuristic_16
        self.INF = 999999
        self._parse_domain()

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
        """
        Parses the domain structure to build lookup tables and the distance graph.
        """
        self.rloc_facts = {}  # map room_name -> global_id
        self.holding_facts = {} # map obj_name -> global_id
        self.in_facts = {}      # map (obj, loc) -> global_id
        self.obj_loc_fids = {}  # map obj -> list of (loc, global_id)
        self.goal_locs = {}     # map obj -> goal_room (derived from model.goals)
        self.rooms = set()
        
        adj = {}

        # 1. Parse all facts to map predicates to IDs and build static graph
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            
            if pred == 'rloc':
                r = args[0]
                self.rloc_facts[r] = fact.global_id
                self.rooms.add(r)
            elif pred == 'in':
                obj, loc = args[0], args[1]
                self.in_facts[(obj, loc)] = fact.global_id
                if obj not in self.obj_loc_fids:
                    self.obj_loc_fids[obj] = []
                self.obj_loc_fids[obj].append((loc, fact.global_id))
                self.rooms.add(loc)
            elif pred == 'holding':
                obj = args[0]
                self.holding_facts[obj] = fact.global_id
            elif pred == 'door':
                # Use initial state to confirm static door connectivity
                if (self.model.initial_state >> fact.global_id) & 1:
                    r1, r2 = args[0], args[1]
                    if r1 not in adj: adj[r1] = []
                    adj[r1].append(r2)
                    # Assuming symmetric for graph traversal
                    if r2 not in adj: adj[r2] = []
                    adj[r2].append(r1)
                    self.rooms.add(r1)
                    self.rooms.add(r2)

        # 2. Build All-Pairs Shortest Paths (Floyd-Warshall)
        # We ignore closed doors for the admissible heuristic (lower bound)
        sorted_rooms = sorted(list(self.rooms))
        self.room_map = {r: i for i, r in enumerate(sorted_rooms)}
        n = len(sorted_rooms)
        self.dist = [[self.INF] * n for _ in range(n)]
        
        for i in range(n):
            self.dist[i][i] = 0
            
        for r, neighbors in adj.items():
            u = self.room_map.get(r)
            if u is None: continue
            for neighbor in neighbors:
                v = self.room_map.get(neighbor)
                if v is not None:
                    self.dist[u][v] = 1 # Edge weight 1 for move action

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.dist[i][j] > self.dist[i][k] + self.dist[k][j]:
                        self.dist[i][j] = self.dist[i][k] + self.dist[k][j]

        # 3. Identify Goal Configuration
        # Scan in_facts against model.goals bitmask
        goal_mask = self.model.goals
        for (obj, loc), fid in self.in_facts.items():
            if (goal_mask >> fid) & 1:
                self.goal_locs[obj] = loc

    def _get_dist(self, r1, r2):
        u = self.room_map.get(r1)
        v = self.room_map.get(r2)
        if u is not None and v is not None:
            return self.dist[u][v]
        return self.INF

    def h(self, node):
        """
        Calculates the heuristic value.
        Estimates the cost to transport all pending packages to their destinations.
        Accounts for robot travel time to the nearest package.
        """
        state = node.state
        
        # 1. Locate Robot
        robot_loc = None
        for r, fid in self.rloc_facts.items():
            if (state >> fid) & 1:
                robot_loc = r
                break
        
        if robot_loc is None:
            return self.INF

        total_cost = 0
        
        # If robot is holding an object, it effectively starts the next sequence from that object's goal
        # because it must deliver it first (capacity=1).
        current_effective_loc = robot_loc
        
        pending_pickup_locs = []
        
        # 2. Iterate over all goals
        for obj, goal_loc in self.goal_locs.items():
            # Check if goal satisfied
            goal_fid = self.in_facts.get((obj, goal_loc))
            if goal_fid is not None and ((state >> goal_fid) & 1):
                continue
            
            # Check if holding
            is_held = False
            hid = self.holding_facts.get(obj)
            if hid is not None and ((state >> hid) & 1):
                is_held = True
            
            if is_held:
                # Cost: Dist(Current, Goal) + Putdown(1)
                d = self._get_dist(robot_loc, goal_loc)
                if d >= self.INF: return self.INF
                total_cost += d + 1
                
                # Since we are holding this, we must drop it off before doing anything else.
                # The robot will end up at goal_loc.
                current_effective_loc = goal_loc
            else:
                # Not held, not at goal -> Must be picked up
                obj_loc = None
                if obj in self.obj_loc_fids:
                    for loc, fid in self.obj_loc_fids[obj]:
                        if (state >> fid) & 1:
                            obj_loc = loc
                            break
                
                if obj_loc:
                    # Transport Cost: Pickup(1) + Move(Loc->Goal) + Putdown(1)
                    d_trans = self._get_dist(obj_loc, goal_loc)
                    if d_trans >= self.INF: return self.INF
                    total_cost += d_trans + 2
                    
                    # We need to travel to obj_loc eventually
                    pending_pickup_locs.append(obj_loc)
                else:
                    # Object lost? Should not happen in valid state.
                    pass

        # 3. Add travel cost to the nearest next task
        # If we have pending pickups, we must travel to at least one of them.
        # We use a greedy estimate: min distance from our current effective location.
        if pending_pickup_locs:
            min_d = self.INF
            for loc in pending_pickup_locs:
                d = self._get_dist(current_effective_loc, loc)
                if d < min_d:
                    min_d = d
            
            if min_d < self.INF:
                total_cost += min_d

        return total_cost
