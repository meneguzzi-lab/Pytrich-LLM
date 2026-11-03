from pytrich.heuristics.heuristic import Heuristic

class RobotGemini3ProHeuristic15(Heuristic):
    def __init__(self, model, robot_gemini3pro_heuristic_12=None):
        super().__init__(model)
        self.model = model
        
        # Data structures for domain analysis
        self.graph = {}
        self.goals = {} # pkg -> target_room
        self.fact_lookup = [None] * len(model.facts)
        
        # 1. Parse static facts and build lookup table
        # We check model.initial_state for static facts like door and goal_in
        # to ensure they are true in the problem definition.
        init_state = model.initial_state
        
        for fact in model.facts:
            gid = fact.global_id
            name = fact.name
            
            predicate, args = self._parse_fact_name(name)
            is_initially_true = (init_state >> gid) & 1
            
            # Static Graph Construction (Door connectivity)
            if predicate == "door":
                # Format: +door[loc1, loc2, d]
                # We treat doors as directed edges for the move action
                if is_initially_true and len(args) >= 2:
                    u, v = args[0], args[1]
                    if u not in self.graph: self.graph[u] = []
                    if v not in self.graph: self.graph[v] = [] # Ensure nodes exist
                    self.graph[u].append(v)
            
            # Goal Mapping
            elif predicate == "goal_in":
                # Format: +goal_in[pkg, room]
                if is_initially_true and len(args) == 2:
                    self.goals[args[0]] = args[1]
            
            # Dynamic Fact Lookup Table
            # We map global_ids to tuples for fast decoding during search
            if predicate == "rloc":
                self.fact_lookup[gid] = ("rloc", args[0])
            elif predicate == "in":
                self.fact_lookup[gid] = ("in", args[0], args[1]) # pkg, loc
            elif predicate == "holding":
                self.fact_lookup[gid] = ("holding", args[0]) # pkg

        # 2. Precompute All-Pairs Shortest Paths (BFS)
        # This handles the distance estimation efficiently
        self.distances = {}
        self._compute_distances()

    def _parse_fact_name(self, name: str):
        """Parses a grounded fact string like +predicate[arg1,arg2]"""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        idx = name.find("[")
        if idx != -1 and name.endswith("]"):
            pred = name[:idx]
            arg_str = name[idx+1:-1]
            args = arg_str.split(",") if arg_str else []
            return pred, args
        return name, []

    def _compute_distances(self):
        """Computes BFS distances between all rooms"""
        nodes = list(self.graph.keys())
        # Ensure goal rooms are included even if they have no outgoing edges
        for g_room in self.goals.values():
            if g_room not in self.graph:
                self.graph[g_room] = []
                nodes.append(g_room)
        nodes = list(set(nodes))
        
        for start in nodes:
            self.distances[(start, start)] = 0
            queue = [(start, 0)]
            visited = {start}
            
            idx = 0
            while idx < len(queue):
                curr, d = queue[idx]
                idx += 1
                
                self.distances[(start, curr)] = d
                
                if curr in self.graph:
                    for neighbor in self.graph[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))

    def __call__(self, node):
        """
        Heuristic Function: h(n)
        Estimates cost: Sum of transport costs for all misplaced packages.
        Cost = (Fetch Robot) + (Pickup) + (Transport) + (Putdown)
        """
        state = node.state
        
        # Decode current state variables efficiently
        robot_loc = None
        pkg_locs = {}
        held_pkg = None
        
        temp_state = state
        while temp_state:
            # Get lowest set bit
            lsb = temp_state & -temp_state
            gid = lsb.bit_length() - 1
            temp_state ^= lsb # Clear bit
            
            info = self.fact_lookup[gid]
            if info:
                itype = info[0]
                if itype == "rloc":
                    robot_loc = info[1]
                elif itype == "in":
                    pkg_locs[info[1]] = info[2]
                elif itype == "holding":
                    held_pkg = info[1]
        
        if robot_loc is None:
            # Invalid state (robot missing), return max cost
            return float('inf')
            
        total_cost = 0
        min_fetch_dist = float('inf')
        pending_work = False
        
        # Calculate cost for each goal
        for pkg, target_room in self.goals.items():
            # If package is already at goal, cost is 0
            if pkg in pkg_locs and pkg_locs[pkg] == target_room:
                continue
                
            # Package is not at goal
            pending_work = True
            
            if held_pkg == pkg:
                # Robot is holding the package
                # Cost: Move to goal + Putdown (cost 1)
                d = self.distances.get((robot_loc, target_room), float('inf'))
                if d == float('inf'): return float('inf')
                
                total_cost += d + 1
                # Robot is effectively fetching this package (distance 0)
                min_fetch_dist = 0
                
            elif pkg in pkg_locs:
                # Package is on the floor elsewhere
                curr_room = pkg_locs[pkg]
                
                # Transport Cost: Move Pkg to Goal + Pickup(1) + Putdown(1)
                d_trans = self.distances.get((curr_room, target_room), float('inf'))
                if d_trans == float('inf'): return float('inf')
                
                total_cost += d_trans + 2
                
                # Fetch Cost: Move Robot to Pkg
                d_fetch = self.distances.get((robot_loc, curr_room), float('inf'))
                if d_fetch < min_fetch_dist:
                    min_fetch_dist = d_fetch
                    
        # Add the cost to reach the nearest relevant package
        # If we are holding a package, min_fetch_dist is 0, which is correct
        if pending_work and min_fetch_dist != float('inf'):
            total_cost += min_fetch_dist
            
        return total_cost
