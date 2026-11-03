from Pytrich.Heuristics.heuristic import Heuristic

class RobotGemini3ProHeuristic7(Heuristic):
    def __init__(self, robot_gemini3pro_heuristic_4):
        super().__init__(robot_gemini3pro_heuristic_4)
        self.model = robot_gemini3pro_heuristic_4.model
        
        # Debug/Config
        self.MAX_COST = 999999
        
        # Static Data Structures
        self.rooms = []
        self.room_to_id = {}
        self.packages = []
        
        # Fast Lookups for Heuristic Calculation
        # rloc_checks: list of (global_id, room_id)
        self.rloc_checks = [] 
        # pkg_location_checks: dict[pkg_name] -> list of (global_id, room_id)
        self.pkg_location_checks = {}
        # holding_checks: dict[pkg_name] -> global_id
        self.holding_checks = {}
        
        # Goal State: pkg_name -> room_id
        self.goal_state_map = {}
        
        # Graph for distances
        self.adj = {}
        self.distances = []
        
        # Initialization
        self._initialize_static_data()
        self._calculate_distances()
        self._parse_goals()

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

    def _initialize_static_data(self):
        """
        Parses all facts in the model to identify objects (rooms, packages)
        and build the necessary lookups and adjacency graph.
        """
        room_names = set()
        pkg_names = set()
        
        # 1. Identification Pass
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
                
            if pred == 'rloc':
                room_names.add(args[0])
            elif pred == 'in':
                pkg_names.add(args[0])
                room_names.add(args[1])
            elif pred == 'door':
                room_names.add(args[0])
                room_names.add(args[1])
        
        self.rooms = sorted(list(room_names))
        self.room_to_id = {name: i for i, name in enumerate(self.rooms)}
        self.packages = sorted(list(pkg_names))
        
        # Initialize structures
        num_rooms = len(self.rooms)
        for i in range(num_rooms):
            self.adj[i] = []
        
        for pkg in self.packages:
            self.pkg_location_checks[pkg] = []
            
        # 2. Fact Mapping Pass
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            
            gid = fact.global_id
            
            if pred == 'rloc':
                r_id = self.room_to_id.get(args[0])
                if r_id is not None:
                    self.rloc_checks.append((gid, r_id))
            
            elif pred == 'in':
                pkg = args[0]
                r_id = self.room_to_id.get(args[1])
                if r_id is not None and pkg in self.pkg_location_checks:
                    self.pkg_location_checks[pkg].append((gid, r_id))
            
            elif pred == 'holding':
                pkg = args[0]
                self.holding_checks[pkg] = gid
                
            elif pred == 'door':
                # door(loc1, loc2, d)
                r1 = self.room_to_id.get(args[0])
                r2 = self.room_to_id.get(args[1])
                if r1 is not None and r2 is not None:
                    # Add edge if not exists (doors might be defined twice for symmetry)
                    if r2 not in self.adj[r1]:
                        self.adj[r1].append(r2)
                    if r1 not in self.adj[r2]:
                        self.adj[r2].append(r1)

    def _calculate_distances(self):
        """
        Computes all-pairs shortest paths using BFS.
        We ignore door status (closed/open) for the heuristic distance,
        assuming they can be opened (cost included in action or ignored for relaxation).
        """
        num_rooms = len(self.rooms)
        self.distances = [[self.MAX_COST] * num_rooms for _ in range(num_rooms)]
        
        for start_node in range(num_rooms):
            self.distances[start_node][start_node] = 0
            queue = [start_node]
            while queue:
                u = queue.pop(0)
                dist_u = self.distances[start_node][u]
                
                for v in self.adj[u]:
                    if self.distances[start_node][v] == self.MAX_COST:
                        self.distances[start_node][v] = dist_u + 1
                        queue.append(v)

    def _parse_goals(self):
        """
        Identifies goal locations for packages.
        First checks PDDL goals (model.goals).
        Fallback to 'goal_in' predicates in initial state if PDDL goals are empty.
        """
        goal_bitmask = self.model.goals
        found_pddl_goals = False
        
        # Check PDDL goals
        for pkg in self.packages:
            for gid, r_id in self.pkg_location_checks[pkg]:
                if (goal_bitmask >> gid) & 1:
                    self.goal_state_map[pkg] = r_id
                    found_pddl_goals = True
                    break
        
        if found_pddl_goals:
            return
            
        # Fallback: Check 'goal_in' in initial state
        init_state = self.model.initial_state
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'goal_in':
                pkg = args[0]
                r_id = self.room_to_id.get(args[1])
                if (init_state >> fact.global_id) & 1 and r_id is not None:
                    self.goal_state_map[pkg] = r_id

    def h(self, node):
        """
        Heuristic Estimate:
        Sum of transport costs for all misplaced packages.
        Cost = Distance(pkg_loc, goal_loc) + handling_cost.
        If robot is not holding a package, adds min distance to a pending package.
        """
        state = node.state
        
        # 1. Determine Robot Location
        robot_loc = -1
        for gid, r_id in self.rloc_checks:
            if (state >> gid) & 1:
                robot_loc = r_id
                break
        
        if robot_loc == -1:
            return self.MAX_COST # Should not happen in reachable states
            
        total_cost = 0
        pending_pickup_locs = []
        holding_pkg = None
        holding_goal_loc = None
        
        # 2. Evaluate Packages
        for pkg, goal_rid in self.goal_state_map.items():
            # Check if holding
            h_gid = self.holding_checks.get(pkg)
            if h_gid is not None and (state >> h_gid) & 1:
                # Robot is holding this package
                holding_pkg = pkg
                holding_goal_loc = goal_rid
                
                # Cost: Move to goal + Putdown (1)
                d = self.distances[robot_loc][goal_rid]
                if d >= self.MAX_COST:
                    return self.MAX_COST
                total_cost += d + 1
                continue
            
            # Check if in a room
            current_rid = -1
            # Fast check: iterate known location facts for this package
            for gid, r_id in self.pkg_location_checks.get(pkg, []):
                if (state >> gid) & 1:
                    current_rid = r_id
                    break
            
            if current_rid != -1:
                if current_rid == goal_rid:
                    continue # Already satisfied
                else:
                    # Package needs transport
                    # Cost: Transport + Pickup (1) + Putdown (1)
                    # We count pickup/putdown as 2 ops.
                    d = self.distances[current_rid][goal_rid]
                    if d >= self.MAX_COST:
                        return self.MAX_COST
                    total_cost += d + 2
                    pending_pickup_locs.append(current_rid)
        
        # 3. Add Robot Travel Cost (The "Empty Moves")
        # If robot is holding a package, it MUST deliver it first (capacity 1 assumption).
        # After delivery, it is at holding_goal_loc. Then it must go to a pending package.
        # If robot is empty, it must go to a pending package immediately.
        
        if pending_pickup_locs:
            min_dist_to_next = self.MAX_COST
            
            origin_loc = robot_loc
            if holding_pkg:
                # We effectively start searching for the next package from the destination of the current one
                # because the cost to get there is already added.
                origin_loc = holding_goal_loc
            
            # Find nearest pending package from the logical origin
            for p_loc in pending_pickup_locs:
                d = self.distances[origin_loc][p_loc]
                if d < min_dist_to_next:
                    min_dist_to_next = d
            
            if min_dist_to_next < self.MAX_COST:
                total_cost += min_dist_to_next
                
        return total_cost
