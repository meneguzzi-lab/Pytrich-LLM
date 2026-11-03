import sys
from Pytrich.Heuristics.heuristic import Heuristic

class RobotGemini3ProHeuristic12(Heuristic):
    def __init__(self, robot_gemini3pro_heuristic_9, model):
        super().__init__(robot_gemini3pro_heuristic_9, model)
        self.model = model
        self.rooms = set()
        self.packages = set()
        self.distances = {}
        
        # Mappings for fast state lookup via bitmasks
        # List of (room_name, global_id) for robot location predicates
        self.rloc_bits = [] 
        # Dictionary mapping package_name -> list of (room_name, global_id) for 'in' predicates
        self.in_bits = {}
        # Dictionary mapping package_name -> global_id for 'holding' predicates
        self.holding_bits = {}
        
        # Goal information: package_name -> target_room_name
        self.package_goals = {}

        self._initialize()

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

    def _initialize(self):
        # 1. Parse Facts to build bitmask mappings
        # This allows us to check specific predicates in O(1) during heuristic evaluation
        for i, fact in enumerate(self.model.facts):
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == 'rloc':
                # Robot location facts: +rloc[room]
                r = args[0]
                self.rooms.add(r)
                self.rloc_bits.append((r, i))
            elif pred == 'in':
                # Package location facts: +in[pkg, room]
                if len(args) == 2:
                    pkg, r = args
                    self.packages.add(pkg)
                    self.rooms.add(r)
                    if pkg not in self.in_bits: 
                        self.in_bits[pkg] = []
                    self.in_bits[pkg].append((r, i))
            elif pred == 'holding':
                # Robot holding facts: +holding[pkg]
                if len(args) == 1:
                    pkg = args[0]
                    self.packages.add(pkg)
                    self.holding_bits[pkg] = i
        
        # 2. Build Topology Graph and Compute All-Pairs Shortest Paths (APSP)
        # We assume move cost is 1. We treat doors as edges.
        # Closed doors are treated as passable for distance estimation (admissible lower bound).
        adj = {r: [] for r in self.rooms}
        
        # Extract topology from initial state facts (door definitions are static)
        init_fact_names = self.model.state_explicit_repr(self.model.initial_state)
        for fname in init_fact_names:
            pred, args = self._parse_fact_name(fname)
            if pred == 'door' and len(args) == 3:
                u, v, d = args
                if u in adj:
                    adj[u].append(v)
                # Although domain might define doors both ways, we ensure symmetry if not explicit
                if v in adj and u not in adj[v]:
                    adj[v].append(u)

        # Run BFS from every room to compute shortest path distances
        for start_node in self.rooms:
            dists = {r: float('inf') for r in self.rooms}
            dists[start_node] = 0
            queue = [start_node]
            while queue:
                u = queue.pop(0)
                d_val = dists[u]
                for v in adj.get(u, []):
                    if dists[v] == float('inf'):
                        dists[v] = d_val + 1
                        queue.append(v)
            self.distances[start_node] = dists

        # 3. Parse Goal State
        # Extract target locations for packages from the goal bitmask
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for fname in goal_fact_names:
            pred, args = self._parse_fact_name(fname)
            if pred == 'in' and len(args) == 2:
                pkg, r = args
                self.package_goals[pkg] = r

    def h(self, node):
        """
        Heuristic estimate: Sum of transport costs for all misplaced packages.
        For each package not at goal:
            Cost += 1 (Pickup) + Distance(PkgLoc, Goal) + 1 (Putdown)
        Plus: Cost to move robot to the nearest relevant location.
        """
        state = node.state
        cost = 0.0
        
        # 1. Determine Robot Location
        robot_loc = None
        for r, fid in self.rloc_bits:
            if (state >> fid) & 1:
                robot_loc = r
                break
        
        if robot_loc is None:
            # Invalid state (robot must be somewhere), return infinity
            return float('inf')

        # 2. Check if Robot is holding anything
        holding_pkg = None
        for pkg, fid in self.holding_bits.items():
            if (state >> fid) & 1:
                holding_pkg = pkg
                break
        
        # 3. Calculate Heuristic Cost
        virtual_robot_loc = robot_loc
        
        # Cost for currently held package
        if holding_pkg:
            goal = self.package_goals.get(holding_pkg)
            if goal:
                # Robot has the package, needs to move to goal and put it down
                d_to_goal = self.distances.get(robot_loc, {}).get(goal, float('inf'))
                if d_to_goal == float('inf'):
                    return float('inf')
                cost += d_to_goal + 1 # Move + Putdown
                # After this task, robot is effectively at the goal location
                virtual_robot_loc = goal
            else:
                # Holding package with no goal (rare/unlikely), assume cost 0 to drop
                pass
        
        # Cost for other packages
        min_dist_to_pickup = float('inf')
        has_pending_packages = False
        
        for pkg in self.packages:
            if pkg == holding_pkg: 
                continue
            
            goal = self.package_goals.get(pkg)
            if not goal:
                continue # This package has no goal constraint
            
            # Find current location of this package
            pkg_loc = None
            is_at_goal = False
            
            if pkg in self.in_bits:
                for r, fid in self.in_bits[pkg]:
                    if (state >> fid) & 1:
                        pkg_loc = r
                        if r == goal:
                            is_at_goal = True
                        break
            
            if is_at_goal:
                continue
            
            if pkg_loc:
                has_pending_packages = True
                # Transport cost: Pickup + Move to Goal + Putdown
                # Note: We ignore robot travel TO the package here, handled by min_dist below
                d_transport = self.distances.get(pkg_loc, {}).get(goal, float('inf'))
                if d_transport == float('inf'):
                    return float('inf')
                
                cost += 1 + d_transport + 1
                
                # Calculate distance from 'virtual_robot_loc' to this package's pickup
                # This estimates the travel cost to start the NEXT task
                d_to_pickup = self.distances.get(virtual_robot_loc, {}).get(pkg_loc, float('inf'))
                if d_to_pickup < min_dist_to_pickup:
                    min_dist_to_pickup = d_to_pickup
            
            # If pkg_loc is None, we might have partial info or it's somewhere else.
            # We assume cost 0 if unknown to remain admissible.

        # Add the cost to travel to the nearest pickup location
        if has_pending_packages:
            if min_dist_to_pickup != float('inf'):
                cost += min_dist_to_pickup
            else:
                return float('inf')
                
        return cost
