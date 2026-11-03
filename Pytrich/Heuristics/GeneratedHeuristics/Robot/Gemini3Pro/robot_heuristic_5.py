from Pytrich.Heuristics.heuristic import Heuristic
from collections import defaultdict
import sys

class RobotGemini3ProHeuristic5(Heuristic):
    def __init__(self, robot_gemini3pro_heuristic_2):
        super().__init__(robot_gemini3pro_heuristic_2)
        self.model = robot_gemini3pro_heuristic_2.model
        self.inf = float('inf')
        
        # Initialize static domain data and lookup tables
        self._initialize_domain_data()

    def _parse_fact_name(self, name):
        """
        Parses a grounded fact string into predicate and arguments.
        Format: +predicate[arg1,arg2]
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_start = name.find("[")
            if bracket_start > 0 and name.endswith("]"):
                pred = name[:bracket_start]
                args_str = name[bracket_start+1:-1]
                args = args_str.split(",") if args_str else []
                return pred, args
        return None, []

    def _initialize_domain_data(self):
        """
        Pre-computes static connectivity, goal maps, and fact indices for fast lookup.
        """
        # 1. Parse Goals from the model's goal bitmask
        # Goals are stored as a bitmask in model.goals. We need explicit names.
        self.package_goals = {}
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        
        needed_rooms = set()
        
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "goal_in" and len(args) == 2:
                # goal_in ?obj ?loc
                pkg, room = args[0], args[1]
                self.package_goals[pkg] = room
                needed_rooms.add(room)
        
        # 2. Build Static Connectivity Graph and Fact Lookups
        self.rooms = set()
        adj = defaultdict(list)
        
        self.id_to_rloc = {}
        self.id_to_holding = {}
        self.id_to_in = {}
        
        # Iterate through all facts in the model to identify static topology
        # and create ID mappings for dynamic state checking
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: 
                continue
            
            gid = fact.global_id
            
            if pred == "rloc" and len(args) == 1:
                self.id_to_rloc[gid] = args[0]
                self.rooms.add(args[0])
            
            elif pred == "holding" and len(args) == 1:
                self.id_to_holding[gid] = args[0]
            
            elif pred == "in" and len(args) == 2:
                # We only index 'in' facts for packages that actually have goals
                # to save iteration time during heuristic computation
                pkg = args[0]
                if pkg in self.package_goals:
                    self.id_to_in[gid] = (pkg, args[1])
                    self.rooms.add(args[1])
            
            elif pred == "door" and len(args) == 3:
                # (door ?loc1 ?loc2 ?d)
                # Assume undirected connectivity for movement
                r1, r2 = args[0], args[1]
                adj[r1].append(r2)
                adj[r2].append(r1)
                self.rooms.add(r1)
                self.rooms.add(r2)

        # 3. Compute All-Pairs Shortest Paths (BFS)
        # We assume all doors are open (cost 1) for the admissible lower bound.
        # Opening a door adds cost, so this assumption is admissible.
        self.distances = {}
        all_rooms = list(self.rooms)
        
        for start in all_rooms:
            dists = {}
            queue = [(start, 0)]
            visited = {start}
            dists[start] = 0
            
            idx = 0
            while idx < len(queue):
                u, d = queue[idx]
                idx += 1
                
                if u in adj:
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            dists[v] = d + 1
                            queue.append((v, d + 1))
            
            self.distances[start] = dists

    def compute(self, state, task_network=[]):
        """
        Calculates the domain-dependent heuristic.
        Strategy: Sum of transport costs for all unfulfilled packages + robot travel to first task.
        Admissible because robot capacity is 1, enforcing sequential delivery.
        """
        
        # 1. Extract Current State Information using Bitwise Operations
        robot_loc = None
        held_pkg = None
        pkg_locations = {}
        
        # Check Robot Location
        # There is usually only 1 robot location, so we break after finding it
        for gid, loc in self.id_to_rloc.items():
            if (state >> gid) & 1:
                robot_loc = loc
                break
        
        if robot_loc is None:
            # Invalid state (robot lost), return max cost
            return self.inf

        # Check Held Package
        # Robot can hold at most 1 package
        for gid, pkg in self.id_to_holding.items():
            if (state >> gid) & 1:
                held_pkg = pkg
                break
        
        # Check Locations of Relevant Packages
        # Iterate only through 'in' facts related to packages with goals
        for gid, (pkg, loc) in self.id_to_in.items():
            if (state >> gid) & 1:
                pkg_locations[pkg] = loc

        # 2. Calculate Heuristic Cost
        h_val = 0
        pending_pickup_locs = []
        
        for pkg, goal_loc in self.package_goals.items():
            # Case A: Package is currently held by robot
            if held_pkg == pkg:
                # Cost = Move to Goal + Putdown
                dist = self.distances.get(robot_loc, {}).get(goal_loc, self.inf)
                if dist == self.inf:
                    return self.inf
                h_val += dist + 1
            
            # Case B: Package is at a location (and not held)
            elif pkg in pkg_locations:
                curr_loc = pkg_locations[pkg]
                
                # If not at goal, we need to move it
                if curr_loc != goal_loc:
                    # Cost = Pickup + Move to Goal + Putdown
                    # (Pickup cost=1, Putdown cost=1)
                    dist_transport = self.distances.get(curr_loc, {}).get(goal_loc, self.inf)
                    if dist_transport == self.inf:
                        return self.inf
                    
                    h_val += dist_transport + 2
                    pending_pickup_locs.append(curr_loc)
            
            # Case C: Package location unknown. 
            # In a valid state, it must be either held or in a room. 
            # If it's already at goal and we didn't track it above? 
            # Logic handles 'in' check. If it's in goal, curr_loc == goal_loc -> cost 0.

        # 3. Add Robot Transit Cost
        # If the robot is not holding anything, it must travel to a pending pickup.
        # We take the minimum distance to any pending pickup as a lower bound.
        if held_pkg is None and pending_pickup_locs:
            min_transit = self.inf
            dists_from_robot = self.distances.get(robot_loc, {})
            
            for loc in pending_pickup_locs:
                d = dists_from_robot.get(loc, self.inf)
                if d < min_transit:
                    min_transit = d
            
            if min_transit == self.inf:
                return self.inf
            
            h_val += min_transit

        return h_val
