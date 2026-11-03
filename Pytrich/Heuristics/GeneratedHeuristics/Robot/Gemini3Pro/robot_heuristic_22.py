import sys
from typing import List, Dict, Set, Tuple
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic

class RobotGemini3ProHeuristic22(Heuristic):
    def __init__(self, robot_gemini3pro_heuristic_19, model):
        super().__init__(robot_gemini3pro_heuristic_19, model)
        self.model = model
        
        # Lookup tables for fast state decoding
        # Maps global_id -> type ('rloc', 'holding', 'in')
        self.fact_type = {} 
        # Maps global_id -> args (e.g. room name, obj name)
        self.fact_args = {}
        
        # Domain static structures
        self.rooms = set()
        self.doors = [] 
        self.goals = {} # obj -> goal_room
        
        # 1. Preprocess facts to populate lookup tables and identify rooms
        self._preprocess_facts()
        
        # 2. Parse initial state to find static relations (door, goal_in)
        self._parse_static_init()
        
        # 3. Compute All-Pairs Shortest Paths for room distances
        self.distances = self._compute_distances()

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

    def _preprocess_facts(self):
        """Iterate all grounded facts to build direct lookup tables."""
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
                
            gid = fact.global_id
            
            if pred == 'rloc':
                self.fact_type[gid] = 'rloc'
                self.fact_args[gid] = args[0] # room
                self.rooms.add(args[0])
            elif pred == 'holding':
                self.fact_type[gid] = 'holding'
                self.fact_args[gid] = args[0] # obj
            elif pred == 'in':
                self.fact_type[gid] = 'in'
                self.fact_args[gid] = (args[0], args[1]) # obj, room
                self.rooms.add(args[1])
            # Static predicates (door, goal_in) are handled in _parse_static_init

    def _parse_static_init(self):
        """Extract static relations from the initial state."""
        init_state = self.model.initial_state
        # Iterate model facts again to check if they are in init_state
        # This is safe because this runs only once during initialization.
        for fact in self.model.facts:
            if not ((init_state >> fact.global_id) & 1):
                continue
                
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            
            if pred == 'door':
                # args: loc1, loc2, d. We treat graph as undirected for movement.
                self.doors.append((args[0], args[1]))
                self.doors.append((args[1], args[0]))
                self.rooms.add(args[0])
                self.rooms.add(args[1])
            elif pred == 'goal_in':
                # args: obj, loc
                self.goals[args[0]] = args[1]

    def _compute_distances(self):
        """Compute All-Pairs Shortest Path using BFS (unweighted graph)."""
        sorted_rooms = sorted(list(self.rooms))
        self.room_to_idx = {r: i for i, r in enumerate(sorted_rooms)}
        n = len(sorted_rooms)
        
        # Build adjacency list
        adj = {i: [] for i in range(n)}
        for r1, r2 in self.doors:
            if r1 in self.room_to_idx and r2 in self.room_to_idx:
                u, v = self.room_to_idx[r1], self.room_to_idx[r2]
                adj[u].append(v)
        
        # Initialize distances with infinity
        inf = 999999
        dists = [[inf] * n for _ in range(n)]
        
        # Run BFS from each node
        for i in range(n):
            dists[i][i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                d = dists[i][u]
                for v in adj[u]:
                    if dists[i][v] == inf:
                        dists[i][v] = d + 1
                        q.append(v)
        return dists

    def get_dist(self, r1, r2):
        """Get precomputed distance between two rooms."""
        if r1 == r2: return 0
        if r1 not in self.room_to_idx or r2 not in self.room_to_idx:
            return 999 # Fallback for disconnected/unknown
        return self.distances[self.room_to_idx[r1]][self.room_to_idx[r2]]

    def h(self, node, task_network, state):
        """
        Heuristic calculation:
        1. Decode current state (robot loc, objects held/in rooms).
        2. Identify unsatisfied goals.
        3. Cost = Transport costs + Pickup transition costs.
           - If holding obj: Dist(robot, goal) + 1 (putdown).
           - If obj not at goal: Dist(obj_loc, goal) + 2 (pickup/putdown).
           - Fetch cost: Dist(robot, nearest_pickup).
        """
        # Decode state using bitwise operations
        robot_loc = None
        held_obj = None
        obj_locs = {}
        
        temp_state = state
        while temp_state:
            # Get lowest set bit efficiently
            lsb = temp_state & -temp_state
            idx = lsb.bit_length() - 1
            temp_state ^= lsb
            
            # Retrieve fact info from lookup tables
            ftype = self.fact_type.get(idx)
            if ftype == 'rloc':
                robot_loc = self.fact_args[idx]
            elif ftype == 'holding':
                held_obj = self.fact_args[idx]
            elif ftype == 'in':
                o, r = self.fact_args[idx]
                obj_locs[o] = r
        
        if robot_loc is None:
            return 0 # Should not happen in valid states
            
        cost = 0
        pending_pickups = []
        
        # Calculate cost for all goals
        for obj, goal_room in self.goals.items():
            # Case 1: Robot is holding the object
            if obj == held_obj:
                # Must deliver this first (capacity 1)
                d = self.get_dist(robot_loc, goal_room)
                cost += d + 1 # Move + Putdown
                # Effectively robot moves to goal_room
                robot_loc = goal_room
                held_obj = None
            
            # Case 2: Object is on the ground
            elif obj in obj_locs:
                curr_room = obj_locs[obj]
                if curr_room != goal_room:
                    # Needs transport
                    d = self.get_dist(curr_room, goal_room)
                    cost += d + 2 # Pickup + Move + Putdown
                    pending_pickups.append(curr_room)
        
        # Add fetch cost: Robot needs to get to the next pickup location
        if pending_pickups:
            # Admissible heuristic: minimum distance to any required pickup
            min_fetch = 999999
            for loc in pending_pickups:
                d = self.get_dist(robot_loc, loc)
                if d < min_fetch:
                    min_fetch = d
            
            if min_fetch < 999999:
                cost += min_fetch
                
        return cost