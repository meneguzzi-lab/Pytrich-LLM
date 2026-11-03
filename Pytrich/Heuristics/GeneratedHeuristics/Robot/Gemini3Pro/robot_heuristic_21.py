import heapq
import sys
from collections import deque
from pytrich_planner.core.heuristic import Heuristic

class RobotGemini3ProHeuristic21(Heuristic):
    def __init__(self, model, robot_gemini3pro_heuristic_18=None):
        super().__init__(model)
        self.model = model
        self.facts = model.facts
        
        # Initialize data structures
        self.room_to_id = {}
        self.id_to_room = []
        self.pkg_to_id = {}
        self.door_to_id = {}
        
        self.adj = [] # room_id -> list of (neighbor_id, door_id)
        
        # Fact Lookups (Mappings to Global IDs)
        self.rloc_facts = {}       # room_id -> global_id
        self.holding_facts = {}    # pkg_id -> global_id
        self.closed_facts = {}     # door_id -> global_id
        self.in_facts = {}         # pkg_id -> {room_id: global_id}
        self.goal_in_facts = []    # list of (pkg_id, room_id, global_id)
        
        # Parse the domain structure from facts
        self._parse_domain_facts()
        
        # Determine static goals (from initial state)
        self.pkg_goals = self._get_static_goals()
        
        # Precompute static distances (ignoring door status)
        self.static_dist = self._compute_static_apsp()

    def _get_obj_id(self, name, mapping, reverse_mapping=None):
        if name not in mapping:
            idx = len(mapping)
            mapping[name] = idx
            if reverse_mapping is not None:
                reverse_mapping.append(name)
                # Resize adjacency list if it's a room
                if mapping is self.room_to_id:
                    self.adj.append([])
        return mapping[name]

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

    def _parse_domain_facts(self):
        for fact in self.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
                
            if pred == "door":
                # door[loc1, loc2, d]
                u = self._get_obj_id(args[0], self.room_to_id, self.id_to_room)
                v = self._get_obj_id(args[1], self.room_to_id, self.id_to_room)
                d = self._get_obj_id(args[2], self.door_to_id)
                # Add edge
                self.adj[u].append((v, d))
                
            elif pred == "rloc":
                r = self._get_obj_id(args[0], self.room_to_id, self.id_to_room)
                self.rloc_facts[r] = fact.global_id
                
            elif pred == "holding":
                p = self._get_obj_id(args[0], self.pkg_to_id)
                self.holding_facts[p] = fact.global_id
                
            elif pred == "closed":
                d = self._get_obj_id(args[0], self.door_to_id)
                self.closed_facts[d] = fact.global_id
                
            elif pred == "in":
                p = self._get_obj_id(args[0], self.pkg_to_id)
                r = self._get_obj_id(args[1], self.room_to_id, self.id_to_room)
                if p not in self.in_facts:
                    self.in_facts[p] = {}
                self.in_facts[p][r] = fact.global_id
                
            elif pred == "goal_in":
                p = self._get_obj_id(args[0], self.pkg_to_id)
                r = self._get_obj_id(args[1], self.room_to_id, self.id_to_room)
                self.goal_in_facts.append((p, r, fact.global_id))

    def _get_static_goals(self):
        # Identify which goal_in facts are true in the initial state
        pkg_goals = {}
        init_state = self.model.initial_state
        for p_id, r_id, g_id in self.goal_in_facts:
            if (init_state >> g_id) & 1:
                pkg_goals[p_id] = r_id
        return pkg_goals

    def _compute_static_apsp(self):
        # All-Pairs Shortest Path on the room graph (unweighted)
        # Used as a lower bound for movement costs
        n = len(self.room_to_id)
        dists = [[999999] * n for _ in range(n)]
        
        for i in range(n):
            dists[i][i] = 0
            queue = deque([i])
            visited = {i}
            while queue:
                u = queue.popleft()
                for v, _ in self.adj[u]:
                    if v not in visited:
                        visited.add(v)
                        dists[i][v] = dists[i][u] + 1
                        queue.append(v)
        return dists

    def _compute_dynamic_dists(self, start_node, closed_doors):
        # Dijkstra from start_node considering closed doors cost
        # Move = 1, Open = 1. 
        # Traversing a closed door edge (u, v) costs 2 (open + move)
        # Traversing an open door edge (u, v) costs 1 (move)
        
        dists = {}
        pq = [(0, start_node)]
        visited_min = {}
        
        while pq:
            cost, u = heapq.heappop(pq)
            
            if u in visited_min and visited_min[u] <= cost:
                continue
            visited_min[u] = cost
            dists[u] = cost
            
            if u < len(self.adj):
                for v, d_id in self.adj[u]:
                    edge_cost = 1
                    if d_id in closed_doors:
                        edge_cost = 2
                    
                    new_cost = cost + edge_cost
                    if v not in visited_min or new_cost < visited_min.get(v, float('inf')):
                        heapq.heappush(pq, (new_cost, v))
        return dists

    def __call__(self, node):
        state = node.state
        h_val = 0
        
        # 1. Identify Robot Location
        robot_loc = -1
        for r_id, g_id in self.rloc_facts.items():
            if (state >> g_id) & 1:
                robot_loc = r_id
                break
        
        if robot_loc == -1:
            return float('inf')
            
        # 2. Identify Closed Doors
        closed_doors = set()
        for d_id, g_id in self.closed_facts.items():
            if (state >> g_id) & 1:
                closed_doors.add(d_id)
                
        # 3. Compute dynamic distances from robot
        # This gives accurate costs to reach packages or destinations from current loc
        robot_dists = self._compute_dynamic_dists(robot_loc, closed_doors)
        
        # 4. Check Held Package
        held_pkg = -1
        for p_id, g_id in self.holding_facts.items():
            if (state >> g_id) & 1:
                held_pkg = p_id
                break
        
        current_virtual_loc = robot_loc
        
        # If holding a package, we must deliver it first
        if held_pkg != -1:
            if held_pkg in self.pkg_goals:
                goal_r = self.pkg_goals[held_pkg]
                # Cost to deliver: dist(robot, goal) + putdown(1)
                dist_to_goal = robot_dists.get(goal_r, 999999)
                h_val += dist_to_goal + 1
                current_virtual_loc = goal_r
            # If held package has no goal, just assume we drop it? (Shouldn't happen in valid tasks)
        
        # 5. Process Waiting Packages
        pending_tasks = [] # list of (pickup_loc, delivery_cost)
        
        for p_id, goal_r in self.pkg_goals.items():
            if p_id == held_pkg:
                continue
                
            # Check if already at goal
            at_goal = False
            if p_id in self.in_facts and goal_r in self.in_facts[p_id]:
                g_id = self.in_facts[p_id][goal_r]
                if (state >> g_id) & 1:
                    at_goal = True
            
            if at_goal:
                continue
                
            # Where is the package?
            pkg_loc = -1
            if p_id in self.in_facts:
                for r_id, g_id in self.in_facts[p_id].items():
                    if (state >> g_id) & 1:
                        pkg_loc = r_id
                        break
            
            if pkg_loc != -1:
                # Cost for this package: Pickup(1) + Move(static_dist) + Putdown(1)
                # We use static_dist for the move because we assume doors will be opened
                # during the process, or it's a lower bound.
                move_cost = self.static_dist[pkg_loc][goal_r]
                task_cost = 1 + move_cost + 1
                
                pending_tasks.append((pkg_loc, task_cost))
                h_val += task_cost
                
        # 6. Add Transit Cost
        # We need to move from 'current_virtual_loc' to the start of a pending task.
        # To keep it admissible and simple, we take the minimum distance to ANY pending package.
        if pending_tasks:
            min_transit = float('inf')
            for start_loc, _ in pending_tasks:
                dist = 0
                if held_pkg != -1:
                    # We are virtually at the held package's goal. 
                    # Use static distance for future transit estimate.
                    if current_virtual_loc < len(self.static_dist) and start_loc < len(self.static_dist):
                        dist = self.static_dist[current_virtual_loc][start_loc]
                    else:
                         dist = 999999
                else:
                    # We are at robot_loc, use dynamic distance (considering closed doors)
                    dist = robot_dists.get(start_loc, 999999)
                
                if dist < min_transit:
                    min_transit = dist
            
            if min_transit != float('inf'):
                h_val += min_transit
                
        return h_val
