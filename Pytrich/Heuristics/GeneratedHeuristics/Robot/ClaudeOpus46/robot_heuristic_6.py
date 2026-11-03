# Robot Domain Heuristic
# Bottleneck: Single-arm carry constraint — robot holds one package at a time,
# requiring separate pickup-travel-putdown cycles per misplaced package.
#
# Admissible lower bounds:
# T1: Each misplaced package requires >= 2 actions (pickup + putdown). Admissible
#     because no decomposition can avoid these primitive actions.
# T2: Travel lower bound — sum of minimum distances the robot must travel to
#     service each misplaced package (visit source, then destination). We use
#     a greedy TSP-like lower bound: for each misplaced package we need at least
#     the distance from source to dest (the robot must carry it). The overall
#     lower bound also accounts for the robot needing to reach each source.
# T3: Task network depth — remaining abstract tasks give a structural lower bound.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque


class RobotAnthropicOpus46Heuristic6(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_6"):
        super().__init__(name)
        self.model = None
        # Fact ID maps
        self.in_fact_ids = {}       # (pkg, room) -> global_id
        self.goal_in_fact_ids = {}  # (pkg, room) -> global_id
        self.rloc_fact_ids = {}     # room -> global_id
        self.holding_fact_ids = {}  # pkg -> global_id
        self.armempty_id = None
        self.closed_fact_ids = {}   # door -> global_id
        self.door_facts = {}        # (room1, room2) -> door
        
        # Room graph and distances
        self.room_adj = {}          # room -> set of (neighbor_room, door)
        self.rooms = set()
        self.packages = set()
        self.dist_matrix = {}       # (room1, room2) -> shortest distance (ignoring closed doors)
        
        # Goal info
        self.pkg_goal_room = {}     # pkg -> goal_room
        
        # Operator cost info
        self.min_op_cost = {}

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

    def _bfs_distances(self, start_room):
        """BFS from start_room, treating all doors as passable (optimistic/admissible)."""
        dist = {start_room: 0}
        queue = deque([start_room])
        while queue:
            room = queue.popleft()
            d = dist[room]
            if room in self.room_adj:
                for neighbor, door in self.room_adj[room]:
                    if neighbor not in dist:
                        dist[neighbor] = d + 1
                        queue.append(neighbor)
        return dist

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in":
                self.in_fact_ids[(args[0], args[1])] = f.global_id
                self.packages.add(args[0])
                self.rooms.add(args[1])
            elif pred == "goal_in":
                self.goal_in_fact_ids[(args[0], args[1])] = f.global_id
                self.pkg_goal_room[args[0]] = args[1]
            elif pred == "rloc":
                self.rloc_fact_ids[args[0]] = f.global_id
                self.rooms.add(args[0])
            elif pred == "holding":
                self.holding_fact_ids[args[0]] = f.global_id
            elif pred == "armempty":
                self.armempty_id = f.global_id
            elif pred == "closed":
                self.closed_fact_ids[args[0]] = f.global_id
            elif pred == "door":
                r1, r2, d = args[0], args[1], args[2]
                self.rooms.add(r1)
                self.rooms.add(r2)
                if r1 not in self.room_adj:
                    self.room_adj[r1] = []
                self.room_adj[r1].append((r2, d))
                self.door_facts[(r1, r2)] = d
        
        # Precompute all-pairs shortest distances using BFS (ignoring closed doors = admissible)
        for room in self.rooms:
            self.dist_matrix[room] = self._bfs_distances(room)
        
        # Build a quick lookup: for each package, which rooms could it be in?
        self.pkg_in_rooms = {}
        for (pkg, room), gid in self.in_fact_ids.items():
            if pkg not in self.pkg_in_rooms:
                self.pkg_in_rooms[pkg] = []
            self.pkg_in_rooms[pkg].append((room, gid))
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_location(self, state):
        for room, gid in self.rloc_fact_ids.items():
            if (state >> gid) & 1:
                return room
        return None

    def _get_pkg_location(self, state, pkg):
        """Return current room of package, or None if being held."""
        if pkg in self.pkg_in_rooms:
            for room, gid in self.pkg_in_rooms[pkg]:
                if (state >> gid) & 1:
                    return room
        return None

    def _compute(self, state, task_network):
        # Find misplaced packages
        misplaced = []
        held_pkg = None
        
        # Check if robot is holding something
        for pkg, gid in self.holding_fact_ids.items():
            if (state >> gid) & 1:
                held_pkg = pkg
                break
        
        robot_loc = self._get_robot_location(state)
        
        for pkg, goal_room in self.pkg_goal_room.items():
            # Check if package is already at goal
            goal_key = (pkg, goal_room)
            if goal_key in self.in_fact_ids:
                gid = self.in_fact_ids[goal_key]
                if (state >> gid) & 1:
                    continue  # Already at goal
            
            # Package is misplaced (either in wrong room or being held)
            current_room = self._get_pkg_location(state, pkg)
            misplaced.append((pkg, current_room, goal_room))
        
        if not misplaced:
            # Check task network - if there are still tasks, return small value
            if task_network:
                return len(task_network)
            return 0
        
        num_misplaced = len(misplaced)
        
        # T1: Each misplaced package needs at least pickup + putdown = 2 actions
        # But if one is currently held, it only needs putdown (+ possibly travel)
        t1 = num_misplaced * 2
        if held_pkg is not None:
            # The held package only needs putdown (1 action) not pickup+putdown
            for pkg, src, dst in misplaced:
                if pkg == held_pkg:
                    t1 -= 1  # only putdown needed, not pickup
                    break
        
        # T2: Travel distance lower bound
        # For each misplaced package, robot must travel from source to dest carrying it.
        # This is a lower bound on total move actions.
        # We compute: for currently held package, distance from robot to goal.
        # For other packages: distance from their current room to their goal room.
        # Plus the robot must visit each source room.
        # 
        # Admissible approximation: sum of (source->dest) distances for all misplaced,
        # which would be exact if robot could teleport between packages.
        # Better: greedy nearest-neighbor TSP approximation.
        
        travel_cost = 0
        
        if robot_loc is None:
            # Shouldn't happen, but be safe
            t2 = 0
        else:
            # For held package: just need to go to goal
            held_travel = 0
            non_held_misplaced = []
            
            for pkg, src, dst in misplaced:
                if pkg == held_pkg:
                    # Robot is holding this, needs to travel to dst
                    if robot_loc in self.dist_matrix and dst in self.dist_matrix[robot_loc]:
                        held_travel = self.dist_matrix[robot_loc].get(dst, 0)
                    else:
                        held_travel = 0
                else:
                    non_held_misplaced.append((pkg, src, dst))
            
            if not non_held_misplaced:
                t2 = held_travel
            else:
                # Greedy nearest-neighbor from robot's current position
                # For each non-held misplaced package, we need:
                #   travel to src + pickup + travel src->dst + putdown
                # The travel component: visit src, then travel to dst
                # After delivering, robot is at dst, then goes to next src
                
                # Use greedy nearest-neighbor TSP for ordering
                current_pos = robot_loc
                remaining = list(non_held_misplaced)
                total_travel = held_travel  # first deliver held package if any
                
                if held_pkg is not None and held_travel > 0:
                    # After delivering held package, robot is at held pkg's goal
                    for pkg, src, dst in misplaced:
                        if pkg == held_pkg:
                            current_pos = dst
                            break
                
                # Now greedily pick nearest remaining package
                while remaining:
                    best_idx = -1
                    best_cost = float('inf')
                    
                    for i, (pkg, src, dst) in enumerate(remaining):
                        if src is None:
                            # Being held by someone else? shouldn't happen
                            cost_to_src = 0
                        elif current_pos in self.dist_matrix and src in self.dist_matrix[current_pos]:
                            cost_to_src = self.dist_matrix[current_pos].get(src, 50)
                        else:
                            cost_to_src = 50  # large default
                        
                        if src is not None and src in self.dist_matrix and dst in self.dist_matrix.get(src, {}):
                            cost_src_dst = self.dist_matrix[src].get(dst, 0)
                        else:
                            cost_src_dst = 0
                        
                        total_pkg_cost = cost_to_src + cost_src_dst
                        if total_pkg_cost < best_cost:
                            best_cost = total_pkg_cost
                            best_idx = i
                    
                    if best_idx >= 0:
                        pkg, src, dst = remaining[best_idx]
                        remaining.pop(best_idx)
                        total_travel += best_cost
                        current_pos = dst
                    else:
                        break
                
                # The greedy solution is an upper bound on actual travel.
                # For admissibility, use the minimum per-package travel as lower bound.
                # sum of max(1, dist(src, dst)) for each package is admissible
                min_travel = 0
                for pkg, src, dst in non_held_misplaced:
                    if src is not None and src in self.dist_matrix:
                        d = self.dist_matrix[src].get(dst, 1)
                    else:
                        d = 1
                    if src != dst:
                        min_travel += d
                
                min_travel += held_travel
                
                # Also need robot to reach the first package
                if non_held_misplaced:
                    min_reach = float('inf')
                    start = robot_loc
                    if held_pkg is not None:
                        for pkg, src, dst in misplaced:
                            if pkg == held_pkg:
                                start = dst
                                break
                    for pkg, src, dst in non_held_misplaced:
                        if src is not None and start in self.dist_matrix:
                            d = self.dist_matrix[start].get(src, 50)
                            min_reach = min(min_reach, d)
                    if min_reach < float('inf'):
                        min_travel += min_reach
                
                t2 = min_travel
        
        # T3: Count closed doors on shortest paths between critical rooms
        # Each closed door that must be opened costs 1 action
        # This is hard to compute exactly, so we just count doors on BFS paths
        # that need opening. Simple version: count closed doors the robot will
        # likely need to pass through.
        t3_closed = 0
        needed_doors = set()
        if robot_loc is not None:
            # For each misplaced package, find doors on shortest path from src to dst
            for pkg, src, dst in misplaced:
                if pkg == held_pkg:
                    # Path from robot_loc to dst
                    self._find_path_doors(robot_loc, dst, needed_doors)
                elif src is not None:
                    self._find_path_doors(src, dst, needed_doors)
        
        for door in needed_doors:
            if door in self.closed_fact_ids:
                gid = self.closed_fact_ids[door]
                if (state >> gid) & 1:
                    t3_closed += 1
        
        # T4: Task network based lower bound
        t4 = 0
        if task_network:
            for task in task_network:
                task_name = task.name if hasattr(task, 'name') else ''
                if 'pickup' in task_name and '__' not in task_name:
                    t4 += 1
                elif 'putdown' in task_name and '__' not in task_name:
                    t4 += 1
                elif 'move' in task_name and '__' not in task_name:
                    t4 += 1
                elif 'open' in task_name and '__' not in task_name:
                    t4 += 1
                elif 'achieve-goals' in task_name or 'release' in task_name:
                    # These are abstract tasks that will decompose
                    pass
        
        # Combine: max of independent lower bounds
        h = max(t1 + t3_closed, t2 + t1, t4)
        
        return h

    def _find_path_doors(self, src, dst, needed_doors):
        """BFS to find shortest path and record doors used."""
        if src == dst:
            return
        if src not in self.room_adj:
            return
        
        visited = {src: None}  # room -> (prev_room, door)
        queue = deque([src])
        found = False
        
        while queue:
            room = queue.popleft()
            if room == dst:
                found = True
                break
            if room in self.room_adj:
                for neighbor, door in self.room_adj[room]:
                    if neighbor not in visited:
                        visited[neighbor] = (room, door)
                        queue.append(neighbor)
        
        if found:
            # Trace back path
            current = dst
            while visited[current] is not None:
                prev_room, door = visited[current]
                needed_doors.add(door)
                current = prev_room

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
