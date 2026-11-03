# Robot Domain Heuristic
# Bottleneck: Single-carry constraint. Robot holds at most one package, serializing all deliveries.
#
# Admissible lower bounds:
# T1: Sum of displaced packages * 2 (each needs at least pickup + putdown)
#     This is admissible because each misplaced package needs at minimum these 2 actions.
# T2: For displaced packages, estimate minimum travel cost using BFS distances.
#     We compute a TSP-like lower bound: the robot must visit each source and destination.
#     A simple admissible bound: sum over displaced packages of (dist(src, dst)) + 
#     min over displaced packages of dist(robot_loc, src). Each package needs at least
#     dist(src,dst) moves to transport it.
# T3: Number of closed doors on critical shortest paths that must be opened.
#
# We return max of several combined admissible estimates.

from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotAnthropicOpus46Heuristic13(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_13"):
        super().__init__(name)
        self.rloc_fact_ids = {}    # room_name -> fact global_id for rloc[room]
        self.in_fact_ids = {}      # (pkg, room) -> fact global_id
        self.goal_in_ids = {}      # (pkg, room) -> fact global_id
        self.holding_ids = {}      # pkg -> fact global_id
        self.armempty_id = None
        self.closed_ids = {}       # door_name -> fact global_id
        self.door_info = {}        # (room1, room2) -> door_name
        self.rooms = set()
        self.packages = set()
        self.doors = set()
        self.room_adj = {}         # room -> set of (neighbor_room, door_name)
        self.bfs_dist = {}         # (room1, room2) -> shortest distance ignoring door states
        self.bfs_dist_doors = {}   # (room1, room2) -> (distance, set of doors on shortest path)
        self.goal_room = {}        # pkg -> goal_room (from goal_in predicates)
        self.pkg_goals = []        # list of (pkg, goal_room) pairs
        # For fast lookup during __call__
        self.in_facts_by_pkg = {}  # pkg -> list of (room, global_id)
        self.room_list = []
        self.room_to_idx = {}
        self.dist_matrix = None    # room_idx x room_idx -> int distance

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

    def _bfs_all_pairs(self):
        """Compute shortest distances between all pairs of rooms ignoring door states."""
        n = len(self.room_list)
        # Initialize distance matrix with infinity
        INF = 999999
        self.dist_matrix = [[INF] * n for _ in range(n)]
        for i in range(n):
            self.dist_matrix[i][i] = 0

        for start_idx in range(n):
            start_room = self.room_list[start_idx]
            # BFS from start_room
            queue = deque()
            queue.append((start_room, 0))
            visited = {start_room: 0}
            while queue:
                room, dist = queue.popleft()
                room_idx = self.room_to_idx[room]
                self.dist_matrix[start_idx][room_idx] = dist
                if room in self.room_adj:
                    for neighbor, door_name in self.room_adj[room]:
                        if neighbor not in visited:
                            visited[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

    def _bfs_with_doors(self, start_room):
        """BFS from start_room, counting closed doors on the path.
        Returns dict: room -> (distance, num_closed_doors_on_shortest_path)
        """
        # We do a BFS where cost = number of moves, and we also track closed doors needed
        result = {}
        queue = deque()
        queue.append((start_room, 0, 0))  # room, dist, closed_doors_count
        visited = {start_room}
        result[start_room] = (0, 0)
        while queue:
            room, dist, cd = queue.popleft()
            if room in self.room_adj:
                for neighbor, door_name in self.room_adj[room]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        # Check if this door is initially closed - we'll check during call
                        # For precomputation, just store the path info
                        result[neighbor] = (dist + 1, cd)  # simplified
                        queue.append((neighbor, dist + 1, cd))
        return result

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model

        # Parse all facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                self.rloc_fact_ids[args[0]] = fact.global_id
                self.rooms.add(args[0])
            elif pred == "in":
                self.in_fact_ids[(args[0], args[1])] = fact.global_id
                self.packages.add(args[0])
                self.rooms.add(args[1])
                if args[0] not in self.in_facts_by_pkg:
                    self.in_facts_by_pkg[args[0]] = []
                self.in_facts_by_pkg[args[0]].append((args[1], fact.global_id))
            elif pred == "goal_in":
                self.goal_in_ids[(args[0], args[1])] = fact.global_id
                self.goal_room[args[0]] = args[1]
                self.packages.add(args[0])
                self.rooms.add(args[1])
            elif pred == "holding":
                self.holding_ids[args[0]] = fact.global_id
                self.packages.add(args[0])
            elif pred == "armempty":
                self.armempty_id = fact.global_id
            elif pred == "closed":
                self.closed_ids[args[0]] = fact.global_id
                self.doors.add(args[0])
            elif pred == "door":
                self.rooms.add(args[0])
                self.rooms.add(args[1])
                self.doors.add(args[2])
                key = (args[0], args[1])
                self.door_info[key] = args[2]
                if args[0] not in self.room_adj:
                    self.room_adj[args[0]] = []
                self.room_adj[args[0]].append((args[1], args[2]))

        # Build room list and index
        self.room_list = sorted(self.rooms)
        self.room_to_idx = {r: i for i, r in enumerate(self.room_list)}

        # Compute all-pairs shortest distances
        self._bfs_all_pairs()

        # Build goal packages list
        self.pkg_goals = [(pkg, room) for (pkg, room) in self.goal_room.items()]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_room_dist(self, r1, r2):
        """Get precomputed distance between two rooms."""
        if r1 not in self.room_to_idx or r2 not in self.room_to_idx:
            return 999999
        return self.dist_matrix[self.room_to_idx[r1]][self.room_to_idx[r2]]

    def _get_robot_location(self, state):
        """Find current robot location from state."""
        for room, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1:
                return room
        return None

    def _get_pkg_location(self, state, pkg):
        """Find where a package currently is. Returns room name or None if being held."""
        if pkg in self.in_facts_by_pkg:
            for room, fid in self.in_facts_by_pkg[pkg]:
                if (state >> fid) & 1:
                    return room
        return None  # Being held

    def _is_holding(self, state, pkg):
        """Check if robot is holding a specific package."""
        if pkg in self.holding_ids:
            return (state >> self.holding_ids[pkg]) & 1
        return False

    def _is_arm_empty(self, state):
        """Check if arm is empty."""
        if self.armempty_id is not None:
            return (state >> self.armempty_id) & 1
        return True

    def _is_door_closed(self, state, door_name):
        """Check if a door is closed."""
        if door_name in self.closed_ids:
            return (state >> self.closed_ids[door_name]) & 1
        return False

    def _compute(self, state, task_network):
        robot_loc = self._get_robot_location(state)
        if robot_loc is None:
            return 0

        # Find all displaced packages
        displaced = []  # (pkg, current_room_or_None_if_held, goal_room)
        held_pkg = None
        held_goal = None

        for pkg, goal_room in self.pkg_goals:
            # Check if package is already at goal
            goal_key = (pkg, goal_room)
            if goal_key in self.in_fact_ids:
                fid = self.in_fact_ids[goal_key]
                if (state >> fid) & 1:
                    continue  # Already at goal

            # Package is displaced
            if self._is_holding(state, pkg):
                held_pkg = pkg
                held_goal = goal_room
            else:
                current_room = self._get_pkg_location(state, pkg)
                if current_room is not None:
                    displaced.append((pkg, current_room, goal_room))

        num_displaced = len(displaced) + (1 if held_pkg else 0)

        if num_displaced == 0:
            # Check if task network is empty
            if len(task_network) == 0:
                return 0
            # Might still have tasks to do
            return len(task_network)

        # ---- Term 1: Action count lower bound ----
        # Each displaced package needs at least: pickup(1) + putdown(1) = 2 actions
        # Plus travel from source to dest: dist(src, dst) moves
        # Plus if doors are closed on the path, open actions
        # The held package needs: travel to goal + putdown = dist(robot, goal) + 1

        total_transport_cost = 0
        min_dist_to_src = 999999

        if held_pkg is not None:
            # Robot is holding a package, needs to travel to goal and put down
            dist_to_goal = self._get_room_dist(robot_loc, held_goal)
            total_transport_cost += dist_to_goal + 1  # moves + putdown
            # After putting down, robot is at held_goal
            effective_robot_loc_after_held = held_goal
        else:
            effective_robot_loc_after_held = robot_loc

        for pkg, src, dst in displaced:
            # Transport cost: dist(src, dst) moves + pickup + putdown = dist + 2
            transport_dist = self._get_room_dist(src, dst)
            total_transport_cost += transport_dist + 2  # pickup + moves + putdown

            # Track minimum distance from robot to any source
            d = self._get_room_dist(robot_loc, src)
            if d < min_dist_to_src:
                min_dist_to_src = d

        # The robot needs to reach at least the first package source
        if len(displaced) > 0:
            if held_pkg is not None:
                # After delivering held package, need to reach nearest displaced source
                min_from_held_goal = 999999
                for pkg, src, dst in displaced:
                    d = self._get_room_dist(held_goal, src)
                    if d < min_from_held_goal:
                        min_from_held_goal = d
                travel_to_first = min(min_dist_to_src, min_from_held_goal)
            else:
                travel_to_first = min_dist_to_src
            total_transport_cost += travel_to_first

        # ---- Term 2: Task network based estimate ----
        # Count remaining abstract tasks as a simple lower bound
        tn_cost = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):
                # Abstract task - at minimum contributes 0 actions (finished method)
                # But if it's achieve-goals or release, it needs work proportional to displaced
                tn_cost += 0  # We rely on Term 1 for the main estimate
            else:
                # Primitive operator
                tn_cost += 1

        # ---- Term 3: Closed doors that must be opened ----
        # For each displaced package, check if doors on shortest path are closed
        # This is expensive to compute exactly, so we use a simpler bound:
        # Count unique closed doors that lie on at least one critical shortest path
        # For simplicity, we just count the minimum doors that need opening for each package
        closed_door_cost = 0
        # For each displaced package, find doors on shortest path that are closed
        # Use BFS with door tracking would be ideal but expensive
        # Simple approximation: for each displaced package's route, check direct door if adjacent
        if held_pkg is not None and self._get_room_dist(robot_loc, held_goal) == 1:
            door_key = (robot_loc, held_goal)
            if door_key in self.door_info:
                door_name = self.door_info[door_key]
                if self._is_door_closed(state, door_name):
                    closed_door_cost += 1

        for pkg, src, dst in displaced:
            d = self._get_room_dist(src, dst)
            if d == 1:
                door_key = (src, dst)
                if door_key in self.door_info:
                    door_name = self.door_info[door_key]
                    if self._is_door_closed(state, door_name):
                        closed_door_cost += 1

        # ---- Term 4: Serialization bound ----
        # Since robot carries one package at a time, if N packages are displaced
        # and not held, the robot must do N pickup-travel-putdown cycles.
        # Between cycles, robot must travel from last putdown location to next pickup.
        # A lower bound on inter-delivery travel: use nearest-neighbor from current pos
        # through all (src -> dst) pairs. But that's expensive.
        # Simple bound: at least (N-1) * min_inter_delivery_distance
        # where min_inter_delivery_distance = min over all (dst_i, src_j) pairs
        if len(displaced) > 1:
            min_inter = 999999
            for i in range(len(displaced)):
                dst_i = displaced[i][2]
                for j in range(len(displaced)):
                    if i != j:
                        src_j = displaced[j][1]
                        d = self._get_room_dist(dst_i, src_j)
                        if d < min_inter:
                            min_inter = d
            if min_inter < 999999:
                # We can't just multiply by (N-1) as that double counts
                # But we can add at least (N-1) * min_inter as a lower bound
                # on the transitions between deliveries
                # Actually this overcounts. A better bound:
                # total cost >= sum(transport) + travel_to_first + (N-1)*min_inter
                # But sum(transport) already includes each package's own transport.
                # The inter-delivery travel is additional.
                # However, to keep admissibility, we should be careful.
                # Let's just use it as a separate term.
                serialization_cost = total_transport_cost  # already includes travel_to_first
                # Don't add inter-delivery here as total_transport_cost is already good
            else:
                serialization_cost = total_transport_cost
        else:
            serialization_cost = total_transport_cost

        h = max(total_transport_cost + closed_door_cost, tn_cost, serialization_cost)
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
