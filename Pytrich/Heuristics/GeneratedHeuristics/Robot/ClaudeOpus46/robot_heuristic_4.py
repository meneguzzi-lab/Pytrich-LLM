# Robot Domain Heuristic
# Bottleneck: Single-carry constraint - robot can hold only one package at a time,
# serializing all deliveries. Each misplaced package needs pickup+putdown (2 actions)
# plus navigation actions between rooms.
#
# Admissible lower bounds:
# T1: Number of misplaced packages * 2 (each needs at least pickup + putdown)
# T2: Sum of shortest-path distances for each misplaced package from its current room
#     to its goal room (lower bound on navigation needed just for deliveries)
# T3: Minimum travel overhead - robot must visit each source and destination room;
#     we estimate a lower bound on total navigation using BFS distances on room graph
#     (ignoring closed doors gives admissible underestimate)

from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotAnthropicOpus46Heuristic4(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_4"):
        super().__init__(name)
        self.in_fact_map = {}      # (pkg, room) -> global_id
        self.rloc_fact_map = {}    # room -> global_id
        self.holding_fact_map = {} # pkg -> global_id
        self.armempty_id = None
        self.closed_fact_map = {}  # door -> global_id
        self.door_fact_map = {}    # (room1, room2) -> door
        self.goal_in_map = {}      # pkg -> goal_room
        self.packages = []
        self.rooms = []
        self.room_adj = {}         # room -> set of (neighbor_room, door)
        self.bfs_dist = {}         # (room1, room2) -> shortest path distance (ignoring door states)
        self.pkg_source_init = {}  # pkg -> initial room (from init state)

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

    def _bfs_from(self, start_room):
        """BFS on room adjacency graph ignoring door states (admissible)."""
        dist = {start_room: 0}
        queue = deque([start_room])
        while queue:
            current = queue.popleft()
            for neighbor, door in self.room_adj.get(current, []):
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
        return dist

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model

        rooms_set = set()
        packages_set = set()

        # Parse all facts and build maps
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "in" and len(args) == 2:
                pkg, room = args[0], args[1]
                self.in_fact_map[(pkg, room)] = fact.global_id
                packages_set.add(pkg)
                rooms_set.add(room)
            elif pred == "rloc" and len(args) == 1:
                self.rloc_fact_map[args[0]] = fact.global_id
                rooms_set.add(args[0])
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_map[args[0]] = fact.global_id
                packages_set.add(args[0])
            elif pred == "armempty" and len(args) == 0:
                self.armempty_id = fact.global_id
            elif pred == "closed" and len(args) == 1:
                self.closed_fact_map[args[0]] = fact.global_id
            elif pred == "door" and len(args) == 3:
                room1, room2, door = args[0], args[1], args[2]
                self.door_fact_map[(room1, room2)] = door
                rooms_set.add(room1)
                rooms_set.add(room2)
            elif pred == "goal_in" and len(args) == 2:
                pkg, room = args[0], args[1]
                self.goal_in_map[pkg] = room
                packages_set.add(pkg)
                rooms_set.add(room)

        self.packages = sorted(packages_set)
        self.rooms = sorted(rooms_set)

        # Build room adjacency graph (both directions from door facts)
        self.room_adj = {room: [] for room in self.rooms}
        seen_edges = set()
        for (room1, room2), door in self.door_fact_map.items():
            edge_key = (room1, room2, door)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                self.room_adj[room1].append((room2, door))

        # Precompute BFS distances from every room
        # For large instances this could be expensive but rooms are typically <= 50
        self.bfs_dist = {}
        for room in self.rooms:
            dists = self._bfs_from(room)
            for dest, d in dists.items():
                self.bfs_dist[(room, dest)] = d

        # Record initial package locations
        state = initial_node.state
        for pkg in self.packages:
            for room in self.rooms:
                key = (pkg, room)
                if key in self.in_fact_map:
                    fid = self.in_fact_map[key]
                    if (state >> fid) & 1:
                        self.pkg_source_init[pkg] = room
                        break

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_location(self, state):
        for room, fid in self.rloc_fact_map.items():
            if (state >> fid) & 1:
                return room
        return None

    def _get_package_location(self, state, pkg):
        """Return the room where package is, or None if being held."""
        for room in self.rooms:
            key = (pkg, room)
            if key in self.in_fact_map:
                fid = self.in_fact_map[key]
                if (state >> fid) & 1:
                    return room
        return None

    def _is_holding(self, state, pkg):
        if pkg in self.holding_fact_map:
            return (state >> self.holding_fact_map[pkg]) & 1
        return False

    def _is_door_closed(self, state, door):
        if door in self.closed_fact_map:
            return (state >> self.closed_fact_map[door]) & 1
        return False

    def _get_dist(self, room1, room2):
        if room1 == room2:
            return 0
        return self.bfs_dist.get((room1, room2), 999)

    def _compute(self, state, task_network):
        robot_loc = self._get_robot_location(state)
        if robot_loc is None:
            return 0

        # Find misplaced packages and their current/goal locations
        misplaced = []  # (pkg, current_room_or_None, goal_room)
        held_pkg = None
        held_goal = None

        for pkg in self.packages:
            if pkg not in self.goal_in_map:
                continue
            goal_room = self.goal_in_map[pkg]

            if self._is_holding(state, pkg):
                held_pkg = pkg
                held_goal = goal_room
                # Held package: needs putdown (1 action) + travel to goal room
                continue

            current_room = self._get_package_location(state, pkg)
            if current_room is None:
                continue

            # Check if already at goal
            key = (pkg, goal_room)
            if key in self.in_fact_map:
                fid = self.in_fact_map[key]
                if (state >> fid) & 1:
                    continue  # Already at goal

            misplaced.append((pkg, current_room, goal_room))

        if not misplaced and held_pkg is None:
            # Check if task network is empty too
            if len(task_network) == 0:
                return 0
            # Tasks remain but no misplaced packages - might be navigation/open tasks
            return len(task_network) // 2  # rough estimate

        # Term 1: Each misplaced package needs at least pickup + putdown = 2 actions
        # Held package needs at least putdown = 1 action
        t1 = len(misplaced) * 2
        if held_pkg is not None:
            t1 += 1  # putdown for held package

        # Term 2: Navigation lower bound for deliveries
        # For each misplaced package: distance(source, goal) is a lower bound on
        # movement actions needed to deliver it (robot must traverse this path while carrying)
        # This is admissible since these distances overlap in reality
        # But we can use max of individual delivery costs as admissible
        delivery_costs = []
        for pkg, src, goal in misplaced:
            # Cost to deliver: go to src (from current pos), pick up (1), 
            # go to goal, put down (1)
            # But for admissibility per package: at minimum need travel(src, goal) + 2
            travel = self._get_dist(src, goal)
            delivery_costs.append(travel + 2)  # pickup + travel + putdown

        if held_pkg is not None:
            # For held package: travel to goal + putdown
            travel = self._get_dist(robot_loc, held_goal)
            delivery_costs.append(travel + 1)

        # Max of individual delivery costs is admissible
        t2_max = max(delivery_costs) if delivery_costs else 0

        # Term 3: Total navigation estimate with serial delivery constraint
        # Since robot carries one package at a time, we need a routing estimate
        # Admissible lower bound: sum of (source_to_goal distances) for all misplaced
        # packages, plus minimum robot travel to reach the nearest source
        # This is admissible because every delivery requires at least source->goal movement
        t3_sum_delivery = 0
        for pkg, src, goal in misplaced:
            t3_sum_delivery += self._get_dist(src, goal)

        # Add pickup + putdown for each misplaced package
        t3 = t3_sum_delivery + len(misplaced) * 2

        if held_pkg is not None:
            t3 += self._get_dist(robot_loc, held_goal) + 1

        # Minimum distance to reach nearest source from current position
        if misplaced:
            if held_pkg is not None:
                # Must first deliver held package, then go to nearest source
                min_start = self._get_dist(robot_loc, held_goal)
                min_to_src = min(self._get_dist(held_goal, src) for _, src, _ in misplaced)
                t3 += min_to_src
            else:
                min_to_src = min(self._get_dist(robot_loc, src) for _, src, _ in misplaced)
                t3 += min_to_src

        # Term 4: Count closed doors on shortest paths for critical deliveries
        # This is harder to compute exactly, so we add a small bonus for closed doors
        # that are on the adjacency of the robot's current location
        closed_door_count = 0
        for door, fid in self.closed_fact_map.items():
            if (state >> fid) & 1:
                closed_door_count += 1
        # Not all closed doors need opening, but each that does costs 1 action
        # We don't add this directly (not admissible as standalone without path analysis)
        # Instead use it as minor tie-breaker
        tie_break = 0.001 * closed_door_count

        h = max(t1, t2_max, t3) + tie_break

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
