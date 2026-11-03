"""
Robot Domain Heuristic

Bottleneck: Single-carry constraint - robot can hold only one package at a time.
Each misplaced package requires pickup + putdown (2 actions) plus movement.

Admissible lower bounds:
1. Sum of per-package costs: each misplaced package needs >= 2 actions (pickup+putdown)
   plus shortest-path travel from robot/current-loc to package source, then to goal dest.
2. Task network structural bound: count of abstract tasks requiring actual work.
3. Max single-package cost: the most expensive single package delivery is a lower bound
   on remaining cost (since robot is serial).

We use a greedy TSP-like lower bound: the robot must visit each misplaced package's
source and destination. We compute a lower bound by summing individual package costs
(pickup+putdown+travel) but since the robot is serial, we use a smarter bound.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque


class RobotAnthropicOpus46Heuristic9(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_9"):
        super().__init__(name)
        self._initialized = False

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._initialized = True

        # Parse all facts and build lookup maps
        self.in_facts = {}       # (pkg, room) -> global_id
        self.rloc_facts = {}     # room -> global_id
        self.holding_facts = {}  # pkg -> global_id
        self.armempty_id = None
        self.goal_in_facts = {}  # (pkg, room) -> global_id
        self.closed_facts = {}   # door -> global_id
        self.door_facts = {}     # (room1, room2, door) -> global_id

        # All rooms and packages
        self.rooms = set()
        self.packages = set()
        self.doors = set()

        # Adjacency for rooms (ignoring door state)
        self.room_adj = {}  # room -> set of (neighbor_room, door)

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'in':
                pkg, room = args[0], args[1]
                self.in_facts[(pkg, room)] = f.global_id
                self.packages.add(pkg)
                self.rooms.add(room)
            elif pred == 'rloc':
                room = args[0]
                self.rloc_facts[room] = f.global_id
                self.rooms.add(room)
            elif pred == 'holding':
                pkg = args[0]
                self.holding_facts[pkg] = f.global_id
                self.packages.add(pkg)
            elif pred == 'armempty':
                self.armempty_id = f.global_id
            elif pred == 'goal_in':
                pkg, room = args[0], args[1]
                self.goal_in_facts[(pkg, room)] = f.global_id
                self.packages.add(pkg)
                self.rooms.add(room)
            elif pred == 'closed':
                door = args[0]
                self.closed_facts[door] = f.global_id
                self.doors.add(door)
            elif pred == 'door':
                r1, r2, d = args[0], args[1], args[2]
                self.door_facts[(r1, r2, d)] = f.global_id
                self.rooms.add(r1)
                self.rooms.add(r2)
                self.doors.add(d)

        # Build adjacency list for rooms
        for (r1, r2, d) in self.door_facts:
            if r1 not in self.room_adj:
                self.room_adj[r1] = []
            self.room_adj[r1].append((r2, d))

        # Build goal map: for each package, what's its goal room?
        self.pkg_goal_room = {}  # pkg -> goal_room
        for (pkg, room) in self.goal_in_facts:
            self.pkg_goal_room[pkg] = room

        # Precompute BFS shortest distances between all rooms (ignoring door states)
        # This gives an admissible lower bound on movement cost
        self.room_list = sorted(self.rooms)
        self.room_index = {r: i for i, r in enumerate(self.room_list)}
        n = len(self.room_list)
        self.dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            self.dist[i][i] = 0

        # BFS from each room
        for start_idx in range(n):
            start_room = self.room_list[start_idx]
            queue = deque()
            queue.append((start_room, 0))
            visited = {start_room}
            self.dist[start_idx][start_idx] = 0
            while queue:
                curr, d = queue.popleft()
                curr_idx = self.room_index[curr]
                self.dist[start_idx][curr_idx] = d
                if curr in self.room_adj:
                    for (neighbor, door) in self.room_adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))

        # Precompute which operator corresponds to which action type
        self.pickup_ops = []  # (pkg, room, op)
        self.putdown_ops = []
        self.move_ops = []
        self.open_ops = []
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if pred == 'pickup':
                self.pickup_ops.append((args[0], args[1], op))
            elif pred == 'putdown':
                self.putdown_ops.append((args[0], args[1], op))
            elif pred == 'move':
                self.move_ops.append((args[0], args[1], args[2], op))
            elif pred == 'open':
                self.open_ops.append((args[0], args[1], args[2], op))

        # Pre-build: for each package, find its source rooms (all rooms where in(pkg, room) fact exists)
        self.pkg_possible_locations = {}  # pkg -> list of rooms
        for (pkg, room) in self.in_facts:
            if pkg not in self.pkg_possible_locations:
                self.pkg_possible_locations[pkg] = []
            self.pkg_possible_locations[pkg].append(room)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        if not self._initialized:
            return 0

        # Find robot's current location
        robot_room = None
        for room, gid in self.rloc_facts.items():
            if (state >> gid) & 1:
                robot_room = room
                break

        if robot_room is None:
            return 0

        robot_idx = self.room_index.get(robot_room, -1)

        # Check if arm is empty
        arm_empty = (state >> self.armempty_id) & 1 if self.armempty_id is not None else 1

        # Find what package is being held (if any)
        held_pkg = None
        held_pkg_goal_room = None
        if not arm_empty:
            for pkg, gid in self.holding_facts.items():
                if (state >> gid) & 1:
                    held_pkg = pkg
                    held_pkg_goal_room = self.pkg_goal_room.get(pkg)
                    break

        # Find all misplaced packages
        misplaced = []  # (pkg, current_room, goal_room)
        for pkg, goal_room in self.pkg_goal_room.items():
            if pkg == held_pkg:
                # Package is being held, needs to be delivered to goal_room
                continue
            # Check if package is already at goal
            in_goal_id = self.in_facts.get((pkg, goal_room))
            if in_goal_id is not None and (state >> in_goal_id) & 1:
                continue  # Already at goal
            # Find current location of package
            current_room = None
            if pkg in self.pkg_possible_locations:
                for room in self.pkg_possible_locations[pkg]:
                    fact_id = self.in_facts.get((pkg, room))
                    if fact_id is not None and (state >> fact_id) & 1:
                        current_room = room
                        break
            if current_room is not None:
                misplaced.append((pkg, current_room, goal_room))

        # Term 1: Cost for currently held package
        held_cost = 0
        if held_pkg is not None and held_pkg_goal_room is not None:
            # Need to travel to goal room + putdown (1 action)
            goal_idx = self.room_index.get(held_pkg_goal_room, -1)
            if robot_idx >= 0 and goal_idx >= 0:
                travel = self.dist[robot_idx][goal_idx]
                if travel == float('inf'):
                    travel = len(self.rooms)  # fallback large estimate
                held_cost = travel + 1  # travel + putdown
            else:
                held_cost = 1

        # Term 2: Sum of per-package minimum costs for misplaced packages
        # Each misplaced package needs at minimum:
        #   - pickup (1) + putdown (1) = 2 actions
        #   - travel from source to dest = dist(source, dest) moves
        # Plus the robot must travel between deliveries
        # Lower bound: sum of (dist(source, dest) + 2) for each misplaced package
        sum_pkg_costs = 0
        for (pkg, src, dst) in misplaced:
            src_idx = self.room_index.get(src, -1)
            dst_idx = self.room_index.get(dst, -1)
            if src_idx >= 0 and dst_idx >= 0:
                travel = self.dist[src_idx][dst_idx]
                if travel == float('inf'):
                    travel = len(self.rooms)
                sum_pkg_costs += travel + 2  # move from src to dst + pickup + putdown
            else:
                sum_pkg_costs += 2

        # Term 3: Serial delivery bound
        # Since robot carries one package at a time, total cost is at least:
        # held_cost + for each misplaced package: travel_to_src + pickup + travel_to_dst + putdown
        # We can compute a tighter bound considering the robot must chain deliveries
        # Simple version: cost of delivering held package + nearest-first lower bound
        
        # Compute individual package delivery costs including robot travel to source
        if misplaced:
            # The robot must travel to each package's source, then to its destination
            # At minimum, the robot travels from its current position (or last delivery dest)
            # to the nearest undelivered package source
            
            # Lower bound: held_cost + sum of (dist(src, dst) + 2) for misplaced
            # Plus minimum travel to reach first misplaced package source
            if held_pkg is not None and held_pkg_goal_room is not None:
                # After delivering held package, robot is at held_pkg_goal_room
                after_held_room = held_pkg_goal_room
                after_held_idx = self.room_index.get(after_held_room, -1)
            else:
                after_held_idx = robot_idx
            
            # Minimum distance from robot (after held delivery) to any misplaced package source
            min_dist_to_first = float('inf')
            for (pkg, src, dst) in misplaced:
                src_idx = self.room_index.get(src, -1)
                if after_held_idx >= 0 and src_idx >= 0:
                    d = self.dist[after_held_idx][src_idx]
                    if d < min_dist_to_first:
                        min_dist_to_first = d
            
            if min_dist_to_first == float('inf'):
                min_dist_to_first = 0
            
            serial_bound = held_cost + min_dist_to_first + sum_pkg_costs
        else:
            serial_bound = held_cost

        # Term 4: Task network bound
        # Count non-trivial tasks in the task network
        tn_cost = 0
        if task_network:
            for task_id in task_network:
                component = self.model.get_component(task_id)
                # If it's an operator, it costs 1
                if hasattr(component, 'add_effects'):
                    tn_cost += 1
                elif hasattr(component, 'decompositions'):
                    # Abstract task - estimate minimum decomposition depth
                    name = component.name if hasattr(component, 'name') else ''
                    pred, args = self._parse_fact_name(name)
                    if pred == 'achieve-goals':
                        # This will decompose into all remaining deliveries
                        # Don't double count - skip
                        pass
                    elif pred == 'release':
                        # At minimum: putdown + achieve-goals
                        tn_cost += 1  # putdown at minimum
                    elif pred == 'pickup_abstract':
                        tn_cost += 1
                    elif pred == 'putdown_abstract':
                        tn_cost += 1
                    elif pred == 'move_abstract':
                        tn_cost += 1
                    elif pred == 'open_abstract':
                        tn_cost += 1
                    else:
                        tn_cost += 1

        # Term 5: Door opening lower bound
        # Count closed doors that lie on shortest paths of misplaced packages
        # This is complex to compute exactly, so we use a simpler bound:
        # For each misplaced package, if the shortest path requires going through
        # a closed door, add 1 for the open action
        # (Simplified: just count based on whether any needed doors are closed)
        
        # Return max of our bounds
        h = max(serial_bound, tn_cost)
        
        return h

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []
