# Robot Domain Heuristic
# Bottleneck: Single-carry constraint - robot can hold only one package at a time,
# serializing all deliveries. Each misplaced package needs pickup + travel + putdown.
#
# Admissible lower bounds:
# T1: For each misplaced package, at least 2 actions (pickup + putdown). Sum over all misplaced.
# T2: For each misplaced package, shortest path from package's current room to goal room.
#     Sum of these gives a lower bound on total navigation actions (each must be done separately).
# T3: Task network depth - minimum remaining decomposition steps.
#
# We precompute BFS shortest distances between all room pairs in initialize().
# In __call__, we do O(num_packages) bit tests + lookups.

from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotAnthropicOpus46Heuristic14(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_14"):
        super().__init__(name)
        self.in_fact_ids = {}      # (package, room) -> global_id
        self.goal_in_facts = {}    # (package, room) -> global_id
        self.rloc_fact_ids = {}    # room -> global_id
        self.holding_fact_ids = {} # package -> global_id
        self.armempty_id = None
        self.door_edges = {}       # (room1, room2) -> door_name
        self.closed_fact_ids = {}  # door_name -> global_id
        self.rooms = set()
        self.packages = set()
        self.package_goals = {}    # package -> goal_room
        self.dist = {}             # (room1, room2) -> shortest distance (ignoring closed doors)
        self.dist_open = {}        # (room1, room2) -> min doors that must be opened
        self.adj = {}              # room -> list of (neighbor, door)

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

    def _bfs_distances(self):
        """Precompute shortest distances between all room pairs.
        We compute two metrics:
        - dist: number of move actions (treating all edges as traversable)
        - We don't need door-aware distances separately; instead we add
          the minimum number of closed doors on shortest paths as a separate term.
        """
        room_list = list(self.rooms)
        # BFS from each room
        for src in room_list:
            self.dist[src] = {}
            self.dist[src][src] = 0
            queue = deque([src])
            while queue:
                curr = queue.popleft()
                for neighbor in self.adj.get(curr, []):
                    if neighbor not in self.dist[src]:
                        self.dist[src][neighbor] = self.dist[src][curr] + 1
                        queue.append(neighbor)

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model

        # Parse all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "goal_in" and len(args) == 2:
                self.goal_in_facts[(args[0], args[1])] = f.global_id
                self.packages.add(args[0])
                self.package_goals[args[0]] = args[1]
                self.rooms.add(args[1])
            elif pred == "rloc" and len(args) == 1:
                self.rloc_fact_ids[args[0]] = f.global_id
                self.rooms.add(args[0])
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_ids[args[0]] = f.global_id
            elif pred == "armempty" and len(args) == 0:
                self.armempty_id = f.global_id
            elif pred == "door" and len(args) == 3:
                room1, room2, door = args[0], args[1], args[2]
                self.rooms.add(room1)
                self.rooms.add(room2)
                self.door_edges[(room1, room2)] = door
                if room1 not in self.adj:
                    self.adj[room1] = []
                self.adj[room1].append(room2)
            elif pred == "closed" and len(args) == 1:
                self.closed_fact_ids[args[0]] = f.global_id

        # Build room-to-fact mapping for 'in' predicates per package
        self.package_room_facts = {}  # package -> list of (room, global_id)
        for (pkg, room), gid in self.in_fact_ids.items():
            if pkg not in self.package_room_facts:
                self.package_room_facts[pkg] = []
            self.package_room_facts[pkg].append((room, gid))

        # Precompute BFS distances
        self._bfs_distances()

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_location(self, state):
        for room, gid in self.rloc_fact_ids.items():
            if (state >> gid) & 1:
                return room
        return None

    def _get_package_location(self, state, package):
        """Return the room where the package currently is, or None if being held."""
        if package in self.package_room_facts:
            for room, gid in self.package_room_facts[package]:
                if (state >> gid) & 1:
                    return room
        return None  # being held

    def _room_dist(self, r1, r2):
        if r1 == r2:
            return 0
        if r1 in self.dist and r2 in self.dist[r1]:
            return self.dist[r1][r2]
        return 999  # unreachable (shouldn't happen in valid problems)

    def _compute(self, state, task_network):
        # Find misplaced packages
        misplaced = []
        held_package = None
        held_package_goal = None

        # Check if robot is holding something
        for pkg, gid in self.holding_fact_ids.items():
            if (state >> gid) & 1:
                held_package = pkg
                held_package_goal = self.package_goals.get(pkg)
                break

        robot_loc = self._get_robot_location(state)

        for pkg, goal_room in self.package_goals.items():
            if pkg == held_package:
                # Package is being held, needs to be delivered to goal_room
                continue
            goal_fact_key = (pkg, goal_room)
            in_gid = self.in_fact_ids.get(goal_fact_key)
            if in_gid is not None and (state >> in_gid) & 1:
                # Already at goal
                continue
            # Package is misplaced
            current_room = self._get_package_location(state, pkg)
            if current_room is not None:
                misplaced.append((pkg, current_room, goal_room))

        # T1: Each misplaced package needs at least pickup + putdown = 2 actions
        # Plus the held package needs at least putdown = 1 action (if not at goal)
        action_count = len(misplaced) * 2
        if held_package is not None:
            action_count += 1  # putdown

        # T2: Navigation lower bound
        # For held package: robot must travel to goal room (putdown there)
        # For each misplaced package: travel from pkg location to goal = at least dist moves
        # Plus robot must visit each source and each destination
        # 
        # Better: compute a greedy TSP-like lower bound.
        # Admissible lower bound: sum of (source-to-dest distances) for each misplaced pkg
        # This is admissible because each package delivery requires at least that many moves,
        # and the robot can only do one at a time.
        # However, between deliveries, the robot travels from last dest to next source.
        # The sum of source-to-dest distances alone is a valid lower bound.

        nav_cost = 0
        if held_package is not None and held_package_goal is not None and robot_loc is not None:
            nav_cost += self._room_dist(robot_loc, held_package_goal)

        for pkg, src, dst in misplaced:
            nav_cost += self._room_dist(src, dst)

        # T3: Minimum travel between deliveries (inter-delivery travel)
        # We need to get to each package's source room. A lower bound on the
        # additional navigation is: for N misplaced packages, the robot needs
        # to travel to each source. The cheapest additional cost is at least:
        # If currently holding, after putdown at held_goal, need to reach first source.
        # This is complex to compute exactly, so we approximate:
        # If there are misplaced packages, robot must reach at least one source.
        # Minimum distance from robot's effective position to nearest source.
        min_to_nearest_src = 0
        if len(misplaced) > 0 and robot_loc is not None:
            effective_loc = robot_loc
            if held_package is not None and held_package_goal is not None:
                effective_loc = held_package_goal  # after delivering held package
            min_dist_to_src = min(self._room_dist(effective_loc, src) for _, src, _ in misplaced)
            min_to_nearest_src = min_dist_to_src

        # T4: Count closed doors on shortest paths that must be opened
        # This is expensive to compute exactly; approximate by checking if
        # any required door is closed.
        # Simple version: count unique closed doors that lie on direct connections
        # between rooms the robot needs to traverse.
        # For efficiency, skip this detailed computation and use a simpler bound.

        # T5: Task network based lower bound
        # Count primitive tasks and estimate abstract task costs
        tn_cost = 0
        if task_network is not None:
            for task_id in task_network:
                component = self.model.get_component(task_id)
                if hasattr(component, 'cost'):  # Operator
                    tn_cost += 1
                elif hasattr(component, 'decompositions'):  # AbstractTask
                    # Each abstract task will produce at least one action
                    name = component.name
                    if 'achieve-goals' in name or 'release' in name:
                        # These might be finished with 0 actions
                        pass
                    elif 'pickup_abstract' in name or 'putdown_abstract' in name:
                        tn_cost += 1
                    elif 'move_abstract' in name or 'open_abstract' in name:
                        tn_cost += 1

        # Combine terms
        # term1: action count (pickup/putdown) + navigation
        term1 = action_count + nav_cost + min_to_nearest_src

        # term2: task network estimate
        term2 = tn_cost

        return max(term1, term2)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
