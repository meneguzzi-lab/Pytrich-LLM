# Robot Domain Heuristic
# BOTTLENECK: Robot can hold only ONE package at a time, serializing all deliveries.
# Each misplaced package requires: travel to source + pickup + travel to dest + putdown.
#
# Admissible Lower Bounds:
# T1: Sum of (2 + min_distance(src, dst)) for each misplaced package. This is admissible
#     because each misplaced package needs at least pickup(1) + putdown(1) + travel(>=0).
#     Since only one package carried at a time, these costs are serialized (additive).
#     However, the robot can combine trips, so we use a relaxed version.
# T2: Max single-package delivery cost - admissible as a single term.
# T3: Task network depth estimate based on remaining abstract tasks.
#
# We use max(T1_relaxed, T2, T3) where T1_relaxed accounts for shared travel.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque


class RobotAnthropicOpus46Heuristic15(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_15"):
        super().__init__(name)
        self.goal_in_map = {}    # package -> goal_room
        self.in_fact_ids = {}    # (package, room) -> global_id
        self.rloc_fact_ids = {}  # room -> global_id
        self.holding_fact_ids = {}  # package -> global_id
        self.armempty_id = None
        self.closed_fact_ids = {}  # door -> global_id
        self.door_connections = {}  # room -> list of (neighbor_room, door)
        self.room_names = set()
        self.package_names = set()
        self.door_names = set()
        self.bfs_dist = {}  # (room1, room2) -> shortest distance ignoring closed doors
        self.bfs_dist_with_doors = {}  # includes cost of opening doors

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

    def _compute_bfs_distances(self):
        """Compute all-pairs shortest distances between rooms.
        Distance = number of move actions needed (each door opening costs 1 extra)."""
        # BFS ignoring door states (optimistic - all doors open)
        for src in self.room_names:
            dist = {src: 0}
            queue = deque([src])
            while queue:
                current = queue.popleft()
                if current in self.door_connections:
                    for neighbor, door in self.door_connections[current]:
                        if neighbor not in dist:
                            dist[neighbor] = dist[current] + 1
                            queue.append(neighbor)
            for dst in self.room_names:
                if dst in dist:
                    self.bfs_dist[(src, dst)] = dist[dst]
                else:
                    self.bfs_dist[(src, dst)] = 999999  # unreachable

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model

        # Parse all facts and build lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "rloc" and len(args) == 1:
                self.rloc_fact_ids[args[0]] = f.global_id
                self.room_names.add(args[0])
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_ids[args[0]] = f.global_id
                self.package_names.add(args[0])
            elif pred == "armempty" and len(args) == 0:
                self.armempty_id = f.global_id
            elif pred == "closed" and len(args) == 1:
                self.closed_fact_ids[args[0]] = f.global_id
                self.door_names.add(args[0])
            elif pred == "goal_in" and len(args) == 2:
                self.goal_in_map[args[0]] = args[1]
                self.package_names.add(args[0])
                self.room_names.add(args[1])
            elif pred == "door" and len(args) == 3:
                self.room_names.add(args[0])
                self.room_names.add(args[1])
                self.door_names.add(args[2])
                if args[0] not in self.door_connections:
                    self.door_connections[args[0]] = []
                self.door_connections[args[0]].append((args[1], args[2]))

        # Also extract rooms from in_fact_ids
        for (pkg, room) in self.in_fact_ids:
            self.room_names.add(room)
            self.package_names.add(pkg)

        # Compute BFS distances
        self._compute_bfs_distances()

        # Precompute which packages actually need to move
        # (goal_in differs from current location)
        self.packages_with_goals = set(self.goal_in_map.keys())

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_location(self, state):
        """Get current robot location from state."""
        for room, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1:
                return room
        return None

    def _get_package_location(self, state, pkg):
        """Get current location of a package, or None if being held."""
        for (p, room), fid in self.in_fact_ids.items():
            if p == pkg and (state >> fid) & 1:
                return room
        return None

    def _is_holding(self, state, pkg):
        """Check if robot is holding this package."""
        if pkg in self.holding_fact_ids:
            return (state >> self.holding_fact_ids[pkg]) & 1
        return False

    def _compute(self, state, task_network):
        """Compute heuristic value."""
        robot_loc = self._get_robot_location(state)
        if robot_loc is None:
            return 0

        # Find all misplaced packages
        misplaced = []
        held_pkg = None
        held_goal = None

        for pkg in self.packages_with_goals:
            goal_room = self.goal_in_map[pkg]

            # Check if package is being held
            if self._is_holding(state, pkg):
                held_pkg = pkg
                held_goal = goal_room
                continue

            # Check if package is already at goal
            key = (pkg, goal_room)
            if key in self.in_fact_ids:
                fid = self.in_fact_ids[key]
                if (state >> fid) & 1:
                    continue  # Already at goal

            # Package is misplaced - find its current location
            current_room = self._get_package_location(state, pkg)
            if current_room is not None:
                misplaced.append((pkg, current_room, goal_room))

        # Term 1: Cost for held package
        held_cost = 0
        if held_pkg is not None:
            # Need to deliver: travel to goal + putdown
            dist_to_goal = self.bfs_dist.get((robot_loc, held_goal), 999)
            held_cost = dist_to_goal + 1  # travel + putdown

        # Term 2: Sum of minimum delivery costs for misplaced packages
        # Each needs: travel_to_src + pickup + travel_to_dst + putdown
        # = dist(robot_or_prev, src) + 1 + dist(src, dst) + 1
        # Lower bound: just dist(src, dst) + 2 for each (pickup + putdown)
        sum_delivery = 0
        max_single_delivery = 0
        for pkg, src, dst in misplaced:
            travel = self.bfs_dist.get((src, dst), 999)
            # Minimum cost: pickup(1) + travel(src->dst) + putdown(1)
            pkg_cost = 2 + travel
            sum_delivery += pkg_cost
            if pkg_cost > max_single_delivery:
                max_single_delivery = pkg_cost

        # Term 3: Account for robot needing to visit each source
        # TSP-like lower bound: at minimum robot must travel to nearest misplaced pkg source
        min_travel_to_any_src = 0
        if misplaced and held_pkg is None:
            # Robot needs to reach at least one misplaced package
            effective_loc = robot_loc
            min_travel_to_any_src = min(
                self.bfs_dist.get((effective_loc, src), 999)
                for _, src, _ in misplaced
            )

        # Term 4: Count closed doors that MUST be opened
        # This is hard to compute exactly, but we can count doors on shortest paths
        # For now, just count as a minor additive term
        # Each closed door on any required path costs 1 action to open

        # Combine terms
        # T1: held_cost + sum_delivery + min_travel_to_any_src
        #     This is admissible because:
        #     - held package must be delivered (held_cost)
        #     - each misplaced package needs at least its individual cost
        #     - robot must reach at least one source
        #     BUT sum_delivery overcounts shared travel, so we use a relaxed version
        
        # Relaxed admissible bound: since robot carries one at a time,
        # total cost >= held_cost + sum_delivery (each package's delivery is serialized)
        # But robot can pick up a package at the destination of a previous delivery,
        # saving some travel. A safe lower bound is:
        # max(sum_delivery + held_cost, task_network_estimate)

        # Task network term
        tn_cost = 0
        if task_network:
            for task_id in task_network:
                component = self.model.get_component(task_id)
                # Check if it's an operator (primitive)
                if hasattr(component, 'add_effects'):
                    tn_cost += 1
                elif hasattr(component, 'decompositions'):
                    # Abstract task - estimate minimum decomposition depth
                    task_name = component.name
                    if 'achieve-goals' in task_name or 'release' in task_name:
                        # These recursively decompose; cost depends on state
                        pass  # handled by package analysis
                    elif 'pickup_abstract' in task_name or 'putdown_abstract' in task_name:
                        tn_cost += 1
                    elif 'move_abstract' in task_name or 'open_abstract' in task_name:
                        tn_cost += 1

        n_misplaced = len(misplaced)
        
        # Main heuristic: admissible combination
        # Each misplaced package is serialized (one at a time), so sum is admissible
        h1 = held_cost + sum_delivery + min_travel_to_any_src
        
        # Alternative: use task network info
        h2 = tn_cost
        
        # Max of independent lower bounds
        h = max(h1, h2)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
