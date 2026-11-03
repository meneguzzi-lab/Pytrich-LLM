# Robot Domain Heuristic
# Bottleneck: Single arm means all package deliveries are serialized.
# Each misplaced package needs: travel to source, pickup, travel to dest, putdown.
#
# Admissible lower bounds:
# T1: Sum over misplaced packages of (2 + shortest_path(src, dest)). This is a lower
#     bound because each package independently requires pickup+putdown+travel, and
#     the single-arm constraint serializes them (no overlap in action costs).
# T2: Number of remaining abstract tasks that correspond to pickup/putdown/move/open
#     in the task network (loose structural bound).
# T3: If robot is not at a useful location, add minimum distance to nearest relevant room.
#
# We use max(T1, T2) with T1 being the dominant, state-aware term.

from collections import deque

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator as ModelOperator
from Pytrich.model import AbstractTask as ModelAbstractTask


class RobotAnthropicOpus46Heuristic8(Heuristic):
    """Domain-dependent heuristic for the Robot domain.
    
    Computes an admissible lower bound based on:
    - Per-package delivery cost (pickup + putdown + shortest path from current to goal room)
    - Minimum robot travel overhead to reach packages
    - Task network structural lower bound
    """

    def __init__(self, name="robot_anthropicopus46_heuristic_8"):
        super().__init__(name)
        # Fact ID maps
        self.in_fact_ids = {}      # (pkg, room) -> global_id
        self.goal_in_fact_ids = {} # (pkg, room) -> global_id
        self.rloc_fact_ids = {}    # room -> global_id
        self.holding_fact_ids = {} # pkg -> global_id
        self.armempty_id = None
        self.closed_fact_ids = {}  # door -> global_id
        self.door_info = {}        # (room1, room2) -> door_name
        
        # Room graph and shortest paths
        self.rooms = set()
        self.packages = set()
        self.adjacency = {}        # room -> set of (neighbor_room, door)
        self.shortest_paths = {}   # (room1, room2) -> int (shortest distance)
        
        # Goal info
        self.package_goals = {}    # pkg -> goal_room
        
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

    def _build_shortest_paths(self):
        """BFS shortest path between all pairs of rooms."""
        room_list = list(self.rooms)
        # Initialize with infinity
        for r1 in room_list:
            self.shortest_paths[(r1, r1)] = 0
        
        # BFS from each room
        for start in room_list:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                current_dist = visited[current]
                if current in self.adjacency:
                    for neighbor, door in self.adjacency[current]:
                        if neighbor not in visited:
                            # Each move costs 1, but if door is closed we need +1 for open
                            # For admissibility, we use 1 per hop (ignoring door costs
                            # since we account for them separately or they may already be open)
                            visited[neighbor] = current_dist + 1
                            queue.append(neighbor)
            for room, dist in visited.items():
                self.shortest_paths[(start, room)] = dist

    def _get_shortest_path(self, room1, room2):
        """Get shortest path distance between two rooms."""
        return self.shortest_paths.get((room1, room2), 999)

    def _build_shortest_paths_with_doors(self):
        """BFS shortest path accounting for the minimum number of closed doors 
        that must be opened along the way. Each move = 1 action, each door open = 1 action.
        We precompute with ALL doors potentially needing to be opened (worst case),
        but at query time we use current state to check which doors are closed.
        
        Instead, we precompute the graph structure and do per-query BFS only when needed.
        For efficiency, we precompute the simple hop distance and add door-opening costs
        at query time based on a precomputed shortest path.
        
        Actually, for large instances, per-query BFS would be too slow.
        Let's precompute simple hop distances and also store which doors are on 
        shortest paths isn't feasible.
        
        Simpler approach: precompute hop distances ignoring doors. The actual cost
        is at least hop_distance (admissible). We can add door costs separately.
        """
        pass  # We'll use the simple BFS above

    def initialize(self, model: Model, initial_node: HTNNode):
        """Parse all facts, build room graph, compute shortest paths."""
        self.model = model
        
        # Parse all facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "in" and len(args) == 2:
                pkg, room = args[0], args[1]
                self.in_fact_ids[(pkg, room)] = fact.global_id
                self.packages.add(pkg)
                self.rooms.add(room)
            elif pred == "goal_in" and len(args) == 2:
                pkg, room = args[0], args[1]
                self.goal_in_fact_ids[(pkg, room)] = fact.global_id
                self.package_goals[pkg] = room
            elif pred == "rloc" and len(args) == 1:
                room = args[0]
                self.rloc_fact_ids[room] = fact.global_id
                self.rooms.add(room)
            elif pred == "holding" and len(args) == 1:
                pkg = args[0]
                self.holding_fact_ids[pkg] = fact.global_id
            elif pred == "armempty" and len(args) == 0:
                self.armempty_id = fact.global_id
            elif pred == "closed" and len(args) == 1:
                door = args[0]
                self.closed_fact_ids[door] = fact.global_id
            elif pred == "door" and len(args) == 3:
                room1, room2, door = args[0], args[1], args[2]
                self.rooms.add(room1)
                self.rooms.add(room2)
                if room1 not in self.adjacency:
                    self.adjacency[room1] = []
                self.adjacency[room1].append((room2, door))
                self.door_info[(room1, room2)] = door
        
        # Build shortest path matrix
        self._build_shortest_paths()
        
        # Precompute: for each package, which (pkg, room) in_fact_ids exist
        # so we can quickly find where a package currently is
        self.pkg_possible_rooms = {}
        for (pkg, room), fid in self.in_fact_ids.items():
            if pkg not in self.pkg_possible_rooms:
                self.pkg_possible_rooms[pkg] = []
            self.pkg_possible_rooms[pkg].append((room, fid))
        
        # Compute initial heuristic
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_location(self, state):
        """Get current robot location from state."""
        for room, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1:
                return room
        return None

    def _get_package_location(self, state, pkg):
        """Get current location of a package (room or being held)."""
        if pkg in self.holding_fact_ids:
            if (state >> self.holding_fact_ids[pkg]) & 1:
                return None  # Being held
        if pkg in self.pkg_possible_rooms:
            for room, fid in self.pkg_possible_rooms[pkg]:
                if (state >> fid) & 1:
                    return room
        return None

    def _is_package_at_goal(self, state, pkg):
        """Check if package is at its goal location."""
        if pkg not in self.package_goals:
            return True  # No goal for this package
        goal_room = self.package_goals[pkg]
        key = (pkg, goal_room)
        if key in self.in_fact_ids:
            return (state >> self.in_fact_ids[key]) & 1
        return False

    def _compute(self, state, task_network):
        """Compute heuristic value."""
        # Term 1: Per-package delivery cost
        # For each misplaced package: at least pickup(1) + putdown(1) + travel(src->dest)
        misplaced_packages = []
        held_package = None
        held_package_goal = None
        
        robot_loc = self._get_robot_location(state)
        
        for pkg in self.package_goals:
            goal_room = self.package_goals[pkg]
            
            # Check if being held
            if pkg in self.holding_fact_ids and (state >> self.holding_fact_ids[pkg]) & 1:
                held_package = pkg
                held_package_goal = goal_room
                continue
            
            # Check if already at goal
            key = (pkg, goal_room)
            if key in self.in_fact_ids and (state >> self.in_fact_ids[key]) & 1:
                continue  # Already at goal
            
            # Find current location
            pkg_room = self._get_package_location(state, pkg)
            if pkg_room is None:
                continue  # Can't find it, skip
            
            # Cost: pickup(1) + travel(src->dest) + putdown(1)
            travel_dist = self._get_shortest_path(pkg_room, goal_room)
            pkg_cost = 2 + travel_dist  # pickup + putdown + travel
            misplaced_packages.append((pkg, pkg_room, goal_room, pkg_cost))
        
        # Handle held package
        held_cost = 0
        if held_package is not None:
            if robot_loc is not None:
                travel_to_goal = self._get_shortest_path(robot_loc, held_package_goal)
                held_cost = 1 + travel_to_goal  # putdown + travel to goal
            else:
                held_cost = 1  # at least putdown
        
        # Compute delivery cost sum
        # This is admissible because each package needs independent pickup+putdown+travel
        # and they can't overlap due to single arm
        total_delivery_cost = held_cost
        
        if len(misplaced_packages) > 0:
            # Sort by some order for TSP-like lower bound
            # Simple admissible bound: sum of per-package costs
            # But we also need robot travel between deliveries
            # 
            # More sophisticated: each delivery requires the robot to be at the source.
            # Minimum additional travel = we need to visit all source rooms and all dest rooms.
            # A lower bound on the inter-delivery travel: 
            # After delivering one package, robot is at dest. Next pickup is at some src.
            # We can't easily compute optimal TSP, but we can add the cost of reaching
            # the nearest misplaced package from current position.
            
            for pkg, src, dest, cost in misplaced_packages:
                total_delivery_cost += cost
            
            # Add minimum travel to reach the first package (if not holding anything)
            if held_package is None and robot_loc is not None and len(misplaced_packages) > 0:
                min_dist_to_any_src = min(
                    self._get_shortest_path(robot_loc, src)
                    for pkg, src, dest, cost in misplaced_packages
                )
                total_delivery_cost += min_dist_to_any_src
            elif held_package is not None and len(misplaced_packages) > 0:
                # After delivering held package, robot is at held_package_goal
                # Then needs to reach nearest misplaced package source
                min_dist_from_held_dest = min(
                    self._get_shortest_path(held_package_goal, src)
                    for pkg, src, dest, cost in misplaced_packages
                )
                total_delivery_cost += min_dist_from_held_dest
            
            # Additional inter-delivery travel lower bound:
            # After delivering each package, robot must travel to next source.
            # A lower bound: for N deliveries, we need N-1 inter-delivery travels.
            # Minimum inter-delivery travel >= 0 (if src_i+1 == dest_i)
            # We can compute a greedy lower bound or just use 0 (already admissible)
            # 
            # Better: use the minimum spanning tree of delivery sources as lower bound
            # on inter-delivery travel, but that's complex. Instead, add nothing extra
            # to maintain admissibility and simplicity.
        
        # Term 2: Count closed doors on shortest paths that need to be traversed
        # This is complex to compute exactly, so we skip for now and rely on Term 1
        
        # Term 3: Task network structural bound
        # Count primitive tasks and abstract tasks in the network
        tn_cost = 0
        if task_network is not None:
            for task_id in task_network:
                component = self.model.get_component(task_id)
                if isinstance(component, ModelOperator):
                    tn_cost += 1  # Each primitive task = 1 action
                elif isinstance(component, ModelAbstractTask):
                    # Estimate minimum decomposition cost
                    task_name = component.name
                    if "achieve-goals" in task_name:
                        # Could be finished (0 cost) or need work
                        # Don't add anything here to avoid double counting
                        pass
                    elif "pickup_abstract" in task_name:
                        tn_cost += 1  # At least 1 action (pickup)
                    elif "putdown_abstract" in task_name:
                        tn_cost += 1  # At least 1 action (putdown)
                    elif "move_abstract" in task_name:
                        tn_cost += 1  # At least 1 action (move)
                    elif "open_abstract" in task_name:
                        tn_cost += 1  # At least 1 action (open)
                    elif "release" in task_name:
                        tn_cost += 1  # At least putdown (1 action)
                    else:
                        tn_cost += 1
        
        # Return max of the two bounds
        return max(total_delivery_cost, tn_cost)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
