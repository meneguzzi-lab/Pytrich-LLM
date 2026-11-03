# Robot Domain Heuristic
# Bottleneck: Single-carry constraint — robot holds one package at a time,
# serializing all deliveries. Each misplaced package needs pickup + travel + putdown.
#
# Admissible lower bounds:
# T1: 2 * misplaced_packages — each needs at least pickup + putdown
# T2: max over all misplaced packages of (dist(pkg_loc, goal_loc)) — at least
#     that many move actions are needed for the hardest package
# T3: sum of per-package travel costs (source->goal) as a delivery lower bound,
#     since each package's travel is independent unavoidable work when carried alone
#     We use max(T1, T2, T3_adjusted) for the final heuristic.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque


class RobotAnthropicOpus46Heuristic5(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_5"):
        super().__init__(name)
        self.room_ids = {}        # room_name -> set of fact global_ids for rloc
        self.in_fact_ids = {}     # (pkg_name, room_name) -> global_id
        self.goal_in_map = {}     # pkg_name -> goal_room_name
        self.holding_fact_ids = {}  # pkg_name -> global_id
        self.armempty_id = None
        self.rloc_fact_ids = {}   # room_name -> global_id for rloc
        self.closed_fact_ids = {} # door_name -> global_id
        self.door_map = {}        # (room1, room2) -> door_name
        self.adjacency = {}       # room_name -> set of (neighbor_room, door_name)
        self.all_rooms = set()
        self.all_packages = set()
        self.dist_matrix = {}     # (room1, room2) -> shortest distance (ignoring closed doors)
        self.pkg_list = []        # list of package names with goals

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
        """BFS from start_room, returns dict room->distance. Ignores closed doors (optimistic)."""
        dist = {start_room: 0}
        queue = deque([start_room])
        while queue:
            current = queue.popleft()
            if current not in self.adjacency:
                continue
            for neighbor, door in self.adjacency[current]:
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
        return dist

    def _bfs_distances_with_doors(self, start_room, state):
        """BFS from start_room considering closed doors need an open action."""
        # Each move costs 1. If door is closed, opening costs 1 additional action.
        # Use Dijkstra-like BFS with costs.
        import heapq
        dist = {}
        heap = [(0, start_room)]
        while heap:
            cost, current = heapq.heappop(heap)
            if current in dist:
                continue
            dist[current] = cost
            if current not in self.adjacency:
                continue
            for neighbor, door in self.adjacency[current]:
                if neighbor in dist:
                    continue
                door_closed = False
                if door in self.closed_fact_ids:
                    door_closed = (state >> self.closed_fact_ids[door]) & 1
                move_cost = 1  # move action
                if door_closed:
                    move_cost += 1  # open action
                new_cost = cost + move_cost
                if neighbor not in dist:
                    heapq.heappush(heap, (new_cost, neighbor))
        return dist

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model

        # Parse all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in":
                pkg, room = args[0], args[1]
                self.in_fact_ids[(pkg, room)] = f.global_id
                self.all_packages.add(pkg)
                self.all_rooms.add(room)
            elif pred == "rloc":
                room = args[0]
                self.rloc_fact_ids[room] = f.global_id
                self.all_rooms.add(room)
            elif pred == "holding":
                pkg = args[0]
                self.holding_fact_ids[pkg] = f.global_id
            elif pred == "armempty":
                self.armempty_id = f.global_id
            elif pred == "closed":
                door = args[0]
                self.closed_fact_ids[door] = f.global_id
            elif pred == "goal_in":
                pkg, room = args[0], args[1]
                self.goal_in_map[pkg] = room
                self.all_packages.add(pkg)
                self.all_rooms.add(room)
            elif pred == "door":
                room1, room2, door = args[0], args[1], args[2]
                self.all_rooms.add(room1)
                self.all_rooms.add(room2)
                if room1 not in self.adjacency:
                    self.adjacency[room1] = set()
                self.adjacency[room1].add((room2, door))
                self.door_map[(room1, room2)] = door

        # Build list of packages that have goals
        self.pkg_list = [pkg for pkg in self.all_packages if pkg in self.goal_in_map]

        # Precompute all-pairs shortest distances (optimistic, ignoring doors)
        # This is used as admissible lower bound on travel
        self.optimistic_dist = {}
        for room in self.all_rooms:
            self.optimistic_dist[room] = self._bfs_distances(room)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_room(self, state):
        """Get current robot room from state."""
        for room, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1:
                return room
        return None

    def _get_pkg_room(self, state, pkg):
        """Get the room a package is in, or None if being held."""
        for (p, room), fid in self.in_fact_ids.items():
            if p == pkg and (state >> fid) & 1:
                return room
        return None

    def _is_holding(self, state, pkg):
        """Check if robot is holding a specific package."""
        if pkg in self.holding_fact_ids:
            return (state >> self.holding_fact_ids[pkg]) & 1
        return False

    def _is_armempty(self, state):
        if self.armempty_id is not None:
            return (state >> self.armempty_id) & 1
        return True

    def _opt_dist(self, r1, r2):
        """Optimistic distance between two rooms (ignoring doors)."""
        if r1 == r2:
            return 0
        if r1 in self.optimistic_dist and r2 in self.optimistic_dist[r1]:
            return self.optimistic_dist[r1][r2]
        return 999  # unreachable

    def _compute(self, state, task_network):
        # If no tasks remain, we're done
        if len(task_network) == 0:
            return 0

        robot_room = self._get_robot_room(state)
        if robot_room is None:
            return 0

        # Find misplaced packages and held package
        misplaced = []  # (pkg, current_room_or_None, goal_room)
        held_pkg = None
        held_goal = None

        for pkg in self.pkg_list:
            goal_room = self.goal_in_map[pkg]
            if self._is_holding(state, pkg):
                held_pkg = pkg
                held_goal = goal_room
                # Held package is misplaced (not yet delivered)
                misplaced.append((pkg, None, goal_room))
                continue
            pkg_room = self._get_pkg_room(state, pkg)
            if pkg_room is not None and pkg_room != goal_room:
                misplaced.append((pkg, pkg_room, goal_room))

        num_misplaced = len(misplaced)
        if num_misplaced == 0:
            # No misplaced packages — check task network
            # Return small value based on remaining tasks
            return 0

        # T1: Each misplaced package needs at minimum pickup + putdown = 2 actions
        # If currently holding one, it only needs putdown = 1 action less overhead
        t1 = 2 * num_misplaced
        if held_pkg is not None:
            t1 -= 1  # held package only needs putdown, not pickup

        # T2: Travel distance lower bound
        # For held package: dist(robot, goal)
        # For each grounded package: dist(source, goal) (optimistic, robot must travel there)
        # Plus robot must go to each source
        # Admissible: max single package cost = dist(robot, src) + dist(src, goal) + 2 (pickup/putdown)
        max_single_pkg_cost = 0
        total_delivery_cost = 0

        if held_pkg is not None:
            # Cost for held package: travel to goal + putdown
            held_cost = self._opt_dist(robot_room, held_goal) + 1  # move + putdown
            max_single_pkg_cost = max(max_single_pkg_cost, held_cost)
            total_delivery_cost += self._opt_dist(robot_room, held_goal)

        for pkg, src, goal in misplaced:
            if src is None:  # held package, already handled
                continue
            # Minimum for this package: travel src->goal
            pkg_travel = self._opt_dist(src, goal)
            # Full cost: go to src + pickup + travel to goal + putdown
            full_cost = self._opt_dist(robot_room, src) + 1 + pkg_travel + 1
            max_single_pkg_cost = max(max_single_pkg_cost, full_cost)
            total_delivery_cost += pkg_travel

        t2 = max_single_pkg_cost

        # T3: Total delivery travel lower bound
        # Each package needs at least dist(src, goal) moves while being carried
        # Plus pickup + putdown for each = 2 * num_misplaced (already in t1)
        # Since robot carries one at a time, these travels are sequential
        # So total_delivery_cost + 2*num_misplaced is a lower bound
        # But we must also account for robot traveling between deliveries
        # Minimum: robot goes to nearest misplaced, then chains deliveries
        # Admissible approximation: sum of all source->goal distances + pickup/putdown
        t3 = total_delivery_cost + t1

        # T4: Consider closed doors that MUST be opened
        # For each misplaced package, check if shortest path requires opening doors
        # Count minimum door openings needed
        # This is expensive to compute per call, so we use a simpler bound:
        # For held package, compute actual distance considering current door states
        door_cost = 0
        if held_pkg is not None and held_goal is not None:
            actual_dist = self._bfs_distances_with_doors(robot_room, state)
            if held_goal in actual_dist:
                # The difference between actual (with doors) and optimistic gives door opening cost
                opt = self._opt_dist(robot_room, held_goal)
                door_cost = max(0, actual_dist[held_goal] - opt)

        # T5: Nearest misplaced package distance (robot must reach at least one)
        if not self._is_armempty(state) and held_pkg is None:
            # Shouldn't happen but handle gracefully
            nearest_dist = 0
        elif held_pkg is not None:
            nearest_dist = 0  # already holding something
        else:
            nearest_dist = 999
            for pkg, src, goal in misplaced:
                if src is not None:
                    d = self._opt_dist(robot_room, src)
                    nearest_dist = min(nearest_dist, d)
            if nearest_dist == 999:
                nearest_dist = 0

        # Combine: return max of independent lower bounds
        # t1 = 2*misplaced (pickup+putdown)
        # t2 = max single package full cost
        # t3 = total delivery + pickup/putdown
        # nearest_dist is additive to t1 if arm is empty
        t1_adj = t1 + nearest_dist + door_cost

        result = max(t1_adj, t2, t3)

        # Task network depth penalty (small tie-breaker)
        # Count abstract tasks in network as indicator of remaining work
        abstract_count = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):
                abstract_count += 1

        # Add tiny tie-breaker for abstract tasks remaining
        result = result + 0.001 * abstract_count

        return result

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
