# Robot Domain Heuristic
# 
# BOTTLENECK: Single arm (armempty) serializes all package deliveries.
# Each misplaced package requires pickup + travel + putdown.
#
# ADMISSIBLE LOWER BOUNDS:
# 1. misplaced_count * 2: Each misplaced package needs at least pickup + putdown (2 actions).
# 2. travel_lower_bound: Sum of BFS shortest-path distances for each misplaced package
#    (robot must visit source and destination). Since robot is serial, we use
#    a TSP-style lower bound: max individual trip cost.
# 3. task_network_bound: Minimum actions implied by remaining abstract tasks.
#
# We return max(term1, term2, term3) for strong guidance.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque


class RobotAnthropicOpus46Heuristic0(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_0"):
        super().__init__(name)
        self.model = None
        # Fact ID maps
        self.in_fact_ids = {}       # (obj, room) -> global_id
        self.goal_in_fact_ids = {}  # (obj, room) -> global_id
        self.rloc_fact_ids = {}     # room -> global_id
        self.armempty_id = None
        self.holding_fact_ids = {}  # obj -> global_id
        self.closed_fact_ids = {}   # door -> global_id
        self.door_facts = {}        # (room1, room2) -> door
        
        # Goal info
        self.goal_packages = {}     # obj -> goal_room
        
        # Room graph and BFS distances
        self.rooms = set()
        self.adjacency = {}         # room -> [(neighbor_room, door)]
        self.bfs_dist = {}          # (room1, room2) -> shortest distance (ignoring closed doors)
        self.bfs_dist_with_doors = {}  # includes +1 for each closed door that needs opening
        
        # Abstract task type detection
        self.achieve_goals_ids = set()
        self.release_ids = set()
        self.pickup_abstract_ids = set()
        self.putdown_abstract_ids = set()
        self.move_abstract_ids = set()
        self.open_abstract_ids = set()
        
        # Operator type detection
        self.pickup_op_ids = set()
        self.putdown_op_ids = set()
        self.move_op_ids = set()
        self.open_op_ids = set()

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
        # No brackets - predicate with no args
        return name, []

    def _bfs_all_pairs(self):
        """Compute BFS shortest path distances between all rooms.
        Distance = number of move actions needed (each move = 1 action).
        We ignore door states for the base distance."""
        room_list = list(self.rooms)
        for src in room_list:
            if src not in self.bfs_dist:
                self.bfs_dist[src] = {}
            self.bfs_dist[src][src] = 0
            queue = deque([src])
            visited = {src}
            while queue:
                current = queue.popleft()
                for neighbor, door in self.adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.bfs_dist[src][neighbor] = self.bfs_dist[src][current] + 1
                        queue.append(neighbor)
            # For unreachable rooms
            for r in room_list:
                if r not in self.bfs_dist[src]:
                    self.bfs_dist[src][r] = 999

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "goal_in" and len(args) == 2:
                self.goal_in_fact_ids[(args[0], args[1])] = fact.global_id
                self.goal_packages[args[0]] = args[1]
            elif pred == "rloc" and len(args) == 1:
                self.rloc_fact_ids[args[0]] = fact.global_id
                self.rooms.add(args[0])
            elif pred == "armempty":
                self.armempty_id = fact.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_ids[args[0]] = fact.global_id
            elif pred == "closed" and len(args) == 1:
                self.closed_fact_ids[args[0]] = fact.global_id
            elif pred == "door" and len(args) == 3:
                r1, r2, d = args[0], args[1], args[2]
                self.rooms.add(r1)
                self.rooms.add(r2)
                if r1 not in self.adjacency:
                    self.adjacency[r1] = []
                self.adjacency[r1].append((r2, d))
                self.door_facts[(r1, r2)] = d
        
        # Classify abstract tasks
        for at in model.abstract_tasks:
            name_lower = at.name.lower()
            if "achieve-goals" in name_lower or "achieve_goals" in name_lower:
                self.achieve_goals_ids.add(at.global_id)
            elif "release" in name_lower:
                self.release_ids.add(at.global_id)
            elif "pickup_abstract" in name_lower or "pickup-abstract" in name_lower:
                self.pickup_abstract_ids.add(at.global_id)
            elif "putdown_abstract" in name_lower or "putdown-abstract" in name_lower:
                self.putdown_abstract_ids.add(at.global_id)
            elif "move_abstract" in name_lower or "move-abstract" in name_lower:
                self.move_abstract_ids.add(at.global_id)
            elif "open_abstract" in name_lower or "open-abstract" in name_lower:
                self.open_abstract_ids.add(at.global_id)
        
        # Classify operators
        for op in model.operators:
            name_lower = op.name.lower()
            if "pickup" in name_lower:
                self.pickup_op_ids.add(op.global_id)
            elif "putdown" in name_lower:
                self.putdown_op_ids.add(op.global_id)
            elif name_lower.startswith("move") or "__move" in name_lower or "_move_" in name_lower or name_lower.endswith("move"):
                self.move_op_ids.add(op.global_id)
            elif "open" in name_lower:
                self.open_op_ids.add(op.global_id)
        
        # Actually let's be more careful with operator classification
        self.move_op_ids = set()
        self.open_op_ids = set()
        for op in model.operators:
            # Parse the operator name to find the action type
            op_name = op.name
            # Operator names are like "move[r1,r2,d01]" or "open[r1,r2,d01]"
            # or could be "move_loc1_loc2_door" format
            if op_name.startswith("+") or op_name.startswith("-"):
                op_name = op_name[1:]
            bracket_idx = op_name.find("[")
            if bracket_idx > 0:
                action_name = op_name[:bracket_idx].lower()
            else:
                # Try underscore split
                action_name = op_name.split("_")[0].lower() if "_" in op_name else op_name.lower()
            
            if action_name == "move":
                self.move_op_ids.add(op.global_id)
            elif action_name == "open":
                self.open_op_ids.add(op.global_id)
            elif action_name == "pickup":
                self.pickup_op_ids.add(op.global_id)
            elif action_name == "putdown":
                self.putdown_op_ids.add(op.global_id)
        
        # Compute BFS distances
        self._bfs_all_pairs()
        
        # Precompute: for each package, find the set of rooms it could be in
        self.package_rooms = {}  # obj -> list of rooms it has in_fact_ids for
        for (obj, room), gid in self.in_fact_ids.items():
            if obj not in self.package_rooms:
                self.package_rooms[obj] = []
            self.package_rooms[obj].append(room)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_robot_location(self, state):
        """Find current robot location from state."""
        for room, gid in self.rloc_fact_ids.items():
            if (state >> gid) & 1:
                return room
        return None

    def _get_holding(self, state):
        """Find what package robot is holding, if any."""
        for obj, gid in self.holding_fact_ids.items():
            if (state >> gid) & 1:
                return obj
        return None

    def _compute(self, state, task_network):
        """Compute heuristic value."""
        if not task_network:
            return 0
        
        # Term 1: Count misplaced packages and compute pickup+putdown lower bound
        misplaced = []
        robot_loc = self._get_robot_location(state)
        holding = self._get_holding(state)
        arm_empty = (state >> self.armempty_id) & 1 if self.armempty_id is not None else 1
        
        for obj, goal_room in self.goal_packages.items():
            in_key = (obj, goal_room)
            if in_key in self.in_fact_ids:
                gid = self.in_fact_ids[in_key]
                if (state >> gid) & 1:
                    # Already at goal
                    continue
            # Package is misplaced
            # Find where the package currently is
            current_room = None
            if obj == holding:
                current_room = robot_loc  # effectively at robot's location
                misplaced.append((obj, current_room, goal_room, True))
            else:
                for room in self.package_rooms.get(obj, []):
                    key = (obj, room)
                    if key in self.in_fact_ids:
                        gid = self.in_fact_ids[key]
                        if (state >> gid) & 1:
                            current_room = room
                            break
                if current_room is not None:
                    misplaced.append((obj, current_room, goal_room, False))
        
        if not misplaced:
            # No misplaced packages - check if task network is empty
            if not task_network:
                return 0
            # Still have tasks but no misplaced packages
            # Could be tasks like move_abstract, open_abstract to finalize
            return 0
        
        num_misplaced = len(misplaced)
        
        # Term 1: Each misplaced package needs at least pickup + putdown = 2 actions
        # If currently holding one, it only needs putdown = 1 action for that one
        term1 = 0
        for obj, src, dst, is_held in misplaced:
            if is_held:
                term1 += 1  # just putdown (+ possible travel)
            else:
                term1 += 2  # pickup + putdown
        
        # Term 2: Travel-based lower bound
        # For each misplaced package, robot must travel to source, then to destination.
        # Since robot is serial, the total travel is at least the sum of
        # source-to-destination distances for each package.
        # But more accurately, we compute a lower bound considering the robot
        # must visit all sources and destinations.
        #
        # Simple admissible bound: for each misplaced package not held,
        # add distance(src, dst) as minimum travel for that package.
        # For the held package, add distance(robot_loc, dst).
        # Then the robot also needs to reach the first package, so add min distance to any source.
        
        term2 = 0
        if robot_loc is not None:
            # For each misplaced package, compute minimum actions:
            # - If held: travel to goal_room + putdown = dist(robot_loc, goal_room) + 1
            # - If not held: travel to src + pickup + travel to dst + putdown
            #   = dist(?, src) + 1 + dist(src, dst) + 1
            #   The dist(?, src) part depends on robot state after previous deliveries.
            
            # Admissible lower bound: sum of dist(src, dst) for each non-held package
            # plus dist(robot_loc, goal_room) for held package
            # plus minimum distance from robot to nearest misplaced package source
            
            total_travel = 0
            min_dist_to_first = 999
            has_held = False
            
            for obj, src, dst, is_held in misplaced:
                if is_held:
                    has_held = True
                    # Must travel from robot_loc to dst
                    d = self.bfs_dist.get(robot_loc, {}).get(dst, 999)
                    total_travel += d + 1  # + 1 for putdown
                else:
                    # Must travel src->dst at minimum
                    d = self.bfs_dist.get(src, {}).get(dst, 999)
                    total_travel += d + 2  # +2 for pickup and putdown
                    # Distance from robot to this source
                    d_to_src = self.bfs_dist.get(robot_loc, {}).get(src, 999)
                    min_dist_to_first = min(min_dist_to_first, d_to_src)
            
            if not has_held and min_dist_to_first < 999:
                # Robot must travel to at least one source first
                total_travel += min_dist_to_first
            
            term2 = total_travel
        
        # Term 3: Closed doors penalty
        # Count doors on shortest paths that are closed - each needs an open action
        # This is harder to compute exactly. Instead, we use a simpler bound:
        # For BFS distance computation considering closed doors, each closed door
        # on the path adds +1 for the open action.
        # We'll use a simpler approach: just count closed doors that are on
        # any path the robot might need.
        term3 = 0
        if robot_loc is not None:
            # For a better bound, compute shortest path with door-opening costs
            # using a modified BFS/Dijkstra
            total_travel_with_doors = 0
            min_dist_to_first_wd = 999
            has_held_wd = False
            
            for obj, src, dst, is_held in misplaced:
                if is_held:
                    has_held_wd = True
                    d = self._shortest_path_with_doors(robot_loc, dst, state)
                    total_travel_with_doors += d + 1
                else:
                    d = self._shortest_path_with_doors(src, dst, state)
                    total_travel_with_doors += d + 2
                    d_to_src = self._shortest_path_with_doors(robot_loc, src, state)
                    min_dist_to_first_wd = min(min_dist_to_first_wd, d_to_src)
            
            if not has_held_wd and min_dist_to_first_wd < 999:
                total_travel_with_doors += min_dist_to_first_wd
            
            term3 = total_travel_with_doors
        
        # Term 4: Task network based bound
        # Count primitive actions remaining in task network
        term4 = self._task_network_bound(task_network, state)
        
        return max(term1, term2, term3, term4)

    def _shortest_path_with_doors(self, src, dst, state):
        """BFS shortest path from src to dst, adding +1 for each closed door on the path.
        Uses Dijkstra-like approach since door opening adds cost."""
        if src == dst:
            return 0
        if src not in self.rooms or dst not in self.rooms:
            return 999
        
        # Use a simple BFS with cost tracking
        # Cost = number of moves + number of door opens needed
        import heapq
        dist = {src: 0}
        pq = [(0, src)]
        
        while pq:
            cost, current = heapq.heappop(pq)
            if current == dst:
                return cost
            if cost > dist.get(current, 999):
                continue
            for neighbor, door in self.adjacency.get(current, []):
                # Cost: 1 for move
                move_cost = 1
                # If door is closed, +1 for open action
                if door in self.closed_fact_ids:
                    door_gid = self.closed_fact_ids[door]
                    if (state >> door_gid) & 1:
                        move_cost += 1
                new_cost = cost + move_cost
                if new_cost < dist.get(neighbor, 999):
                    dist[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor))
        
        return dist.get(dst, 999)

    def _task_network_bound(self, task_network, state):
        """Compute a lower bound based on the task network.
        Each pickup_abstract implies at least 1 action (pickup).
        Each putdown_abstract implies at least 1 action (putdown).
        Each move_abstract implies at least 1 action (move).
        Each open_abstract implies at least 1 action (open).
        Each release implies at least 1 action (putdown) if holding something.
        Each achieve-goals implies at least 2 actions per remaining misplaced package.
        """
        bound = 0
        for task in task_network:
            gid = task.global_id
            if gid in self.pickup_op_ids or gid in self.putdown_op_ids or \
               gid in self.move_op_ids or gid in self.open_op_ids:
                # Primitive operator
                bound += 1
            elif gid in self.pickup_abstract_ids:
                bound += 1
            elif gid in self.putdown_abstract_ids:
                bound += 1
            elif gid in self.move_abstract_ids:
                bound += 1
            elif gid in self.open_abstract_ids:
                bound += 1
            elif gid in self.release_ids:
                # Release must eventually putdown + possibly move
                bound += 1
            elif gid in self.achieve_goals_ids:
                # This is the main recursive task
                # Count misplaced packages for a lower bound
                misplaced_count = 0
                holding = self._get_holding(state)
                for obj, goal_room in self.goal_packages.items():
                    in_key = (obj, goal_room)
                    if in_key in self.in_fact_ids:
                        fact_gid = self.in_fact_ids[in_key]
                        if (state >> fact_gid) & 1:
                            continue
                    misplaced_count += 1
                # Each misplaced needs at least 2 actions (pickup + putdown)
                # But if holding one, that one needs only 1
                if holding is not None and holding in self.goal_packages:
                    goal_room = self.goal_packages[holding]
                    in_key = (holding, goal_room)
                    if in_key in self.in_fact_ids:
                        fact_gid = self.in_fact_ids[in_key]
                        if not ((state >> fact_gid) & 1):
                            bound += (misplaced_count - 1) * 2 + 1
                        else:
                            bound += misplaced_count * 2
                    else:
                        bound += misplaced_count * 2
                else:
                    bound += misplaced_count * 2
                break  # achieve-goals subsumes remaining tasks
        
        return bound
