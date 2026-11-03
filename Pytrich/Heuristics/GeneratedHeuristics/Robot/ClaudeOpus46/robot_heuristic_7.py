# Robot Domain Heuristic
# Bottleneck: Single arm serializes all package moves. Each misplaced package needs
# pickup + travel + putdown. Closed doors need opening. Robot must visit every source
# and destination room.
#
# Lower bounds:
# 1. Misplaced packages * 2 (pickup + putdown per package) - admissible since each
#    misplaced package requires at least these two actions.
# 2. TSP-style lower bound: robot must visit all source/dest rooms. We approximate
#    with sum of min distances from each package source to dest, plus a connectivity
#    penalty for closed doors on shortest paths.
# 3. Task network bound: count of remaining primitive-equivalent tasks.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque


class RobotAnthropicOpus46Heuristic7(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_7"):
        super().__init__(name)
        self.model = None
        # Fact ID maps
        self.in_fact_ids = {}       # (pkg, room) -> global_id
        self.goal_in_fact_ids = {}  # (pkg, room) -> global_id
        self.rloc_fact_ids = {}     # room -> global_id
        self.holding_fact_ids = {}  # pkg -> global_id
        self.armempty_id = None
        self.closed_fact_ids = {}   # door -> global_id
        self.door_facts = {}        # (room1, room2) -> door_name
        
        # Goal info
        self.goal_packages = {}     # pkg -> goal_room
        
        # Room graph and distances
        self.rooms = set()
        self.packages = set()
        self.adjacency = {}         # room -> list of (neighbor_room, door_name)
        self.dist = {}              # (room1, room2) -> shortest distance (all doors open)
        
        # Operator info for task network estimation
        self.op_name_to_type = {}   # operator global_id -> action type string

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

    def _bfs_distances(self, start):
        """BFS from start room to all reachable rooms, treating all doors as open."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current in self.adjacency:
                for neighbor, door in self.adjacency[current]:
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
        return dist

    def _bfs_distances_with_closed(self, start, state):
        """BFS from start room considering closed doors need +1 action to open."""
        # Dijkstra-like BFS where closed doors cost 2 (open + move) and open doors cost 1 (move)
        import heapq
        dist = {}
        heap = [(0, start)]
        while heap:
            d, current = heapq.heappop(heap)
            if current in dist:
                continue
            dist[current] = d
            if current in self.adjacency:
                for neighbor, door in self.adjacency[current]:
                    if neighbor not in dist:
                        # Check if door is closed
                        cost = 1  # move action
                        if door in self.closed_fact_ids:
                            door_gid = self.closed_fact_ids[door]
                            if (state >> door_gid) & 1:
                                cost = 2  # open + move
                        new_d = d + cost
                        heapq.heappush(heap, (new_d, neighbor))
        return dist

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = fact.global_id
                self.packages.add(args[0])
                self.rooms.add(args[1])
            elif pred == "goal_in" and len(args) == 2:
                self.goal_in_fact_ids[(args[0], args[1])] = fact.global_id
                self.goal_packages[args[0]] = args[1]
                self.packages.add(args[0])
                self.rooms.add(args[1])
            elif pred == "rloc" and len(args) == 1:
                self.rloc_fact_ids[args[0]] = fact.global_id
                self.rooms.add(args[0])
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_ids[args[0]] = fact.global_id
                self.packages.add(args[0])
            elif pred == "armempty" and len(args) == 0:
                self.armempty_id = fact.global_id
            elif pred == "closed" and len(args) == 1:
                self.closed_fact_ids[args[0]] = fact.global_id
            elif pred == "door" and len(args) == 3:
                self.door_facts[(args[0], args[1])] = args[2]
                self.rooms.add(args[0])
                self.rooms.add(args[1])
                if args[0] not in self.adjacency:
                    self.adjacency[args[0]] = []
                self.adjacency[args[0]].append((args[1], args[2]))
        
        # Precompute all-pairs shortest distances (all doors open)
        self.rooms_list = sorted(self.rooms)
        self.all_dist = {}
        for room in self.rooms_list:
            self.all_dist[room] = self._bfs_distances(room)
        
        # Categorize operators by action type
        for op in model.operators:
            name = op.name
            if name.startswith("+") or name.startswith("-"):
                name = name[1:]
            # Parse operator name to get action type
            paren_idx = name.find("(")
            if paren_idx > 0:
                action_type = name[:paren_idx].strip().lower()
            else:
                bracket_idx = name.find("[")
                if bracket_idx > 0:
                    action_type = name[:bracket_idx].strip().lower()
                else:
                    action_type = name.strip().lower()
            self.op_name_to_type[op.global_id] = action_type
        
        # Categorize abstract tasks
        self.abstract_task_types = {}
        for at in model.abstract_tasks:
            name = at.name
            if name.startswith("+") or name.startswith("-"):
                name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0:
                task_type = name[:bracket_idx].strip().lower()
            else:
                paren_idx = name.find("(")
                if paren_idx > 0:
                    task_type = name[:paren_idx].strip().lower()
                else:
                    task_type = name.strip().lower()
            self.abstract_task_types[at.global_id] = task_type
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_location(self, state):
        """Get current robot room from state."""
        for room, gid in self.rloc_fact_ids.items():
            if (state >> gid) & 1:
                return room
        return None

    def _get_held_package(self, state):
        """Get currently held package, or None."""
        for pkg, gid in self.holding_fact_ids.items():
            if (state >> gid) & 1:
                return pkg
        return None

    def _get_package_location(self, state, pkg):
        """Get room where package is currently located, or None if held."""
        for (p, room), gid in self.in_fact_ids.items():
            if p == pkg and (state >> gid) & 1:
                return room
        return None

    def _count_misplaced(self, state):
        """Count packages not at their goal location."""
        count = 0
        for pkg, goal_room in self.goal_packages.items():
            key = (pkg, goal_room)
            if key in self.in_fact_ids:
                gid = self.in_fact_ids[key]
                if not ((state >> gid) & 1):
                    count += 1
            else:
                # Package goal fact doesn't exist as in-fact, so it's misplaced
                count += 1
        return count

    def _compute(self, state, task_network):
        """Compute heuristic value."""
        misplaced = self._count_misplaced(state)
        
        if misplaced == 0 and len(task_network) == 0:
            return 0
        
        # Term 1: Each misplaced package needs at least pickup + putdown = 2 actions
        # Plus if holding a package that's misplaced, it still needs putdown + possibly travel
        term1 = misplaced * 2
        
        # Check if currently holding a package
        held_pkg = self._get_held_package(state)
        if held_pkg is not None:
            # Held package needs at least putdown (1 action) if it's at goal room,
            # or travel + putdown if not
            if held_pkg in self.goal_packages:
                goal_room = self.goal_packages[held_pkg]
                robot_loc = self._get_robot_location(state)
                if robot_loc and robot_loc != goal_room:
                    if robot_loc in self.all_dist and goal_room in self.all_dist[robot_loc]:
                        travel = self.all_dist[robot_loc][goal_room]
                    else:
                        travel = 1
                    # We already counted this as 2 in term1, but it needs travel + putdown
                    # Adjust: it needs travel_dist + 1 (putdown) instead of 2
                    extra = max(0, travel + 1 - 2)
                    term1 += extra
        
        # Term 2: Travel-based lower bound
        # For each misplaced package, compute min travel cost from source to dest
        # The robot must traverse all these paths, but can chain them
        robot_loc = self._get_robot_location(state)
        
        travel_costs = []
        source_rooms = []
        dest_rooms = []
        
        for pkg, goal_room in self.goal_packages.items():
            key = (pkg, goal_room)
            if key in self.in_fact_ids:
                gid = self.in_fact_ids[key]
                if (state >> gid) & 1:
                    continue  # Already at goal
            
            # Package is misplaced
            if held_pkg == pkg:
                # Package is being held, source is robot location
                src = robot_loc
            else:
                src = self._get_package_location(state, pkg)
            
            if src is None:
                continue
            
            dst = goal_room
            source_rooms.append(src)
            dest_rooms.append(dst)
            
            if src == dst:
                travel_costs.append(0)
            elif src in self.all_dist and dst in self.all_dist[src]:
                travel_costs.append(self.all_dist[src][dst])
            else:
                travel_costs.append(1)  # Fallback
        
        # The robot needs to visit each source then each destination
        # Lower bound: sum of all source->dest travels (since robot carries one at a time)
        # Plus the robot needs to get to each source from current position or previous dest
        # A tight lower bound: for each package, the cost is at least the travel from
        # its current room to its goal room. Since the robot can only carry one package,
        # the total travel is at least the sum of all individual travel distances.
        term2_travel = sum(travel_costs)
        
        # Add cost of robot reaching the first package (minimum over all sources)
        if source_rooms and robot_loc:
            min_to_first = float('inf')
            for src in source_rooms:
                if robot_loc in self.all_dist and src in self.all_dist[robot_loc]:
                    d = self.all_dist[robot_loc][src]
                    min_to_first = min(min_to_first, d)
            if min_to_first < float('inf'):
                term2_travel += min_to_first
        
        term2 = term2_travel + misplaced * 2  # travel + pickup + putdown for each
        
        # Term 3: Count closed doors that are on shortest paths and need opening
        closed_door_count = 0
        for door, gid in self.closed_fact_ids.items():
            if (state >> gid) & 1:
                closed_door_count += 1
        
        # Not all closed doors need opening, but we can estimate:
        # Each misplaced package path might need some doors opened
        # As a lower bound, we just note that needed door openings add actions
        # We'll use a more refined bound: compute travel with closed doors for the held package
        term3 = 0
        if held_pkg and held_pkg in self.goal_packages:
            goal_room = self.goal_packages[held_pkg]
            if robot_loc:
                dist_with_closed = self._bfs_distances_with_closed(robot_loc, state)
                if goal_room in dist_with_closed:
                    term3 = dist_with_closed[goal_room] + 1  # travel + putdown
        
        # Term 4: Task network based bound
        term4 = 0
        for task in task_network:
            task_gid = task.global_id
            if task_gid in self.abstract_task_types:
                ttype = self.abstract_task_types[task_gid]
                if ttype == "achieve-goals":
                    # This task handles all remaining goals
                    # At minimum needs pickup+travel+putdown per misplaced package
                    term4 += misplaced * 2
                elif ttype == "release":
                    # Release = putdown + achieve-goals or move/open + release
                    term4 += 1  # At least one putdown
                elif ttype == "pickup_abstract":
                    term4 += 1
                elif ttype == "putdown_abstract":
                    term4 += 1
                elif ttype == "move_abstract":
                    term4 += 1
                elif ttype == "open_abstract":
                    term4 += 1
                else:
                    term4 += 1
            elif task_gid in self.op_name_to_type:
                # Primitive task - costs exactly 1
                term4 += 1
            else:
                # Check if it's in operator range (primitive)
                if self.model.iop_init <= task_gid <= self.model.iop_end:
                    term4 += 1
                else:
                    term4 += 1
        
        return max(term1, term2, term3, term4)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
