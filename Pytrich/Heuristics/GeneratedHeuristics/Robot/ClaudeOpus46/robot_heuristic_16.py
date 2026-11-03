# Robot Domain Heuristic
# Bottleneck: Single arm (armempty) serializes all package deliveries.
# Each misplaced package requires pickup + travel + putdown.
#
# Admissible lower bounds:
# T1: 2 * num_misplaced_packages (each needs pickup + putdown minimum)
# T2: max individual package cost = 2 + shortest_path(src, dest) for worst package
# T3: task-network based estimate counting minimum primitive tasks
#
# Preprocessing: BFS all-pairs shortest path on room graph (ignoring closed doors
# gives optimistic distance), build fact maps for fast lookup.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque


class RobotAnthropicOpus46Heuristic16(Heuristic):
    """Domain-dependent heuristic for the Robot domain.
    
    Uses multi-component max of:
    1. Sum-based misplaced package count (2 per misplaced package)
    2. Max individual package displacement cost
    3. Task network primitive action lower bound
    Combined with BFS shortest paths between rooms.
    """

    def __init__(self, name="robot_anthropicopus46_heuristic_16"):
        super().__init__(name)
        self.in_fact_ids = {}      # (obj, room) -> global_id
        self.goal_in_facts = {}    # (obj, room) -> global_id
        self.holding_fact_ids = {} # obj -> global_id
        self.rloc_fact_ids = {}    # room -> global_id
        self.armempty_id = None
        self.closed_fact_ids = {}  # door -> global_id
        self.door_facts = {}       # (room1, room2) -> door
        
        # Room graph for BFS
        self.room_adj = {}         # room -> set of (neighbor_room, door)
        self.rooms = set()
        self.packages = set()
        
        # Goal info
        self.goal_locations = {}   # obj -> goal_room
        
        # Shortest path cache
        self.shortest_paths = {}   # (room1, room2) -> distance (ignoring closed doors)
        
        # For task network analysis
        self.operator_name_map = {}  # operator global_id -> action name

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

    def _bfs_shortest_paths(self, start_room):
        """BFS from start_room to all other rooms (optimistic: ignoring closed doors)."""
        dist = {start_room: 0}
        queue = deque([start_room])
        while queue:
            room = queue.popleft()
            if room in self.room_adj:
                for neighbor, door in self.room_adj[room]:
                    if neighbor not in dist:
                        dist[neighbor] = dist[room] + 1
                        queue.append(neighbor)
        return dist

    def _bfs_shortest_paths_with_closed(self, start_room, state):
        """BFS from start_room considering closed doors need an open action."""
        # Dijkstra-like with cost 1 for open doors, 2 for closed doors (open + move)
        dist = {start_room: 0}
        queue = deque([(start_room, 0)])
        # Simple BFS where closed doors cost +1 extra
        # Use a priority queue approach with deque (0-1 BFS style)
        import heapq
        heap = [(0, start_room)]
        dist = {start_room: 0}
        while heap:
            d, room = heapq.heappop(heap)
            if d > dist.get(room, float('inf')):
                continue
            if room in self.room_adj:
                for neighbor, door in self.room_adj[room]:
                    # Check if door is closed
                    door_closed = False
                    if door in self.closed_fact_ids:
                        gid = self.closed_fact_ids[door]
                        door_closed = (state >> gid) & 1
                    
                    cost = 2 if door_closed else 1  # open+move vs just move
                    new_dist = d + cost
                    if new_dist < dist.get(neighbor, float('inf')):
                        dist[neighbor] = new_dist
                        heapq.heappush(heap, (new_dist, neighbor))
        return dist

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "goal_in" and len(args) == 2:
                self.goal_in_facts[(args[0], args[1])] = fact.global_id
                self.goal_locations[args[0]] = args[1]
                self.packages.add(args[0])
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_ids[args[0]] = fact.global_id
            elif pred == "rloc" and len(args) == 1:
                self.rloc_fact_ids[args[0]] = fact.global_id
                self.rooms.add(args[0])
            elif pred == "armempty" and len(args) == 0:
                self.armempty_id = fact.global_id
            elif pred == "closed" and len(args) == 1:
                self.closed_fact_ids[args[0]] = fact.global_id
            elif pred == "door" and len(args) == 3:
                room1, room2, door = args[0], args[1], args[2]
                self.door_facts[(room1, room2)] = door
                if room1 not in self.room_adj:
                    self.room_adj[room1] = []
                self.room_adj[room1].append((room2, door))
                self.rooms.add(room1)
                self.rooms.add(room2)
        
        # Precompute optimistic all-pairs shortest paths (ignoring closed doors)
        self.optimistic_dist = {}
        for room in self.rooms:
            self.optimistic_dist[room] = self._bfs_shortest_paths(room)
        
        # Map operators by name for task network analysis
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if pred:
                self.operator_name_map[op.global_id] = pred
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_location(self, state):
        """Get the current robot location from state."""
        for room, gid in self.rloc_fact_ids.items():
            if (state >> gid) & 1:
                return room
        return None

    def _get_current_locations(self, state):
        """Get current location of each package."""
        locations = {}
        for (obj, room), gid in self.in_fact_ids.items():
            if (state >> gid) & 1:
                locations[obj] = room
        return locations

    def _get_held_package(self, state):
        """Get the package currently being held, if any."""
        for obj, gid in self.holding_fact_ids.items():
            if (state >> gid) & 1:
                return obj
        return None

    def _compute(self, state, task_network):
        """Compute heuristic value."""
        # Get current state info
        robot_loc = self._get_robot_location(state)
        pkg_locations = self._get_current_locations(state)
        held_pkg = self._get_held_package(state)
        arm_empty = (state >> self.armempty_id) & 1 if self.armempty_id is not None else True
        
        # Find misplaced packages
        misplaced = []
        for obj in self.packages:
            goal_room = self.goal_locations.get(obj)
            if goal_room is None:
                continue
            
            # Check if package is at goal
            in_goal_gid = self.in_fact_ids.get((obj, goal_room))
            if in_goal_gid is not None and (state >> in_goal_gid) & 1:
                continue  # Already at goal
            
            # Package is misplaced
            current_room = pkg_locations.get(obj)
            if current_room is not None:
                misplaced.append((obj, current_room, goal_room))
            elif held_pkg == obj:
                # Package is being held
                misplaced.append((obj, None, goal_room))  # None means held
        
        if not misplaced:
            # Check task network - if there are still tasks, give small value
            if len(task_network) > 0:
                return len(task_network)
            return 0
        
        num_misplaced = len(misplaced)
        
        # Term 1: Each misplaced package needs at least pickup + putdown = 2 actions
        # But if held, only putdown = 1 action needed for the held one
        t1 = 0
        for obj, cur_room, goal_room in misplaced:
            if cur_room is None:  # held
                t1 += 1  # just putdown
            else:
                t1 += 2  # pickup + putdown
        
        # Term 2: Maximum individual package cost (2 + travel for pickup source->dest)
        # This is admissible as a lower bound on total cost
        max_individual = 0
        total_travel = 0
        
        for obj, cur_room, goal_room in misplaced:
            if cur_room is None:  # held
                # Need to travel to goal_room and putdown
                if robot_loc and goal_room in self.optimistic_dist.get(robot_loc, {}):
                    pkg_cost = 1 + self.optimistic_dist[robot_loc][goal_room]
                else:
                    pkg_cost = 1
                max_individual = max(max_individual, pkg_cost)
            else:
                # Need to travel to cur_room, pickup, travel to goal_room, putdown
                src_to_dest = self.optimistic_dist.get(cur_room, {}).get(goal_room, 0)
                pkg_cost = 2 + src_to_dest  # pickup + putdown + travel
                max_individual = max(max_individual, pkg_cost)
                total_travel += src_to_dest
        
        # Term 3: Sum-based estimate
        # Each package needs pickup+putdown+travel. But robot is serial, so
        # between consecutive deliveries, robot must travel from last dest to next src.
        # Lower bound: sum of all (pickup + putdown) + sum of all src->dest travel
        # This overcounts robot transit between packages, but pickup+putdown is exact.
        t3 = t1 + total_travel
        
        # Term 4: State-aware travel estimate
        # Robot needs to reach first misplaced package, then chain through all.
        # Minimum: robot travel to nearest misplaced + per-package costs
        if robot_loc:
            min_robot_to_pkg = float('inf')
            for obj, cur_room, goal_room in misplaced:
                target = goal_room if cur_room is None else cur_room
                d = self.optimistic_dist.get(robot_loc, {}).get(target, 100)
                min_robot_to_pkg = min(min_robot_to_pkg, d)
            if min_robot_to_pkg == float('inf'):
                min_robot_to_pkg = 0
        else:
            min_robot_to_pkg = 0
        
        # Term 5: Task network lower bound
        # Count minimum primitive actions needed from task network
        tn_cost = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):
                # Abstract task - estimate minimum decomposition cost
                task_name = task.name if hasattr(task, 'name') else ''
                # Parse task name to understand what it does
                pred, args = self._parse_fact_name(task_name)
                if pred == 'achieve-goals':
                    # This will need to handle remaining misplaced packages
                    # Don't double count - just add 0 here
                    pass
                elif pred == 'release':
                    # At minimum: putdown (1 action)
                    tn_cost += 1
                elif pred == 'pickup_abstract':
                    tn_cost += 1
                elif pred == 'putdown_abstract':
                    tn_cost += 1
                elif pred == 'move_abstract':
                    tn_cost += 1
                elif pred == 'open_abstract':
                    tn_cost += 1
            else:
                # Primitive operator
                tn_cost += 1
        
        # Final heuristic: max of multiple admissible bounds
        h_val = max(
            t1 + min_robot_to_pkg,  # pickup+putdown for all + travel to first
            max_individual,          # single hardest package
            t3,                      # sum of per-package travel + pickup/putdown
            tn_cost                  # task network lower bound
        )
        
        return h_val

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
