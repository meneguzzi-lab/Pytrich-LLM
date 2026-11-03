import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic

class RobotGemini25ProHeuristic2(Heuristic):
    """
    A domain-dependent heuristic for the Robot domain.

    This heuristic estimates the cost to achieve the goal state by summing the estimated
    costs of delivering all misplaced packages. The cost is composed of:
    1.  Action Costs: The number of required pickup and putdown actions.
    2.  Movement Costs: An admissible estimate of the total moves required to transport packages.
    3.  Door Costs: The number of unique closed doors that must be opened.

    The heuristic pre-computes all-pairs shortest paths between rooms to make the per-node
    calculation efficient. It is designed to be admissible by consistently underestimating
    the true cost, primarily by ignoring the robot's travel to pick up packages.
    """

    def __init__(self, robot_gemini25pro_heuristic_2="robot_gemini25pro_heuristic_2", move_cost=1, open_cost=1, pickup_cost=1, putdown_cost=1):
        super().__init__(name=robot_gemini25pro_heuristic_2)
        self.move_cost = move_cost
        self.open_cost = open_cost
        self.pickup_cost = pickup_cost
        self.putdown_cost = putdown_cost
        self.preprocessing_time = 0

        # Pre-computation results
        self.dist = {}
        self.paths = {}
        self.door_map = {}
        self.package_goals = {}
        self.rooms = set()
        self.packages = set()

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model

        # 1. Parse static domain info (doors, rooms, package goals) from the model
        self._parse_static_info()

        # 2. Build room graph and compute all-pairs shortest paths
        self._compute_shortest_paths()

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value for the root node
        h_value = self.__call__(None, initial_node)
        return super().initialize(model, h_value)

    def _parse_static_info(self):
        """Parses all ground predicates from the model to extract static information."""
        if not hasattr(self.model, 'facts'):
            return

        for fact in self.model.facts:
            parts = fact.name.strip('()').split()
            pred = parts[0]
            args = parts[1:]

            if pred == 'door' and len(args) == 3:
                r1, r2, d = args[0], args[1], args[2]
                self.rooms.add(r1)
                self.rooms.add(r2)
                self.door_map[(r1, r2)] = d
            elif pred == 'goal_in' and len(args) == 2:
                pkg, room = args[0], args[1]
                self.package_goals[pkg] = room
                self.packages.add(pkg)
            elif pred == 'in' and len(args) == 2: 
                self.packages.add(args[0])

    def _compute_shortest_paths(self):
        """Computes all-pairs shortest paths using BFS from each room."""
        adj = {room: [] for room in self.rooms}
        for (r1, r2), door in self.door_map.items():
            if r1 in adj:
                adj[r1].append(r2)

        for r1_rev, r2_rev in list(self.door_map.keys()):
            if (r2_rev, r1_rev) in self.door_map:
                if r2_rev in adj and r1_rev not in adj[r2_rev]:
                    adj[r2_rev].append(r1_rev)

        for start_node in self.rooms:
            self.dist[start_node] = {room: float('inf') for room in self.rooms}
            self.paths[start_node] = {room: [] for room in self.rooms}
            if start_node not in self.dist[start_node]: continue
            self.dist[start_node][start_node] = 0
            
            queue = deque([start_node])
            parent = {start_node: None}

            while queue:
                u = queue.popleft()
                for v in adj.get(u, []):
                    if self.dist[start_node][v] == float('inf'):
                        self.dist[start_node][v] = self.dist[start_node][u] + 1
                        parent[v] = u
                        queue.append(v)
            
            for end_node in self.rooms:
                if self.dist[start_node].get(end_node, float('inf')) != float('inf'):
                    path = []
                    curr = end_node
                    while curr is not None:
                        path.append(curr)
                        curr = parent.get(curr)
                    self.paths[start_node][end_node] = path[::-1]

    def _parse_state(self, state_bitwise):
        """Parses a bitwise state into a more usable format."""
        state_facts = self.model.state_explicit_repr(state_bitwise)
        
        robot_loc = None
        package_locs = {}
        held_pkg = None
        closed_doors = set()

        for fact_str in state_facts:
            parts = fact_str.strip('()').split()
            pred = parts[0]
            args = parts[1:]

            if pred == 'rloc':
                robot_loc = args[0]
            elif pred == 'in':
                package_locs[args[0]] = args[1]
            elif pred == 'holding':
                held_pkg = args[0]
            elif pred == 'closed':
                closed_doors.add(args[0])

        if held_pkg and robot_loc:
            package_locs[held_pkg] = robot_loc
            
        return robot_loc, package_locs, held_pkg, closed_doors

    def __call__(self, parent_node, node):
        if not hasattr(self, 'model'):
            return 0

        robot_loc, package_locs, held_pkg, closed_doors = self._parse_state(node.state)
        
        if not robot_loc:
            return 0

        misplaced_packages = {
            p for p, loc in package_locs.items()
            if p in self.package_goals and loc != self.package_goals[p]
        }

        if not misplaced_packages:
            return 0

        h = 0
        
        if held_pkg and held_pkg in misplaced_packages:
            h += self.putdown_cost
            h += (len(misplaced_packages) - 1) * (self.pickup_cost + self.putdown_cost)
        else:
            h += len(misplaced_packages) * (self.pickup_cost + self.putdown_cost)

        required_legs = set()
        if held_pkg and held_pkg in misplaced_packages:
            goal_loc = self.package_goals[held_pkg]
            if robot_loc != goal_loc:
                required_legs.add((robot_loc, goal_loc))

        for p in misplaced_packages:
            if p != held_pkg:
                current_loc = package_locs[p]
                goal_loc = self.package_goals[p]
                if current_loc != goal_loc:
                    required_legs.add((current_loc, goal_loc))
        
        total_move_cost = 0
        doors_to_open = set()
        for start, end in required_legs:
            if self.dist.get(start, {}).get(end, float('inf')) == float('inf'):
                return float('inf')

            total_move_cost += self.dist[start][end]
            path = self.paths[start][end]
            for i in range(len(path) - 1):
                r1, r2 = path[i], path[i+1]
                door = self.door_map.get((r1, r2)) or self.door_map.get((r2, r1))
                if door and door in closed_doors:
                    doors_to_open.add(door)

        h += total_move_cost * self.move_cost
        h += len(doors_to_open) * self.open_cost

        super().update_info(h)
        return h
