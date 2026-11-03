import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic

class RobotGemini25ProHeuristic9(Heuristic):
    """
    A domain-dependent heuristic for the Robot domain.

    This heuristic estimates the minimum number of actions required to reach the goal state.
    It is state-based, meaning it calculates a value based on the current state of the planning problem.
    The heuristic is designed to be admissible for use with A* search.

    The heuristic value is calculated as the sum of two components:
    1. Manipulation Cost: The number of pickup and putdown actions required.
       - Each misplaced package on the floor requires one pickup and one putdown (+2 cost).
       - A package currently held by the robot requires one putdown (+1 cost).
    2. Transportation Cost: The total travel distance for all packages.
       - This is the sum of the shortest path distances from each package's current location to its goal location.
       - The path cost includes both 'move' and 'open' actions. Moving through an open door costs 1, while a closed door costs 2 (1 to open, 1 to move).

    This heuristic assumes that the node object passed to __call__ has a 'state' attribute
    containing a bitwise representation of the current world state.
    """

    def __init__(self, robot_gemini25pro_heuristic_9):
        super().__init__(name="robot_gemini25pro_heuristic_9")
        self.config = robot_gemini25pro_heuristic_9  # Store config if any, as per convention
        self.model = None
        self.goal_locations = {}
        self.room_graph = {}
        self.fact_maps = {}

    def initialize(self, model, initial_node):
        """Pre-parses static information from the model for efficient lookups later."""
        start_time = time.time()
        self.model = model
        self._parse_static_facts()

        h_value = self.__call__(None, initial_node)
        self.preprocessing_time = time.time() - start_time
        return super().initialize(model, h_value)

    def _parse_static_facts(self):
        """Builds internal data structures from the model's facts for quick access."""
        self.fact_maps = {
            'rloc': {},       # fact_id -> room_name
            'in': {},         # fact_id -> (package_name, room_name)
            'holding': {},    # fact_id -> package_name
            'closed': {},     # fact_id -> door_name
            'door_map': {},   # door_name -> (room1, room2)
        }

        for fact in self.model.facts:
            parts = fact.name.strip("()").split()
            pred_name = parts[0]

            if pred_name == "goal_in":
                pkg, room = parts[1], parts[2]
                self.goal_locations[pkg] = room
            elif pred_name == "door":
                r1, r2, d = parts[1], parts[2], parts[3]
                if r1 not in self.room_graph: self.room_graph[r1] = []
                if r2 not in self.room_graph: self.room_graph[r2] = []
                self.room_graph[r1].append((r2, d))
                self.room_graph[r2].append((r1, d))
                self.fact_maps['door_map'][d] = (r1, r2)
            elif pred_name == "rloc":
                self.fact_maps['rloc'][fact.global_id] = parts[1]
            elif pred_name == "in":
                self.fact_maps['in'][fact.global_id] = (parts[1], parts[2])
            elif pred_name == "holding":
                self.fact_maps['holding'][fact.global_id] = parts[1]
            elif pred_name == "closed":
                self.fact_maps['closed'][fact.global_id] = parts[1]

    def _get_path_cost(self, start_room, end_room, closed_doors):
        """Calculates shortest path cost between rooms using BFS, considering closed doors."""
        if start_room == end_room:
            return 0
        
        # BFS queue stores (room, cost)
        q = deque([(start_room, 0)])
        visited = {start_room}
        
        while q:
            current_room, cost = q.popleft()
            
            if current_room == end_room:
                return cost
                
            if current_room in self.room_graph:
                for neighbor_room, door in self.room_graph[current_room]:
                    if neighbor_room not in visited:
                        # Cost is 2 if door is closed (open + move), 1 otherwise (move)
                        edge_cost = 2 if door in closed_doors else 1
                        visited.add(neighbor_room)
                        q.append((neighbor_room, cost + edge_cost))
        
        return float('inf') # Path not found

    def __call__(self, parent_node, node):
        """Calculates the heuristic value for the given node's state."""
        state = node.state
        
        # 1. Parse dynamic state information from the bitwise state
        robot_loc = None
        held_package = None
        package_locs = {}
        closed_doors = set()

        for fact_id, room in self.fact_maps['rloc'].items():
            if (state >> fact_id) & 1:
                robot_loc = room
                break
        
        for fact_id, pkg in self.fact_maps['holding'].items():
            if (state >> fact_id) & 1:
                held_package = pkg
                break
                
        for fact_id, (pkg, room) in self.fact_maps['in'].items():
            if (state >> fact_id) & 1:
                package_locs[pkg] = room

        for fact_id, door in self.fact_maps['closed'].items():
            if (state >> fact_id) & 1:
                closed_doors.add(door)

        # 2. Calculate heuristic value
        h_manipulate = 0
        h_transport = 0

        # Identify all misplaced packages
        misplaced_packages = []
        for pkg, goal_loc in self.goal_locations.items():
            current_loc = package_locs.get(pkg)
            # A package is misplaced if it's not at its goal location.
            # If it's held, its location is effectively the robot's location.
            if pkg != held_package and current_loc != goal_loc:
                misplaced_packages.append(pkg)

        # Account for the held package
        if held_package:
            goal_loc = self.goal_locations.get(held_package)
            if goal_loc: # Ensure package has a goal
                h_manipulate += 1  # 1 putdown action
                h_transport += self._get_path_cost(robot_loc, goal_loc, closed_doors)
        
        # Account for packages on the floor
        if misplaced_packages:
            h_manipulate += 2 * len(misplaced_packages) # 1 pickup + 1 putdown per package
            
            for pkg in misplaced_packages:
                loc = package_locs[pkg]
                goal_loc = self.goal_locations[pkg]
                h_transport += self._get_path_cost(loc, goal_loc, closed_doors)
            
        h = h_manipulate + h_transport

        super().update_info(h)
        return h

    def __repr__(self):
        return f"RobotGemini25ProHeuristic9()"
