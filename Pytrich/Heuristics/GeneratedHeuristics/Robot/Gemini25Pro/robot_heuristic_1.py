import time
from Pytrich.Heuristics.heuristic import Heuristic

class RobotGemini25ProHeuristic1(Heuristic):
    """
    A domain-dependent heuristic for the Robot domain.

    This heuristic estimates the cost to achieve the goals (delivering packages)
    by calculating the shortest path for the robot and packages, considering the
    current state of doors (open or closed).

    The core idea is to calculate an admissible estimate of the required actions:
    - If the robot is holding a package, the heuristic is the cost to move to the
      package's goal location and put it down.
    - If the robot's arm is empty, the heuristic is the minimum cost to go to any
      misplaced package, pick it up, carry it to its destination, and put it down.

    Distances between rooms are calculated dynamically for each state using the
    Floyd-Warshall all-pairs shortest path algorithm, where moving through a closed
    door has a higher cost (open + move) than an open one.
    """
    def __init__(self, robot_gemini25pro_heuristic_1=None, name="robot_gemini25pro_heuristic_1"):
        super().__init__(name=name)
        self.model = None
        self.preprocessing_time = 0

        # Statically pre-parsed information
        self.rooms = []
        self.room_to_idx = {}
        self.packages = []
        self.doors = {}  # {door_name: (room1, room2)}
        self.package_goals = {}  # {package_name: goal_room}

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-parsing static information from the model,
        such as room layouts, door connections, and package goals.
        """
        start_time = time.time()
        self.model = model

        all_fact_names = [f.name for f in self.model.facts]
        rooms_set = set()
        packages_set = set()

        for fact_name in all_fact_names:
            parts = fact_name.replace(')', '').split('(')
            pred_name = parts[0]
            if len(parts) > 1:
                args = [a.strip() for a in parts[1].split(',')]
            else:
                continue # Skip facts with no arguments like 'armempty'

            if pred_name == 'door':
                r1, r2, d = args[0], args[1], args[2]
                rooms_set.add(r1)
                rooms_set.add(r2)
                self.doors[d] = (r1, r2)
            elif pred_name == 'goal_in':
                pkg, room = args[0], args[1]
                packages_set.add(pkg)
                rooms_set.add(room)
                self.package_goals[pkg] = room
            elif pred_name == 'in':
                packages_set.add(args[0])
                rooms_set.add(args[1])
            elif pred_name == 'rloc':
                rooms_set.add(args[0])

        self.rooms = sorted(list(rooms_set))
        self.packages = sorted(list(packages_set))
        self.room_to_idx = {name: i for i, name in enumerate(self.rooms)}

        self.preprocessing_time = time.time() - start_time

        # Calculate heuristic for the initial node
        h_value = self.__call__(None, initial_node)
        return super().initialize(model, h_value)

    def _parse_state(self, state_bitmask):
        """
        Parses the bitmask state into a more usable format by identifying the
        robot's location, held package, package locations, and closed doors.
        """
        true_predicates = self.model.state_explicit_repr(state_bitmask)
        
        robot_loc = None
        held_package = None
        package_locs = {}
        closed_doors = set()

        for pred_str in true_predicates:
            parts = pred_str.replace(')', '').split('(')
            pred_name = parts[0]
            if len(parts) > 1:
                args = [a.strip() for a in parts[1].split(',')]
            else:
                continue

            if pred_name == 'rloc':
                robot_loc = args[0]
            elif pred_name == 'holding':
                held_package = args[0]
            elif pred_name == 'in':
                package_locs[args[0]] = args[1]
            elif pred_name == 'closed':
                closed_doors.add(args[0])

        return robot_loc, held_package, package_locs, closed_doors

    def _floyd_warshall(self, closed_doors):
        """
        Computes all-pairs shortest paths for the room graph.
        Edge weights are 1 for open doors and 2 for closed doors (1 to open, 1 to move).
        """
        num_rooms = len(self.rooms)
        dist = [[float('inf')] * num_rooms for _ in range(num_rooms)]

        for i in range(num_rooms):
            dist[i][i] = 0

        for door_name, (r1, r2) in self.doors.items():
            weight = 2 if door_name in closed_doors else 1
            idx1, idx2 = self.room_to_idx[r1], self.room_to_idx[r2]
            dist[idx1][idx2] = min(dist[idx1][idx2], weight)
            dist[idx2][idx1] = min(dist[idx2][idx1], weight)

        for k in range(num_rooms):
            for i in range(num_rooms):
                for j in range(num_rooms):
                    if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        return dist

    def __call__(self, parent_node, node):
        """Calculates the heuristic value for a given search node."""
        # Step A: Parse the current state from the bitmask
        robot_loc, held_package, package_locs, closed_doors = self._parse_state(node.state)

        # Step B: Compute state-dependent distances
        dist_matrix = self._floyd_warshall(closed_doors)

        # Step C: Calculate heuristic value based on state
        h = 0.0

        if held_package:
            # Heuristic is the cost to deliver the currently held package.
            goal_loc = self.package_goals.get(held_package)
            if goal_loc and robot_loc:
                r_idx = self.room_to_idx.get(robot_loc)
                g_idx = self.room_to_idx.get(goal_loc)
                if r_idx is not None and g_idx is not None:
                    # Cost = distance to travel + 1 action for 'putdown'
                    h = float(dist_matrix[r_idx][g_idx] + 1)
        else:
            # Heuristic is the minimum cost to deliver one of the misplaced packages.
            misplaced = [p for p, loc in package_locs.items() if loc != self.package_goals.get(p)]
            
            if not misplaced:
                h = 0.0
            elif robot_loc:
                min_cost = float('inf')
                r_idx = self.room_to_idx[robot_loc]

                for p_name in misplaced:
                    p_loc = package_locs[p_name]
                    g_loc = self.package_goals[p_name]
                    
                    p_idx = self.room_to_idx[p_loc]
                    g_idx = self.room_to_idx[g_loc]

                    # Cost = (robot->pkg) + (pkg->goal) + pickup action + putdown action
                    cost = dist_matrix[r_idx][p_idx] + dist_matrix[p_idx][g_idx] + 2
                    min_cost = min(min_cost, cost)
                
                h = float(min_cost) if min_cost != float('inf') else 0.0

        super().update_info(h)
        return h
