import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic19(Heuristic):
    """
    A domain-dependent heuristic for the Robot domain.

    This heuristic estimates the cost to reach the goal state by considering:
    1. The number of misplaced packages, which determines the necessary pickup and putdown actions.
    2. The travel distance (number of move and open actions) required for the robot's next
       necessary trip. This is either moving a held package to its destination or moving to
       the nearest misplaced package to pick it up.

    The heuristic pre-processes the static information of the domain (room connectivity and
    goal locations) and then calculates the heuristic value for each state by:
    - Counting required pickup/putdown actions for all misplaced packages.
    - Calculating the shortest path for the robot's current task using a live BFS that
      accounts for the current state of the doors.

    This combination provides an admissible and informative estimate of the remaining plan cost.
    """
    def __init__(self, robot_gemini25pro_heuristic_19):
        super().__init__(name="robot_gemini25pro_heuristic_19")
        self.preprocessing_time = 0.0
        self.adj = {}
        self.goal_locs = {}
        self.fact_id_to_name = {}
        self.package_names = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by parsing static information from the model.
        This includes room connections and package goal locations.
        """
        start_time = time.time()
        self.model = model
        self.adj = {}
        self.goal_locs = {}
        self.package_names = set()

        self.fact_id_to_name = {f.global_id: f.name for f in model.facts}

        for fact in model.facts:
            fact_name = fact.name
            if fact_name.startswith("(door "):
                try:
                    _, r1, r2, d = fact_name.strip("() ").split()
                    if r1 not in self.adj: self.adj[r1] = []
                    self.adj[r1].append((r2, d))
                except ValueError:
                    continue # Ignore malformed facts
            elif fact_name.startswith("(goal_in "):
                try:
                    _, pkg, room = fact_name.strip("() ").split()
                    self.goal_locs[pkg] = room
                    self.package_names.add(pkg)
                except ValueError:
                    continue # Ignore malformed facts

        self.preprocessing_time = time.time() - start_time
        h_value = self.__call__(None, initial_node)
        return super().initialize(model, h_value)

    def _get_distance(self, start_room, end_room, closed_doors_names):
        """
        Calculates the shortest path cost between two rooms using BFS.
        The cost is the number of 'move' and 'open' actions required.
        An edge traversal costs 1 (move). If the door is closed, it costs an additional 1 (open).
        """
        if start_room == end_room:
            return 0

        queue = deque([(start_room, 0)])  # (room, cost)
        visited = {start_room}

        while queue:
            current_room, cost = queue.popleft()

            if current_room not in self.adj:
                continue

            for neighbor, door in self.adj[current_room]:
                if neighbor not in visited:
                    edge_cost = 1  # Cost for the 'move' action
                    if door in closed_doors_names:
                        edge_cost += 1  # Additional cost for the 'open' action

                    if neighbor == end_room:
                        return cost + edge_cost
                    
                    visited.add(neighbor)
                    queue.append((neighbor, cost + edge_cost))
        
        return float('inf')  # No path found

    def __call__(self, parent_node: HTNNode, node: HTNNode):
        """
        Calculates the heuristic value for a given search node.
        """
        state_bits = node.state
        
        # 1. Decode current state from bitwise representation
        robot_loc = None
        held_pkg = None
        pkg_locs = {}
        closed_doors = set()

        for fact_id, fact_name in self.fact_id_to_name.items():
            if (state_bits >> fact_id) & 1:  # Check if the fact is true in the current state
                if fact_name.startswith("(rloc "):
                    robot_loc = fact_name.strip("() ").split()[1]
                elif fact_name.startswith("(holding "):
                    held_pkg = fact_name.strip("() ").split()[1]
                elif fact_name.startswith("(in "):
                    parts = fact_name.strip("() ").split()
                    if len(parts) == 3 and parts[1] in self.package_names:
                        _, pkg, loc = parts
                        pkg_locs[pkg] = loc
                elif fact_name.startswith("(closed "):
                    closed_doors.add(fact_name.strip("() ").split()[1])
        
        if not robot_loc:
            return float('inf') # Invalid state

        # 2. Identify all packages that are not in their goal location
        misplaced_pkgs = []
        for pkg, goal_loc in self.goal_locs.items():
            if pkg == held_pkg or pkg_locs.get(pkg) != goal_loc:
                misplaced_pkgs.append(pkg)
        
        # 3. Calculate heuristic value
        h = 0
        
        # Add cost for necessary pickup and putdown actions
        num_putdowns = len(misplaced_pkgs)
        num_pickups = len([p for p in misplaced_pkgs if p != held_pkg])
        h += num_pickups + num_putdowns

        # Add admissible travel cost estimate for the next task
        if held_pkg and held_pkg in self.goal_locs:
            # If holding a package, must travel to its destination
            goal_loc = self.goal_locs[held_pkg]
            h += self._get_distance(robot_loc, goal_loc, closed_doors)
        elif misplaced_pkgs:
            # If arm is empty, must travel to the nearest misplaced package
            min_dist = float('inf')
            pkgs_on_ground = [p for p in misplaced_pkgs if p in pkg_locs]
            if pkgs_on_ground:
                for pkg in pkgs_on_ground:
                    dist = self._get_distance(robot_loc, pkg_locs[pkg], closed_doors)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist != float('inf'):
                    h += min_dist
            
        super().update_info(h)
        return h
