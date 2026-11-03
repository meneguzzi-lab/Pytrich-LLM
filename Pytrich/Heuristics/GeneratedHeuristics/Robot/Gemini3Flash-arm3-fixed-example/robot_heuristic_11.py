import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic11(Heuristic):
    """
    Domain-dependent heuristic for the Robot planning domain.

    Bottleneck Analysis:
    The robot has a single arm (armempty / holding package) and moves between rooms.
    Every package must be individually transported from its location to its target room.
    This creates a strictly serialized sequence of tasks: 
    move to package -> pickup -> move to goal -> putdown.

    Admissibility Sketch:
    1. Package Handling Cost: Each package not at its goal location requires at least a 'pickup' and 
       a 'putdown' action (2 actions). If a package is already being held, it requires 1 'putdown'.
    2. Carrying Distance: For every package, the robot must traverse the shortest path between 
       the package's current location and its goal location. These distances are additive because 
       the robot can only carry one package at a time.
    3. Transition Distance: The robot must move to the location of the first package it intends to pick up.

    Total H = (Sum of Handling Costs) + (Sum of Carrying Distances) + (Distance to nearest unplaced package).
    """

    def __init__(self, robot_gemini3flash_heuristic_11=None, name="RobotGemini3FlashHeuristic11"):
        super().__init__(name=name)
        self.pkg_goal_room = {}      # pkg -> goal_room
        self.pkg_in_ids = defaultdict(dict)  # pkg -> {room: fact_id}
        self.pkg_holding_ids = {}    # pkg -> fact_id
        self.robot_loc_ids = {}      # room -> fact_id
        self.arm_empty_id = None
        self.room_dist = defaultdict(lambda: defaultdict(lambda: 1000))
        self.rooms = set()
        self.packages = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse Predicates and IDs
        room_graph = defaultdict(list)
        
        # Check initial state and facts for goal_in and topology
        initial_state = initial_node.state
        
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
                
            if pred == "goal_in":
                # goal_in ?obj ?loc
                if (initial_state >> fact.global_id) & 1:
                    self.pkg_goal_room[args[0]] = args[1]
                    self.packages.add(args[0])
                    self.rooms.add(args[1])
            
            elif pred == "in":
                self.pkg_in_ids[args[0]][args[1]] = fact.global_id
                self.packages.add(args[0])
                self.rooms.add(args[1])
                
            elif pred == "holding":
                self.pkg_holding_ids[args[0]] = fact.global_id
                self.packages.add(args[0])
                
            elif pred == "rloc":
                self.robot_loc_ids[args[0]] = fact.global_id
                self.rooms.add(args[0])
                
            elif pred == "armempty":
                self.arm_empty_id = fact.global_id
                
            elif pred == "door":
                # door ?loc1 ?loc2 ?d
                room_graph[args[0]].append(args[1])
                self.rooms.add(args[0])
                self.rooms.add(args[1])

        # 2. Precompute All-Pairs Shortest Paths (BFS)
        for start_node in self.rooms:
            self.room_dist[start_node][start_node] = 0
            queue = deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                u, d = queue.popleft()
                for v in room_graph[u]:
                    if v not in visited:
                        visited.add(v)
                        self.room_dist[start_node][v] = d + 1
                        queue.append((v, d + 1))

        h_initial = self._compute_h(initial_node.state)
        return super().initialize(model, h_initial)

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

    def _compute_h(self, state: int) -> int:
        # 1. Find robot location
        robot_room = None
        for room, f_id in self.robot_loc_ids.items():
            if (state >> f_id) & 1:
                robot_room = room
                break
        
        if robot_room is None:
            return 0

        # 2. Find currently held package
        held_pkg = None
        for pkg, f_id in self.pkg_holding_ids.items():
            if (state >> f_id) & 1:
                held_pkg = pkg
                break

        h_value = 0
        unplaced_package_locs = []

        # 3. Calculate costs for all packages in goals
        for pkg, goal_room in self.pkg_goal_room.items():
            # Is package at its goal location?
            at_goal = False
            if pkg in self.pkg_in_ids and goal_room in self.pkg_in_ids[pkg]:
                at_goal = (state >> self.pkg_in_ids[pkg][goal_room]) & 1
            
            if at_goal:
                continue

            # If not at goal, calculate handling + carrying cost
            if held_pkg == pkg:
                # Handling: 1 (putdown)
                # Carrying: robot to goal
                h_value += 1 + self.room_dist[robot_room][goal_room]
            else:
                # Handling: 2 (pickup + putdown)
                # Carrying: current loc to goal
                h_value += 2
                # Find where it is currently
                curr_loc = None
                for room, f_id in self.pkg_in_ids[pkg].items():
                    if (state >> f_id) & 1:
                        curr_loc = room
                        break
                
                if curr_loc:
                    h_value += self.room_dist[curr_loc][goal_room]
                    unplaced_package_locs.append(curr_loc)
                else:
                    # Safety: should not happen in a valid state
                    h_value += 1

        # 4. Add transition distance to nearest package if robot is empty
        if held_pkg is None and unplaced_package_locs:
            min_dist = min(self.room_dist[robot_room][loc] for loc in unplaced_package_locs)
            h_value += min_dist

        return h_value

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute_h(node.state)
        super().update_info(h_val)
        return h_val
