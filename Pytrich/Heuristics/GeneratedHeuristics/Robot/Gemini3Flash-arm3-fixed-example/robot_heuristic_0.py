import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic0(Heuristic):
    """
    Domain-dependent heuristic for the Robot planning domain.

    Bottleneck Analysis:
    The primary bottleneck is the single robot arm, which serializes the transport of packages.
    The robot must sequentially pick up, transport, and put down each package that is not at its goal room.
    This serial nature allows for an additive (and thus admissible) cost estimation.

    Admissibility Sketches:
    1. Action Cost: Every package not in its goal room requires at least a 'pickup' and a 'putdown' (2 actions).
       If the robot is already holding the package, only a 'putdown' is required (1 action).
    2. Travel Cost: For every package, the robot must move from the package's current room to its goal room.
       The number of 'move' actions is at least the shortest path distance in the room-door graph.
    3. Sequence Cost: To transport the first package, the robot must first move from its current location to the package.

    Total h = Sum_{p in wrong_rooms} (actions(p) + dist(p_start, p_goal)) + dist(robot, nearest_p_start).
    This is admissible because the arm constraint prevents any parallel action execution.
    """

    def __init__(self, robot_gemini3flash_heuristic_0=None, name="RobotGemini3FlashHeuristic0"):
        super().__init__(name=name)
        self.package_goal_room = {} # package -> room_name
        self.room_adj = {}          # room -> [neighbor_rooms]
        self.dists = {}             # room -> {room: distance}
        self.package_names = []
        self.in_fact_ids = {}       # (package, room) -> global_id
        self.holding_fact_ids = {}  # package -> global_id
        self.rloc_fact_ids = {}     # room -> global_id
        self.armempty_id = None

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        rooms = set()
        packages = set()

        # 1. Map predicates to facts and IDs
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                rooms.add(args[0])
                self.rloc_fact_ids[args[0]] = fact.global_id
            elif pred == "in":
                packages.add(args[0])
                self.in_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "holding":
                self.holding_fact_ids[args[0]] = fact.global_id
            elif pred == "armempty":
                self.armempty_id = fact.global_id
            elif pred == "door":
                r1, r2 = args[0], args[1]
                if r1 not in self.room_adj: self.room_adj[r1] = []
                self.room_adj[r1].append(r2)
            elif pred == "goal_in":
                # Use the initial state to determine target destinations as defined in domain logic
                if (model.initial_state >> fact.global_id) & 1:
                    self.package_goal_room[args[0]] = args[1]

        self.package_names = list(packages)

        # 2. Precompute all-pairs shortest paths for rooms
        for start_node in rooms:
            self.dists[start_node] = {start_node: 0}
            queue = deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                curr, d = queue.popleft()
                for neighbor in self.room_adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dists[start_node][neighbor] = d + 1
                        queue.append((neighbor, d + 1))

        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

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

    def _compute(self, state: int) -> int:
        # Determine current robot location
        current_room = None
        for room, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1:
                current_room = room
                break

        if not current_room:
            return 0

        # Determine if holding a package
        held_package = None
        for p, fid in self.holding_fact_ids.items():
            if (state >> fid) & 1:
                held_package = p
                break

        unplaced_packages = []
        for p in self.package_names:
            goal_room = self.package_goal_room.get(p)
            if not goal_room: continue
            
            # A package is unplaced if it is not in its goal room
            in_goal_fid = self.in_fact_ids.get((p, goal_room))
            if in_goal_fid is not None and not ((state >> in_goal_fid) & 1):
                unplaced_packages.append(p)

        if not unplaced_packages and held_package is None:
            return 0

        h = 0
        # Find rooms where unplaced packages are currently located
        unplaced_info = [] # (package, current_room)
        for p in unplaced_packages:
            if p == held_package: continue
            p_curr_room = None
            for r in self.room_adj.keys():
                fid = self.in_fact_ids.get((p, r))
                if fid is not None and (state >> fid) & 1:
                    p_curr_room = r
                    break
            if p_curr_room:
                unplaced_info.append((p, p_curr_room))

        # Admissible calculation: sum of unavoidable steps
        min_dist_to_first = float('inf')

        # Case 1: Robot is holding a package
        if held_package:
            goal = self.package_goal_room[held_package]
            h += 1 + self.dists.get(current_room, {}).get(goal, 100) # putdown + travel to goal
            # Cost for all other unplaced packages
            for p, p_room in unplaced_info:
                p_goal = self.package_goal_room[p]
                h += 2 + self.dists.get(p_room, {}).get(p_goal, 100) # pickup + putdown + travel
        
        # Case 2: Robot is empty
        else:
            for p, p_room in unplaced_info:
                p_goal = self.package_goal_room[p]
                h += 2 + self.dists.get(p_room, {}).get(p_goal, 100)
                dist_to_p = self.dists.get(current_room, {}).get(p_room, 100)
                if dist_to_p < min_dist_to_first:
                    min_dist_to_first = dist_to_p
            if unplaced_info:
                h += min_dist_to_first

        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value
