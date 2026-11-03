from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic8(Heuristic):
    """
    Heuristic for the Robot domain.
    
    Bottleneck Analysis:
    The primary bottleneck in this domain is that the robot has only one arm and can carry 
    at most one package at a time. This forces all package transportation tasks to be 
    performed sequentially.

    Admissibility Sketch:
    1. Each package not currently at its goal location requires at least one 'pickup' and one 'putdown' action.
    2. Moving a package from its current location to its goal requires a sequence of 'move' actions 
       equal to the shortest path distance between the two rooms.
    3. Since the robot carries only one package, the total number of actions required is at least the sum 
       of (pickup + putdown + travel_distance) for every package that needs to be moved.
    4. Additionally, if the robot is currently empty, it must travel to at least one package's current 
       location to start moving it. Adding the minimum distance to any unmoved package is also admissible.
    5. Doors that are closed require an 'open' action, but for simplicity and guaranteed admissibility, 
       we treat moves as the baseline cost.
    """

    def __init__(self, robot_gemini3flash_heuristic_8="robot_gemini3flash_heuristic_8"):
        super().__init__(robot_gemini3flash_heuristic_8)
        self.pkg_goal_room = {}
        self.rooms = set()
        self.packages = set()
        self.adj = collections.defaultdict(list)
        self.dist = {}
        
        self.armempty_id = None
        self.rloc_ids = {} # room -> global_id
        self.holding_ids = {} # package -> global_id
        self.in_ids = [] # list of (package, room, global_id)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args, prefix
        return None, [], None

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates to IDs and objects
        for f_id, fact in enumerate(model.facts):
            pred, args, prefix = self._parse_fact_name(fact.name)
            if prefix != "+":
                continue
            
            if pred == "armempty":
                self.armempty_id = fact.global_id
            elif pred == "rloc":
                room = args[0]
                self.rooms.add(room)
                self.rloc_ids[room] = fact.global_id
            elif pred == "holding":
                pkg = args[0]
                self.packages.add(pkg)
                self.holding_ids[pkg] = fact.global_id
            elif pred == "in":
                pkg, room = args[0], args[1]
                self.packages.add(pkg)
                self.rooms.add(room)
                self.in_ids.append((pkg, room, fact.global_id))
            elif pred == "goal_in":
                pkg, room = args[0], args[1]
                self.pkg_goal_room[pkg] = room
            elif pred == "door":
                r1, r2 = args[0], args[1]
                self.adj[r1].append(r2)

        # 2. Precompute shortest path distances between rooms using BFS
        for start_node in self.rooms:
            self.dist[start_node] = {r: 1000 for r in self.rooms}
            self.dist[start_node][start_node] = 0
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                u, d = queue.popleft()
                for v in self.adj[u]:
                    if v not in visited:
                        visited.add(v)
                        self.dist[start_node][v] = d + 1
                        queue.append((v, d + 1))

        robot_gemini3flash_heuristic_8 = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, robot_gemini3flash_heuristic_8)

    def _compute(self, state, task_network):
        if not task_network and self.model.goal_reached(state):
            return 0

        # Find robot location
        robot_loc = None
        for room, r_id in self.rloc_ids.items():
            if (state >> r_id) & 1:
                robot_loc = room
                break
        
        if robot_loc is None:
            return 1000

        # Check if holding anything
        held_pkg = None
        for pkg, h_id in self.holding_ids.items():
            if (state >> h_id) & 1:
                held_pkg = pkg
                break
        
        # Map current package locations
        pkg_locs = {}
        for pkg, room, in_id in self.in_ids:
            if (state >> in_id) & 1:
                pkg_locs[pkg] = room

        total_h = 0
        min_dist_to_next_pkg = 1000
        needs_pickup = False

        for pkg, target_room in self.pkg_goal_room.items():
            if held_pkg == pkg:
                # Already picked up: cost is moving to target + putdown
                total_h += 1 + self.dist[robot_loc][target_room]
            else:
                curr_room = pkg_locs.get(pkg)
                if curr_room and curr_room != target_room:
                    # Needs pickup, transport, and putdown
                    total_h += 2 + self.dist[curr_room][target_room]
                    min_dist_to_next_pkg = min(min_dist_to_next_pkg, self.dist[robot_loc][curr_room])
                    needs_pickup = True
        
        # If arm is empty and we have packages to move, we must travel to one first
        if held_pkg is None and needs_pickup:
            total_h += min_dist_to_next_pkg

        return total_h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        robot_gemini3flash_heuristic_8 = self._compute(node.state, node.task_network)
        super().update_info(robot_gemini3flash_heuristic_8)
        return robot_gemini3flash_heuristic_8
