from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic13(Heuristic):
    """
    Robot Domain Heuristic
    
    BOTTLENECK: Single-carry constraint. 
    The robot holds exactly one package at a time. This serializes all deliveries into 
    distinct round-trip sequences (move to package, pickup, move to goal, putdown).

    ADMISSIBILITY SKETCH:
    Term 1 (Delivery Actions): Each undelivered package p currently in a room requires 
    at least 1 pickup and 1 putdown (2 actions). If a package is being held, it requires 
    1 putdown. Summing these is admissible as they must occur sequentially.
    
    Term 2 (Travel Distance): For each package p, the robot must traverse at least 
    the shortest-path distance between p's current location and p's goal location. 
    Additionally, the robot must traverse the distance to the first target (either 
    the held package's goal or the nearest undelivered package). Since the robot 
    can only carry one package, these travel segments are disjoint and sequential.
    
    Term 3 (Door Opening): (Optional but ignored for strict admissibility) Opening a 
    closed door on a path adds a cost. To remain strictly admissible, we use BFS 
    distance as a lower bound assuming all doors are open.
    """

    def __init__(self, robot_gemini3flash_heuristic_13="RobotGemini3FlashHeuristic13"):
        super().__init__(robot_gemini3flash_heuristic_13)
        self.rooms = []
        self.packages = []
        self.dist = {}
        self.pkg_goal = {} # package -> goal_room
        self.rloc_ids = {} # room -> global_id
        self.holding_ids = {} # package -> global_id
        self.in_ids = {} # package -> {room -> global_id}
        self.armempty_id = None
        self.initialized = False

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

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates and objects
        rooms_set = set()
        packages_set = set()
        adj = collections.defaultdict(list)
        
        # Use initial state for goal_in and door logic
        init_facts = model.state_explicit_repr(initial_node.state)
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "rloc":
                rooms_set.add(args[0])
                self.rloc_ids[args[0]] = f.global_id
            elif pred == "holding":
                packages_set.add(args[0])
                self.holding_ids[args[0]] = f.global_id
            elif pred == "in":
                packages_set.add(args[0])
                rooms_set.add(args[1])
                if args[0] not in self.in_ids: self.in_ids[args[0]] = {}
                self.in_ids[args[0]][args[1]] = f.global_id
            elif pred == "armempty":
                self.armempty_id = f.global_id
            elif pred == "door":
                adj[args[0]].append(args[1])
            elif pred == "goal_in":
                self.pkg_goal[args[0]] = args[1]

        self.rooms = list(rooms_set)
        self.packages = list(packages_set)

        # 2. Precompute BFS distances (all-pairs shortest paths, assuming all doors open)
        for start_node in self.rooms:
            self.dist[start_node] = {r: 999999 for r in self.rooms}
            self.dist[start_node][start_node] = 0
            queue = collections.deque([start_node])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if self.dist[start_node][v] == 999999:
                        self.dist[start_node][v] = self.dist[start_node][u] + 1
                        queue.append(v)

        self.initialized = True
        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        if not self.initialized:
            return 0
            
        # Find robot location
        robot_loc = None
        for r in self.rooms:
            if (state >> self.rloc_ids[r]) & 1:
                robot_loc = r
                break
        
        if not robot_loc: return 0

        # Find what robot is holding
        held_pkg = None
        arm_empty = (state >> self.armempty_id) & 1 if self.armempty_id is not None else True
        if not arm_empty:
            for p in self.packages:
                if (state >> self.holding_ids[p]) & 1:
                    held_pkg = p
                    break

        # Find undelivered packages
        undelivered = []
        h_delivery = 0
        
        for p in self.packages:
            if p == held_pkg:
                # Holding it: cost is 1 (putdown) + travel to goal
                goal_room = self.pkg_goal.get(p)
                if goal_room:
                    h_delivery += 1 + self.dist[robot_loc][goal_room]
                continue
            
            # Where is it?
            p_loc = None
            for r, f_id in self.in_ids.get(p, {}).items():
                if (state >> f_id) & 1:
                    p_loc = r
                    break
            
            if p_loc and p_loc != self.pkg_goal.get(p):
                undelivered.append((p, p_loc))
                goal_room = self.pkg_goal.get(p)
                # Every undelivered requires 1 pickup + 1 putdown + travel to goal
                h_delivery += 2 + self.dist[p_loc][goal_room]

        # If arm is empty, we must also travel to the first package
        if arm_empty and undelivered:
            min_travel_to_pkg = min(self.dist[robot_loc][p_loc] for p, p_loc in undelivered)
            h_delivery += min_travel_to_pkg
        
        # Basic task network count to differentiate states with same delivery progress
        h_tn = len(task_network)

        # Scale task network count so it acts as a tie-breaker
        return int(h_delivery + 0.1 * h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
