import time
from collections import deque, defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic6(Heuristic):
    """
    Domain-dependent heuristic for the Robot domain.

    Bottleneck Analysis:
    The Robot domain is a sequential logistics problem where a single robot can only carry one package at a time. 
    The total cost is dominated by the robot's movement between rooms and the pickup/putdown actions. 
    Doors must also be opened, adding a potential cost of 1 action per move.

    Admissible Lower Bounds:
    1. Transport Cost (State-based): For each package not at its goal, the robot must at least pick it up (1), 
       move it to the goal (min_moves), and put it down (1). If holding the package, only move and putdown are needed.
    2. Robot Travel: If the robot is empty and packages are unsettled, it must travel to the nearest package.

    Summing these costs is admissible for a single robot because all pickup, putdown, and transport moves 
    are distinct and must be performed sequentially.
    """

    def __init__(self, robot_gemini3flash_heuristic_6=None, name="RobotGemini3FlashHeuristic6"):
        super().__init__(name=name)
        self.package_goals = {}  # pkg -> target_room
        self.dist_matrix = defaultdict(lambda: defaultdict(lambda: 999999))
        self.p_in_fids = defaultdict(dict)  # pkg -> room -> global_id
        self.p_holding_fids = {}  # pkg -> global_id
        self.r_loc_fids = {}  # room -> global_id
        self.arm_empty_fid = None
        self.rooms = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess(model)
        h_value = self._calculate_heuristic(initial_node.state, initial_node.task_network)
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

    def _preprocess(self, model: Model):
        adj = defaultdict(list)
        all_rooms = set()
        packages = set()

        # 1. Map predicates to global IDs and identify entities
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue

            if pred == "rloc":
                room = args[0]
                self.r_loc_fids[room] = fact.global_id
                all_rooms.add(room)
            elif pred == "in":
                pkg, room = args[0], args[1]
                self.p_in_fids[pkg][room] = fact.global_id
                packages.add(pkg)
            elif pred == "holding":
                pkg = args[0]
                self.p_holding_fids[pkg] = fact.global_id
                packages.add(pkg)
            elif pred == "armempty":
                self.arm_empty_fid = fact.global_id
            elif pred == "door":
                u, v = args[0], args[1]
                adj[u].append(v)
                adj[v].append(u)
            elif pred == "goal_in":
                if (model.initial_state >> fact.global_id) & 1:
                    self.package_goals[args[0]] = args[1]

        self.rooms = list(all_rooms)
        
        # 2. All-pairs shortest paths using BFS
        for start in self.rooms:
            self.dist_matrix[start][start] = 0
            queue = deque([(start, 0)])
            visited = {start}
            while queue:
                u, d = queue.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        self.dist_matrix[start][v] = d + 1
                        queue.append((v, d + 1))

    def _calculate_heuristic(self, state: int, task_network) -> int:
        h_val = 0
        
        # Find robot location
        robot_loc = None
        for room, fid in self.r_loc_fids.items():
            if (state >> fid) & 1:
                robot_loc = room
                break
        
        if robot_loc is None:
            return 0

        unsettled_locations = []
        arm_is_empty = True if self.arm_empty_fid is not None and (state >> self.arm_empty_fid) & 1 else False
        
        for pkg, goal_room in self.package_goals.items():
            # Case 1: Robot is holding the package
            if (state >> self.p_holding_fids.get(pkg, -1)) & 1 if pkg in self.p_holding_fids else False:
                dist = self.dist_matrix[robot_loc][goal_room]
                h_val += dist + 1 # Moves to goal + putdown
                arm_is_empty = False
                continue
            
            # Case 2: Package is in a room
            current_room = None
            for room, fid in self.p_in_fids.get(pkg, {}).items():
                if (state >> fid) & 1:
                    current_room = room
                    break
            
            if current_room and current_room != goal_room:
                dist = self.dist_matrix[current_room][goal_room]
                h_val += dist + 2 # Pickup + Moves to goal + putdown
                unsettled_locations.append(current_room)

        # Add robot travel to the first unsettled package if empty
        if arm_is_empty and unsettled_locations:
            min_travel = min(self.dist_matrix[robot_loc][loc] for loc in unsettled_locations)
            h_val += min_travel

        return h_val

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
