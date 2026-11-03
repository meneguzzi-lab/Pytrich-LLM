from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic3(Heuristic):
    """
    Heuristic for the Robot domain.
    
    Bottleneck Analysis:
    The primary bottleneck is the single-carry constraint (armempty). The robot can only hold one package at a time. 
    Each delivery requires a sequence: move to source, pickup, move to destination, putdown.
    
    Admissibility Sketch:
    1. Each undelivered package requires at least one pickup and one putdown (+2 actions per package).
    2. For each package p, the robot must traverse the shortest path from its current location to its goal.
    3. Since only one package can be carried at a time, these 'delivery' move segments and 'pick/put' actions 
       are strictly sequential and disjoint. 
    4. The robot must also travel to the first package's source and between deliveries.
    5. The sum of (2 + dist(source, target)) for all undelivered packages, plus the distance from the robot
       to the first available source, is an admissible lower bound on the total number of actions.
    """

    def __init__(self, robot_gemini3flash_heuristic_3="robot_gemini3flash_heuristic_3"):
        super().__init__(robot_gemini3flash_heuristic_3)
        self.pkg_to_goal_room = {}
        self.dist_matrix = {}
        self.armempty_id = -1
        self.all_rloc_mask = 0
        self.rloc_to_room = {}
        self.all_in_mask = 0
        self.in_to_pkg_room = {}
        self.all_holding_mask = 0
        self.holding_to_pkg = {}
        self.inf = 1000

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = [a.strip() for a in args_str.split(",")] if args_str else []
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Identify Target Rooms via 'goal_in' facts in initial state
        init_state_facts = model.state_explicit_repr(model.initial_state)
        rooms = set()
        adj = collections.defaultdict(list)
        
        for f_name in init_state_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == "goal_in":
                self.pkg_to_goal_room[args[0]] = args[1]
            elif pred == "door":
                adj[args[0]].append(args[1])
                rooms.add(args[0])
                rooms.add(args[1])
            elif pred == "rloc":
                rooms.add(args[0])

        # 2. Precompute All-Pairs Shortest Paths (BFS because all moves cost 1)
        for start_node in rooms:
            self.dist_matrix[start_node] = {r: self.inf for r in rooms}
            self.dist_matrix[start_node][start_node] = 0
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                u, d = queue.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        self.dist_matrix[start_node][v] = d + 1
                        queue.append((v, d + 1))

        # 3. Build fast bitwise masks for __call__
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "rloc":
                self.all_rloc_mask |= (1 << f.global_id)
                self.rloc_to_room[f.global_id] = args[0]
            elif pred == "in":
                self.all_in_mask |= (1 << f.global_id)
                self.in_to_pkg_room[f.global_id] = (args[0], args[1])
            elif pred == "holding":
                self.all_holding_mask |= (1 << f.global_id)
                self.holding_to_pkg[f.global_id] = args[0]
            elif pred == "armempty":
                self.armempty_id = f.global_id

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network) -> int:
        # Identify robot location
        robot_bits = state & self.all_rloc_mask
        if not robot_bits:
            return self.inf
        robot_room = self.rloc_to_room[robot_bits.bit_length() - 1]

        # Identify packages not at goal and the one held
        undelivered_pkgs = []
        held_pkg = None
        
        # Check held package
        held_bits = state & self.all_holding_mask
        if held_bits:
            held_pkg = self.holding_to_pkg[held_bits.bit_length() - 1]

        # Check packages in rooms
        temp_in = state & self.all_in_mask
        while temp_in:
            bit = (temp_in & -temp_in).bit_length() - 1
            pkg, room = self.in_to_pkg_room[bit]
            goal_room = self.pkg_to_goal_room.get(pkg)
            if goal_room and room != goal_room:
                undelivered_pkgs.append((pkg, room))
            temp_in &= temp_in - 1

        cost = 0
        if held_pkg:
            # If holding, we MUST put it down at its goal
            goal = self.pkg_to_goal_room.get(held_pkg)
            if goal:
                cost += 1 + self.dist_matrix.get(robot_room, {}).get(goal, self.inf)
            # Plus all other deliveries
            for p, loc in undelivered_pkgs:
                target = self.pkg_to_goal_room[p]
                cost += 2 + self.dist_matrix.get(loc, {}).get(target, self.inf)
        else:
            # If empty, we must travel to the first package
            if undelivered_pkgs:
                dist_to_first = min(self.dist_matrix.get(robot_room, {}).get(loc, self.inf) for p, loc in undelivered_pkgs)
                cost += dist_to_first
                for p, loc in undelivered_pkgs:
                    target = self.pkg_to_goal_room[p]
                    cost += 2 + self.dist_matrix.get(loc, {}).get(target, self.inf)
            else:
                cost = 0

        # Tie-breaker for task network progress
        return int(cost) + (0.001 * len(task_network))
