from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic3(Heuristic):
    """
    Robot Domain Heuristic Analysis:
    1. Bottleneck: The single-arm constraint (robot can carry only one package at a time) is the primary serialized resource.
    2. Admissible Lower Bounds:
       - Package Delivery Cost: Each undelivered package requires at least 1 pickup, N moves (shortest path), and 1 putdown.
         Since the robot moves them one by one, the sum of these transition costs is a valid lower bound.
       - Robot Travel: The robot must travel to the next target (package or goal location).
       - Task Network: A small tie-breaker proportional to the network length helps guide the search toward completion.
    3. Complexity: initialize() runs Floyd-Warshall/BFS in O(V^3), and __call__() performs O(N*V) bitwise checks.
    """

    def __init__(self, robot_gemini3flash_heuristic_3="robot_gemini3flash_heuristic_3"):
        super().__init__(robot_gemini3flash_heuristic_3)
        self.rooms = set()
        self.packages = set()
        self.in_ids = {}        # (pkg, room) -> global_id
        self.holding_ids = {}   # pkg -> global_id
        self.rloc_ids = {}      # room -> global_id
        self.pkg_goal = {}      # pkg -> target_room
        self.adj = {}           # room -> list of neighbors
        self.room_dist = {}     # (r1, r2) -> min moves

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
        # Pre-process facts to identify objects and IDs
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            if pred == "rloc" and len(args) == 1:
                self.rooms.add(args[0])
                self.rloc_ids[args[0]] = fact.global_id
            elif pred == "in" and len(args) == 2:
                self.packages.add(args[0])
                self.rooms.add(args[1])
                self.in_ids[(args[0], args[1])] = fact.global_id
            elif pred == "holding" and len(args) == 1:
                self.packages.add(args[0])
                self.holding_ids[args[0]] = fact.global_id

        # Identify static connectivity and goal locations from initial state
        self.adj = {r: [] for r in self.rooms}
        for fact in model.facts:
            if (initial_node.state >> fact.global_id) & 1:
                pred, args = self._parse_fact_name(fact.name)
                if not pred: continue
                if pred == "door" and len(args) >= 2:
                    r1, r2 = args[0], args[1]
                    if r1 in self.rooms and r2 in self.rooms:
                        self.adj[r1].append(r2)
                elif pred == "goal_in" and len(args) == 2:
                    self.pkg_goal[args[0]] = args[1]

        # Precompute shortest move paths using BFS from each room
        self.room_dist = {r: {r2: 1000 for r2 in self.rooms} for r in self.rooms}
        for start in self.rooms:
            self.room_dist[start][start] = 0
            queue = [(start, 0)]
            visited = {start}
            while queue:
                u, d = queue.pop(0)
                for v in self.adj[u]:
                    if v not in visited:
                        visited.add(v)
                        self.room_dist[start][v] = d + 1
                        queue.append((v, d + 1))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        # 1. Identify robot location
        robot_at = None
        for r in self.rooms:
            if (state >> self.rloc_ids.get(r, -1)) & 1:
                robot_at = r
                break
        if robot_at is None: return 0

        # 2. Check current cargo
        held_p = None
        for p in self.packages:
            if (state >> self.holding_ids.get(p, -1)) & 1:
                held_p = p
                break

        # 3. Identify undelivered packages
        undelivered = []
        for p in self.packages:
            if p == held_p: continue
            target = self.pkg_goal.get(p)
            if target:
                # If not at goal room
                goal_fact_id = self.in_ids.get((p, target))
                if goal_fact_id is None or not ((state >> goal_fact_id) & 1):
                    undelivered.append(p)

        # 4. Calculate admissible cost sum
        h = 0
        curr_robot_r = robot_at
        
        # Cost for currently held package
        if held_p:
            target = self.pkg_goal.get(held_p)
            if target:
                h += self.room_dist[curr_robot_r][target] + 1 # Moves + putdown
                curr_robot_r = target # Robot will be here after delivery

        # Individual costs for undelivered packages (pickup + move + putdown)
        p_locs = []
        for p in undelivered:
            target = self.pkg_goal[p]
            # Find where package is currently located
            ploc = None
            for r in self.rooms:
                in_fact_id = self.in_ids.get((p, r))
                if in_fact_id is not None and (state >> in_fact_id) & 1:
                    ploc = r
                    break
            if ploc:
                h += 1 + self.room_dist[ploc][target] + 1 # pickup + moves + putdown
                p_locs.append(ploc)
        
        # Cost to reach the first undelivered package
        if p_locs:
            h += min(self.room_dist[curr_robot_r][loc] for loc in p_locs)

        # Tie-breaker for progress
        return h + 0.001 * len(task_network)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
