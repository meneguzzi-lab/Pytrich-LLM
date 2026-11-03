import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic10(Heuristic):
    """
    Domain-dependent heuristic for the Robot planning domain.

    BOTTLENECK:
    The primary bottleneck is the single-arm constraint: the robot can only carry one package at a time,
    forcing a sequential pickup-move-putdown cycle for every misplaced package.

    ADMISSIBILITY SKETCHES:
    1. Transfer Cost Lower Bound: Each misplaced package requires at least a 'pickup' and a 'putdown'.
       If not held, this is 2 actions. If currently held, this is 1 action (the putdown).
       Summing these (2 * misplaced - 1 if holding) is a lower bound on required task-specific actions.
    2. Travel Cost Lower Bound: The robot must perform at least one 'move' for every room transition.
       The distance from the current robot location to the first target location (either the goal room 
       of a held package or the source room of a misplaced package) is a lower bound on 'move' actions.
    3. Multi-component Sum: Since moves and pickup/putdown actions are distinct and both necessary,
       their sum provides a tight admissible lower bound.
    """

    def __init__(self, robot_gemini3flash_heuristic_10=None, name="RobotGemini3FlashHeuristic10"):
        super().__init__(name=name)
        self.rooms = set()
        self.packages = set()
        self.package_to_goal_room = {}
        self.in_fact_ids = {}         # (pkg, room) -> global_id
        self.holding_fact_ids = {}    # pkg -> global_id
        self.rloc_fact_ids = {}       # room -> global_id
        self.dist_matrix = defaultdict(lambda: defaultdict(lambda: 999))

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._calculate_heuristic(initial_node.state)
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

    def _preprocess(self):
        adj = defaultdict(set)

        # First pass: map facts and build connectivity
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue

            if pred == "rloc":
                self.rooms.add(args[0])
                self.rloc_fact_ids[args[0]] = fact.global_id
            elif pred == "in":
                self.packages.add(args[0])
                self.rooms.add(args[1])
                self.in_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "holding":
                self.packages.add(args[0])
                self.holding_fact_ids[args[0]] = fact.global_id
            elif pred == "door":
                self.rooms.add(args[0])
                self.rooms.add(args[1])
                adj[args[0]].add(args[1])
                adj[args[1]].add(args[2]) # Bidirectional doors usually, but HDDL shows (door c r1 d01)
                # Based on robot.hddl (door loc1 loc2 d), we treat it as loc1 -> loc2
                adj[args[0]].add(args[1])
            elif pred == "goal_in":
                self.package_to_goal_room[args[0]] = args[1]

        # Backup: check model.goals if goal_in wasn't specified as a predicate
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "in":
                self.package_to_goal_room[args[0]] = args[1]

        # Precompute BFS distances between rooms (moves)
        for start_node in self.rooms:
            self.dist_matrix[start_node][start_node] = 0
            queue = deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                u, d = queue.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        self.dist_matrix[start_node][v] = d + 1
                        queue.append((v, d + 1))

    def _calculate_heuristic(self, state: int) -> int:
        # 1. Identify robot location
        current_rloc = None
        for r in self.rooms:
            if (state >> self.rloc_fact_ids.get(r, -1)) & 1 if r in self.rloc_fact_ids else 0:
                current_rloc = r
                break

        # 2. Identify misplaced packages and holding state
        misplaced = []
        holding_pkg = None
        for p in self.packages:
            is_held = (state >> self.holding_fact_ids[p]) & 1 if p in self.holding_fact_ids else 0
            if is_held:
                holding_pkg = p
                misplaced.append(p)
            else:
                goal_room = self.package_to_goal_room.get(p)
                if goal_room:
                    in_goal = (state >> self.in_fact_ids.get((p, goal_room), -1)) & 1 if (p, goal_room) in self.in_fact_ids else 0
                    if not in_goal:
                        misplaced.append(p)

        if not misplaced:
            return 0

        # 3. Transfer Cost: 2 per misplaced package, -1 if already holding one
        h_transfer = 2 * len(misplaced)
        if holding_pkg:
            h_transfer -= 1

        # 4. Travel Cost: Distance to the current task room
        h_travel = 0
        if current_rloc:
            if holding_pkg:
                # Must travel to goal of held package
                target_room = self.package_to_goal_room.get(holding_pkg)
                if target_room:
                    h_travel = self.dist_matrix[current_rloc][target_room]
            else:
                # Must travel to pick up some misplaced package
                min_dist = 999
                for p in misplaced:
                    # Find current room of package p
                    p_room = None
                    for r in self.rooms:
                        if (state >> self.in_fact_ids.get((p, r), -1)) & 1 if (p, r) in self.in_fact_ids else 0:
                            p_room = r
                            break
                    if p_room:
                        d = self.dist_matrix[current_rloc][p_room]
                        if d < min_dist: min_dist = d
                h_travel = min_dist if min_dist != 999 else 0

        # Final admissible heuristic value
        return h_transfer + h_travel

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value
