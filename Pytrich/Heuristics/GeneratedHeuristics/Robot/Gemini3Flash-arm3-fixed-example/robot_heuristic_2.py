import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic2(Heuristic):
    """
    Domain-dependent heuristic for the Robot domain.

    Bottleneck Analysis:
    The primary bottleneck is the robot's single arm, which serializes the delivery 
    of packages. Each misplaced package requires exactly one pickup and one putdown, 
    and the robot must move between rooms to fetch and deliver them.

    Admissibility Sketches:
    1. Delivery Actions: Each package not at its goal room requires a 'pickup' and a 'putdown'.
       If the robot is already holding the package, only 'putdown' is required. These are 
       distinct actions, providing a lower bound of (2 * unsettled + 1 * held).
    2. Travel Distance: The robot must perform 'move' actions to travel between the current room,
       the package source room, and the goal room. BFS distances provide the minimum move actions required.
    3. Door Opening: If a door is closed, an 'open' action is mandatory. Adding 1 for each 
       unavoidable closed door is admissible as 'open' is a distinct operator.
    """

    def __init__(self, robot_gemini3flash_heuristic_2=None, name="RobotGemini3FlashHeuristic2"):
        super().__init__(name=name)
        self.package_goals = {}  # pkg -> room_name
        self.rooms = []
        self.packages = []
        self.adj = defaultdict(list)
        self.dists = {}

        # Global IDs
        self.rloc_ids = {}      # room -> id
        self.in_ids = {}        # (pkg, room) -> id
        self.holding_ids = {}   # pkg -> id
        self.closed_ids = {}    # door -> id

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute(initial_node.state, initial_node.task_network)
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
        rooms_set = set()
        pkgs_set = set()
        doors_data = [] # (r1, r2, d)

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                rooms_set.add(args[0])
                self.rloc_ids[args[0]] = fact.global_id
            elif pred == "in":
                pkgs_set.add(args[0])
                rooms_set.add(args[1])
                self.in_ids[(args[0], args[1])] = fact.global_id
            elif pred == "holding":
                pkgs_set.add(args[0])
                self.holding_ids[args[0]] = fact.global_id
            elif pred == "goal_in":
                self.package_goals[args[0]] = args[1]
            elif pred == "door":
                doors_data.append((args[0], args[1], args[2]))
                self.adj[args[0]].append(args[1])
                self.adj[args[1]].append(args[0])
            elif pred == "closed":
                self.closed_ids[args[2]] = fact.global_id

        self.rooms = list(rooms_set)
        self.packages = list(pkgs_set)

        # Precompute All-Pairs Shortest Paths (BFS)
        for start_node in self.rooms:
            self.dists[start_node] = {r: 999999 for r in self.rooms}
            self.dists[start_node][start_node] = 0
            queue = deque([start_node])
            while queue:
                u = queue.popleft()
                for v in self.adj[u]:
                    if self.dists[start_node][v] == 999999:
                        self.dists[start_node][v] = self.dists[start_node][u] + 1
                        queue.append(v)

    def _compute(self, state, task_network):
        if not task_network and (state & self.model.goals == self.model.goals):
            return 0

        robot_loc = None
        for room in self.rooms:
            if (state >> self.rloc_ids[room]) & 1:
                robot_loc = room
                break
        
        if robot_loc is None:
            return 0

        pkgs_unsettled = [] # List of (current_loc, goal_loc)
        pkg_held_goal = None

        for pkg in self.packages:
            goal = self.package_goals.get(pkg)
            if not goal: continue

            # Check if held
            if (state >> self.holding_ids[pkg]) & 1:
                pkg_held_goal = goal
            else:
                # Check where it is
                at_goal = False
                for room in self.rooms:
                    if (state >> self.in_ids[(pkg, room)]) & 1:
                        if room == goal:
                            at_goal = True
                        else:
                            pkgs_unsettled.append((room, goal))
                        break

        # Admissible Action Count: 2 actions per misplaced pkg, 1 if held
        h_delivery = (len(pkgs_unsettled) * 2) + (1 if pkg_held_goal else 0)

        # Admissible Travel Count: moves to first target and between source/goal
        h_travel = 0
        if pkg_held_goal:
            h_travel = self.dists[robot_loc][pkg_held_goal]
        elif pkgs_unsettled:
            min_path = 999999
            for curr, goal in pkgs_unsettled:
                dist_to_pkg = self.dists[robot_loc][curr]
                dist_to_goal = self.dists[curr][goal]
                min_path = min(min_path, dist_to_pkg + dist_to_goal)
            h_travel = min_path if min_path != 999999 else 0

        return h_delivery + h_travel

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
