import time
from collections import deque, defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic8(Heuristic):
    """
    Domain-dependent heuristic for the Robot planning domain.

    1. Bottleneck Analysis:
       The primary bottleneck is the single robot arm (armempty). The robot can only carry 
       one package at a time. This forces a complete serialization of package deliveries: 
       each package not at its goal requires a sequence of: 
       travel to package -> pickup -> travel to goal -> putdown.

    2. Admissible Lower Bounds:
       - h_manip: Each unsettled package (not at goal location) requires exactly one 'pickup' and one 'putdown' (cost 2). 
         If a package is already being held, it only requires a 'putdown' (cost 1).
       - h_move: Each unsettled package at room A with goal room B requires at least Dist(A, B) 'move' actions.
         Since only one package can be carried at a time, the sum of these distances is an admissible lower bound.
       - h_robot_travel: If not holding a package, the robot must travel to the nearest unsettled package.
         If holding a package, the robot must travel to that package's goal.

    3. Admissibility Sketch:
       Summing (Dist(curr, goal) + 2) for all unsettled packages is admissible because the robot cannot 
       combine trips or carry multiple packages. Each package's delivery sequence is disjoint in terms 
       of required actions.
    """

    def __init__(self, robot_gemini3flash_heuristic_8=None, name="RobotGemini3FlashHeuristic8"):
        super().__init__(name=name)
        self.package_goals = {}  # pkg -> goal_room
        self.rooms = []
        self.packages = []
        self.dist_matrix = {}
        self.rloc_facts = {}     # room -> global_id
        self.in_facts = {}       # pkg -> {room -> global_id}
        self.holding_facts = {}  # pkg -> global_id
        self.armempty_fact = None

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        adj = defaultdict(list)
        
        # 1. Parse all facts to identify entities and static structure
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue

            if pred == "rloc":
                room = args[0]
                self.rooms.append(room)
                self.rloc_facts[room] = fact.global_id
            elif pred == "in":
                pkg, room = args[0], args[1]
                if pkg not in self.packages: self.packages.append(pkg)
                if pkg not in self.in_facts: self.in_facts[pkg] = {}
                self.in_facts[pkg][room] = fact.global_id
            elif pred == "holding":
                pkg = args[0]
                self.holding_facts[pkg] = fact.global_id
            elif pred == "armempty":
                self.armempty_fact = fact.global_id
            elif pred == "door":
                r1, r2, d = args[0], args[1], args[2]
                adj[r1].append(r2)
                adj[r2].append(r1)
            elif pred == "goal_in":
                self.package_goals[args[0]] = args[1]

        # 2. Precompute All-Pairs Shortest Paths (BFS on rooms)
        self.rooms = list(set(self.rooms))
        for start_room in self.rooms:
            self.dist_matrix[start_room] = {r: 999 for r in self.rooms}
            self.dist_matrix[start_room][start_room] = 0
            queue = deque([(start_room, 0)])
            visited = {start_room}
            while queue:
                curr, d = queue.popleft()
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.dist_matrix[start_room][neighbor] = d + 1
                        queue.append((neighbor, d + 1))

        h_initial = self._compute(initial_node.state, initial_node.task_network)
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

    def _compute(self, state, task_network):
        # 1. Identify Robot Location
        robot_loc = None
        for room, fact_id in self.rloc_facts.items():
            if (state >> fact_id) & 1:
                robot_loc = room
                break
        
        if not robot_loc: return 1000 # Should not happen in valid state

        # 2. Identify Package Status
        holding_pkg = None
        for pkg, fact_id in self.holding_facts.items():
            if (state >> fact_id) & 1:
                holding_pkg = pkg
                break

        unsettled = [] # Packages not at goal room and not currently held
        for pkg in self.packages:
            if pkg == holding_pkg: continue
            goal_room = self.package_goals.get(pkg)
            if not goal_room: continue
            
            # Check if package is at its goal room
            goal_fact_id = self.in_facts[pkg].get(goal_room)
            if goal_fact_id is not None and not ((state >> goal_fact_id) & 1):
                # Find its current room
                curr_room = None
                for room, in_id in self.in_facts[pkg].items():
                    if (state >> in_id) & 1:
                        curr_room = room
                        break
                if curr_room:
                    unsettled.append((pkg, curr_room, goal_room))

        # 3. Calculate Heuristic Value
        h = 0
        
        # Cost for package currently held
        if holding_pkg:
            goal_room = self.package_goals.get(holding_pkg)
            if goal_room:
                h += self.dist_matrix[robot_loc][goal_room] + 1 # Travel to goal + putdown
            else:
                h += 1 # Just putdown

        # Cost for all other unsettled packages
        for pkg, curr, goal in unsettled:
            # Robot must: travel to curr, pickup, travel to goal, putdown
            # We sum Dist(curr, goal) + 2 for each. 
            # (Travel between packages is handled separately for admissibility)
            h += self.dist_matrix[curr][goal] + 2

        # Additional cost: Robot travel to first target
        if unsettled:
            if not holding_pkg:
                # Robot must travel to the closest package to begin
                h += min(self.dist_matrix[robot_loc][pkg_curr] for _, pkg_curr, _ in unsettled)
            else:
                # After dropping current package, must travel to the next package
                goal_of_held = self.package_goals.get(holding_pkg)
                if goal_of_held:
                    h += min(self.dist_matrix[goal_of_held][pkg_curr] for _, pkg_curr, _ in unsettled)

        # Tie-breaker: prefer smaller task networks
        h += len(task_network) * 0.001

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return int(h_value)
