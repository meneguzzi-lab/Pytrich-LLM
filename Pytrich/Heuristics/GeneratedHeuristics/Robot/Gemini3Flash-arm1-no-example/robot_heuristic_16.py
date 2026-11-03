from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini3FlashHeuristic16(Heuristic):
    """
    Heuristic for the Robot domain.
    
    Bottleneck Analysis:
    The primary bottleneck in this domain is the single-arm robot capacity. The robot can only hold one package 
    at a time, forcing a sequential approach where each package delivery (pickup, travel, putdown) must be 
    completed (or the package dropped) before another can be moved.
    
    Admissible Lower Bounds:
    1. Action Sum: Each package not at its goal requires at least one pickup and one putdown (2 actions).
       If the robot is currently holding a package, it requires at least one putdown (1 action).
       The sum of these is admissible because each pickup/putdown action is unique to that package.
    2. Disjoint Travel: Each delivery requires the robot to move from the package's origin to its goal room.
       Since only one package is carried, these move-sets are disjoint in time. The sum of shortest-path 
       distances (hops) between origin and goal for all packages is a lower bound on movement actions.
    3. Robot Circuit: The robot must travel from its current location to the first package's origin.
    """

    def __init__(self, robot_gemini3flash_heuristic_16="robot_gemini3flash_heuristic_16"):
        super().__init__(name=robot_gemini3flash_heuristic_16)
        self.pkg_goal_room = {}
        self.rloc_fact_ids = {}
        self.holding_fact_ids = {}
        self.pkg_room_to_in_fact = {}
        self.room_dist = {}
        self.package_names = []
        self.all_rooms = []

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
        temp_adj = {}
        all_rooms_set = set()
        all_pkgs_set = set()

        # Parse facts from the initial state to find static properties and object names
        explicit_init = model.state_explicit_repr(initial_node.state)
        for fact_name in explicit_init:
            pred, args = self._parse_fact_name(fact_name)
            if pred == "goal_in":
                self.pkg_goal_room[args[0]] = args[1]
            if pred == "door":
                r1, r2, d = args
                if r1 not in temp_adj: temp_adj[r1] = []
                temp_adj[r1].append(r2)
                all_rooms_set.add(r1)
                all_rooms_set.add(r2)

        # 2. Index dynamic facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                room = args[0]
                self.rloc_fact_ids[room] = fact.global_id
                all_rooms_set.add(room)
            elif pred == "holding":
                pkg = args[0]
                self.holding_fact_ids[pkg] = fact.global_id
                all_pkgs_set.add(pkg)
            elif pred == "in":
                pkg, room = args
                if pkg not in self.pkg_room_to_in_fact: self.pkg_room_to_in_fact[pkg] = []
                self.pkg_room_to_in_fact[pkg].append((room, fact.global_id))
                all_pkgs_set.add(pkg)
                all_rooms_set.add(room)

        self.package_names = list(all_pkgs_set)
        self.all_rooms = list(all_rooms_set)

        # 3. Precompute All-Pairs Shortest Paths (Shortest move distance ignoring door status for admissibility)
        for r1 in self.all_rooms:
            self.room_dist[r1] = {r2: 1000 for r2 in self.all_rooms}
            self.room_dist[r1][r1] = 0
            queue = deque([r1])
            while queue:
                curr = queue.popleft()
                for neighbor in temp_adj.get(curr, []):
                    if self.room_dist[r1][neighbor] == 1000:
                        self.room_dist[r1][neighbor] = self.room_dist[r1][curr] + 1
                        queue.append(neighbor)

        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def _compute(self, state: int) -> int:
        # Determine robot state
        robot_loc = None
        for r in self.all_rooms:
            if (state >> self.rloc_fact_ids.get(r, -1)) & 1:
                robot_loc = r
                break
        
        held_pkg = None
        for p in self.package_names:
            if (state >> self.holding_fact_ids.get(p, -1)) & 1:
                held_pkg = p
                break

        if robot_loc is None:
            return 0

        h = 0
        robot_is_at = robot_loc
        wrong_packages = [] # list of (origin_room, goal_room)

        # 1. Work for held package
        if held_pkg:
            goal_r = self.pkg_goal_room.get(held_pkg)
            if goal_r:
                h += 1 + self.room_dist[robot_loc][goal_r]
                robot_is_at = goal_r
            else:
                h += 1 # Just a putdown if no goal defined (should not happen)

        # 2. Work for packages at rest but not at goal
        for p in self.package_names:
            if p == held_pkg: continue
            
            p_loc = None
            for r, f_id in self.pkg_room_to_in_fact.get(p, []):
                if (state >> f_id) & 1:
                    p_loc = r
                    break
            
            target_r = self.pkg_goal_room.get(p)
            if p_loc and target_r and p_loc != target_r:
                # Pickup (1) + moves (dist) + Putdown (1)
                h += 2 + self.room_dist[p_loc][target_r]
                wrong_packages.append(p_loc)

        # 3. Circuit cost: distance to start the next delivery
        if wrong_packages:
            min_d_start = min(self.room_dist[robot_is_at][p_loc] for p_loc in wrong_packages)
            if min_d_start < 1000:
                h += min_d_start

        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        # Add a tiny tie-breaker for task network depth to favor progressing decompositions
        h_with_tie_breaker = h_value + 0.001 * len(node.task_network)
        super().update_info(h_value)
        return h_value
