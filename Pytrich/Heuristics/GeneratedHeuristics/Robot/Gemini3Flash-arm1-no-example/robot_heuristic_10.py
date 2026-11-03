import collections
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic10(Heuristic):
    """
    Domain-Dependent Heuristic for the Robot Domain.
    
    Bottleneck Analysis:
    The primary bottleneck in this domain is the single robot arm, which serializes the transport 
    of packages. Each misallocated package requires a sequence of: move to pickup, pickup, 
    move to goal, and putdown. Since only one package can be held at a time, these actions 
    (specifically the moves with a package and the pickup/putdown actions) are strictly disjoint 
    and additive across all misallocated packages.

    Admissible Terms:
    1. Transport Cost: Each package not at its goal room requires at least a pickup (if not held) 
       and a putdown (1 or 2 actions). 
    2. Carriage Distance: The minimum number of move actions required to carry each misplaced 
       package from its current (or pickup) location to its goal location. 
    3. Reach Distance: The distance the robot must travel to reach the next package to be moved.

    The heuristic sums these costs because the robot must perform them sequentially.
    """

    def __init__(self, robot_gemini3flash_heuristic_10="robot_gemini3flash_heuristic_10"):
        super().__init__(robot_gemini3flash_heuristic_10)
        self.goal_in_map = {} # pkg_name -> target_room_name
        self.rooms = []
        self.packages = []
        self.rloc_fact_list = [] # List of (room_name, global_id)
        self.holding_fact_list = [] # List of (pkg_name, global_id)
        self.pkg_in_fact_list = collections.defaultdict(list) # pkg_name -> [(room_name, global_id)]
        self.room_dist = {}

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
        # 1. Parse goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "goal_in":
                self.goal_in_map[args[0]] = args[1]

        # 2. Map facts and build room adjacency
        adj = collections.defaultdict(list)
        all_rooms = set()
        pkgs = set()

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                self.rloc_fact_list.append((args[0], fact.global_id))
                all_rooms.add(args[0])
            elif pred == "holding":
                self.holding_fact_list.append((args[0], fact.global_id))
                pkgs.add(args[0])
            elif pred == "in":
                self.pkg_in_fact_list[args[0]].append((args[1], fact.global_id))
                pkgs.add(args[0])
                all_rooms.add(args[1])
            elif pred == "door":
                adj[args[0]].append(args[1])
                adj[args[1]].append(args[0])
                all_rooms.add(args[0])
                all_rooms.add(args[1])
        
        self.rooms = list(all_rooms)
        self.packages = list(self.goal_in_map.keys())

        # 3. Precompute All-Pairs Shortest Paths for Rooms
        for start_node in self.rooms:
            self.room_dist[start_node] = {r: 1000 for r in self.rooms}
            self.room_dist[start_node][start_node] = 0
            queue = collections.deque([start_node])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if self.room_dist[start_node][v] == 1000:
                        self.room_dist[start_node][v] = self.room_dist[start_node][u] + 1
                        queue.append(v)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        h = 0
        
        # Identify current robot location
        curr_rloc = None
        for loc_name, fid in self.rloc_fact_list:
            if (state >> fid) & 1:
                curr_rloc = loc_name
                break
        
        if curr_rloc is None: 
            return 1000

        # Identify what is being held
        held_pkg = None
        for pkg_name, fid in self.holding_fact_list:
            if (state >> fid) & 1:
                held_pkg = pkg_name
                break

        pkgs_to_move_locs = []

        # Case 1: Robot is holding a package
        if held_pkg:
            target = self.goal_in_map.get(held_pkg)
            if target:
                # Cost: 1 (putdown) + travel to target
                h += 1 + self.room_dist[curr_rloc].get(target, 100)

        # Case 2: Packages in rooms
        for pkg in self.packages:
            if pkg == held_pkg: 
                continue
            
            # Current location of package
            pkg_loc = None
            for loc_name, fid in self.pkg_in_fact_list[pkg]:
                if (state >> fid) & 1:
                    pkg_loc = loc_name
                    break
            
            target = self.goal_in_map[pkg]
            if pkg_loc and pkg_loc != target:
                # Cost: 1 (pickup) + 1 (putdown) + travel from pickup to target
                h += 2 + self.room_dist[pkg_loc].get(target, 100)
                pkgs_to_move_locs.append(pkg_loc)

        # Case 3: Robot reach distance (to the nearest pickup location)
        if not held_pkg and pkgs_to_move_locs:
            min_reach = min(self.room_dist[curr_rloc].get(loc, 100) for loc in pkgs_to_move_locs)
            h += min_reach

        # Tie-breaker: small penalty for items in the task network
        h += 0.001 * len(task_network)

        return int(h) if h < 1000000 else 1000000
