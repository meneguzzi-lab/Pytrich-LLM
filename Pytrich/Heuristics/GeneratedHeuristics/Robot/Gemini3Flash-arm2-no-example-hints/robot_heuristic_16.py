from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic16(Heuristic):
    """
    Robot Domain Heuristic
    
    Bottleneck: Single-carry constraint. The robot can only hold one package at a time, 
    forcing a strictly sequential delivery process (Pickup -> Move -> Putdown -> Move back).
    
    Admissibility Sketches:
    1. Action Lower Bound: Each undelivered package requires at least a pickup and a putdown (2 actions).
       If a package is already held, it needs 1 putdown. This is a hard floor on plan length.
    2. Navigation Lower Bound: The robot must traverse the shortest path between a package's start 
       and its goal. These distances (ignoring door states) are unavoidable move actions.
    3. Connectivity: Total distance from robot to its first target plus the sum of all 
       package-to-goal paths is a lower bound on robot travel.
    """
    
    def __init__(self, robot_gemini3flash_heuristic_16="RobotGemini3FlashHeuristic16"):
        super().__init__(robot_gemini3flash_heuristic_16)
        self.package_targets = {} # pkg -> room_name
        self.room_dist = {} # room1 -> room2 -> int
        self.pkg_ids = []
        self.room_ids = []
        
        # Fact Maps
        self.rloc_map = {} # room_name -> global_id
        self.in_map = {} # (pkg_name, room_name) -> global_id
        self.holding_map = {} # pkg_name -> global_id
        self.armempty_id = None
        
        self.room_names = []
        self.package_names = []

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
        # 1. Parse all facts to build the room graph and identify objects
        rooms = set()
        packages = set()
        adj = collections.defaultdict(list)
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "rloc":
                rooms.add(args[0])
                self.rloc_map[args[0]] = f.global_id
            elif pred == "in":
                packages.add(args[0])
                rooms.add(args[1])
                self.in_map[(args[0], args[1])] = f.global_id
            elif pred == "holding":
                packages.add(args[0])
                self.holding_map[args[0]] = f.global_id
            elif pred == "armempty":
                self.armempty_id = f.global_id
            elif pred == "door":
                rooms.add(args[0])
                rooms.add(args[1])
                adj[args[0]].append(args[1])
        
        self.room_names = list(rooms)
        self.package_names = list(packages)
        
        # 2. Precompute All-Pairs Shortest Paths (BFS since all costs are 1)
        for start in self.room_names:
            self.room_dist[start] = {r: 9999 for r in self.room_names}
            self.room_dist[start][start] = 0
            queue = collections.deque([(start, 0)])
            visited = {start}
            while queue:
                u, d = queue.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        self.room_dist[start][v] = d + 1
                        queue.append((v, d + 1))

        # 3. Determine Goal Locations
        # Check model.goals first (bitwise integer)
        goal_facts = model.state_explicit_repr(model.goals)
        for g_name in goal_facts:
            pred, args = self._parse_fact_name(g_name)
            if pred == "in":
                self.package_targets[args[0]] = args[1]
        
        # Fallback to goal_in in initial state if goals list is descriptive rather than explicit
        init_facts = model.state_explicit_repr(initial_node.state)
        for f_name in init_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == "goal_in":
                if args[0] not in self.package_targets:
                    self.package_targets[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        # Fast exit if no packages to deliver
        if not self.package_targets:
            return 0

        # 1. Identify current robot location
        robot_room = None
        for rname in self.room_names:
            if (state >> self.rloc_map[rname]) & 1:
                robot_room = rname
                break
        
        if robot_room is None: # Should not happen in valid states
            return 0

        # 2. Collect package status
        undelivered = [] # (pkg, current_room, target_room)
        holding_pkg = None

        for p in self.package_names:
            target = self.package_targets.get(p)
            if not target:
                continue
            
            # Is robot holding it?
            if (state >> self.holding_map[p]) & 1:
                holding_pkg = (p, target)
                continue
            
            # Is it at its target room?
            in_target_id = self.in_map.get((p, target))
            if in_target_id is not None and (state >> in_target_id) & 1:
                continue
            
            # Otherwise find where it is
            current_loc = None
            for r in self.room_names:
                if (state >> self.in_map[(p, r)]) & 1:
                    current_loc = r
                    break
            
            if current_loc:
                undelivered.append((p, current_loc, target))

        # 3. Calculate heuristic terms
        # Term A: Action count (pickup/putdown)
        h_actions = 2 * len(undelivered)
        if holding_pkg:
            h_actions += 1

        # Term B: Navigation distance (moves)
        h_nav = 0
        
        # Distance for packages not yet picked up
        for _, cur, tar in undelivered:
            h_nav += self.room_dist[cur][tar]
            
        # Distance for currently held package
        if holding_pkg:
            h_nav += self.room_dist[robot_room][holding_pkg[1]]
            # Plus potential travel to the next package after dropping this one
            if undelivered:
                h_nav += min(self.room_dist[holding_pkg[1]][p_start] for _, p_start, _ in undelivered)
        else:
            # If not holding, we must first travel to the nearest undelivered package
            if undelivered:
                h_nav += min(self.room_dist[robot_room][p_start] for _, p_start, _ in undelivered)

        # Term C: Small HTN network penalty to break ties
        h_network = 0.1 * len(task_network)

        return int(h_actions + h_nav + h_network)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
