from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic11(Heuristic):
    """
    Robot Domain Heuristic Analysis:
    
    1. Bottleneck: Single-carry constraint. The robot can only hold one package at a time. 
       Every delivery requires a serial sequence of (travel to package, pickup, travel to goal, putdown).

    2. Admissible Lower Bounds:
       - Delivery Cost: Each undelivered package requires at least one pickup and one putdown (2 actions) 
         plus the distance between its source and destination.
       - Robot Travel: If the robot is not holding a package, it must at least travel to the nearest 
         undelivered package's source room.
       - Single-Carriage Serial Cost: Total actions = Sum(pickup + move_dist(source, dest) + putdown) 
         for all undelivered packages. This is admissible as it ignores travel between delivery legs.

    3. Preprocessing:
       - Build a room connectivity graph and precompute all-pairs shortest paths (BFS).
       - Map packages to their target goal rooms using goal_in facts.
       - Map package/room names to their respective fact global IDs for fast bitwise checking.
    """

    def __init__(self, robot_gemini3flash_heuristic_11="robot_heuristic"):
        super().__init__(robot_gemini3flash_heuristic_11)
        self.dist = {}  # room -> room -> int
        self.goal_in_map = {}  # pkg_name -> room_name
        self.rloc_fact_id = {}  # room_name -> global_id
        self.holding_fact_id = {}  # pkg_name -> global_id
        self.in_fact_id = {}  # (pkg_name, room_name) -> global_id
        self.pkg_in_fact_ids = collections.defaultdict(list)  # pkg_name -> [(global_id, room_name)]
        self.rooms = []
        self.packages = []

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates to IDs and gather objects
        adj = collections.defaultdict(list)
        pkg_set = set()
        room_set = set()

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "rloc":
                self.rloc_fact_id[args[0]] = fact.global_id
                room_set.add(args[0])
            elif pred == "holding":
                self.holding_fact_id[args[0]] = fact.global_id
                pkg_set.add(args[0])
            elif pred == "in":
                self.in_fact_id[(args[0], args[1])] = fact.global_id
                self.pkg_in_fact_ids[args[0]].append((fact.global_id, args[1]))
                pkg_set.add(args[0])
                room_set.add(args[1])
            elif pred == "door":
                adj[args[0]].append(args[1])
                room_set.add(args[0])
                room_set.add(args[1])
            elif pred == "goal_in":
                self.goal_in_map[args[0]] = args[1]

        self.rooms = list(room_set)
        self.packages = list(pkg_set)

        # 2. Extract targets from model goals if goal_in wasn't in state
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "in":
                self.goal_in_map[args[0]] = args[1]

        # 3. All-pairs shortest paths (BFS)
        for start_node in self.rooms:
            dists = {r: 999 for r in self.rooms}
            dists[start_node] = 0
            queue = collections.deque([start_node])
            while queue:
                u = queue.popleft()
                for v in adj.get(u, []):
                    if dists[v] == 999:
                        dists[v] = dists[u] + 1
                        queue.append(v)
            self.dist[start_node] = dists

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        if not task_network:
            return 0

        # 1. Identify Robot Location
        robot_loc = None
        for r_name in self.rooms:
            if (state >> self.rloc_fact_id[r_name]) & 1:
                robot_loc = r_name
                break
        if not robot_loc: return 0

        # 2. Identify Status of Packages
        held_pkg = None
        undelivered_pkgs = []
        
        for p in self.packages:
            if p in self.holding_fact_id and (state >> self.holding_fact_id[p]) & 1:
                held_pkg = p
            else:
                target = self.goal_in_map.get(p)
                if not target: continue
                at_goal_id = self.in_fact_id.get((p, target))
                if at_goal_id is None or not ((state >> at_goal_id) & 1):
                    undelivered_pkgs.append(p)

        # 3. Sum Costs
        h_val = 0
        min_dist_to_pkg = 999 if undelivered_pkgs else 0

        if held_pkg:
            target = self.goal_in_map.get(held_pkg)
            if target:
                h_val += 1 + self.dist[robot_loc][target] # putdown + moves
        
        for p in undelivered_pkgs:
            # Find where p is currently
            p_loc = None
            for f_id, r_name in self.pkg_in_fact_ids[p]:
                if (state >> f_id) & 1:
                    p_loc = r_name
                    break
            
            if p_loc:
                target = self.goal_in_map[p]
                # Cost to deliver: pickup (1) + move_dist + putdown (1)
                h_val += 2 + self.dist[p_loc][target]
                # If robot is not holding anything, must reach a package source
                if not held_pkg:
                    min_dist_to_pkg = min(min_dist_to_pkg, self.dist[robot_loc][p_loc])

        if not held_pkg and undelivered_pkgs:
            h_val += min_dist_to_pkg

        return h_val

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
