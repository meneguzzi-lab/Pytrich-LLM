from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque, defaultdict

class RobotGemini3FlashHeuristic7(Heuristic):
    """
    Robot Domain Heuristic
    
    Bottleneck Analysis:
    The single-carry constraint is the primary bottleneck. Every package delivery requires 
    a serialized round-trip (travel to package, pickup, travel to goal, putdown). 
    Since the robot can only hold one package at a time, the plan length is dominated by 
    the sum of the delivery actions (2 per package) and the travel distances.

    Admissibility Sketches:
    1. Actions Term (h_actions): Every package not yet at its destination requires 1 putdown. 
       If it's not currently held, it also requires 1 pickup. Total: 2*N - (1 if holding). 
       This is a strict lower bound on mandatory actions.
    2. Travel Term (h_moves): The robot must traverse the shortest path for each package delivery.
       The distance sum of dist(loc, goal) for every undelivered package, plus the distance 
       to the current target, is a lower bound because it ignores transition travel between 
       deliveries (from goal A to pickup B).
    3. Door Term (h_opens): Every closed door on a necessary path must be opened at least once.
       Counting unique closed doors required for the calculated shortest paths provides an 
       additional unavoidable cost.
    """

    def __init__(self, robot_gemini3flash_heuristic_7="RobotGemini3FlashHeuristic7"):
        super().__init__(robot_gemini3flash_heuristic_7)
        self.pkg_goals = {}        # pkg -> goal_room
        self.rooms = []
        self.pkgs = []
        self.rloc_fact_ids = {}    # room -> global_id
        self.in_fact_ids = {}      # pkg -> room -> global_id
        self.holding_fact_ids = {} # pkg -> global_id
        self.closed_fact_ids = {}  # door -> global_id
        self.armempty_id = None
        self.dist = {}             # r1 -> r2 -> distance
        self.doors_on_path = {}    # r1 -> r2 -> set(door_names)
        self.pkgs_with_goals = []

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
            else:
                return name, []
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates to global IDs for fast bitwise access
        temp_rooms = set()
        temp_pkgs = set()
        adj = defaultdict(list)

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "rloc":
                self.rloc_fact_ids[args[0]] = f.global_id
                temp_rooms.add(args[0])
            elif pred == "in":
                pkg, room = args
                if pkg not in self.in_fact_ids: self.in_fact_ids[pkg] = {}
                self.in_fact_ids[pkg][room] = f.global_id
                temp_pkgs.add(pkg)
                temp_rooms.add(room)
            elif pred == "holding":
                self.holding_fact_ids[args[0]] = f.global_id
                temp_pkgs.add(args[0])
            elif pred == "closed":
                self.closed_fact_ids[args[0]] = f.global_id
            elif pred == "armempty":
                self.armempty_id = f.global_id
            elif pred == "door":
                r1, r2, d_name = args
                adj[r1].append((r2, d_name))
                temp_rooms.add(r1)
                temp_rooms.add(r2)

        self.rooms = list(temp_rooms)
        self.pkgs = list(temp_pkgs)

        # 2. Parse Goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "in":
                self.pkg_goals[args[0]] = args[1]
        
        self.pkgs_with_goals = [p for p in self.pkgs if p in self.pkg_goals]

        # 3. Precompute shortest paths and mandatory doors using BFS
        for start_room in self.rooms:
            self.dist[start_room] = {start_room: 0}
            self.doors_on_path[start_room] = {start_room: frozenset()}
            queue = deque([start_room])
            while queue:
                u = queue.popleft()
                for v, door_name in adj[u]:
                    if v not in self.dist[start_room]:
                        self.dist[start_room][v] = self.dist[start_room][u] + 1
                        new_doors = set(self.doors_on_path[start_room][u])
                        new_doors.add(door_name)
                        self.doors_on_path[start_room][v] = frozenset(new_doors)
                        queue.append(v)

        robot_gemini3flash_heuristic_7 = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, robot_gemini3flash_heuristic_7)

    def _compute(self, state, task_network):
        # Identify robot location
        robot_room = None
        for r in self.rooms:
            if (state >> self.rloc_fact_ids[r]) & 1:
                robot_room = r
                break
        
        if robot_room is None: return 1000

        # Identify held package
        held_pkg = None
        for p in self.pkgs_with_goals:
            if (state >> self.holding_fact_ids[p]) & 1:
                held_pkg = p
                break

        h_actions = 0
        h_moves = 0
        needed_doors = set()
        undelivered_locs = []

        # Process currently held package first
        if held_pkg:
            h_actions += 1 # putdown
            dest = self.pkg_goals[held_pkg]
            h_moves += self.dist[robot_room].get(dest, 0)
            needed_doors.update(self.doors_on_path[robot_room].get(dest, []))

        # Process all other packages not at goal
        for p in self.pkgs_with_goals:
            if p == held_pkg: continue
            
            # Fast check: is package at its goal room?
            dest = self.pkg_goals[p]
            if (state >> self.in_fact_ids[p][dest]) & 1:
                continue
            
            # Not at goal, find where it is
            curr_loc = None
            for r in self.rooms:
                if (state >> self.in_fact_ids[p][r]) & 1:
                    curr_loc = r
                    break
            
            if curr_loc:
                h_actions += 2 # pickup + putdown
                h_moves += self.dist[curr_loc].get(dest, 0)
                needed_doors.update(self.doors_on_path[curr_loc].get(dest, []))
                undelivered_locs.append(curr_loc)

        # Add travel to the first package if not holding anything
        if not held_pkg and undelivered_locs:
            min_dist_to_pkg = min(self.dist[robot_room].get(loc, 10) for loc in undelivered_locs)
            h_moves += min_dist_to_pkg
            # (Door check for first move is omitted for speed, covered by set update if needed later)

        # Count closed doors required for paths
        h_opens = 0
        for d_name in needed_doors:
            if (state >> self.closed_fact_ids[d_name]) & 1:
                h_opens += 1

        return int(h_actions + h_moves + h_opens)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        robot_gemini3flash_heuristic_7 = self._compute(node.state, node.task_network)
        super().update_info(robot_gemini3flash_heuristic_7)
        return robot_gemini3flash_heuristic_7
