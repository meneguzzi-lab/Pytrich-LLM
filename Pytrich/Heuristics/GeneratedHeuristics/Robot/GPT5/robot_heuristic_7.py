from typing import Dict, List, Set, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic7(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Design summary (admissible lower bound):
    - Delivery base actions: For each package not yet at its goal room, at least 2 primitive actions are needed: pickup + putdown.
      If the package is currently held, only the putdown remains (1 action). Sum across all unsatisfied packages.
    - Move actions: For each unsatisfied package, you must move from its pickup/source room to its goal room while holding it.
      We use the shortest path length in the fully-open door graph (precomputed all-pairs shortest paths) as a lower bound on
      the number of move actions for that package. Summed across all unsatisfied packages, this is still a valid LB (it ignores
      inter-delivery repositioning of the robot, so it underestimates total moves required).
    - Open actions: With currently open doors, rooms partition into connected components. Let required rooms be the union of:
      - the robot's current room
      - each unsatisfied package's current room (or robot's room if it's the held package)
      - each unsatisfied package's goal room
      To visit all required rooms starting from the robot's current room, at least (components - 1) door openings are required
      to connect these components. This is an admissible LB on the number of 'open' actions.

    Final heuristic value = base_pick_put + sum_min_moves + min_opens_needed.

    Implementation details:
    - Uses bitwise checks for state membership (fast), never converts state to strings except at initialization
      for extracting goal mapping via model.goals -> state_explicit_repr.
    - Precomputes:
      * Door endpoints per door-id (undirected edge)
      * All-pairs shortest path lengths on the fully-open room graph
      * Fact id maps for rloc, in(obj,room), holding(obj), and closed(door)
      * Goal mapping p -> goal_room from model.goals (target state is '+in[p,goal_room]')

    Notes:
    - This heuristic is designed to work efficiently on large instances (100 packages, ~50 rooms), keeping per-state
      evaluation linear in the number of relevant packages and rooms.
    - Admissibility is preserved by using independent lower bounds and ignoring interaction costs between different deliveries.
    """

    def __init__(self, model, robot_gpt5_heuristic_7: bool = True):
        super().__init__(model)
        self.robot_gpt5_heuristic_7 = robot_gpt5_heuristic_7  # retained as a flag (no functional effect)
        self.model = model

        # Parsed structures
        self.rooms: Set[str] = set()
        self.packages: Set[str] = set()

        # Fact id maps
        self.rloc_fact_ids: Dict[str, int] = {}              # room -> fact_id for +rloc[room]
        self.in_fact_ids: Dict[str, Dict[str, int]] = {}     # obj -> {room -> fact_id for +in[obj,room]}
        self.holding_fact_ids: Dict[str, int] = {}           # obj -> fact_id for +holding[obj]
        self.closed_fact_ids: Dict[str, int] = {}            # door_id -> fact_id for +closed[door]
        self.armempty_fact_id: Optional[int] = None

        # Door graph data
        self.door_rooms: Dict[str, Set[str]] = {}            # door_id -> {room1, room2}
        self.graph_all_open: Dict[str, Set[str]] = {}        # adjacency with all doors considered open
        self.all_pairs_dist: Dict[str, Dict[str, int]] = {}  # APSP distances in the fully-open graph

        # Goals: package -> goal_room
        self.goal_room_of: Dict[str, str] = {}

        # Prepare all metadata
        self._precompute_static_metadata()
        self._extract_goals_from_model()
        self._build_all_open_graph()
        self._precompute_all_pairs_shortest_paths()

    # ---------------------------- Core Heuristic API ----------------------------
    def evaluate(self, state_bitwise: int, task_network: List) -> int:
        # Determine robot's current room
        robot_room = self._get_robot_room(state_bitwise)

        # Determine which package (if any) is currently held
        held_pkg = self._get_held_package(state_bitwise)

        # Identify unsatisfied packages (those not yet at their goal room)
        unsatisfied: List[str] = []
        for pkg, goal_r in self.goal_room_of.items():
            in_gid = self._get_in_gid(pkg, goal_r)
            if in_gid is None:
                # If we can't find the fact id, be conservative: treat as unsatisfied (keeps admissibility)
                unsatisfied.append(pkg)
            else:
                if not self._bit_is_true(state_bitwise, in_gid):
                    unsatisfied.append(pkg)

        # Base pickup/putdown lower bound: 2 per unsatisfied, minus 1 if one is currently held
        base_pick_put = 2 * len(unsatisfied)
        if held_pkg is not None and held_pkg in unsatisfied:
            base_pick_put -= 1  # pickup already done for the held package

        # Move lower bound: sum of shortest path lengths (fully-open graph) from source to goal for each unsatisfied package
        sum_min_moves = 0
        required_rooms: Set[str] = set()
        if robot_room:
            required_rooms.add(robot_room)

        for pkg in unsatisfied:
            goal_r = self.goal_room_of.get(pkg)
            # Determine source room
            if held_pkg == pkg:
                src_r = robot_room
            else:
                src_r = self._get_current_room_of_pkg(state_bitwise, pkg)

            if src_r is None or goal_r is None:
                # Missing info -> contribute 0 (keeps admissibility)
                continue

            # Record rooms for open-doors connectivity LB
            required_rooms.add(src_r)
            required_rooms.add(goal_r)

            # Distance in the fully-open graph is a LB on number of move actions for this package
            d = self._dist_full_open(src_r, goal_r)
            if d is not None:
                sum_min_moves += d
            # else unreachable in fully open graph (shouldn't happen in valid instances) -> treat as 0 to stay safe

        # Open-door lower bound: at least (#components among required rooms in current-open graph) - 1
        opens_lb = 0
        if required_rooms:
            opens_lb = self._required_opens_lb(state_bitwise, required_rooms)

        h_val = base_pick_put + sum_min_moves + opens_lb
        return int(h_val)

    # ---------------------------- Precomputation ----------------------------
    def _precompute_static_metadata(self):
        # Parse all facts once
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue

            if pred == 'rloc' and len(args) == 1:
                room = args[0]
                self.rooms.add(room)
                self.rloc_fact_ids[room] = fact.global_id

            elif pred == 'in' and len(args) == 2:
                obj, room = args[0], args[1]
                self.rooms.add(room)
                self.packages.add(obj)
                if obj not in self.in_fact_ids:
                    self.in_fact_ids[obj] = {}
                self.in_fact_ids[obj][room] = fact.global_id

            elif pred == 'holding' and len(args) == 1:
                obj = args[0]
                self.packages.add(obj)
                self.holding_fact_ids[obj] = fact.global_id

            elif pred == 'closed' and len(args) == 1:
                door_id = args[0]
                self.closed_fact_ids[door_id] = fact.global_id

            elif pred == 'door' and len(args) == 3:
                loc1, loc2, door_id = args[0], args[1], args[2]
                self.rooms.add(loc1)
                self.rooms.add(loc2)
                if door_id not in self.door_rooms:
                    self.door_rooms[door_id] = set()
                # Collect endpoints per door-id
                self.door_rooms[door_id].add(loc1)
                self.door_rooms[door_id].add(loc2)

            elif pred == 'armempty':
                self.armempty_fact_id = fact.global_id

    def _extract_goals_from_model(self):
        # model.goals is a bitwise integer; convert to explicit fact names
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'in' and len(args) == 2:
                obj, goal_room = args[0], args[1]
                self.goal_room_of[obj] = goal_room

    def _build_all_open_graph(self):
        # Build adjacency assuming every door is open (static, from domain topology)
        self.graph_all_open = {r: set() for r in self.rooms}
        for door_id, endpoints in self.door_rooms.items():
            if len(endpoints) >= 2:
                # Expect exactly two endpoints; if more, connect all pairs (robustness)
                eps = list(endpoints)
                for i in range(len(eps)):
                    for j in range(i + 1, len(eps)):
                        a, b = eps[i], eps[j]
                        self.graph_all_open[a].add(b)
                        self.graph_all_open[b].add(a)

    def _precompute_all_pairs_shortest_paths(self):
        # BFS from each room to compute minimal move counts on fully-open graph
        self.all_pairs_dist = {r: {r2: (0 if r == r2 else None) for r2 in self.rooms} for r in self.rooms}
        for src in self.rooms:
            # BFS
            dist = {src: 0}
            queue = [src]
            head = 0
            while head < len(queue):
                cur = queue[head]
                head += 1
                for nb in self.graph_all_open.get(cur, ()): 
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        queue.append(nb)
            # Store
            for dst, d in dist.items():
                self.all_pairs_dist[src][dst] = d

    # ---------------------------- Helpers ----------------------------
    @staticmethod
    def _parse_fact_name(name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _bit_is_true(state_bitwise: int, fact_gid: int) -> bool:
        if fact_gid is None or fact_gid < 0:
            return False
        return (state_bitwise >> fact_gid) & 1 == 1

    def _get_robot_room(self, state_bitwise: int) -> Optional[str]:
        for room, fid in self.rloc_fact_ids.items():
            if self._bit_is_true(state_bitwise, fid):
                return room
        return None

    def _get_held_package(self, state_bitwise: int) -> Optional[str]:
        for pkg, fid in self.holding_fact_ids.items():
            if self._bit_is_true(state_bitwise, fid):
                return pkg
        return None

    def _get_in_gid(self, pkg: str, room: str) -> Optional[int]:
        d = self.in_fact_ids.get(pkg)
        if not d:
            return None
        return d.get(room)

    def _get_current_room_of_pkg(self, state_bitwise: int, pkg: str) -> Optional[str]:
        rooms_map = self.in_fact_ids.get(pkg)
        if not rooms_map:
            return None
        # Exactly one room should be true
        for room, gid in rooms_map.items():
            if self._bit_is_true(state_bitwise, gid):
                return room
        return None

    def _dist_full_open(self, src: str, dst: str) -> Optional[int]:
        dm = self.all_pairs_dist.get(src)
        if not dm:
            return None
        return dm.get(dst)

    def _required_opens_lb(self, state_bitwise: int, required_rooms: Set[str]) -> int:
        # Build adjacency consisting only of currently-open doors
        open_adj: Dict[str, Set[str]] = {r: set() for r in self.rooms}
        for door_id, endpoints in self.door_rooms.items():
            # A door is open if 'closed[door_id]' is NOT true in the current state
            closed_gid = self.closed_fact_ids.get(door_id)
            is_closed = self._bit_is_true(state_bitwise, closed_gid) if closed_gid is not None else False
            if not is_closed and len(endpoints) >= 2:
                eps = list(endpoints)
                for i in range(len(eps)):
                    for j in range(i + 1, len(eps)):
                        a, b = eps[i], eps[j]
                        open_adj[a].add(b)
                        open_adj[b].add(a)

        # Compute number of connected components among required rooms using open_adj
        visited: Set[str] = set()
        comps = 0

        for r in required_rooms:
            if r in visited:
                continue
            # BFS/DFS from r over open edges, but only mark rooms we can reach
            comps += 1
            stack = [r]
            visited.add(r)
            while stack:
                cur = stack.pop()
                for nb in open_adj.get(cur, ()):  # may be empty if isolated
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)

        return max(0, comps - 1)


# Alias required by some loaders (optional)
HeuristicClass = RobotGPT5Heuristic7
