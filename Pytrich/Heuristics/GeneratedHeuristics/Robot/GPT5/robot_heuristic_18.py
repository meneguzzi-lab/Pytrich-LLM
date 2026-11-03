from collections import deque, defaultdict
from typing import Dict, List, Tuple, Set, Optional

from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic18(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Admissible lower bound combining four components:
    - Putdowns LB: at least one putdown per unsatisfied +in[obj,goal_room]
    - Pickups LB: at least one pickup per unsatisfied package, except the currently held one (if any)
    - Movement LB: a lower bound on the number of move actions as the maximum, over all unsatisfied packages,
      of the shortest-path length (in number of moves) on the static door graph (ignoring closed doors) for:
        * if currently holding that object: dist(robot_loc, goal_loc)
        * else: dist(robot_loc, package_loc) + dist(package_loc, goal_loc)
    - Open-doors LB: considering only currently open doors, compute connected components on the full room graph
      and count how many of the "needed" rooms (robot position + each unsatisfied package's current room and
      goal room) lie in distinct components. At least (components - 1) open actions are necessary to connect them.

    These components are independent kinds of actions (putdown/pickup/move/open), so their sum is an admissible
    lower bound on the number of remaining primitive actions.

    Implementation details:
    - Parses facts once to build fast indices (fact ids for rloc/holding/in/closed/door/goal_in).
    - Uses bitwise checks for state membership (fast) as required by the framework.
    - Builds a static directed adjacency (ignoring closed) for distance computation.
    - For open-lower-bound, dynamically builds the open-edges undirected connectivity under the current state.
    - Efficiently handles large instances by caching per-call BFS results from needed sources only.
    """

    def __init__(self, model, robot_gpt5_heuristic_18: bool = False):
        super().__init__(model)
        self.model = model

        # Name -> Fact object
        self.fact_by_name: Dict[str, object] = {}
        for f in self.model.facts:
            self.fact_by_name[f.name] = f

        # Indices
        self.rloc_fact_id_by_room: Dict[str, int] = {}
        self.holding_fact_id_by_obj: Dict[str, int] = {}
        self.in_fact_ids_by_obj: Dict[str, Dict[str, int]] = defaultdict(dict)  # obj -> room -> fact_id
        self.closed_fact_id_by_door: Dict[str, Optional[int]] = {}  # door_id -> fact_id for +closed[door]
        self.goal_room_by_obj: Dict[str, str] = {}

        # Door structures
        # Directed adjacency ignoring closed: room -> List[(neighbor_room, door_id)]
        self.adj_all_directed: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # For open-edges connectivity (undirected endpoints per door id)
        self.door_endpoints_undirected: Dict[str, Set[str]] = defaultdict(set)

        # Universe of rooms and packages (for robustness)
        self.all_rooms: Set[str] = set()
        self.all_packages: Set[str] = set()

        # Parse all facts once
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            if pred == 'rloc' and len(args) == 1:
                room = args[0]
                self.rloc_fact_id_by_room[room] = f.global_id
                self.all_rooms.add(room)
            elif pred == 'holding' and len(args) == 1:
                obj = args[0]
                self.holding_fact_id_by_obj[obj] = f.global_id
                self.all_packages.add(obj)
            elif pred == 'in' and len(args) == 2:
                obj, room = args
                self.in_fact_ids_by_obj[obj][room] = f.global_id
                self.all_rooms.add(room)
                self.all_packages.add(obj)
            elif pred == 'closed' and len(args) == 1:
                d = args[0]
                self.closed_fact_id_by_door[d] = f.global_id
            elif pred == 'door' and len(args) == 3:
                loc1, loc2, d = args
                # Directed edge ignoring closed, for move distance LB
                self.adj_all_directed[loc1].append((loc2, d))
                self.all_rooms.add(loc1)
                self.all_rooms.add(loc2)
                # For undirected connectivity under open edges
                self.door_endpoints_undirected[d].add(loc1)
                self.door_endpoints_undirected[d].add(loc2)
            elif pred == 'goal_in' and len(args) == 2:
                obj, room = args
                # Static, always true by design in this domain. Needed for mapping obj->goal room
                # If multiple specified, last wins (instances use one per object)
                self.goal_room_by_obj[obj] = room
                self.all_rooms.add(room)
                self.all_packages.add(obj)

        # Some doors may never have an explicit closed fact defined in the model's fact list.
        # Treat those as not-closed facts missing -> assumed open (as a safe relaxation for LB computations).
        # Ensure each known door id has an entry (possibly None)
        for d in list(self.door_endpoints_undirected.keys()):
            if d not in self.closed_fact_id_by_door:
                self.closed_fact_id_by_door[d] = None

        # Precompute goal facts for this specific problem instance to avoid repeated parsing
        self.goal_in_atoms: List[Tuple[int, str, str]] = []  # list of (fact_id for +in[obj,goal_room], obj, goal_room)
        for g_name in self.model.state_explicit_repr(self.model.goals):
            # Goals are +in[obj,room]
            fact = self.fact_by_name.get(g_name, None)
            if fact is None:
                # Defensive: skip if not found (should not happen)
                continue
            pred, args = self._parse_fact_name(g_name)
            if pred == 'in' and len(args) == 2:
                obj, room = args
                self.goal_in_atoms.append((fact.global_id, obj, room))
                # Ensure we know the goal mapping for held-object penalty and general consistency
                if obj not in self.goal_room_by_obj:
                    self.goal_room_by_obj[obj] = room

        # Internal reusable buffers
        self._bfs_cache_sources: Dict[str, Dict[str, int]] = {}

    def _parse_fact_name(self, name: str):
        """Parse fact names in the Pytrich grounded format: +pred[arg1,arg2] or -pred[arg1,arg2]."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name_ = name[1:]
            bracket_idx = name_.find("[")
            if bracket_idx > 0 and name_.endswith("]"):
                predicate = name_[:bracket_idx]
                args_str = name_[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _bit_is_true(self, state_bitwise: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    def _current_robot_room(self, state_bitwise: int) -> Optional[str]:
        for room, fid in self.rloc_fact_id_by_room.items():
            if ((state_bitwise >> fid) & 1) == 1:
                return room
        return None

    def _currently_held_object(self, state_bitwise: int) -> Optional[str]:
        for obj, fid in self.holding_fact_id_by_obj.items():
            if ((state_bitwise >> fid) & 1) == 1:
                return obj
        return None

    def _object_room(self, state_bitwise: int, obj: str) -> Optional[str]:
        # Find the room r such that +in[obj,r] is true in the current state
        room_map = self.in_fact_ids_by_obj.get(obj, {})
        for room, fid in room_map.items():
            if ((state_bitwise >> fid) & 1) == 1:
                return room
        return None

    def _build_open_connectivity(self, state_bitwise: int) -> Dict[str, Set[str]]:
        """Build undirected adjacency based on doors that are currently open (not closed)."""
        adj_open: Dict[str, Set[str]] = {r: set() for r in self.all_rooms}
        for d, endpoints in self.door_endpoints_undirected.items():
            # If we don't have an explicit closed fact for d, treat it as open (safe relaxation)
            fid = self.closed_fact_id_by_door.get(d, None)
            is_closed = ((state_bitwise >> fid) & 1) == 1 if fid is not None else False
            if is_closed:
                continue
            # Connect all endpoints pairwise (typically just two)
            eps = list(endpoints)
            if len(eps) >= 2:
                a = eps[0]
                for b in eps[1:]:
                    adj_open[a].add(b)
                    adj_open[b].add(a)
        return adj_open

    def _components(self, adj_open: Dict[str, Set[str]]) -> Dict[str, int]:
        comp_id: Dict[str, int] = {}
        cid = 0
        for r in self.all_rooms:
            if r in comp_id:
                continue
            # BFS from r
            q = deque([r])
            comp_id[r] = cid
            while q:
                u = q.popleft()
                for v in adj_open.get(u, ()):  # neighbors
                    if v not in comp_id:
                        comp_id[v] = cid
                        q.append(v)
            cid += 1
        return comp_id

    def _bfs_distances_from(self, source: str) -> Dict[str, int]:
        """Shortest-path distances in number of moves on the directed door graph ignoring closed doors.
        Returns a mapping room -> distance (0 for source). Unreachable rooms are absent.
        Uses per-instance cache for repeated sources within a single heuristic call.
        """
        # Static graph; cache across heuristic calls as it doesn't depend on state
        key = source
        cached = self._bfs_cache_sources.get(key)
        if cached is not None:
            return cached

        dist: Dict[str, int] = {source: 0}
        dq = deque([source])
        while dq:
            u = dq.popleft()
            du = dist[u]
            for v, _door in self.adj_all_directed.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    dq.append(v)
        self._bfs_cache_sources[key] = dist
        return dist

    def h(self, state_bitwise: int, task_network: List) -> int:
        # Reset per-call BFS cache (sources) to avoid unbounded growth across different states
        self._bfs_cache_sources = {}

        # 1) Identify robot room and any held object
        robot_room = self._current_robot_room(state_bitwise)
        held_obj = self._currently_held_object(state_bitwise)

        # 2) Collect unsatisfied goals (+in[obj,goal_room] that are not true yet)
        unsat: List[Tuple[str, str]] = []  # (obj, goal_room)
        for fact_id, obj, goal_room in self.goal_in_atoms:
            if ((state_bitwise >> fact_id) & 1) == 0:
                unsat.append((obj, goal_room))

        if not unsat:
            return 0

        # 3) Putdown and Pickup lower bounds
        putdowns_lb = len(unsat)
        pickups_lb = len(unsat) - (1 if held_obj is not None else 0)
        if pickups_lb < 0:
            pickups_lb = 0

        # 4) Open-doors lower bound via components under currently open doors
        opens_lb = 0
        if robot_room is not None:
            adj_open = self._build_open_connectivity(state_bitwise)
            comp_id = self._components(adj_open)
            needed_rooms: Set[str] = set()
            needed_rooms.add(robot_room)
            for obj, goal_room in unsat:
                needed_rooms.add(goal_room)
                if obj != held_obj:
                    # If not currently held, it resides in some room we must reach to pick it up
                    src_room = self._object_room(state_bitwise, obj)
                    if src_room is not None:
                        needed_rooms.add(src_room)
            # Count distinct open-components among needed rooms
            comp_set: Set[int] = set()
            for r in needed_rooms:
                cid = comp_id.get(r)
                if cid is not None:
                    comp_set.add(cid)
            k = len(comp_set)
            if k >= 2:
                opens_lb = k - 1

        # 5) Movement lower bound: max over packages of shortest path distances ignoring closed
        dmax_moves = 0
        # Precompute distance maps only for used sources
        # We'll fetch on-demand with caching
        for obj, goal_room in unsat:
            if goal_room not in self.all_rooms:
                continue
            if obj == held_obj:
                if robot_room is None:
                    continue
                dist_from_robot = self._bfs_distances_from(robot_room)
                d = dist_from_robot.get(goal_room)
                if d is not None and d > dmax_moves:
                    dmax_moves = d
            else:
                # Need dist(robot->pkg) + dist(pkg->goal)
                src_room = self._object_room(state_bitwise, obj)
                if src_room is None or robot_room is None:
                    continue
                dist_from_robot = self._bfs_distances_from(robot_room)
                d1 = dist_from_robot.get(src_room)
                if d1 is None:
                    continue
                dist_from_src = self._bfs_distances_from(src_room)
                d2 = dist_from_src.get(goal_room)
                if d2 is None:
                    continue
                dtotal = d1 + d2
                if dtotal > dmax_moves:
                    dmax_moves = dtotal

        # Sum of independent LBs (admissible):
        # - putdowns_lb   (one per unsatisfied goal)
        # - pickups_lb    (one per unsatisfied except the currently held one)
        # - dmax_moves    (lower bound on number of move actions needed)
        # - opens_lb      (at least to connect needed components under current open doors)
        h_value = putdowns_lb + pickups_lb + dmax_moves + opens_lb
        if h_value < 0:
            h_value = 0
        return h_value
