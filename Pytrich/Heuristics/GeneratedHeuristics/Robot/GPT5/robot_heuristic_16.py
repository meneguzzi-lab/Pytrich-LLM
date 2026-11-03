from collections import deque, defaultdict
from typing import Dict, List, Tuple, Set, Optional

from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic16(Heuristic):
    """
    Domain-dependent heuristic for the HDDL 'robot' domain.

    Key lower-bound components (all admissible and fast):
    - Sum of carrying distances: For each unsatisfied package, the robot must carry it
      from its current room to its goal room without intermediate putdowns. Hence the
      number of move actions while holding is at least the shortest path length
      between these rooms (ignoring door statuses). We sum this over all unsatisfied
      packages (additive, disjoint carrying segments).
    - Mandatory putdowns/pickups: Every unsatisfied package needs a putdown (1).
      If it is not currently held, it also needs a pickup (1). These are additive
      and unavoidable across all unsatisfied packages.
    - First-pickup travel LB: If no unsatisfied package is being held, the robot must
      travel from its current room to some package's location before its first pickup.
      We add the minimum distance to any unsatisfied package's location (ignoring door
      statuses). This is a safe lower bound for the upcoming travel.
    - Max required opens (state-dependent): For each unsatisfied package, we compute
      the minimal number of still-closed doors along any path from its (relevant) start
      room (robot room if the package is held, else the package's room) to its goal room,
      where an edge contributes 1 if its door is currently closed in the state, else 0.
      Taking the maximum over all unsatisfied packages yields an admissible lower bound
      on how many open actions remain necessary overall (at least one package will
      require that many opens).

    The heuristic balances accuracy with computation time:
    - Precomputes all-pairs unweighted shortest-path distances over rooms (ignoring door
      statuses) for quick distance lookups.
    - Uses fast bitwise checks for state membership.
    - Parses and indexes grounded facts for O(1) access.

    Notes on admissibility and hierarchy:
    - The estimates consider the hierarchical structure implicitly: the domain enforces
      that packages cannot be put down except at their goal (via goal_in precondition),
      so each package inevitably requires a contiguous carry (moves), plus exactly one
      putdown and typically one pickup. These are independent of method choices and thus
      form solid lower bounds for HTN refinement cost.
    - Door openings are monotonic (once opened, they stay open), so counting the maximum
      number of required opens for any single unsatisfied package remains admissible.
    - Travel before the first pickup is unavoidable unless an unsatisfied package is
      already held, captured by the first-pickup LB term.
    """

    def __init__(self, model, robot_gpt5_heuristic_16: bool = True):
        super().__init__(model)
        self.model = model

        # Internal indices
        self.facts_by_id = model.facts
        self.fact_name_to_id: Dict[str, int] = {}

        # Parsed structures
        self.rooms: Set[str] = set()
        self.packages: Set[str] = set()
        self.doors: Set[str] = set()

        # Fact indices for quick bitwise checks
        self.rloc_fact_by_room: Dict[str, int] = {}                # '+rloc[room]' -> id
        self.holding_fact_by_obj: Dict[str, int] = {}              # '+holding[obj]' -> id
        self.in_fact_by_obj_loc: Dict[str, Dict[str, int]] = {}    # obj -> (room -> id)
        self.in_goal_fact_id_by_obj: Dict[str, int] = {}           # obj -> id of '+in[obj, goal_room]'
        self.goal_loc_by_obj: Dict[str, str] = {}                  # obj -> goal room name
        self.closed_fact_by_door: Dict[str, int] = {}              # door -> id of '+closed[door]'

        # Door graph (undirected via bidirectional arcs as present in instances)
        # Adjacency stores per direction (u -> [(v, door), ...])
        self.adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        # Build indices from grounded facts
        self._index_facts_and_build_graph()

        # Precompute all-pairs unweighted shortest path lengths (ignoring door status)
        self.all_pairs_dist: Dict[str, Dict[str, int]] = self._compute_all_pairs_unweighted_dist()

    # ------------------------------ Parsing & Precomputation ------------------------------

    @staticmethod
    def _parse_fact_name(name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _index_facts_and_build_graph(self):
        # Build name->id map and parse relevant predicates
        for f in self.facts_by_id:
            self.fact_name_to_id[f.name] = f.global_id

        # First pass: collect all relevant facts
        for f in self.facts_by_id:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue

            if pred == 'rloc' and len(args) == 1:
                room = args[0]
                self.rooms.add(room)
                self.rloc_fact_by_room[room] = f.global_id

            elif pred == 'in' and len(args) == 2:
                obj, room = args
                self.packages.add(obj)
                self.rooms.add(room)
                self.in_fact_by_obj_loc.setdefault(obj, {})[room] = f.global_id

            elif pred == 'holding' and len(args) == 1:
                obj = args[0]
                self.packages.add(obj)
                self.holding_fact_by_obj[obj] = f.global_id

            elif pred == 'goal_in' and len(args) == 2:
                obj, room = args
                self.packages.add(obj)
                self.rooms.add(room)
                # Record goal location for obj
                # Also map to the grounded '+in[obj,room]' fact id for quick check
                self.goal_loc_by_obj[obj] = room
                in_name = f"+in[{obj},{room}]"
                if in_name in self.fact_name_to_id:
                    self.in_goal_fact_id_by_obj[obj] = self.fact_name_to_id[in_name]

            elif pred == 'door' and len(args) == 3:
                loc1, loc2, door = args
                self.rooms.add(loc1)
                self.rooms.add(loc2)
                self.doors.add(door)
                self.adj[loc1].append((loc2, door))
                # In the instances, both directions are explicitly provided. We keep as-is.

            elif pred == 'closed' and len(args) == 1:
                door = args[0]
                self.doors.add(door)
                self.closed_fact_by_door[door] = f.global_id

        # Ensure all objects have in-fact dicts even if empty (robustness)
        for obj in list(self.packages):
            self.in_fact_by_obj_loc.setdefault(obj, {})

    def _compute_all_pairs_unweighted_dist(self) -> Dict[str, Dict[str, int]]:
        # BFS from every room, edge cost = 1 (ignore door status)
        dists: Dict[str, Dict[str, int]] = {r: {} for r in self.rooms}
        for src in self.rooms:
            d = self._bfs_unweighted(src)
            dists[src] = d
        return dists

    def _bfs_unweighted(self, src: str) -> Dict[str, int]:
        dist: Dict[str, int] = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            du = dist[u]
            for v, _door in self.adj.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        return dist

    # ------------------------------ Utilities ------------------------------

    @staticmethod
    def _bit_is_true(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    def _current_robot_room(self, state_bitwise: int) -> Optional[str]:
        # Scan rloc facts and find the one true in state
        for room, fid in self.rloc_fact_by_room.items():
            if self._bit_is_true(state_bitwise, fid):
                return room
        return None

    def _current_object_room(self, state_bitwise: int, obj: str) -> Optional[str]:
        # If object is held, it is not 'in' any room; caller should check holding first
        # Otherwise, scan '+in[obj,room]' facts to find the one true in state
        room_to_id = self.in_fact_by_obj_loc.get(obj, {})
        for room, fid in room_to_id.items():
            if self._bit_is_true(state_bitwise, fid):
                return room
        return None

    def _get_unweighted_dist(self, a: str, b: str) -> int:
        if a == b:
            return 0
        return self.all_pairs_dist.get(a, {}).get(b, 0)

    def _min_closed_on_path(self, state_bitwise: int, start: str, goal: str) -> int:
        # 0-1 BFS where edge weight is 1 if the door is currently closed, else 0.
        if start == goal:
            return 0
        # Distances (min number of closed doors encountered)
        INF = 10**9
        dist: Dict[str, int] = {room: INF for room in self.rooms}
        dist[start] = 0
        dq = deque([start])
        while dq:
            u = dq.popleft()
            du = dist[u]
            # Early exit
            if u == goal:
                return du
            for v, door in self.adj.get(u, []):
                # door closed?
                closed_id = self.closed_fact_by_door.get(door, None)
                w = 1 if (closed_id is not None and self._bit_is_true(state_bitwise, closed_id)) else 0
                nv = du + w
                if nv < dist[v]:
                    dist[v] = nv
                    if w == 0:
                        dq.appendleft(v)
                    else:
                        dq.append(v)
        return dist.get(goal, 0 if start == goal else 0)

    # ------------------------------ Main Heuristic ------------------------------

    def evaluate(self, state_bitwise: int, task_network=None) -> int:
        # If goals reached and no tasks, return 0 quickly
        try:
            if self.model.goal_reached(state_bitwise, task_network or []):
                return 0
        except Exception:
            # Fallback: proceed to compute h
            pass

        robot_room = self._current_robot_room(state_bitwise)
        if robot_room is None:
            # No robot location found; return neutral admissible value
            return 0

        carry_dist_sum = 0
        pickup_count = 0
        putdown_count = 0
        first_pickup_travel_lb: int = 0
        has_held_unsatisfied = False
        opens_lb_max = 0

        # Compute min distance from robot to some unsatisfied object's location if needed
        min_robot_to_obj_dist: Optional[int] = None

        for obj in self.packages:
            goal_room = self.goal_loc_by_obj.get(obj, None)
            in_goal_fid = self.in_goal_fact_id_by_obj.get(obj, None)
            if goal_room is None or in_goal_fid is None:
                # If no declared goal for this object, ignore it (robustness)
                continue

            # If already in goal room, nothing to do for this object
            if self._bit_is_true(state_bitwise, in_goal_fid):
                continue

            # Must do a putdown eventually for this object
            putdown_count += 1

            held_fid = self.holding_fact_by_obj.get(obj, None)
            is_held = (held_fid is not None) and self._bit_is_true(state_bitwise, held_fid)

            if is_held:
                # No pickup needed
                start_room = robot_room
                has_held_unsatisfied = True
            else:
                # Will need a pickup
                pickup_count += 1
                # Find current object room
                obj_room = self._current_object_room(state_bitwise, obj)
                if obj_room is None:
                    # If unknown, be conservative: no distance added (still admissible)
                    continue
                start_room = obj_room
                # For first-pickup LB, consider distance from robot to this object
                d0 = self._get_unweighted_dist(robot_room, obj_room)
                if min_robot_to_obj_dist is None or d0 < min_robot_to_obj_dist:
                    min_robot_to_obj_dist = d0

            # Carrying distance LB (moves while holding): shortest path len ignoring door status
            d = self._get_unweighted_dist(start_room, goal_room)
            carry_dist_sum += d

            # Required opens LB for this package from current state: minimal #closed doors along any path
            opens_needed = self._min_closed_on_path(state_bitwise, start_room, goal_room)
            if opens_needed > opens_lb_max:
                opens_lb_max = opens_needed

        # First-pickup travel LB applies only if no unsatisfied package is currently held
        if not has_held_unsatisfied and min_robot_to_obj_dist is not None:
            first_pickup_travel_lb = min_robot_to_obj_dist
        else:
            first_pickup_travel_lb = 0

        h_val = carry_dist_sum + pickup_count + putdown_count + first_pickup_travel_lb + opens_lb_max
        # Ensure non-negative integer
        if h_val < 0:
            h_val = 0
        return int(h_val)

    # Optional: make the object callable
    def __call__(self, state_bitwise: int, task_network=None) -> int:
        return self.evaluate(state_bitwise, task_network)
