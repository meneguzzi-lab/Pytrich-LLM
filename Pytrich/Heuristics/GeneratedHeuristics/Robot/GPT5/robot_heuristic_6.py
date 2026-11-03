from typing import Dict, List, Tuple, Optional, Set
import heapq

# Robust import of the base Heuristic class across possible project layouts
from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic6(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Design rationale and domain insights:
    - Tasks revolve around moving a single robot through rooms connected by doors, optionally opening closed doors,
      picking up a package, and putting it down at its goal room.
    - Primitive action costs (uniform cost):
        open: 1, move: 1, pickup: 1, putdown: 1
      Crossing a closed door minimally costs 1 (open) + 1 (move) = 2, while an open door costs 1 (move).
    - HTN structure (achieve-goals -> pickup/release -> ...): Each unsatisfied object necessarily needs a pickup and
      a putdown, so we get a universal per-object lower bound of 2 actions, ignoring robot travel and door openings.
    - To sharpen guidance, we compute a one-object lookahead lower bound for the "current" next delivery:
        If holding some object h:   LB = dist(current_rloc -> goal(h)) with door-sensitive costs + 1 (putdown)
        If not holding: For every unsatisfied object o, consider
                         LB(o) = dist(current_rloc -> loc(o)) with door-sensitive costs
                                 + 1 (pickup)
                                 + dist(loc(o) -> goal(o)) with unit-cost edges (ignoring closed doors)
                                 + 1 (putdown)
                       and take min_o LB(o).
      For all remaining unsatisfied objects, add 2 each (pickup + putdown) as a conservative lower bound (ignoring
      travel and openings that could be shared).
    - Door-sensitive distances are computed per heuristic call from the current robot location using Dijkstra with
      edge cost = 1 + is_closed(door). Distances for the second leg (object -> goal) use a precomputed all-pairs
      unit-edge distance matrix that ignores door closures; this is admissible and very fast to query.

    Admissibility:
    - Every unsatisfied object requires at least two actions (pickup and putdown) -> safe lower bound.
    - For the immediate next transfer (or the currently held object), we account for movement/opening costs that must
      be paid from the current state. For the leg from object to goal we use the unit-edge graph distance ignoring
      closed doors (never overestimates the true door-sensitive movement cost), preserving admissibility.

    Efficiency:
    - Precomputation at init: ground model parsing, door graph adjacency, all-pairs unit-edge distances.
    - Per call: one Dijkstra from current rloc (door-sensitive), plus O(#packages * avg #rooms per object) bit checks,
      which is fine up to the largest instance provided.

    Notes:
    - This heuristic does not attempt to add extra lower bounds from the abstract task network to remain safe across
      different decompositions and to avoid inadmissible additions.
    - It balances informativeness (door-aware distance for the next move) and speed (precomputed unit-edge distances).
    """

    def __init__(self, model, robot_gpt5_heuristic_6: Optional[dict] = None):
        super().__init__(model)

        # Parsed structures
        self.rooms: Set[str] = set()
        self.packages: Set[str] = set()

        # Facts of interest
        self.armempty_fact_id: Optional[int] = None
        self.rloc_fact_ids: Dict[str, int] = {}                 # room -> fact id for +rloc[room]
        self.in_facts_by_obj: Dict[str, List[Tuple[str, int]]] = {}  # obj -> list of (room, fact id) for +in[obj,room]
        self.holding_fact_ids: Dict[str, int] = {}              # obj -> fact id for +holding[obj]
        self.closed_fact_ids: Dict[str, int] = {}               # door_symbol -> fact id for +closed[door]
        self.goal_room_of_obj: Dict[str, str] = {}              # obj -> goal room

        # Door graph
        self.adj_doors: Dict[str, List[Tuple[str, str]]] = {}   # room -> list of (neighbor_room, door_symbol)
        self.adj_unit: Dict[str, Set[str]] = {}                 # room -> set of neighbor rooms (unit edges, ignore closed)

        # All-pairs unit-edge distances (ignore closed)
        self.unit_dist: Dict[str, Dict[str, int]] = {}

        # Build all static caches from the model
        self._analyze_model()
        self._build_unit_all_pairs_distances()

    # -------------- Parsing and preprocessing --------------
    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        name = name.strip()
        if name and (name[0] == '+' or name[0] == '-'):
            core = name[1:]
            bidx = core.find('[')
            if bidx > 0 and core.endswith(']'):
                predicate = core[:bidx]
                args_str = core[bidx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []

    def _analyze_model(self):
        # Iterate over all facts to populate structures
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == 'armempty':
                self.armempty_fact_id = fact.global_id

            elif pred == 'rloc' and len(args) == 1:
                room = args[0]
                self.rooms.add(room)
                self.rloc_fact_ids[room] = fact.global_id

            elif pred == 'in' and len(args) == 2:
                obj, room = args
                self.packages.add(obj)
                self.rooms.add(room)
                self.in_facts_by_obj.setdefault(obj, []).append((room, fact.global_id))

            elif pred == 'holding' and len(args) == 1:
                obj = args[0]
                self.packages.add(obj)
                self.holding_fact_ids[obj] = fact.global_id

            elif pred == 'closed' and len(args) == 1:
                door = args[0]
                self.closed_fact_ids[door] = fact.global_id

            elif pred == 'door' and len(args) == 3:
                r1, r2, d = args
                self.rooms.add(r1)
                self.rooms.add(r2)
                self.adj_doors.setdefault(r1, []).append((r2, d))
                # We'll rely on the model providing both directions, but also capture unit adjacency regardless
                self.adj_unit.setdefault(r1, set()).add(r2)

            elif pred == 'goal_in' and len(args) == 2:
                obj, room = args
                self.packages.add(obj)
                self.rooms.add(room)
                # If multiple are present, we keep the last; normally there's exactly one per object
                self.goal_room_of_obj[obj] = room

        # Ensure all rooms are keys in adjacency maps
        for r in self.rooms:
            self.adj_doors.setdefault(r, [])
            self.adj_unit.setdefault(r, set())

        # Some domains might not include door facts for both directions; ensure undirected unit adjacency
        # based on observed door relations
        undirected_adj: Dict[str, Set[str]] = {r: set(nbrs) for r, nbrs in self.adj_unit.items()}
        for r, nbrs in list(undirected_adj.items()):
            for nb in nbrs:
                undirected_adj.setdefault(nb, set()).add(r)
        self.adj_unit = undirected_adj

    def _build_unit_all_pairs_distances(self):
        # BFS from each room on unit-edge adjacency, store distances
        for src in self.rooms:
            self.unit_dist[src] = self._bfs_unit(src)

    def _bfs_unit(self, src: str) -> Dict[str, int]:
        from collections import deque
        dist: Dict[str, int] = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            du = dist[u]
            for v in self.adj_unit.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        return dist

    # -------------- Helpers over bitwise state --------------
    @staticmethod
    def _is_true(state: int, fid: Optional[int]) -> bool:
        if fid is None:
            return False
        return ((state >> fid) & 1) != 0

    def _current_robot_room(self, state: int) -> Optional[str]:
        # Only one rloc should be true
        for room, fid in self.rloc_fact_ids.items():
            if self._is_true(state, fid):
                return room
        return None

    def _held_object(self, state: int) -> Optional[str]:
        for obj, fid in self.holding_fact_ids.items():
            if self._is_true(state, fid):
                return obj
        return None

    def _object_location(self, state: int, obj: str) -> Optional[str]:
        # Returns current room where obj is, or None if currently held
        for room, fid in self.in_facts_by_obj.get(obj, []):
            if self._is_true(state, fid):
                return room
        return None

    def _door_closed_cost(self, state: int, door: str) -> int:
        # Returns 1 if door is currently closed (we must open), else 0
        fid = self.closed_fact_ids.get(door)
        return 1 if self._is_true(state, fid) else 0

    def _dijkstra_closed_from(self, state: int, src: str) -> Dict[str, int]:
        # Dijkstra where edge cost = 1 + (door_closed ? 1 : 0)
        dist: Dict[str, int] = {src: 0}
        pq: List[Tuple[int, str]] = [(0, src)]
        visited: Set[str] = set()

        while pq:
            du, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            for v, door in self.adj_doors.get(u, []):
                # Skip if not a real adjacency (shouldn't happen, but safe-guard)
                move_cost = 1 + self._door_closed_cost(state, door)
                alt = du + move_cost
                if v not in dist or alt < dist[v]:
                    dist[v] = alt
                    heapq.heappush(pq, (alt, v))
        return dist

    # -------------- Heuristic main --------------
    def h(self, state_bitwise: int, task_network: List) -> int:
        # If goals are already reached and no remaining tasks, return 0 quickly
        if self.model.goal_reached(state_bitwise, task_network):
            return 0

        # Gather current robot room
        robot_room = self._current_robot_room(state_bitwise)
        if robot_room is None:
            # Should not happen; be safe (return 0 so as not to break search, but conservative large would be ok too)
            return 0

        # Find the currently held object if any
        held_obj = self._held_object(state_bitwise)

        # Build unsatisfied set (only objects that have a goal specified)
        unsatisfied: List[str] = []
        for obj, goal_room in self.goal_room_of_obj.items():
            if obj == held_obj:
                unsatisfied.append(obj)
                continue
            cur_loc = self._object_location(state_bitwise, obj)
            if cur_loc != goal_room:
                unsatisfied.append(obj)

        # If nothing is left to do in terms of goals, return 0
        if not unsatisfied:
            return 0

        # Precompute door-sensitive distances from current robot room
        dist_from_robot_closed = self._dijkstra_closed_from(state_bitwise, robot_room)

        INF = 10 ** 9

        # If holding: cost to deliver held object now (move to its goal, then putdown)
        if held_obj is not None:
            goal_room = self.goal_room_of_obj.get(held_obj)
            # If held object has no specified goal, we cannot reason -> return 0 as safe lower bound
            if goal_room is None:
                return 0
            d_to_goal = dist_from_robot_closed.get(goal_room, INF)
            cost_first = (d_to_goal if d_to_goal < INF else self.unit_dist.get(robot_room, {}).get(goal_room, INF))
            if cost_first >= INF:
                # Disconnected in model (shouldn't happen); return 0 to remain safe
                return 0
            # Putdown action
            cost_first += 1
            # Remaining objects at least need 2 actions each
            total = cost_first + 2 * (len(unsatisfied) - 1)
            return int(total)

        # Not holding: choose the cheapest next object to deliver
        best = INF
        for obj in unsatisfied:
            goal_room = self.goal_room_of_obj.get(obj)
            if goal_room is None:
                # No goal specified -> ignore for cost (treat as satisfied)
                continue
            cur_loc = self._object_location(state_bitwise, obj)
            if cur_loc is None:
                # Should not occur when not holding; skip conservatively
                continue

            d1 = dist_from_robot_closed.get(cur_loc, INF)
            if d1 >= INF:
                # Fallback to unit-edge distance if door-sensitive graph seems disconnected for this query
                d1 = self.unit_dist.get(robot_room, {}).get(cur_loc, INF)
            d2 = self.unit_dist.get(cur_loc, {}).get(goal_room, INF)

            if d1 < INF and d2 < INF:
                # pickup + putdown add 2
                total_i = d1 + 1 + d2 + 1
                if total_i < best:
                    best = total_i

        if best >= INF:
            # If we couldn't compute a reasonable bound, return 0 (safe but uninformative)
            return 0

        # Add 2 per each remaining unsatisfied object after the first one
        total = best + 2 * (len(unsatisfied) - 1)
        return int(total)

    # Provide __call__ alias if the planner expects callables
    def __call__(self, state_bitwise: int, task_network: List) -> int:
        return self.h(state_bitwise, task_network)
