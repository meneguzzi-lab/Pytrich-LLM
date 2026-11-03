from collections import defaultdict, deque
import heapq
import math

from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic4(Heuristic):
    """
    Robot domain-dependent heuristic (admissible, consistent) for Pytrich.

    Core idea (lower-bound cost-to-go in primitive actions):
    - Each package not yet at its goal must be picked up and put down at least once (2 actions).
    - If currently holding a package, only a putdown is needed for that package (1 action) instead of 2.
    - Movement is necessary: estimate the inevitable movement and door openings for the next/ongoing delivery only,
      keeping admissibility by not summing movement costs across all remaining packages.
    - We approximate movement cost with shortest-path distances over the room graph:
        * Edge traversal (room -> room via door) costs 1 move action.
        * If the door is currently closed in the given state, opening it costs 1 additional action the first time.
      This is captured by a Dijkstra run with per-edge weight = 1 + is_closed(door,state) for the first leg.
    - To balance accuracy and speed:
        * If holding an object h: add Dijkstra distance from current robot room to goal(h).
        * If not holding: add min_p [ Dijkstra distance from robot room to loc(p) + (static) unweighted distance from
          loc(p) to goal(p) ] over all unsatisfied unheld packages p. We only charge movement for the next delivery,
          which is an admissible lower bound.

    Implementation details:
    - Parses facts once to build:
        * Room graph from door facts.
        * Mappings for rloc, in(obj,room), holding(obj), closed(door), and each package's goal room from goal_in facts.
    - Precomputes all-pairs unweighted shortest path distances (moves-only) ignoring door closed info.
    - Uses fast bitwise checks for state membership via fact global ids (no string lookups at runtime).

    Admissibility reasoning:
    - Baseline action count: 2 per unsatisfied unheld package; 1 for the currently held package if any.
    - Movement+opening cost is added only for the current/next delivery path, never summed across all remaining
      packages; thus we underestimate total motion costs, ensuring a lower bound.
    - Door openings are counted at most once per traversed door along the chosen path via Dijkstra, which is a
      necessary minimal cost to traverse currently-closed doors.
    """

    def __init__(self, model, robot_gpt5_heuristic_4=True):
        super().__init__(model)
        self.robot_gpt5_heuristic_4 = robot_gpt5_heuristic_4

        # Domain caches
        self.rooms = set()
        self.packages = set()
        self.doors = set()

        self.graph = defaultdict(list)  # room -> List[(neighbor_room, door_name)] (directed)

        # Fact id caches for O(1) bitwise checks
        self.closed_gid_by_door = {}        # door -> gid for +closed[door]
        self.rloc_gid_by_room = {}          # room -> gid for +rloc[room]
        self.holding_gid_by_obj = {}        # obj -> gid for +holding[obj]
        self.armempty_gid = None            # gid for +armempty
        self.in_gid_by_obj_room = defaultdict(dict)   # obj -> {room: gid of +in[obj,room]}
        self.in_goal_gid_by_obj = {}        # obj -> gid of +in[obj, goal_room]
        self.goal_room_by_obj = {}          # obj -> goal room (from +goal_in facts in init)

        # Build caches from model facts
        self._build_domain_caches()

        # Precompute static (moves-only) all-pairs shortest path distances ignoring door closures
        self._precompute_unweighted_apsp()

    # ----------------------------
    # Heuristic API
    # ----------------------------
    def compute(self, state_bitwise: int, task_network):
        # Baseline mandatory actions: putdown for held, pickup+putdown for other unsatisfied packages
        held_obj = self._get_held_object(state_bitwise)
        baseline = 0

        # Count unsatisfied packages and add baseline action costs
        unsatisfied_unheld = []  # list of objects not at their goal and not held
        for obj in self.packages:
            gid_goal_in = self.in_goal_gid_by_obj.get(obj)
            if gid_goal_in is None:
                # Safety: if no goal known, skip (should not happen in given instances)
                continue
            at_goal = self._is_true(state_bitwise, gid_goal_in)
            if not at_goal:
                if held_obj is not None and obj == held_obj:
                    # Currently holding this object: only a putdown is inevitably required
                    # (pickup already happened earlier)
                    pass
                else:
                    unsatisfied_unheld.append(obj)

        # Baseline: 1 for the held (if unsatisfied), 2 per unheld unsatisfied
        if held_obj is not None:
            # Held must still be put down; check if it is already at goal and on floor? If held, it's not on floor,
            # so it is necessarily unsatisfied and needs a putdown.
            baseline += 1
        baseline += 2 * len(unsatisfied_unheld)

        # If nothing remains, return baseline (likely 0)
        if baseline == 0:
            return 0

        # Movement lower bound: only charge for the ongoing/next delivery (admissible)
        rloc_room = self._get_robot_room(state_bitwise)

        move_lb = 0
        if rloc_room is not None:
            if held_obj is not None:
                # Must deliver held_obj to its goal
                goal_room = self.goal_room_by_obj.get(held_obj)
                if goal_room is not None:
                    dist_from_rloc = self._dijkstra_from(rloc_room, state_bitwise)
                    d = dist_from_rloc.get(goal_room, math.inf)
                    if math.isfinite(d):
                        move_lb += d
                    else:
                        # Fallback to moves-only if weighted distance is unavailable
                        move_lb += self._dist0(rloc_room, goal_room)
            else:
                # Choose best next package to pick and deliver
                if unsatisfied_unheld:
                    dist_from_rloc = self._dijkstra_from(rloc_room, state_bitwise)
                    best = math.inf
                    for obj in unsatisfied_unheld:
                        cur_room = self._get_object_room(state_bitwise, obj)
                        goal_room = self.goal_room_by_obj.get(obj)
                        if cur_room is None or goal_room is None:
                            continue
                        d1 = dist_from_rloc.get(cur_room, math.inf)  # includes moves + openings for the first leg
                        d2 = self._dist0(cur_room, goal_room)        # moves-only for second leg (admissible)
                        cand = d1 + d2
                        if cand < best:
                            best = cand
                    if math.isfinite(best):
                        move_lb += best
        # Return combined lower bound (baseline actions + minimal movement for the next/ongoing delivery)
        h_val = baseline + int(move_lb)
        if h_val < 0:
            h_val = 0
        return h_val

    # ----------------------------
    # Domain parsing and caches
    # ----------------------------
    def _build_domain_caches(self):
        # 1) Parse all facts to collect identifiers
        for fact in self.model.facts:
            name = fact.name
            pred, args = self._parse_fact_name(name)
            if pred is None:
                continue

            if pred == 'rloc' and len(args) == 1:
                room = args[0]
                self.rooms.add(room)
                self.rloc_gid_by_room[room] = fact.global_id

            elif pred == 'armempty' and len(args) == 0:
                self.armempty_gid = fact.global_id

            elif pred == 'holding' and len(args) == 1:
                obj = args[0]
                self.packages.add(obj)
                self.holding_gid_by_obj[obj] = fact.global_id

            elif pred == 'in' and len(args) == 2:
                obj, room = args
                self.packages.add(obj)
                self.rooms.add(room)
                self.in_gid_by_obj_room[obj][room] = fact.global_id

            elif pred == 'door' and len(args) == 3:
                loc1, loc2, d = args
                self.rooms.add(loc1)
                self.rooms.add(loc2)
                self.doors.add(d)
                # Directed edge loc1 -> loc2 via door d
                self.graph[loc1].append((loc2, d))

            elif pred == 'closed' and len(args) == 1:
                d = args[0]
                self.doors.add(d)
                self.closed_gid_by_door[d] = fact.global_id

            elif pred == 'goal_in' and len(args) == 2:
                # We set goal rooms from the initial state only (static goal facts)
                pass

        # 2) Extract package goals from the initial state (goal_in facts true in init)
        try:
            init_names = set(self.model.state_explicit_repr(self.model.initial_state))
        except Exception:
            init_names = set()
        for name in init_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'goal_in' and len(args) == 2:
                obj, room = args
                self.goal_room_by_obj[obj] = room

        # 3) For each object, precompute the gid for the goal in-fact (in[obj, goal_room]) to speed up checks
        for obj, goal_room in self.goal_room_by_obj.items():
            gid = self.in_gid_by_obj_room.get(obj, {}).get(goal_room)
            if gid is not None:
                self.in_goal_gid_by_obj[obj] = gid

    def _precompute_unweighted_apsp(self):
        # Precompute unweighted (moves-only) distances ignoring door closures
        # self._apsp[room][room2] = shortest number of moves (>=0) or math.inf if unreachable
        self._apsp = {room: {} for room in self.rooms}
        for src in self.rooms:
            self._apsp[src] = self._bfs_unweighted(src)

    # ----------------------------
    # Distance helpers
    # ----------------------------
    def _bfs_unweighted(self, src):
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v, _d in self.graph.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        # For rooms not reached, value will be missing: callers treat as inf
        return dist

    def _dist0(self, src, dst):
        if src == dst:
            return 0
        d = self._apsp.get(src, {}).get(dst, math.inf)
        return d

    def _dijkstra_from(self, src, state_bitwise):
        # Edge weight = 1 (move) + is_closed(door)
        dist = {src: 0}
        pq = [(0, src)]
        INF = 10**12
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist.get(u, INF):
                continue
            for v, door in self.graph.get(u, []):
                w = 1 + self._is_closed(state_bitwise, door)
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    # ----------------------------
    # State queries
    # ----------------------------
    def _get_robot_room(self, state_bitwise: int):
        # Exactly one rloc[...] should be true. Find it via bitwise check.
        for room, gid in self.rloc_gid_by_room.items():
            if self._is_true(state_bitwise, gid):
                return room
        return None

    def _get_held_object(self, state_bitwise: int):
        for obj, gid in self.holding_gid_by_obj.items():
            if self._is_true(state_bitwise, gid):
                return obj
        return None

    def _get_object_room(self, state_bitwise: int, obj: str):
        # Scan in-facts for this object; the domain ensures exactly one location if not being held
        for room, gid in self.in_gid_by_obj_room.get(obj, {}).items():
            if self._is_true(state_bitwise, gid):
                return room
        # If none found, it may be currently held
        return None

    def _is_closed(self, state_bitwise: int, door: str) -> int:
        gid = self.closed_gid_by_door.get(door)
        if gid is None:
            return 0
        return 1 if self._is_true(state_bitwise, gid) else 0

    def _is_true(self, state_bitwise: int, gid: int) -> bool:
        if gid is None:
            return False
        return (state_bitwise >> gid) & 1 == 1

    # ----------------------------
    # Fact name parsing
    # ----------------------------
    def _parse_fact_name(self, name: str):
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if not name:
            return None, []
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
            bracket_idx = s.find('[')
            if bracket_idx > 0 and s.endswith(']'):
                predicate = s[:bracket_idx]
                args_str = s[bracket_idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []
