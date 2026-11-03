from collections import deque
from typing import Dict, List, Optional, Tuple

from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic2(Heuristic):
    """Domain-dependent heuristic for the HDDL *robot* domain.

    Rationale / domain insights
    --------------------------
    The domain is a single-robot fetch&place problem on a graph of rooms connected by doors.

    Key structure:
      - Primitive actions: move(loc1,loc2,door), open(loc1,loc2,door), pickup(obj,loc), putdown(obj,loc)
      - The HTN control loops on (achieve-goals) and (release) with abstract tasks move/open/pickup/putdown.
      - Constraints: the robot can carry at most one package (armempty/holding).
      - putdown is only allowed when the object is at its goal location (goal_in).

    Planning challenges:
      1) Navigation with closed doors: to traverse an edge whose door is closed you must first open it.
      2) Many objects (up to 100): a heuristic must be O(#facts + #doors) per state, not O(#objects * #doors)
         with expensive string-based checks.
      3) HTN task network: the task list always contains some abstract tasks, but the real progress is measured
         by satisfying the classical goal facts (in obj goal_room). In this domain, the HTN methods essentially
         enforce iterating over unsatisfied goals.

    Heuristic idea (admissible lower bound):
      - Count unsatisfied (in obj goal_room) goals => each requires at least one pickup + one putdown = 2 actions.
        (Even if an object is already held at its goal room, putdown is still required; if already correctly in
         place, cost 0.)
      - Add a navigation lower bound to *reach* the next useful room:
          - If holding an object not yet placed, you must reach its goal room.
          - Else you must reach some room that contains an unsatisfied-goal object (to pick it up).
        We compute shortest path distances in the room graph where traversing an edge costs:
            1 if door open, 2 if door closed (need open+move)
        This is a delete-relaxation style cost and is admissible.

    The final heuristic is:
        h = 2 * (#unsatisfied goals) + min_nav
    where min_nav is computed as above (0 if already at a required room).

    This remains admissible because we are summing independent lower bounds:
      - Each unsatisfied object must be picked and placed (>=2 primitive actions).
      - Additionally, to perform the next required pickup/putdown, some amount of navigation is necessary
        (>= min_nav). Navigation actions can overlap with different objects later, but min_nav only accounts
        for the first necessary reachability, so it does not overcount.

    Performance notes:
      - All state queries use bitwise checks on global fact ids.
      - Door graph is precomputed once from static (door ...) facts.
      - Per-state BFS is on rooms only (<= 50 in provided instances), very cheap.
    """

    def __init__(self, model, robot_gpt52_heuristic_2: str = "robot_gpt52_heuristic_2"):
        super().__init__(model, robot_gpt52_heuristic_2)
        self.model = model

        # --- Fact indices for fast access ---
        self._armempty_fid: Optional[int] = None

        # Maps
        self._in_fact_id: Dict[Tuple[str, str], int] = {}         # (obj, room) -> fact_id
        self._goal_room: Dict[str, str] = {}                      # obj -> goal_room
        self._holding_fact_id: Dict[str, int] = {}                # obj -> fact_id
        self._rloc_fact_id: Dict[str, int] = {}                   # room -> fact_id
        self._closed_fact_id: Dict[str, int] = {}                 # door -> fact_id

        # Static door graph: room -> list of (neighbor_room, door_id)
        self._adj: Dict[str, List[Tuple[str, str]]] = {}

        # Cache list of objects that appear in goal_in
        self._goal_objects: List[str] = []

        self._build_indices_and_graph()

    # ----------------------------- parsing helpers -----------------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _is_true(self, state: int, fact_id: int) -> int:
        return (state >> fact_id) & 1

    # ----------------------------- precomputation -----------------------------
    def _build_indices_and_graph(self):
        # First pass: index relevant facts and collect static doors
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "armempty":
                # There is only one armempty fact
                self._armempty_fid = fact.global_id

            elif pred == "in" and len(args) == 2:
                obj, room = args
                self._in_fact_id[(obj, room)] = fact.global_id

            elif pred == "goal_in" and len(args) == 2:
                obj, room = args
                # There is exactly one goal room per object in provided instances
                self._goal_room[obj] = room

            elif pred == "holding" and len(args) == 1:
                (obj,) = args
                self._holding_fact_id[obj] = fact.global_id

            elif pred == "rloc" and len(args) == 1:
                (room,) = args
                self._rloc_fact_id[room] = fact.global_id

            elif pred == "closed" and len(args) == 1:
                (door_id,) = args
                self._closed_fact_id[door_id] = fact.global_id

            elif pred == "door" and len(args) == 3:
                r1, r2, d = args
                self._adj.setdefault(r1, []).append((r2, d))

        self._goal_objects = list(self._goal_room.keys())

    # ----------------------------- core computations -----------------------------
    def _current_room(self, state: int) -> Optional[str]:
        # Exactly one rloc is true.
        for room, fid in self._rloc_fact_id.items():
            if self._is_true(state, fid):
                return room
        return None

    def _held_object(self, state: int) -> Optional[str]:
        # Either none (armempty) or exactly one holding.
        for obj, fid in self._holding_fact_id.items():
            if self._is_true(state, fid):
                return obj
        return None

    def _unsatisfied_goal_objects(self, state: int) -> List[str]:
        # Object goal is satisfied iff (in obj goal_room) holds.
        unsat = []
        for obj in self._goal_objects:
            g_room = self._goal_room[obj]
            fid = self._in_fact_id.get((obj, g_room))
            if fid is None:
                # Should not happen, but be robust
                unsat.append(obj)
                continue
            if not self._is_true(state, fid):
                unsat.append(obj)
        return unsat

    def _room_of_object_if_present(self, state: int, obj: str) -> Optional[str]:
        # Find which room currently contains the object (if it is in some room).
        # We avoid scanning all facts; instead we scan rooms for this object by using the indexed in-facts.
        # Complexity: O(#rooms) per query, but we only call it on a small set (targets), and #rooms<=50.
        for room in self._rloc_fact_id.keys():
            fid = self._in_fact_id.get((obj, room))
            if fid is not None and self._is_true(state, fid):
                return room
        return None

    def _bfs_room_distances(self, state: int, start_room: str) -> Dict[str, int]:
        # Edge cost: 1 if door open, 2 if closed (open+move). Compute shortest distances.
        # Since costs are only 1 or 2, we can use a simple deque-based 0-1-2 BFS variant.
        # For simplicity and small graphs, we use Dijkstra with a deque bucket approach.

        # Bucketed distances up to 2*|V| is safe here; but easiest: standard Dijkstra with deque is overkill.
        # We'll do a classic Dijkstra with small overhead.
        import heapq

        dist: Dict[str, int] = {start_room: 0}
        pq = [(0, start_room)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist.get(u, None):
                continue
            for v, door_id in self._adj.get(u, []):
                closed_fid = self._closed_fact_id.get(door_id)
                is_closed = 0
                if closed_fid is not None:
                    is_closed = self._is_true(state, closed_fid)
                w = 2 if is_closed else 1
                nd = d + w
                if nd < dist.get(v, 10**9):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    # ----------------------------- heuristic interface -----------------------------
    def compute_heuristic(self, parent_node, node) -> int:
        state = node.state

        # If classical goals already satisfied, heuristic is 0 (even if HTN tn not empty,
        # the planner will still need to finish tn; but in this domain tn completion aligns
        # with goals. Keeping admissible, we return 0 as a safe lower bound.)
        # Note: model.goal_reached also checks empty tn; we cannot rely on it here.
        unsat = self._unsatisfied_goal_objects(state)
        k = len(unsat)
        if k == 0:
            return 0

        # Base lower bound: each unsatisfied object needs at least pickup+putdown.
        base = 2 * k

        # Navigation lower bound to the *next* useful room
        cur_room = self._current_room(state)
        if cur_room is None:
            return base

        held = self._held_object(state)

        # If holding an object that is not yet in its goal room, we must (at least) reach its goal room.
        targets: List[str] = []
        if held is not None:
            g_room = self._goal_room.get(held)
            if g_room is not None:
                fid_goal_in = self._in_fact_id.get((held, g_room))
                if fid_goal_in is None or not self._is_true(state, fid_goal_in):
                    targets = [g_room]

        # Otherwise, we need to reach some room that contains an unsatisfied-goal object to pick up.
        if not targets:
            target_rooms = set()
            for obj in unsat:
                loc = self._room_of_object_if_present(state, obj)
                if loc is not None:
                    target_rooms.add(loc)
            targets = list(target_rooms)

        # If we cannot identify any target room (should be rare), skip nav term.
        if not targets:
            return base

        dist = self._bfs_room_distances(state, cur_room)
        min_nav = min((dist.get(t, 10**9) for t in targets), default=0)
        if min_nav >= 10**9:
            # disconnected: still admissible to ignore nav
            min_nav = 0

        return base + min_nav
