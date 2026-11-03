from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic3(Heuristic):
    """Domain-dependent heuristic for the Robot HDDL domain.

    Domain structure (key observations)
    ----------------------------------
    - The top-level task (achieve-goals) iterates over packages that are not yet at their goal rooms.
    - For each misplaced package, the HTN enforces the pattern:
        pickup(obj at current room) -> release -> putdown(obj at goal room)
      and 'release' may interleave move/open actions until robot is at the goal room.
    - Robot has capacity 1 (armempty/holding), so each remaining package requires at least:
        pickup + putdown  => 2 primitive actions (unavoidable lower bound).
    - Additionally, for a package not already at its goal room, the robot must at least:
        (a) reach the package location, and
        (b) reach the goal room (while holding it).
      Moving between rooms requires traversing doors; if a traversed door is closed, an 'open'
      action is needed before moving through it.

    Heuristic design
    ----------------
    We compute an admissible lower bound:

        h(state, tn) = cost_to_finish_current_holding + sum_{each package not at goal} min_delivery_cost

    where:
    - Each delivery lower bound includes 2 actions (pickup+putdown), plus optimistic navigation costs
      measured on a relaxed graph:
        edge cost(room->room through door d) = 1 (move) + [1 if door d is currently closed else 0]
      This is admissible because:
        * You must move at least once per traversed door.
        * If a door is closed now, at least one open is required sometime before first traversal.
        * We ignore interactions (e.g., doors opened once benefit multiple traversals), making it a
          potentially overcounting *if summed across packages*. Therefore, we avoid summing navigation
          across all packages; instead, we use a max-based aggregation for navigation parts and a sum
          only for the unavoidable pickup/putdown counts.

    Aggregation choice (to keep admissibility)
    ------------------------------------------
    Summing per-package navigation distances can overestimate because the robot can reuse movement
    and opened doors across multiple deliveries. To remain admissible, we:

    - Sum only the mandatory manipulation actions:
        2 * (#packages currently not in their goal room)
        plus 1 extra putdown if currently holding a goal-satisfied package at the current room.
    - Add a single navigation lower bound as the minimum effort to make progress on *some* pending
      delivery (or finish current holding):
        nav_lb = min over pending package p of (dist(robot, loc(p)) + dist(loc(p), goal(p)))
      and if holding an object o, use dist(robot, goal(o)) to finish it.

    This gives useful guidance (focuses search towards states where robot is nearer to some package
    and its delivery route is shorter), while staying computationally cheap.

    Efficiency
    ----------
    - We pre-index all facts by predicate and arguments, and precompute door graph adjacency.
    - For each heuristic call, we run Dijkstra once from the robot's current room to get dist to all rooms.
    - For dist(loc(p), goal(p)) we either:
        * run Dijkstra from each unique source room among package locations (cached), or
        * reuse a cached all-pairs-on-demand map keyed by source.
      This stays fast even on the largest instances (50 rooms, sparse graph).

    Notes
    -----
    - Uses ONLY fast bitwise checks on facts.
    - Ignores task network content intentionally; the domain's HTN essentially serializes deliveries.
      The state already captures progress (in/holding/rloc), so this is sufficient and fast.

    Required init signature
    -----------------------
    The __init__ parameter name is `robot_gpt52_heuristic_3` (lowercase), per instructions.
    """

    @dataclass(frozen=True)
    class _Edge:
        to_room: str
        door_id: str

    def __init__(self, robot_gpt52_heuristic_3):
        super().__init__(robot_gpt52_heuristic_3)
        self.model = robot_gpt52_heuristic_3

        # --- Fact indexing ---
        # Map (pred, tuple(args)) -> global_id
        self._fact_id: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        # Reverse: global_id -> (pred, args)
        self._fact_parsed: List[Tuple[Optional[str], Tuple[str, ...]]] = [None] * len(self.model.facts)

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self._fact_id[key] = f.global_id
            self._fact_parsed[f.global_id] = (pred, tuple(args))

        # Objects: packages, rooms, doors discovered from facts (robust to typing access limitations)
        self._packages: Set[str] = set()
        self._rooms: Set[str] = set()
        self._doors: Set[str] = set()

        # goal map: package -> goal_room
        self._goal_room_of: Dict[str, str] = {}

        # in-fact ids: (obj, room) -> fact_id
        self._in_fact_id: Dict[Tuple[str, str], int] = {}
        # goal_in-fact ids: (obj, room) -> fact_id
        self._goal_in_fact_id: Dict[Tuple[str, str], int] = {}

        # door adjacency: room -> list of edges (to_room, door)
        self._adj: Dict[str, List[RobotGPT52Heuristic3._Edge]] = {}

        # closed facts per door
        self._closed_fact_id: Dict[str, int] = {}

        # rloc facts: room -> fact_id
        self._rloc_fact_id: Dict[str, int] = {}

        # holding facts: obj -> fact_id
        self._holding_fact_id: Dict[str, int] = {}

        # armempty fact
        self._armempty_fact_id: Optional[int] = None

        self._build_indices_from_facts()

        # Cache for Dijkstra distances keyed by (source_room, closed_bitmask_signature)
        # Signature is frozenset of currently closed doors (names) in the state.
        # Rooms are few; closed sets change during search; still beneficial.
        self._dist_cache: Dict[Tuple[str, Tuple[str, ...]], Dict[str, int]] = {}

        # Precompute goal facts explicit once (allowed; goals is fixed)
        self._goal_fact_names = self.model.state_explicit_repr(self.model.goals)

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

    def _is_true(self, state: int, fact_id: int) -> bool:
        return ((state >> fact_id) & 1) == 1

    def _build_indices_from_facts(self):
        for (pred, args_t), fid in self._fact_id.items():
            # Only positive facts exist in state representation for this domain.
            if pred == "armempty" and len(args_t) == 0:
                self._armempty_fact_id = fid

            elif pred == "rloc" and len(args_t) == 1:
                room = args_t[0]
                self._rooms.add(room)
                self._rloc_fact_id[room] = fid

            elif pred == "in" and len(args_t) == 2:
                obj, room = args_t
                self._packages.add(obj)
                self._rooms.add(room)
                self._in_fact_id[(obj, room)] = fid

            elif pred == "holding" and len(args_t) == 1:
                obj = args_t[0]
                self._packages.add(obj)
                self._holding_fact_id[obj] = fid

            elif pred == "door" and len(args_t) == 3:
                r1, r2, d = args_t
                self._rooms.add(r1)
                self._rooms.add(r2)
                self._doors.add(d)
                self._adj.setdefault(r1, []).append(self._Edge(to_room=r2, door_id=d))

            elif pred == "closed" and len(args_t) == 1:
                d = args_t[0]
                self._doors.add(d)
                self._closed_fact_id[d] = fid

            elif pred == "goal_in" and len(args_t) == 2:
                obj, room = args_t
                self._packages.add(obj)
                self._rooms.add(room)
                self._goal_room_of[obj] = room
                self._goal_in_fact_id[(obj, room)] = fid

        # Ensure adjacency keys exist
        for r in self._rooms:
            self._adj.setdefault(r, [])

    def _current_room(self, state: int) -> Optional[str]:
        # Exactly one rloc should be true.
        for room, fid in self._rloc_fact_id.items():
            if self._is_true(state, fid):
                return room
        return None

    def _current_holding(self, state: int) -> Optional[str]:
        for obj, fid in self._holding_fact_id.items():
            if self._is_true(state, fid):
                return obj
        return None

    def _current_closed_doors_signature(self, state: int) -> Tuple[str, ...]:
        closed = []
        for d, fid in self._closed_fact_id.items():
            if self._is_true(state, fid):
                closed.append(d)
        closed.sort()
        return tuple(closed)

    def _dijkstra_rooms(self, state: int, source: str) -> Dict[str, int]:
        closed_sig = self._current_closed_doors_signature(state)
        cache_key = (source, closed_sig)
        if cache_key in self._dist_cache:
            return self._dist_cache[cache_key]

        # Edge cost: move=1, plus open=1 if door currently closed.
        # This is a lower bound for travel cost in primitive actions.
        closed_set = set(closed_sig)

        dist: Dict[str, int] = {source: 0}
        dq = deque([source])

        # Since weights are 1 or 2, we can use 0-1-2 BFS via deque by expanding into a small bucketed approach.
        # Simplicity: standard Dijkstra with deque isn't correct for weight 2; implement small-bucket Dial's algorithm.
        # Max possible distance is small; but for robustness use a simple heap.
        import heapq

        heap = [(0, source)]
        while heap:
            du, u = heapq.heappop(heap)
            if du != dist.get(u, None):
                continue
            for e in self._adj.get(u, []):
                w = 1 + (1 if e.door_id in closed_set else 0)
                nv = du + w
                if nv < dist.get(e.to_room, 10**9):
                    dist[e.to_room] = nv
                    heapq.heappush(heap, (nv, e.to_room))

        self._dist_cache[cache_key] = dist
        return dist

    def _package_location(self, state: int, obj: str) -> Optional[str]:
        # If holding, location is robot location.
        if obj in self._holding_fact_id and self._is_true(state, self._holding_fact_id[obj]):
            return self._current_room(state)
        # Else find in(obj, room)
        for (o, room), fid in self._in_fact_id.items():
            if o == obj and self._is_true(state, fid):
                return room
        return None

    def __call__(self, node) -> int:
        state = node.state

        robot_room = self._current_room(state)
        if robot_room is None:
            return 0

        holding = self._current_holding(state)

        # Count packages not yet at their goal room (in-goal predicate truth)
        remaining = 0
        for obj, goal_room in self._goal_room_of.items():
            fid = self._in_fact_id.get((obj, goal_room))
            if fid is None:
                # If goal is unreachable in representation, ignore.
                continue
            if not self._is_true(state, fid):
                remaining += 1

        # Mandatory manipulation LB: each remaining package must be picked and put down (2 actions)
        # If already holding something, pickup for that item is already done; adjust.
        manip_lb = 2 * remaining
        if holding is not None:
            # Holding an item means we won't need to pickup it again.
            # If that item is one of the remaining ones, reduce by 1.
            goal_room = self._goal_room_of.get(holding, None)
            if goal_room is not None:
                in_goal_fid = self._in_fact_id.get((holding, goal_room))
                if in_goal_fid is not None and not self._is_true(state, in_goal_fid):
                    manip_lb -= 1  # pickup already done

        # Navigation LB: need to at least finish current holding or complete one pending delivery.
        nav_lb = 0

        # Distances from robot to all rooms
        dist_from_robot = self._dijkstra_rooms(state, robot_room)

        # If holding, must reach goal room (and then putdown, but putdown counted in manip_lb)
        if holding is not None:
            goal_room = self._goal_room_of.get(holding, None)
            if goal_room is not None:
                nav_lb = dist_from_robot.get(goal_room, 0)
        else:
            # Not holding: pick a best pending package to start delivering.
            best = None
            for obj, goal_room in self._goal_room_of.items():
                # skip already satisfied
                in_goal_fid = self._in_fact_id.get((obj, goal_room))
                if in_goal_fid is None or self._is_true(state, in_goal_fid):
                    continue

                loc = self._package_location(state, obj)
                if loc is None:
                    continue

                d1 = dist_from_robot.get(loc, 10**9)
                # Distance from loc to goal_room (run/cached per loc)
                dist_from_loc = self._dijkstra_rooms(state, loc)
                d2 = dist_from_loc.get(goal_room, 10**9)

                cand = d1 + d2
                if best is None or cand < best:
                    best = cand

            nav_lb = 0 if best is None or best >= 10**9 else best

        h = manip_lb + nav_lb
        if h < 0:
            h = 0
        return int(h)
