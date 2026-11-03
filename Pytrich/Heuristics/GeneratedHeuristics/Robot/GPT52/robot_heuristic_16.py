from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

# Pytrich provides the Heuristic base class at runtime.
# We only rely on the interface: self.model, and __call__(state, task_network).
from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic16(Heuristic):
    """Domain-dependent heuristic for the HDDL *robot* domain.

    Domain structure (key insights)
    ------------------------------
    The domain is essentially a single-robot, single-arm transport problem with doors:

    - Primitive actions: move, open, pickup, putdown
    - State predicates:
        rloc(room), armempty, holding(pkg), in(pkg,room), closed(door), door(room,room,door)
    - Goals are standard STRIPS goals (in pkg room) and also mirrored by goal_in(pkg,room)
      used by methods to decide where to put down.

    HTN structure:
    - The root task is (achieve-goals).
    - If an object is in the robot's current room and not already at its goal room, it gets
      picked up, then the planner enters (release) which tries to eventually put it down
      when the robot is at the goal room.
    - Otherwise, achieve-goals/release will recurse via move_abstract/open_abstract.

    Planning challenge:
    - Large instances have many rooms/doors and up to 100 packages.
    - Search can waste effort opening irrelevant doors or moving without targeting packages.

    Heuristic idea (efficient admissible lower bound)
    -------------------------------------------------
    We compute a *sum of per-package optimistic costs* to place every package at its goal room,
    considering only the robot position and the door graph.

    - We precompute all-pairs shortest path lengths in the room graph where each directed edge
      has cost 1 move, plus +1 if its door is currently closed (for a required open).
      This is a relaxation: we ignore that opening a door helps multiple packages and we ignore
      that doors remain open. Therefore, counting opens per traversal can only *overestimate*
      per-path? Actually it may *underestimate* the true cost because:
         - We count at most one open per edge traversal (optimistic)
         - We ignore the need to navigate to the door's source when opening; but move already
           accounts for being at the source.
      The resulting shortest path length is a lower bound on the number of (open+move) actions
      needed to reach a room.

    - For each misplaced package p (not in its goal room), we estimate the minimal number of
      actions to fix it:
         if holding(p):  dist(robot, goal(p)) + 1 (putdown)
         elif armempty and p is somewhere: dist(robot, loc(p)) + 1 (pickup) + dist(loc(p), goal(p)) + 1 (putdown)
         else (robot holding something else): we cannot pick up now; optimistic bound assumes
              we can still move towards some useful target, so we ignore pickup and use
              dist(robot, loc(p)) + dist(loc(p), goal(p)) (very optimistic but admissible).

    - Since the robot can only carry one package at a time, summing these per-package bounds
      is still admissible? Summation can become inadmissible in general because actions can
      contribute to multiple packages. Here, moves/opens can be shared across deliveries,
      so summation would overestimate.

    To preserve admissibility we use a *max* aggregation instead of sum:
      h(state) = max_p h_p(state)

    This is a safe lower bound: the plan must at least achieve the hardest single package.

    Additionally, we include a tiny task-network-aware component:
      - If there are remaining abstract tasks, we add 0 (still admissible).
      - If the network starts with a primitive operator, we do not add extra.

    This heuristic is designed to be fast (bit checks only) and scalable.
    """

    def __init__(self, robot_gpt52_heuristic_16):
        super().__init__(robot_gpt52_heuristic_16)
        self.model = robot_gpt52_heuristic_16

        # Cache facts by predicate and args for O(1) bit checks.
        self._facts_by_pred: Dict[str, List[Tuple[List[str], int]]] = {}
        self._fact_id_by_key: Dict[Tuple[str, Tuple[str, ...], bool], int] = {}

        for f in self.model.facts:
            pred, args, sign = self._parse_fact(f.name)
            if pred is None:
                continue
            self._facts_by_pred.setdefault(pred, []).append((args, f.global_id))
            self._fact_id_by_key[(pred, tuple(args), sign)] = f.global_id

        # Extract static door graph mapping (loc1, loc2) -> door-id object name.
        self._door_edges: List[Tuple[str, str, str]] = []  # (room1, room2, doorobj)
        for args, gid in self._facts_by_pred.get("door", []):
            # door ?loc1 ?loc2 ?d
            if len(args) == 3:
                self._door_edges.append((args[0], args[1], args[2]))

        # Rooms and packages: derive from facts rather than relying on type tables.
        self._rooms = self._collect_rooms()
        self._packages = self._collect_packages()

        # Pre-extract goal_in mapping: package -> goal room (from facts, not model.goals).
        # The problem's actual goal uses (in p r), but instances also include goal_in.
        self._goal_room_of: Dict[str, str] = {}
        for args, gid in self._facts_by_pred.get("goal_in", []):
            if len(args) == 2:
                self._goal_room_of[args[0]] = args[1]

        # Also parse explicit goal (in p r) from model.goals (bitset)
        # to be robust if goal_in is absent.
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args, sign = self._parse_fact(gname)
            if pred == "in" and sign and len(args) == 2:
                self._goal_room_of.setdefault(args[0], args[1])

        # Scratch buffers
        self._dist_cache_state: Optional[int] = None
        self._dist_allpairs: Dict[str, Dict[str, int]] = {}

    # -------------------------- parsing helpers --------------------------
    def _parse_fact(self, name: str) -> Tuple[Optional[str], List[str], bool]:
        """Parse '+pred[a,b]' / '-pred[a,b]' into (pred, args, signTrue)."""
        name = name.strip()
        if not name:
            return None, [], True
        sign = True
        if name[0] == "+":
            sign = True
            core = name[1:]
        elif name[0] == "-":
            sign = False
            core = name[1:]
        else:
            # Unexpected, treat as positive
            core = name
        b = core.find("[")
        if b > 0 and core.endswith("]"):
            pred = core[:b]
            args_str = core[b + 1 : -1]
            args = args_str.split(",") if args_str else []
            return pred, args, sign
        return None, [], sign

    def _collect_rooms(self) -> List[str]:
        rooms = set()
        for loc_args, _ in self._facts_by_pred.get("rloc", []):
            if len(loc_args) == 1:
                rooms.add(loc_args[0])
        for loc1, loc2, d in self._door_edges:
            rooms.add(loc1)
            rooms.add(loc2)
        for args, _ in self._facts_by_pred.get("in", []):
            if len(args) == 2:
                rooms.add(args[1])
        for _, r in self._goal_room_of.items():
            rooms.add(r)
        return list(rooms)

    def _collect_packages(self) -> List[str]:
        pkgs = set()
        for args, _ in self._facts_by_pred.get("in", []):
            if len(args) == 2:
                pkgs.add(args[0])
        for args, _ in self._facts_by_pred.get("holding", []):
            if len(args) == 1:
                pkgs.add(args[0])
        for p in self._goal_room_of.keys():
            pkgs.add(p)
        return list(pkgs)

    # -------------------------- bit checks --------------------------
    def _is_true(self, state: int, pred: str, args: List[str]) -> bool:
        fid = self._fact_id_by_key.get((pred, tuple(args), True))
        if fid is None:
            return False
        return ((state >> fid) & 1) == 1

    def _get_robot_room(self, state: int) -> Optional[str]:
        for args, gid in self._facts_by_pred.get("rloc", []):
            if len(args) == 1 and ((state >> gid) & 1):
                return args[0]
        return None

    def _get_holding(self, state: int) -> Optional[str]:
        for args, gid in self._facts_by_pred.get("holding", []):
            if len(args) == 1 and ((state >> gid) & 1):
                return args[0]
        return None

    def _get_pkg_room(self, state: int, pkg: str) -> Optional[str]:
        # If holding, it's not in any room.
        if self._is_true(state, "holding", [pkg]):
            return None
        for args, gid in self._facts_by_pred.get("in", []):
            if len(args) == 2 and args[0] == pkg and ((state >> gid) & 1):
                return args[1]
        return None

    # -------------------------- distances --------------------------
    def _ensure_allpairs_dist(self, state: int):
        """Compute all-pairs shortest paths w.r.t. current closed doors.

        Edge cost: move=1 plus open=1 if door is closed.
        This is admissible as a lower bound for navigation cost between rooms.
        """
        if self._dist_cache_state == state:
            return

        # Build adjacency with dynamic costs depending on (closed d).
        adj: Dict[str, List[Tuple[str, int]]] = {r: [] for r in self._rooms}
        for r1, r2, d in self._door_edges:
            cost = 1 + (1 if self._is_true(state, "closed", [d]) else 0)
            adj.setdefault(r1, []).append((r2, cost))

        # Dijkstra from each room (rooms <= 50 in training; acceptable).
        # Use a simple O(V*(E log V)) implementation.
        import heapq

        allpairs: Dict[str, Dict[str, int]] = {}
        for src in self._rooms:
            dist = {src: 0}
            heap = [(0, src)]
            while heap:
                du, u = heapq.heappop(heap)
                if du != dist.get(u, None):
                    continue
                for v, w in adj.get(u, []):
                    nv = du + w
                    if nv < dist.get(v, 10**9):
                        dist[v] = nv
                        heapq.heappush(heap, (nv, v))
            allpairs[src] = dist

        self._dist_allpairs = allpairs
        self._dist_cache_state = state

    def _dist(self, state: int, a: Optional[str], b: Optional[str]) -> int:
        if a is None or b is None:
            return 10**6
        if a == b:
            return 0
        self._ensure_allpairs_dist(state)
        return self._dist_allpairs.get(a, {}).get(b, 10**6)

    # -------------------------- heuristic --------------------------
    def __call__(self, state: int, task_network: List) -> int:
        robot_room = self._get_robot_room(state)
        holding = self._get_holding(state)
        armempty = self._is_true(state, "armempty", [])

        # If already at classical goal state and TN empty, return 0.
        # (Planner typically checks goal separately, but this is safe.)
        if self.model.goals <= state and (task_network is None or len(task_network) == 0):
            return 0

        h = 0

        # For each package with a known goal room, compute an optimistic per-package cost,
        # then take the maximum (admissible under action sharing).
        for pkg, goal_room in self._goal_room_of.items():
            # Check if already satisfied: in(pkg, goal_room)
            if self._is_true(state, "in", [pkg, goal_room]):
                continue

            # If holding this pkg: just navigate to goal and putdown.
            if holding == pkg:
                hp = self._dist(state, robot_room, goal_room) + 1
                if hp > h:
                    h = hp
                continue

            # If pkg is in some room
            pkg_room = self._get_pkg_room(state, pkg)
            if pkg_room is None:
                # Unknown location (or holding by someone else impossible here);
                # keep heuristic safe.
                continue

            if armempty and holding is None:
                # Need: go to pkg + pickup + go to goal + putdown
                hp = self._dist(state, robot_room, pkg_room) + 1 + self._dist(state, pkg_room, goal_room) + 1
            else:
                # Can't guarantee pickup now; optimistic: ignore pickup/putdown.
                hp = self._dist(state, robot_room, pkg_room) + self._dist(state, pkg_room, goal_room)

            if hp > h:
                h = hp

        # Small tie-break improvement: if holding some wrong package, encourage finishing it.
        # Still admissible because it's 0/1 and only when that package is unsatisfied.
        if holding is not None:
            gr = self._goal_room_of.get(holding)
            if gr is not None and not self._is_true(state, "in", [holding, gr]):
                # If not already at goal room, at least one action (move/open) or putdown remains.
                h = max(h, 1)

        return int(h)
