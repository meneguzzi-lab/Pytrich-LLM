from collections import deque
from typing import Dict, List, Tuple, Optional

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic4(Heuristic):
    """Domain-dependent heuristic for the HDDL *robot* domain.

    Domain insight (what makes planning hard here):
    - The only way to satisfy goals is to ensure for every package o: (in o goal_room).
    - Achieve-goals recursively loops over *open/move/pickup/release* until all packages are at goal.
    - Primitive costs are uniform (1). To place a wrong package you must:
        move to its current room (possibly through closed doors), pickup, move to goal room, putdown.
    - Doors may be closed; opening is required before moving through a door.
    - Robot can carry at most one package (armempty/holding).

    Heuristic idea (fast, informative lower bound):
    - Precompute an all-pairs shortest path metric over rooms in terms of *required actions*:
        edge cost = 1 (move) + 1 if the connecting door is currently closed (need open first).
      This is state-dependent because doors can become opened.
    - Estimate remaining work as a sum of independent package delivery lower bounds.
      For a package already at goal: 0.
      For a package not at goal:
        at least 1 pickup + 1 putdown + distance(current_room, goal_room).
      Additionally, we must reach a package to pick it up:
        add distance(robot_room, chosen_pick_room) once (only one package can be carried),
        as a *minimum* extra term: min over all unsatisfied packages.
      If currently holding a package, estimate delivering that held package next.

    This heuristic is admissible as a lower bound because it:
    - Ignores negative interactions between packages (treats deliveries as independent).
    - Uses shortest paths in a relaxation where we only count mandatory open+move actions.
    - Adds at most one “go-to-first-package” movement term.

    It also considers hierarchy implicitly: abstract tasks ultimately realize these primitive steps,
    so estimating primitive effort guides HTN refinement well.
    """

    def __init__(self, robot_gpt52_heuristic_4):
        super().__init__(robot_gpt52_heuristic_4)
        self.model = robot_gpt52_heuristic_4

        # --- Fact indices we care about ---
        self._rloc_facts: Dict[str, int] = {}                 # room -> fact_id
        self._holding_facts: Dict[str, int] = {}              # obj -> fact_id
        self._in_facts: Dict[Tuple[str, str], int] = {}       # (obj, room) -> fact_id
        self._goal_in_map: Dict[str, str] = {}                # obj -> goal_room
        self._closed_facts: Dict[str, int] = {}               # door -> fact_id

        # Door graph (static topology): room -> list of (nbr_room, door)
        self._graph: Dict[str, List[Tuple[str, str]]] = {}

        # Rooms/packages sets (from facts)
        self._rooms = set()
        self._packages = set()
        self._doors = set()

        self._build_static_indices()

    # ---------- Parsing helpers ----------
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

    def _is_true(self, state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    # ---------- Model preprocessing ----------
    def _build_static_indices(self):
        """Index the relevant grounded facts and the room connectivity graph."""
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "rloc" and len(args) == 1:
                room = args[0]
                self._rloc_facts[room] = fact.global_id
                self._rooms.add(room)

            elif pred == "holding" and len(args) == 1:
                obj = args[0]
                self._holding_facts[obj] = fact.global_id
                self._packages.add(obj)

            elif pred == "in" and len(args) == 2:
                obj, room = args
                self._in_facts[(obj, room)] = fact.global_id
                self._packages.add(obj)
                self._rooms.add(room)

            elif pred == "goal_in" and len(args) == 2:
                obj, room = args
                self._goal_in_map[obj] = room
                self._packages.add(obj)
                self._rooms.add(room)

            elif pred == "closed" and len(args) == 1:
                d = args[0]
                self._closed_facts[d] = fact.global_id
                self._doors.add(d)

            elif pred == "door" and len(args) == 3:
                r1, r2, d = args
                self._rooms.add(r1)
                self._rooms.add(r2)
                self._doors.add(d)
                self._graph.setdefault(r1, []).append((r2, d))

        # Ensure all rooms appear as keys
        for r in list(self._rooms):
            self._graph.setdefault(r, [])

        # Cache goal facts list (as names) if needed elsewhere; not used for bit checks.
        self._goal_fact_names = self.model.state_explicit_repr(self.model.goals)

    # ---------- State queries ----------
    def _get_robot_room(self, state: int) -> Optional[str]:
        # rloc is exclusive; find the true one.
        for room, fid in self._rloc_facts.items():
            if self._is_true(state, fid):
                return room
        return None

    def _get_held_object(self, state: int) -> Optional[str]:
        for obj, fid in self._holding_facts.items():
            if self._is_true(state, fid):
                return obj
        return None

    def _get_object_room(self, state: int, obj: str) -> Optional[str]:
        # If held, it is not in any room per domain effects.
        if obj in self._holding_facts and self._is_true(state, self._holding_facts[obj]):
            return None
        # Find room where (in obj room) is true.
        for (o, room), fid in self._in_facts.items():
            if o == obj and self._is_true(state, fid):
                return room
        return None

    # ---------- Shortest paths with door-opening costs ----------
    def _room_distances(self, state: int, start_room: str) -> Dict[str, int]:
        """Dijkstra over rooms with edge weight: move(1) + open(1 if door closed in current state)."""
        # Since weights are 1 or 2, a standard Dijkstra with deque is fine at this scale.
        # Use a simple deque-based SPFA-style; graph is small (<=50 rooms in provided data).
        INF = 10**9
        dist = {r: INF for r in self._rooms}
        dist[start_room] = 0
        q = deque([start_room])
        in_q = {start_room}

        while q:
            u = q.popleft()
            in_q.discard(u)
            du = dist[u]
            for v, d in self._graph.get(u, []):
                extra = 1
                cfid = self._closed_facts.get(d)
                if cfid is not None and self._is_true(state, cfid):
                    extra += 1
                nd = du + extra
                if nd < dist[v]:
                    dist[v] = nd
                    if v not in in_q:
                        q.append(v)
                        in_q.add(v)
        return dist

    # ---------- Main heuristic ----------
    def __call__(self, node) -> int:
        state = node.state

        robot_room = self._get_robot_room(state)
        if robot_room is None:
            return 0

        held = self._get_held_object(state)

        # Precompute distances from robot room once
        dist_from_robot = self._room_distances(state, robot_room)

        # Identify unsatisfied packages and compute per-package lower bounds
        unsat: List[Tuple[str, str, Optional[str]]] = []
        # (obj, goal_room, current_room)
        for obj in self._packages:
            goal_room = self._goal_in_map.get(obj)
            if goal_room is None:
                continue

            if held == obj:
                # Held objects are not "in" any room; certainly not at goal until putdown.
                unsat.append((obj, goal_room, None))
                continue

            cur_room = self._get_object_room(state, obj)
            if cur_room is None:
                # Not found; be safe.
                unsat.append((obj, goal_room, None))
                continue

            if cur_room != goal_room:
                unsat.append((obj, goal_room, cur_room))

        if not unsat:
            return 0

        # If holding something, prioritize delivering it (lower bound for that delivery)
        if held is not None and self._goal_in_map.get(held) is not None:
            g = self._goal_in_map[held]
            dist_to_goal = dist_from_robot.get(g, 10**6)
            # Need at least a putdown at goal; movement/opening accounted by dist.
            return max(0, dist_to_goal + 1)

        # Otherwise, lower bound is:
        #   sum over unsatisfied packages: pickup(1) + putdown(1) + dist(cur, goal)
        # plus one-time cost to reach *some* package to start: min dist(robot, cur)
        total = 0
        best_reach = 10**9

        # Precompute distances from each distinct current room as needed
        dist_cache: Dict[str, Dict[str, int]] = {}

        for obj, goal_room, cur_room in unsat:
            # If location unknown, fall back to minimal 2 actions (pickup+putdown) (still admissible but weak).
            if cur_room is None:
                total += 2
                best_reach = min(best_reach, 0)
                continue

            # reach term candidate
            best_reach = min(best_reach, dist_from_robot.get(cur_room, 10**6))

            # dist(cur, goal)
            if cur_room not in dist_cache:
                dist_cache[cur_room] = self._room_distances(state, cur_room)
            d_cg = dist_cache[cur_room].get(goal_room, 10**6)

            total += 2 + d_cg

        if best_reach == 10**9:
            best_reach = 0

        h = total + best_reach
        if h < 0:
            h = 0
        return int(h)
