from collections import deque
from typing import Dict, List, Tuple, Optional, Set

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic6(Heuristic):
    """Domain-dependent heuristic for the HDDL *robot* domain.

    Rationale (domain insights)
    --------------------------
    The domain has one controlling abstract task (achieve-goals) that repeatedly:
      - chooses an object currently not at its goal location,
      - picks it up (requires robot to be co-located and armempty),
      - then runs release, which repeatedly either moves/open doors until it can
        put the object down at its goal room.

    Key structure:
      * Only one package can be held at a time (armempty/holding).
      * Movement is along a door graph; doors can be closed and require an 'open'
        action from the incident room.
      * Delivering one misplaced object minimally needs:
            pickup (1) + (move/open steps to goal room) + putdown (1)
        We ignore inter-object coupling except the single-arm constraint.

    Heuristic design
    ----------------
    We compute a cheap, admissible lower bound by:
      1) Counting how many goal 'in[obj,goalroom]' facts are currently false.
         Each such object requires at least one 'putdown' eventually => +1 per object.
         Additionally, if that object is not currently held, it must be picked up => +1.
      2) Add a navigation lower bound based on the *minimum* door-steps the robot
         must traverse to make progress.

    Navigation lower bound details (still admissible):
      - Build an undirected room graph from static 'door[room1,room2,d]' facts.
      - Precompute shortest-path distance in number of moves (edges) between all rooms.
      - For any closed door, opening it costs at least 1, but only if we must traverse it.
        We therefore ignore door-opening costs in the distance computation to remain a lower bound.
      - We add at most one distance term:
           * If holding obj: distance from robot room to obj's goal room.
           * Else: choose the single best (minimum) among all misplaced objects:
                 distance(robot -> object's current room) + distance(object room -> goal room)
        This is admissible because at least one misplaced object must be handled next.

    Total heuristic:
        h = min_nav + sum_per_object( putdown_needed + pickup_needed )

    This stays fast on large instances: all-pairs distances are computed once using BFS from
    each room (|V|<=50 in provided instances). State evaluation is O(#misplaced_objects).

    Implementation notes
    --------------------
    * We ONLY use bitwise checks for state facts.
    * Goals are read from model.goals via model.state_explicit_repr(model.goals).
    * Facts are parsed from the grounded string format '+pred[a,b]'.

    """

    def __init__(self, model, robot_gpt52_heuristic_6: str = "robot_gpt52_heuristic_6"):
        super().__init__(model, robot_gpt52_heuristic_6)
        self.model = model

        # ---- Parse static structure from facts ----
        self._fact_by_name: Dict[str, int] = {f.name: f.global_id for f in self.model.facts}

        self._rooms: Set[str] = set()
        self._packages: Set[str] = set()
        self._doors: Set[str] = set()

        # door predicate: door[loc1,loc2,d]
        self._adj: Dict[str, List[str]] = {}

        # For quick lookup of (closed door) fact ids: +closed[d]
        self._closed_fact_id: Dict[str, int] = {}

        # +armempty, +holding[obj], +rloc[room], +in[obj,room], +goal_in[obj,room]
        self._armempty_id: Optional[int] = self._fact_by_name.get("+armempty")

        self._holding_ids: Dict[str, int] = {}
        self._rloc_ids: Dict[str, int] = {}
        self._in_ids: Dict[Tuple[str, str], int] = {}
        self._goal_in_ids: Dict[Tuple[str, str], int] = {}

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "door" and len(args) == 3:
                r1, r2, d = args
                self._rooms.add(r1)
                self._rooms.add(r2)
                self._doors.add(d)
                self._adj.setdefault(r1, []).append(r2)
                # We will also see the reverse direction as a separate grounded fact in instances,
                # but we still treat the graph as undirected by adding both edges when present.

            elif pred == "closed" and len(args) == 1:
                self._doors.add(args[0])
                self._closed_fact_id[args[0]] = fact.global_id

            elif pred == "holding" and len(args) == 1:
                self._packages.add(args[0])
                self._holding_ids[args[0]] = fact.global_id

            elif pred == "rloc" and len(args) == 1:
                self._rooms.add(args[0])
                self._rloc_ids[args[0]] = fact.global_id

            elif pred == "in" and len(args) == 2:
                o, r = args
                self._packages.add(o)
                self._rooms.add(r)
                self._in_ids[(o, r)] = fact.global_id

            elif pred == "goal_in" and len(args) == 2:
                o, r = args
                self._packages.add(o)
                self._rooms.add(r)
                self._goal_in_ids[(o, r)] = fact.global_id

        # Sometimes adjacency might be missing reverse edges in other datasets;
        # symmetrize to be safe.
        for r1, nbrs in list(self._adj.items()):
            for r2 in nbrs:
                self._adj.setdefault(r2, []).append(r1) if r1 not in self._adj.get(r2, []) else None

        # ---- Extract goal mapping obj -> goal_room from model.goals ----
        self._goal_room_of: Dict[str, str] = {}
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "in" and len(args) == 2:
                obj, room = args
                self._goal_room_of[obj] = room

        # Fallback: if goal encoding differs, also accept +goal_in facts.
        if not self._goal_room_of:
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "goal_in" and len(args) == 2:
                    obj, room = args
                    self._goal_room_of[obj] = room

        # Precompute all-pairs shortest path distances on room graph.
        self._dist: Dict[str, Dict[str, int]] = {r: self._bfs_dist(r) for r in self._rooms}

    # --- Required Heuristic API ---
    def __call__(self, node) -> int:
        state = node.state
        return self.h(state, getattr(node, "task_network", []))

    def h(self, state_bitwise: int, task_network=None) -> int:
        # Identify robot location
        robot_room = self._get_robot_room(state_bitwise)

        # Identify held object (if any)
        held_obj = self._get_held_object(state_bitwise)

        # Count misplaced goal objects and mandatory pickup/putdown lower bounds
        misplaced: List[Tuple[str, str, str]] = []  # (obj, cur_room, goal_room)
        base = 0

        for obj, goal_room in self._goal_room_of.items():
            cur_room = self._get_object_room(state_bitwise, obj)

            # If already at goal, no cost for this object
            if cur_room == goal_room:
                continue

            # At least one putdown needed for each misplaced object
            base += 1

            # If not currently holding it, must pick it up at least once
            if held_obj != obj:
                base += 1

            if cur_room is not None:
                misplaced.append((obj, cur_room, goal_room))

        # If no goals are violated, heuristic is 0 (even if tasks remain, this is still a safe LB)
        if not misplaced and held_obj is None:
            return 0

        # Add minimal navigation lower bound
        nav = 0
        if robot_room is None:
            # If rloc is missing (shouldn't happen), stay admissible
            nav = 0
        else:
            if held_obj is not None and held_obj in self._goal_room_of:
                goal_room = self._goal_room_of[held_obj]
                nav = self._room_dist(robot_room, goal_room)
            else:
                # choose best next object to handle
                best = None
                for (obj, cur_room, goal_room) in misplaced:
                    d1 = self._room_dist(robot_room, cur_room)
                    d2 = self._room_dist(cur_room, goal_room)
                    val = d1 + d2
                    if best is None or val < best:
                        best = val
                nav = best if best is not None else 0

        # Small extra lower bound for closed door immediately blocking any movement is not safe to detect cheaply;
        # we keep the heuristic admissible by ignoring door-open costs.
        return int(base + nav)

    # ---- Helpers ----
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

    def _is_true(self, state_bitwise: int, fact_global_id: int) -> bool:
        return ((state_bitwise >> fact_global_id) & 1) == 1

    def _get_robot_room(self, state_bitwise: int) -> Optional[str]:
        # Exactly one rloc should be true
        for room, fid in self._rloc_ids.items():
            if self._is_true(state_bitwise, fid):
                return room
        return None

    def _get_held_object(self, state_bitwise: int) -> Optional[str]:
        for obj, fid in self._holding_ids.items():
            if self._is_true(state_bitwise, fid):
                return obj
        return None

    def _get_object_room(self, state_bitwise: int, obj: str) -> Optional[str]:
        # If held, it has no in[obj,room] true
        if obj in self._holding_ids and self._is_true(state_bitwise, self._holding_ids[obj]):
            return None
        # Otherwise find its in fact
        # (Linear scan over rooms for this obj; rooms <= 50 so cheap)
        for (o, r), fid in self._in_ids.items():
            if o == obj and self._is_true(state_bitwise, fid):
                return r
        return None

    def _bfs_dist(self, start: str) -> Dict[str, int]:
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self._adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def _room_dist(self, a: str, b: str) -> int:
        if a == b:
            return 0
        da = self._dist.get(a)
        if da is None:
            return 0
        return da.get(b, 0)  # if disconnected, keep admissible by returning 0
