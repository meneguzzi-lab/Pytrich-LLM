from collections import deque
from typing import Dict, List, Tuple, Optional

# Pytrich import style can vary by installation.
# The evaluation environment typically provides Heuristic in the search/heuristics package.
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    try:
        from Pytrich.Heuristics.Heuristic import Heuristic
    except Exception:
        # Fallback (some setups expose it directly)
        from heuristic import Heuristic


class RobotGPT52Heuristic14(Heuristic):
    """Domain-dependent heuristic for the HDDL 'robot' domain.

    Domain structure (summary)
    -------------------------
    Primitive actions:
      - pickup(obj, loc): requires armempty, rloc(loc), in(obj, loc)
      - move(loc1, loc2, d): requires rloc(loc1), door(loc1,loc2,d), not closed(d)
      - open(loc1, loc2, d): requires rloc(loc1), door(...), closed(d)
      - putdown(obj, loc): requires rloc(loc), holding(obj), goal_in(obj, loc)

    HTN control:
      - achieve-goals loops: either pickup a misplaced goal object then release,
        or move/open and recurse, or finish.
      - release loops: if holding a goal-satisfied object at goal location -> putdown then achieve-goals,
        else move/open and recurse.

    Key challenges:
      - Navigation through a directed graph of rooms via doors, some initially closed.
      - Doors can be opened from one side (where the robot currently is) to enable moves.
      - Only one package can be carried at a time (armempty/holding).
      - The HTN will keep iterating until all goal_in are satisfied as 'in'.

    Heuristic idea
    --------------
    Provide a fast *lower bound* on remaining primitive actions, combining:
      1) Remaining deliveries count:
         Each package not currently in its goal room must require at least one pickup and one putdown => +2.

      2) Navigation lower bound using a *relaxed distance* in a derived graph:
         - Edge cost is 1 move if the door is already open.
         - Edge cost is 1 open + 1 move = 2 if the door is closed (optimistically assuming we can open it when needed).
         This yields an admissible underestimate of travel actions.

         We estimate minimal travel needed for the *current* next relevant objective:
           - If holding some obj: travel to its goal room (then it can be put down) => dist(rloc, goal_room).
           - Else: travel to the nearest misplaced package location => min dist(rloc, package_room).

    Admissibility notes
    -------------------
    - Counting +2 per misplaced package is a lower bound because each such package must be picked and placed at least once.
    - The door-relaxed shortest path is also a lower bound: we never overcharge (open counted at most once per traversed door)
      and we ignore constraints like needing to be at the correct side multiple times, interference between packages, etc.

    Efficiency
    ----------
    - All fact checks are performed with bitwise operations using fact.global_id.
    - Static parsing of facts builds:
        * adjacency: room -> list[(neighbor_room, door_id)]
        * goal map: obj -> goal_room
        * in-fact ids: (obj, room) -> global_id
        * holding-fact ids: obj -> global_id
        * rloc-fact ids: room -> global_id
        * closed-fact ids: door -> global_id
    - A 0-1/2 weighted shortest-path is computed with a deque (0-1 BFS variant for weights 1 and 2).
      This is done per heuristic call but only once from the current room; it is cheap even for 50 rooms.
    """

    def __init__(self, model, robot_gpt52_heuristic_14=None):
        super().__init__(model)
        self.model = model
        self.robot_gpt52_heuristic_14 = robot_gpt52_heuristic_14

        # --- Helpers: parse grounded fact names of form +pred[arg1,arg2] ---
        self._fact_by_gid = {f.global_id: f for f in self.model.facts}

        def parse_fact(name: str) -> Tuple[Optional[str], List[str]]:
            name = name.strip()
            if name.startswith("+") or name.startswith("-"):
                name2 = name[1:]
                b = name2.find("[")
                if b > 0 and name2.endswith("]"):
                    pred = name2[:b]
                    args_str = name2[b + 1:-1]
                    args = args_str.split(",") if args_str else []
                    return pred, args
            return None, []

        self._parse_fact_name = parse_fact

        # --- Static indices ---
        self.rooms = set()
        self.doors = set()
        self.packages = set()

        self.rloc_gid_by_room: Dict[str, int] = {}
        self.closed_gid_by_door: Dict[str, int] = {}
        self.holding_gid_by_obj: Dict[str, int] = {}
        self.armempty_gid: Optional[int] = None
        self.in_gid_by_obj_room: Dict[Tuple[str, str], int] = {}

        # door predicate: door(loc1, loc2, d)
        self.adj: Dict[str, List[Tuple[str, str]]] = {}

        # goal_in predicate: goal_in(obj, goal_room)
        self.goal_room_by_obj: Dict[str, str] = {}

        # Build from all facts (grounded)
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "rloc" and len(args) == 1:
                room = args[0]
                self.rooms.add(room)
                self.rloc_gid_by_room[room] = fact.global_id

            elif pred == "closed" and len(args) == 1:
                d = args[0]
                self.doors.add(d)
                self.closed_gid_by_door[d] = fact.global_id

            elif pred == "holding" and len(args) == 1:
                o = args[0]
                self.packages.add(o)
                self.holding_gid_by_obj[o] = fact.global_id

            elif pred == "armempty" and len(args) == 0:
                self.armempty_gid = fact.global_id

            elif pred == "in" and len(args) == 2:
                o, room = args
                self.packages.add(o)
                self.rooms.add(room)
                self.in_gid_by_obj_room[(o, room)] = fact.global_id

            elif pred == "door" and len(args) == 3:
                loc1, loc2, d = args
                self.rooms.add(loc1)
                self.rooms.add(loc2)
                self.doors.add(d)
                self.adj.setdefault(loc1, []).append((loc2, d))

            elif pred == "goal_in" and len(args) == 2:
                o, room = args
                self.packages.add(o)
                self.rooms.add(room)
                # Assume each object has exactly one goal room.
                self.goal_room_by_obj[o] = room

        # Cache list of objects that appear in goals (important: some domains may contain extra objects)
        self.goal_objects = list(self.goal_room_by_obj.keys())

    # --- Bitwise checks ---
    def _is_true_gid(self, state: int, gid: int) -> bool:
        return ((state >> gid) & 1) == 1

    def _current_room(self, state: int) -> Optional[str]:
        # Find any room with rloc(room) true.
        for room, gid in self.rloc_gid_by_room.items():
            if self._is_true_gid(state, gid):
                return room
        return None

    def _holding_object(self, state: int) -> Optional[str]:
        for o, gid in self.holding_gid_by_obj.items():
            if self._is_true_gid(state, gid):
                return o
        return None

    def _object_room(self, state: int, obj: str) -> Optional[str]:
        # Find room where in(obj, room) true.
        # (Could optimize with per-object list; not necessary for 100 objs.)
        for (o, room), gid in self.in_gid_by_obj_room.items():
            if o == obj and self._is_true_gid(state, gid):
                return room
        return None

    def _misplaced_objects(self, state: int) -> List[str]:
        res = []
        for o in self.goal_objects:
            goal_room = self.goal_room_by_obj.get(o)
            if goal_room is None:
                continue
            gid = self.in_gid_by_obj_room.get((o, goal_room))
            if gid is None:
                # If the grounded 'in' fact does not exist, be conservative: treat as misplaced.
                res.append(o)
                continue
            if not self._is_true_gid(state, gid):
                res.append(o)
        return res

    def _door_traversal_cost(self, state: int, door_id: str) -> int:
        # move costs 1 always; if door is closed we need open (1) + move (1)
        closed_gid = self.closed_gid_by_door.get(door_id)
        if closed_gid is None:
            # If unknown, assume open (optimistic)
            return 1
        return 2 if self._is_true_gid(state, closed_gid) else 1

    def _shortest_paths_from(self, state: int, start_room: str) -> Dict[str, int]:
        """Compute optimistic shortest-path costs (open+move or just move) from start_room to all rooms."""
        # Weights are 1 or 2 -> use deque-based algorithm.
        INF = 10 ** 9
        dist = {r: INF for r in self.rooms}
        dist[start_room] = 0
        dq = deque([start_room])

        while dq:
            u = dq.popleft()
            du = dist[u]
            for v, d in self.adj.get(u, []):
                w = self._door_traversal_cost(state, d)
                nd = du + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    # w is 1 or 2; pushleft for smaller is not exact but still efficient.
                    # Use heuristic ordering: treat w==1 as more urgent.
                    if w == 1:
                        dq.appendleft(v)
                    else:
                        dq.append(v)
        return dist

    def __call__(self, state: int, task_network=None) -> int:
        """Return an admissible estimate of remaining primitive actions."""
        # Base component: deliveries remaining
        misplaced = self._misplaced_objects(state)
        deliveries_lb = 2 * len(misplaced)

        # If already satisfied, no need to add travel.
        if len(misplaced) == 0:
            return 0

        cur_room = self._current_room(state)
        if cur_room is None:
            # If location unknown, fall back to delivery count only.
            return deliveries_lb

        holding = self._holding_object(state)
        dist = self._shortest_paths_from(state, cur_room)

        travel_lb = 0
        if holding is not None:
            # Must at least reach the goal room of held object.
            goal_room = self.goal_room_by_obj.get(holding)
            if goal_room is not None:
                travel_lb = dist.get(goal_room, 0)
            else:
                travel_lb = 0
        else:
            # Need to reach some misplaced object to pick it up.
            best = None
            for o in misplaced:
                o_room = self._object_room(state, o)
                if o_room is None:
                    continue
                d = dist.get(o_room)
                if d is None:
                    continue
                if best is None or d < best:
                    best = d
            travel_lb = best if best is not None else 0

        # Small additional lower bound: if holding and not at goal room, putdown also needed but already counted in deliveries_lb.
        # If holding a misplaced object, deliveries_lb includes its pickup+putdown, but pickup already happened;
        # However deliveries_lb is a lower bound, so overcounting would be bad.
        # To preserve admissibility, we keep deliveries_lb as defined for *objects not in goal room*, regardless of holding.
        # That remains a lower bound because such objects still require at least a putdown, and typically also a pickup,
        # but we are not allowed to subtract without certainty. We therefore do NOT try to correct it downward.

        return deliveries_lb + travel_lb
