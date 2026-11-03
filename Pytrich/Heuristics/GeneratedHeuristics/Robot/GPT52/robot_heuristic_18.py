from collections import deque
from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic18(Heuristic):
    """Domain-dependent heuristic for the HDDL *robot* domain.

    Domain observations (grounded from the given HDDL):
    - The only real objective is to make all goal facts of the form +in[obj,goal_room] true.
    - An object can be placed in its goal room only via action putdown(obj, goal_room),
      whose preconditions require the robot to be at goal_room, holding obj.
    - To hold obj, action pickup(obj, loc) requires robot at loc, armempty, and obj in loc.
    - Robot motion is along directed door edges door(loc1, loc2, d) and requires the door d not closed.
      If closed, an open action is needed at loc1 for that door.

    Heuristic idea:
    Compute a *lower bound* on remaining primitive actions by combining:
    1) For each object not yet in its goal room, a minimal estimate of actions needed:
       - If object already in goal room: 0.
       - If holding the object: at least travel-to-goal + (maybe open) + 1 putdown.
       - Else (object somewhere else): at least travel-to-object + (maybe open) + 1 pickup
         + travel-to-goal + (maybe open) + 1 putdown.
    2) A task-network penalty: count abstract tasks that necessarily incur at least one primitive action
       (move_abstract/open_abstract/pickup_abstract/putdown_abstract) as a weak but admissible tie-breaker.

    Key admissibility choices:
    - Distances are computed in a relaxed graph where closed doors are traversable with cost 2
      (open + move), and open doors with cost 1 (move). This never overestimates.
    - We *do not* sum costs over all objects (which would overcount shared moves).
      Instead we take the maximum per-object estimate (a safe lower bound on makespan-like serial plan
      because at least one unsatisfied object must be completed and actions are sequential).
    - Additionally, if the robot is holding an object that is not in its goal room, at least one
      putdown must occur for that object; the max-per-object captures that.

    This heuristic is fast:
    - Preprocess door graph once.
    - For each evaluated node, run 0-2 BFS-like shortest path computations from the robot location
      and from a selected set of sources (object rooms). With many objects, we avoid all-pairs by
      computing distances from robot location once and from each distinct object location only when needed,
      with caching.
    """

    def __init__(self, model, robot_gpt52_heuristic_18=None):
        super().__init__(model)
        self.model = model

        # Fact indices by predicate for fast scanning
        self._fact_by_name: Dict[str, int] = {f.name: f.global_id for f in self.model.facts}

        # Parse and store static structures
        self._doors_out: Dict[str, List[Tuple[str, str]]] = {}  # loc1 -> list of (loc2, door_id)
        self._goal_room_of: Dict[str, str] = {}  # obj -> goal_room

        # Dynamic fact ids we will query with bit ops
        self._rloc_fact_ids: List[int] = []
        self._armempty_id: Optional[int] = None

        self._in_fact_ids_by_obj: Dict[str, List[Tuple[str, int]]] = {}  # obj -> [(room, fact_id)]
        self._holding_fact_ids: Dict[str, int] = {}  # obj -> fact_id
        self._closed_fact_ids: Dict[str, int] = {}  # door -> fact_id

        # Build from facts set
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "door" and len(args) == 3:
                loc1, loc2, d = args
                self._doors_out.setdefault(loc1, []).append((loc2, d))
            elif pred == "goal_in" and len(args) == 2:
                obj, room = args
                # goal_in is static in all provided instances
                self._goal_room_of[obj] = room
            elif pred == "rloc" and len(args) == 1:
                self._rloc_fact_ids.append(fact.global_id)
            elif pred == "armempty" and len(args) == 0:
                self._armempty_id = fact.global_id
            elif pred == "in" and len(args) == 2:
                obj, room = args
                self._in_fact_ids_by_obj.setdefault(obj, []).append((room, fact.global_id))
            elif pred == "holding" and len(args) == 1:
                obj = args[0]
                self._holding_fact_ids[obj] = fact.global_id
            elif pred == "closed" and len(args) == 1:
                d = args[0]
                self._closed_fact_ids[d] = fact.global_id

        # Cache for distances given a state: we cache by (state_closed_signature, source_room)
        # where state_closed_signature is a bitmask over known doors that indicates which are closed.
        self._door_list: List[str] = sorted(self._closed_fact_ids.keys())
        self._door_index: Dict[str, int] = {d: i for i, d in enumerate(self._door_list)}
        self._dist_cache: Dict[Tuple[int, str], Dict[str, int]] = {}

        # Cache goal facts bit ids (to detect satisfied goals quickly without string repr)
        self._goal_in_fact_ids_by_obj: Dict[str, int] = {}
        for obj, room in self._goal_room_of.items():
            fname = f"+in[{obj},{room}]"
            if fname in self._fact_by_name:
                self._goal_in_fact_ids_by_obj[obj] = self._fact_by_name[fname]

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

    def _robot_room(self, state: int) -> Optional[str]:
        # Exactly one rloc should be true
        for fid in self._rloc_fact_ids:
            if self._is_true(state, fid):
                pred, args = self._parse_fact_name(self.model.facts[fid].name)
                if pred == "rloc" and len(args) == 1:
                    return args[0]
        return None

    def _door_closed_signature(self, state: int) -> int:
        sig = 0
        for d, fid in self._closed_fact_ids.items():
            if self._is_true(state, fid):
                sig |= (1 << self._door_index[d])
        return sig

    def _all_distances_from(self, state: int, source: str) -> Dict[str, int]:
        """Shortest distances in number of primitive actions from source to all rooms.

        Edge cost:
          - if door open: move costs 1
          - if door closed: must open then move => cost 2

        This is a lower bound on required actions to change rloc.
        """
        sig = self._door_closed_signature(state)
        key = (sig, source)
        cached = self._dist_cache.get(key)
        if cached is not None:
            return cached

        # 0-1-2 BFS (Dijkstra with small integer weights). We'll just use deque-based SPFA-like.
        dist: Dict[str, int] = {source: 0}
        dq = deque([source])

        while dq:
            u = dq.popleft()
            du = dist[u]
            for v, d in self._doors_out.get(u, []):
                # Determine if closed in this state
                closed_fid = self._closed_fact_ids.get(d)
                is_closed = False
                if closed_fid is not None:
                    is_closed = self._is_true(state, closed_fid)
                w = 2 if is_closed else 1
                nd = du + w
                old = dist.get(v)
                if old is None or nd < old:
                    dist[v] = nd
                    dq.append(v)

        self._dist_cache[key] = dist
        return dist

    def _object_room_if_not_holding(self, state: int, obj: str) -> Optional[str]:
        # If holding, it's not in any room
        hid = self._holding_fact_ids.get(obj)
        if hid is not None and self._is_true(state, hid):
            return None
        for room, fid in self._in_fact_ids_by_obj.get(obj, []):
            if self._is_true(state, fid):
                return room
        return None

    def _is_object_in_goal(self, state: int, obj: str) -> bool:
        gid = self._goal_in_fact_ids_by_obj.get(obj)
        if gid is None:
            # If the explicit goal fact wasn't found, fall back to comparing in-facts
            goal_room = self._goal_room_of.get(obj)
            if goal_room is None:
                return True
            for room, fid in self._in_fact_ids_by_obj.get(obj, []):
                if room == goal_room and self._is_true(state, fid):
                    return True
            return False
        return self._is_true(state, gid)

    def __call__(self, node) -> int:
        state = node.state
        tn = node.task_network if hasattr(node, "task_network") else []

        robot_room = self._robot_room(state)
        if robot_room is None:
            return 0

        # Compute distances from robot once
        dist_from_robot = self._all_distances_from(state, robot_room)

        # Per-object admissible estimate; take max to avoid double counting shared travel.
        max_obj_cost = 0

        for obj, goal_room in self._goal_room_of.items():
            if self._is_object_in_goal(state, obj):
                continue

            # If holding obj
            hid = self._holding_fact_ids.get(obj)
            if hid is not None and self._is_true(state, hid):
                travel = dist_from_robot.get(goal_room)
                if travel is None:
                    # unreachable in relaxed sense -> dead end estimate large
                    max_obj_cost = max(max_obj_cost, 10**6)
                    continue
                # need to be at goal_room + putdown
                cost = travel + 1
                if cost > max_obj_cost:
                    max_obj_cost = cost
                continue

            # Not holding: must go to object's current room (if any), pickup, then to goal, putdown.
            obj_room = self._object_room_if_not_holding(state, obj)
            if obj_room is None:
                # If object location unknown (shouldn't happen), don't overestimate.
                continue

            travel_to_obj = dist_from_robot.get(obj_room)
            if travel_to_obj is None:
                max_obj_cost = max(max_obj_cost, 10**6)
                continue

            # Dist from object room to goal room
            dist_from_obj_room = self._all_distances_from(state, obj_room)
            travel_to_goal = dist_from_obj_room.get(goal_room)
            if travel_to_goal is None:
                max_obj_cost = max(max_obj_cost, 10**6)
                continue

            # pickup + putdown are mandatory
            cost = travel_to_obj + 1 + travel_to_goal + 1
            if cost > max_obj_cost:
                max_obj_cost = cost

        # Hierarchy-aware tie-breaker: count abstract tasks that imply at least 1 primitive action.
        # Must remain admissible: we only add 0 (safe) or a very small lower bound.
        # If there is any abstract task among {move_abstract, open_abstract, pickup_abstract, putdown_abstract}
        # then at least one primitive action is needed to reduce it.
        tn_lb = 0
        for t in tn:
            name = getattr(t, "name", "")
            if name.startswith("move_abstract") or name.startswith("open_abstract"):
                tn_lb = 1
                break
            if name.startswith("pickup_abstract") or name.startswith("putdown_abstract"):
                tn_lb = 1
                break

        # Combine: max_obj_cost already dominates; tn_lb only helps when goals are already satisfied
        # but tasks remain (rare), or when max_obj_cost is 0 but still need to execute something.
        h = max(max_obj_cost, tn_lb)
        return int(h)
