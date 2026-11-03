from collections import deque
from typing import Dict, List, Set, Tuple, Optional, Iterable

from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic15(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain, tailored for Pytrich Planner.

    Key ideas (admissible lower bounds):
    - Every misplaced package requires at least one putdown. If not currently held, it also requires a pickup.
      Therefore: 2 actions per misplaced object, except the currently held one (if any) which only needs 1 (putdown).
      This yields: actions_lb = 2 * (#misplaced) - (1 if holding any package else 0)
    - Movement lower bound:
        * If holding an object, you must at least move from current room to its goal room (ignoring door closures)
          before you can put it down. Add the graph distance in door steps (ignoring closures) to the LB.
        * If not holding, you must at least move to the nearest misplaced object's current room to pick it up.
          Add the minimal graph distance (ignoring closures) to any such room.
    - Opening doors lower bound:
        * If movement is needed and there is no path via currently open doors to the needed target(s),
          then at least one 'open' action is required. Add 1 in that case.

    This heuristic uses only bitwise checks (fast) and precomputes all-pairs shortest path distances ignoring closures
    to keep runtime low, even on large instances.

    Admissibility: Each component is a proven lower bound on the number of remaining primitive actions; components are
    combined conservatively to avoid double-counting shared costs. Manipulation actions (pickup/putdown) sum across
    packages because they cannot be shared. Movement and opening costs are added only for the immediate next necessity
    (held package's delivery or the first pickup), which keeps the estimate admissible.
    """

    def __init__(self, model, robot_gpt5_heuristic_15: bool = True, debug: bool = False):
        super().__init__()
        self.model = model
        self.debug = debug
        self.enabled = robot_gpt5_heuristic_15

        # Domain-specific indices
        self.objects: Set[str] = set()
        self.rooms: Set[str] = set()

        # Fact ID maps
        self.armempty_fid: Optional[int] = None
        self.rloc_fid_by_room: Dict[str, int] = {}
        self.holding_fid_by_obj: Dict[str, int] = {}
        self.closed_fid_by_door: Dict[str, int] = {}

        # 'in' predicate facts: (obj, room) -> fid
        self.in_fid_by_obj_room: Dict[Tuple[str, str], int] = {}
        # For each object, map room -> fid of in[obj, room]
        self.in_fids_by_obj: Dict[str, Dict[str, int]] = {}

        # goal_in mapping: obj -> goal_room
        self.goal_room_by_obj: Dict[str, str] = {}
        # For quick at-goal checks: obj -> fid of in[obj, goal_room]
        self.in_goal_fid_by_obj: Dict[str, int] = {}

        # Door graph (directed) ignoring closures
        # adjacency_all[room] = list of neighbor rooms (edges exist regardless of door closed/open)
        self.adjacency_all: Dict[str, List[str]] = {}
        # adjacency_open_check[room] = list of (neighbor, closed_fid) to check open edges in the current state
        self.adjacency_open_check: Dict[str, List[Tuple[str, Optional[int]]]] = {}

        # Precompute all-pairs shortest path distances (edge count) ignoring closures
        # distances[src][dst] = minimal number of moves (ignoring closures)
        self.distances: Dict[str, Dict[str, int]] = {}

        self._precompute()

    def get_name(self) -> str:
        return "RobotGPT5Heuristic15"

    # ---------------------------------------------------------------------
    # Precomputation utilities
    # ---------------------------------------------------------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _precompute(self):
        # First pass: collect facts by predicate
        for f in self.model.facts:
            pred, args = self._parse_fact_name(getattr(f, "name", ""))
            if not pred:
                continue

            if pred == "armempty":
                self.armempty_fid = f.global_id

            elif pred == "rloc":
                # rloc[room]
                if len(args) == 1:
                    room = args[0]
                    self.rooms.add(room)
                    self.rloc_fid_by_room[room] = f.global_id

            elif pred == "holding":
                # holding[obj]
                if len(args) == 1:
                    obj = args[0]
                    self.objects.add(obj)
                    self.holding_fid_by_obj[obj] = f.global_id

            elif pred == "in":
                # in[obj, room]
                if len(args) == 2:
                    obj, room = args
                    self.objects.add(obj)
                    self.rooms.add(room)
                    self.in_fid_by_obj_room[(obj, room)] = f.global_id
                    self.in_fids_by_obj.setdefault(obj, {})[room] = f.global_id

            elif pred == "goal_in":
                # goal_in[obj, room] is static preference (not the actual goal bitset), used to find target room per object
                if len(args) == 2:
                    obj, room = args
                    self.objects.add(obj)
                    self.rooms.add(room)
                    self.goal_room_by_obj[obj] = room

            elif pred == "door":
                # door[room1, room2, d]
                if len(args) == 3:
                    r1, r2, d = args
                    self.rooms.add(r1)
                    self.rooms.add(r2)
                    # For adjacency ignoring closures
                    self.adjacency_all.setdefault(r1, []).append(r2)
                    # place holder for open-check list; closed fid resolved later
                    self.adjacency_open_check.setdefault(r1, []).append((r2, None))

            elif pred == "closed":
                # closed[door]
                if len(args) == 1:
                    d = args[0]
                    self.closed_fid_by_door[d] = f.global_id

        # Second pass: enrich adjacency_open_check with closed fact ids for each directed door edge
        # We need to match edges back to their door name; to do this, we parse door facts again.
        for f in self.model.facts:
            pred, args = self._parse_fact_name(getattr(f, "name", ""))
            if pred == "door" and len(args) == 3:
                r1, r2, d = args
                closed_fid = self.closed_fid_by_door.get(d, None)
                # Update the corresponding entry in adjacency_open_check[r1]
                if r1 in self.adjacency_open_check:
                    # Replace the first (r2, None) with (r2, closed_fid)
                    lst = self.adjacency_open_check[r1]
                    for i, (nbr, cfid) in enumerate(lst):
                        if nbr == r2 and cfid is None:
                            lst[i] = (nbr, closed_fid)
                            break

        # Precompute for quick at-goal checks: in[obj, goal_room] fid
        for obj, goal_room in self.goal_room_by_obj.items():
            fid = self.in_fid_by_obj_room.get((obj, goal_room), None)
            if fid is not None:
                self.in_goal_fid_by_obj[obj] = fid

        # Ensure all rooms appear in adjacency maps
        for r in self.rooms:
            self.adjacency_all.setdefault(r, [])
            self.adjacency_open_check.setdefault(r, [])

        # Precompute all-pairs shortest path distances ignoring closures
        self._compute_all_pairs_distances()

    def _compute_all_pairs_distances(self):
        # BFS from every source room to compute hop-count distances
        for src in self.rooms:
            self.distances[src] = self._bfs_distances_from(src)

    def _bfs_distances_from(self, src: str) -> Dict[str, int]:
        dist: Dict[str, int] = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adjacency_all.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    # ---------------------------------------------------------------------
    # State query helpers (bitwise)
    # ---------------------------------------------------------------------
    def _bit_is_true(self, state_bitwise: int, fid: Optional[int]) -> bool:
        if fid is None:
            return False
        return (state_bitwise >> fid) & 1 == 1

    def _current_robot_room(self, state_bitwise: int) -> Optional[str]:
        # Exactly one rloc[room] should be true
        for room, fid in self.rloc_fid_by_room.items():
            if (state_bitwise >> fid) & 1:
                return room
        return None

    def _currently_held_object(self, state_bitwise: int) -> Optional[str]:
        # At most one
        for obj, fid in self.holding_fid_by_obj.items():
            if (state_bitwise >> fid) & 1:
                return obj
        return None

    def _object_at_goal(self, obj: str, state_bitwise: int) -> bool:
        fid = self.in_goal_fid_by_obj.get(obj, None)
        return self._bit_is_true(state_bitwise, fid)

    def _object_current_room(self, obj: str, state_bitwise: int) -> Optional[str]:
        # If held, no in[obj, room] is true
        rooms_fids = self.in_fids_by_obj.get(obj, {})
        for room, fid in rooms_fids.items():
            if (state_bitwise >> fid) & 1:
                return room
        return None

    # ---------------------------------------------------------------------
    # Graph queries conditioned on current state
    # ---------------------------------------------------------------------
    def _min_distance_ignoring_closures(self, src: Optional[str], targets: Iterable[str]) -> int:
        if not src:
            return 0
        dmap = self.distances.get(src, {})
        best = None
        for t in targets:
            if t == src:
                return 0
            if t in dmap:
                val = dmap[t]
                if best is None or val < best:
                    best = val
        return best if best is not None else 0

    def _is_reachable_via_open_edges(self, state_bitwise: int, src: Optional[str], targets: Set[str]) -> bool:
        if not src:
            return True
        if src in targets:
            return True
        # BFS restricted to edges where closed[door] is FALSE in the given state
        visited = {src}
        q = deque([src])
        while q:
            u = q.popleft()
            for v, closed_fid in self.adjacency_open_check.get(u, []):
                # Edge is traversable if door is not closed in current state
                is_closed = self._bit_is_true(state_bitwise, closed_fid)
                if is_closed:
                    continue
                if v in targets:
                    return True
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        return False

    # ---------------------------------------------------------------------
    # Heuristic value computation
    # ---------------------------------------------------------------------
    def __call__(self, state_bitwise: int, task_network: Optional[List] = None) -> int:
        if not self.enabled:
            return 0

        # Identify current robot location and held object (if any)
        rloc = self._current_robot_room(state_bitwise)
        held_obj = self._currently_held_object(state_bitwise)

        # Count misplaced objects, and collect candidate pickup rooms if not holding
        n_misplaced = 0
        candidate_pick_rooms: Set[str] = set()

        for obj in self.objects:
            at_goal = self._object_at_goal(obj, state_bitwise)
            if at_goal:
                continue
            # Not at goal means either it's in a non-goal room or currently held
            n_misplaced += 1
            if held_obj is None:
                # Only consider current rooms of not-at-goal objects for pickup candidates
                room = self._object_current_room(obj, state_bitwise)
                if room is not None:
                    candidate_pick_rooms.add(room)

        # Base manipulation lower bound: each misplaced object requires at least one putdown;
        # and one pickup unless it is already being held.
        if n_misplaced == 0:
            # All goals achieved regarding 'in' facts
            return 0

        manipulation_lb = 2 * n_misplaced - (1 if held_obj is not None else 0)
        if manipulation_lb < 0:
            manipulation_lb = 0

        # Movement lower bound (ignoring closures):
        movement_lb = 0
        open_lb = 0

        if held_obj is not None:
            goal_room = self.goal_room_by_obj.get(held_obj, None)
            if goal_room is not None and rloc is not None and goal_room != rloc:
                movement_lb = self._min_distance_ignoring_closures(rloc, [goal_room])
                # If there is no path via currently open doors, at least one open action is needed
                if movement_lb > 0:
                    if not self._is_reachable_via_open_edges(state_bitwise, rloc, {goal_room}):
                        open_lb = 1
        else:
            # Not holding: must go to some misplaced object's room (if not already there)
            if len(candidate_pick_rooms) > 0:
                movement_lb = self._min_distance_ignoring_closures(rloc, candidate_pick_rooms)
                # If movement is required and none of the candidate rooms is open-reachable, at least one open needed
                if movement_lb > 0:
                    if not self._is_reachable_via_open_edges(state_bitwise, rloc, candidate_pick_rooms):
                        open_lb = 1

        h_val = manipulation_lb + movement_lb + open_lb

        if self.debug:
            debug_info = {
                "rloc": rloc,
                "held_obj": held_obj,
                "n_misplaced": n_misplaced,
                "manipulation_lb": manipulation_lb,
                "movement_lb": movement_lb,
                "open_lb": open_lb,
                "h": h_val,
            }
            print(f"[RobotGPT5Heuristic15] {debug_info}")

        return int(h_val)

    # For frameworks that call 'evaluate' instead of __call__
    def evaluate(self, state_bitwise: int, task_network: Optional[List] = None) -> int:
        return self.__call__(state_bitwise, task_network)
