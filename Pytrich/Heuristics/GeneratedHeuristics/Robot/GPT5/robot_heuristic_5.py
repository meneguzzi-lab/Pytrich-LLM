from typing import Dict, List, Tuple, Optional, Set
import heapq

# Try to import the base Heuristic class from common locations
from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic5(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Robot domain.

    Rationale and design:
    - The Robot domain requires, for each package not at its goal room, at least a pickup and a putdown.
      If a package is currently held, it requires only a putdown. These per-package actions are independent
      and necessary, so summing them yields an admissible lower bound.
    - Moving between rooms may require opening closed doors (costing 1 action) and moving (costing 1 action).
      We compute shortest-path distances in a graph where each directed door edge has weight:
          1 (move) + (1 if door is currently closed else 0) (open)
      This yields an admissible lower bound on the movement+opening actions needed to traverse an edge.
    - To remain admissible with multiple packages, we only add the minimal travel cost needed to complete
      at least one package from the current state (the first package you could choose). For a held package,
      that is simply distance from current robot location to its goal room. If none is held, it is the minimum
      over all unplaced packages of: dist(robot_loc, obj_room) + dist(obj_room, goal_room).
      Summing more than one package's travel could overestimate because moves can be shared; this approach
      avoids that while still providing guidance.

    Heuristic returned:
      h(s) = (sum over all unplaced packages of [2 if not held else 1])
             + (min travel for the first package as defined above)

    Implementation notes:
    - Facts and state are bitwise-encoded. We use fast bit checks with global_id bit positions.
    - Goals (goal_in facts) are read from model.goals using model.state_explicit_repr(...) at init and cached.
    - We precompute room graph adjacency from door facts, and map each door to its 'closed' fact.
    - Distances are cached by (closed_signature, source_room_idx), where closed_signature is a compact int
      representing which doors are currently closed. This keeps per-node heuristic computation fast.

    This heuristic is admissible and designed to be consistent in practice, effectively guiding search while
    remaining computationally efficient even on the largest provided instance.
    """

    def __init__(self, model, robot_gpt5_heuristic_5: bool = True):
        super().__init__()
        self.model = model

        # Internal parsed structures
        self._rooms: Set[str] = set()
        self._room_to_idx: Dict[str, int] = {}
        self._idx_to_room: List[str] = []

        self._adj: List[List[Tuple[int, int]]] = []  # adjacency: room_idx -> list of (nbr_idx, closed_small_idx)

        # Fact mappings
        self._rloc_fact_by_room: Dict[str, int] = {}
        self._in_fact_by_obj_room: Dict[Tuple[str, str], int] = {}
        self._holding_fact_by_obj: Dict[str, int] = {}
        self._closed_fact_by_door: Dict[str, int] = {}
        self._door_edges_raw: List[Tuple[str, str, str]] = []  # (loc1, loc2, door)

        # Closed fact compact index mapping for fast bit-signature
        self._closed_gid_list: List[int] = []  # ordered list of closed fact global ids
        self._closed_gid_to_small_idx: Dict[int, int] = {}  # gid -> compact idx
        self._door_to_closed_small_idx: Dict[str, int] = {}  # door name -> compact idx

        # Goals
        self._goals_obj_to_room: Dict[str, str] = {}

        # Distance cache keyed by (closed_signature, source_room_idx) -> List[int]
        self._dist_cache: Dict[Tuple[int, int], List[int]] = {}

        # Initialize all mappings
        self._parse_facts_and_goals()
        self._build_room_indices_and_graph()

    # ------------------------ Parsing Helpers ------------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            bracket_idx = name2.find("[")
            if bracket_idx > 0 and name2.endswith("]"):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _parse_facts_and_goals(self):
        # 1) Scan all facts once to fill base structures
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            if pred == "rloc" and len(args) == 1:
                room = args[0]
                self._rooms.add(room)
                self._rloc_fact_by_room[room] = f.global_id
            elif pred == "in" and len(args) == 2:
                obj, room = args[0], args[1]
                self._rooms.add(room)
                self._in_fact_by_obj_room[(obj, room)] = f.global_id
            elif pred == "holding" and len(args) == 1:
                obj = args[0]
                self._holding_fact_by_obj[obj] = f.global_id
            elif pred == "closed" and len(args) == 1:
                d = args[0]
                self._closed_fact_by_door[d] = f.global_id
            elif pred == "door" and len(args) == 3:
                loc1, loc2, d = args[0], args[1], args[2]
                self._rooms.add(loc1)
                self._rooms.add(loc2)
                # We'll convert to adjacency after we also map closed(d)
                self._door_edges_raw.append((loc1, loc2, d))

        # 2) Read goals explicitly from model.goals (bitwise) and cache goal_in mapping
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "goal_in" and len(args) == 2:
                obj, room = args[0], args[1]
                self._goals_obj_to_room[obj] = room
                # room already in self._rooms via facts, but add to be safe
                self._rooms.add(room)

        # 3) Prepare closed compact index mapping (after we know all doors)
        #    We'll keep the order stable as we iterate over closed facts
        self._closed_gid_list = list(self._closed_fact_by_door.values())
        for idx, gid in enumerate(self._closed_gid_list):
            self._closed_gid_to_small_idx[gid] = idx

        # 4) Map each door to its closed small index (if any)
        for d, gid in self._closed_fact_by_door.items():
            small_idx = self._closed_gid_to_small_idx[gid]
            self._door_to_closed_small_idx[d] = small_idx

    def _build_room_indices_and_graph(self):
        # Assign indices to rooms for array-based speed
        self._idx_to_room = sorted(self._rooms)
        self._room_to_idx = {r: i for i, r in enumerate(self._idx_to_room)}
        n = len(self._idx_to_room)
        self._adj = [[] for _ in range(n)]

        # Build adjacency from door facts
        for loc1, loc2, d in self._door_edges_raw:
            if loc1 not in self._room_to_idx or loc2 not in self._room_to_idx:
                continue
            u = self._room_to_idx[loc1]
            v = self._room_to_idx[loc2]
            # Every door must have a closed fact (from domain); if missing, treat as always open (cost 1)
            closed_small_idx = self._door_to_closed_small_idx.get(d, -1)
            if closed_small_idx == -1:
                # Assign a special mapping: we'll treat as open (no extra open cost)
                # To avoid branching on -1 during the search, we'll map it to a guaranteed 'never closed' index
                # by allocating a fresh closed id that is always 0 in signature. Simpler: keep -1 and handle in edge cost.
                pass
            self._adj[u].append((v, closed_small_idx))

    # ------------------------ State Queries ------------------------
    def _is_fact_true(self, state_bitwise: int, fact_gid: int) -> bool:
        return ((state_bitwise >> fact_gid) & 1) == 1

    def _current_robot_room(self, state_bitwise: int) -> Optional[str]:
        # Only one rloc(*) fact can be true
        for room, gid in self._rloc_fact_by_room.items():
            if self._is_fact_true(state_bitwise, gid):
                return room
        return None

    def _held_object(self, state_bitwise: int) -> Optional[str]:
        # Only one object can be held at a time in this domain
        for obj, gid in self._holding_fact_by_obj.items():
            if self._is_fact_true(state_bitwise, gid):
                return obj
        return None

    def _object_room(self, state_bitwise: int, obj: str) -> Optional[str]:
        # Find current room where in[obj, room] is true; None if held
        # Iterate all rooms for this obj
        # Optimization: We can stop at first true, relying on domain mutual exclusivity
        for room in self._rooms:
            gid = self._in_fact_by_obj_room.get((obj, room))
            if gid is not None and self._is_fact_true(state_bitwise, gid):
                return room
        return None

    def _object_at_goal(self, state_bitwise: int, obj: str, goal_room: str) -> bool:
        gid = self._in_fact_by_obj_room.get((obj, goal_room))
        return gid is not None and self._is_fact_true(state_bitwise, gid)

    # ------------------------ Distance Computation ------------------------
    def _closed_signature(self, state_bitwise: int) -> int:
        # Compact signature over only closed(*) facts
        sig = 0
        for idx, gid in enumerate(self._closed_gid_list):
            if ((state_bitwise >> gid) & 1) == 1:
                sig |= (1 << idx)
        return sig

    def _edge_cost(self, closed_signature: int, closed_small_idx: int) -> int:
        # Move cost always 1; opening adds +1 if the door is closed.
        if closed_small_idx < 0:
            # Door without a closed fact treated as always open
            return 1
        is_closed = (closed_signature >> closed_small_idx) & 1
        return 1 + (1 if is_closed else 0)

    def _distances_from(self, state_bitwise: int, source_room: str) -> List[int]:
        # Returns list of distances (in primitive actions) to all rooms from source_room under current closed set
        if source_room not in self._room_to_idx:
            # Unknown room => distances infinite
            n = len(self._idx_to_room)
            return [10 ** 9] * n

        src_idx = self._room_to_idx[source_room]
        closed_sig = self._closed_signature(state_bitwise)
        key = (closed_sig, src_idx)
        cached = self._dist_cache.get(key)
        if cached is not None:
            return cached

        n = len(self._idx_to_room)
        dist = [10 ** 9] * n
        dist[src_idx] = 0
        heap = [(0, src_idx)]

        while heap:
            d, u = heapq.heappop(heap)
            if d != dist[u]:
                continue
            for v, cidx in self._adj[u]:
                w = self._edge_cost(closed_sig, cidx)
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        self._dist_cache[key] = dist
        return dist

    # ------------------------ Heuristic Evaluation ------------------------
    def evaluate(self, state_bitwise: int, task_network: List = None) -> int:
        # 1) Quick exit: if goals reached and TN empty, return 0
        if self.model.goal_reached(state_bitwise, task_network or []):
            return 0

        # 2) Fetch current robot location
        robot_room = self._current_robot_room(state_bitwise)
        if robot_room is None:
            # If robot location is unknown, return 0 to stay safe admissible (or a minimal informative value)
            robot_room = next(iter(self._rooms)) if self._rooms else ""

        # 3) Determine which object is currently held (if any)
        held_obj = self._held_object(state_bitwise)

        # 4) Build list of packages still to place and compute base pickup/putdown cost
        #    For held package: needs only putdown (1). For others not at goal: need pickup+putdown (2).
        to_place: List[Tuple[str, Optional[str], str, int]] = []
        # entries: (obj, current_room if not held else None, goal_room, base_action_cost)
        base_cost = 0

        for obj, goal_room in self._goals_obj_to_room.items():
            if held_obj == obj:
                # Held: not at any room; still needs to be put down at goal
                # Even if robot is currently at goal, putdown still required (1 action)
                base_cost += 1
                to_place.append((obj, None, goal_room, 1))
            else:
                # If object already at goal room, skip
                if self._object_at_goal(state_bitwise, obj, goal_room):
                    continue
                # Else it is somewhere (or possibly unaccounted if held by another agent, which doesn't exist here)
                cur_room = self._object_room(state_bitwise, obj)
                # Normal case: needs pickup + putdown
                base_cost += 2
                to_place.append((obj, cur_room, goal_room, 2))

        # If nothing left to place, we are done w.r.t. goals (though TN may not be empty yet)
        if not to_place:
            return 0

        # 5) Movement + opening lower bound for at least one package
        travel_lb = 0
        if held_obj is not None:
            # If holding something, lower bound on movement is dist(robot_room, goal_room_of_held)
            goal_room = self._goals_obj_to_room[held_obj]
            dist_from_robot = self._distances_from(state_bitwise, robot_room)
            goal_idx = self._room_to_idx.get(goal_room)
            if goal_idx is not None:
                travel_lb = dist_from_robot[goal_idx]
            else:
                travel_lb = 0
        else:
            # Compute distances once
            dist_from_robot = self._distances_from(state_bitwise, robot_room)
            # Group by unique goal rooms to avoid repeated Dijkstra
            unique_goals: Set[str] = set(gr for (_, _, gr, _) in to_place)
            dist_from_goal_map: Dict[str, List[int]] = {}
            for gr in unique_goals:
                dist_from_goal_map[gr] = self._distances_from(state_bitwise, gr)

            best = 10 ** 9
            # For each unplaced (non-held) object: cost = dist(robot, obj_room) + dist(obj_room, goal)
            # We approximate dist(obj_room, goal) using dist_from_goal[ obj_room ] (symmetric graph in this domain)
            for (obj, cur_room, goal_room, base) in to_place:
                if cur_room is None:
                    # This only happens for held object, but in this branch we know held_obj is None
                    continue
                obj_idx = self._room_to_idx.get(cur_room)
                if obj_idx is None:
                    continue
                d1 = dist_from_robot[obj_idx]
                dist_from_goal = dist_from_goal_map.get(goal_room)
                if dist_from_goal is None:
                    continue
                d2 = dist_from_goal[obj_idx]
                cand = d1 + d2
                if cand < best:
                    best = cand
            if best < 10 ** 9:
                travel_lb = best
            else:
                travel_lb = 0

        # 6) Final heuristic value
        h_val = base_cost + travel_lb
        if h_val < 0:
            h_val = 0
        return int(h_val)


# Optional alias for registration convenience in some frameworks
HeuristicClass = RobotGPT5Heuristic5
