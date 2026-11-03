from collections import deque
from typing import Dict, List, Tuple, Optional

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic15(Heuristic):
    """Domain-dependent heuristic for the HDDL Robot domain.

    Domain structure insight
    -----------------------
    The domain implements a single-robot delivery-like problem with these primitives:
      - move(room1, room2, door)
      - open(room1, room2, door)
      - pickup(obj, room)
      - putdown(obj, room)  [only allowed at the *goal room* for that object]

    HTN structure:
      achieve-goals repeatedly chooses an object not yet at its goal room,
      then pickup_abstract(obj) and release.
      release keeps trying move/open until it can putdown the held object in its goal room.

    Key planning challenges
    -----------------------
    1) Navigation dominates: objects can be far apart in a room graph; doors can be closed and
       may require opening before crossing.
    2) Single arm capacity: at most one package can be transported at a time.
    3) Goals are simply (in obj goal_room) for many objects; each misplaced object requires at
       least one pickup+putdown pair, plus travel.

    Heuristic design
    ----------------
    We compute a fast admissible lower bound by combining:
      (A) A mandatory manipulation cost: for each object not in its goal room, at least
          2 actions are needed (pickup + putdown). Sum of these gives 2*k.
      (B) A mandatory "start work" navigation cost: if there exists any misplaced object
          and we are not currently holding one, we must at least reach *some* misplaced object
          location to pick it up. This is at least the minimum relaxed distance from the robot's
          current room to any room containing a misplaced object.
          - We use a relaxed edge-cost: moving through a door costs 1 if open, 2 if closed
            (open+move). This is admissible because any real traversal of a closed door
            requires >= 2 primitive actions.
      (C) If currently holding an object, we must at least reach its goal room and then put it
          down. We add relaxed distance(robot_room, goal_room) + 1.

    Why this helps on large instances
    --------------------------------
    - Uses only bitwise checks for facts.
    - Precomputes a room graph with per-edge cost depending on door-closed facts (evaluated per
      state cheaply).
    - Runs a small number of Dijkstra-like computations per node:
        * one multi-target shortest path to nearest relevant room (or holding goal)
      Typically O(E log V) with V<=50, E small.

    Admissibility
    -------------
    Each term is a lower bound on mandatory actions, and we combine them conservatively:
      - 2*k: each misplaced object must be picked and placed at least once.
      - if holding: +dist_to_goal +1 is mandatory for the held object.
      - else if not holding and k>0: +min_dist_to_any_misplaced_room is mandatory to start.
    We do NOT sum per-object travel distances (would double-count moves across sequential
    transports), keeping the bound safe.

    Notes
    -----
    This heuristic is state-based (it ignores the remaining HTN task network), but it is aligned
    with the domain's only reasonable HTN refinements: achieve-goals will keep servicing misplaced
    objects until none remain.
    """

    def __init__(self, model, robot_gpt52_heuristic_15: str = "robot_gpt52_heuristic_15"):
        super().__init__(model, robot_gpt52_heuristic_15)
        self.model = model

        # Build fast indices for relevant facts
        self._fact_ids_by_pred: Dict[str, List[int]] = {}
        self._rloc_facts: Dict[str, int] = {}              # room -> fact_id for +rloc[room]
        self._armempty_fact: Optional[int] = None          # +armempty
        self._holding_facts: Dict[str, int] = {}           # obj -> fact_id for +holding[obj]

        # +in[obj,room] fact ids
        self._in_fact: Dict[Tuple[str, str], int] = {}

        # +goal_in[obj,room] is static in provided instances; still treat as facts.
        self._goal_in_fact: Dict[Tuple[str, str], int] = {}
        self._goal_room_of_obj: Dict[str, str] = {}

        # door graph: adjacency list with door object name
        # room -> list of (neighbor_room, door_name)
        self._adj: Dict[str, List[Tuple[str, str]]] = {}

        # +closed[door] fact ids
        self._closed_fact: Dict[str, int] = {}

        self._preprocess_facts_and_graph()

        # Cache goal facts explicitly (cheap, done once)
        self._goal_fact_names = self.model.state_explicit_repr(self.model.goals)

    # ---------- Fact parsing helpers ----------
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

    # ---------- Preprocessing ----------
    def _preprocess_facts_and_graph(self):
        # Index facts
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue

            self._fact_ids_by_pred.setdefault(pred, []).append(f.global_id)

            if pred == "armempty" and len(args) == 0 and f.name.startswith("+"):
                self._armempty_fact = f.global_id

            elif pred == "rloc" and len(args) == 1 and f.name.startswith("+"):
                self._rloc_facts[args[0]] = f.global_id

            elif pred == "holding" and len(args) == 1 and f.name.startswith("+"):
                self._holding_facts[args[0]] = f.global_id

            elif pred == "in" and len(args) == 2 and f.name.startswith("+"):
                obj, room = args
                self._in_fact[(obj, room)] = f.global_id

            elif pred == "goal_in" and len(args) == 2 and f.name.startswith("+"):
                obj, room = args
                self._goal_in_fact[(obj, room)] = f.global_id
                # Each package has exactly one goal room in the training data
                self._goal_room_of_obj[obj] = room

            elif pred == "closed" and len(args) == 1 and f.name.startswith("+"):
                self._closed_fact[args[0]] = f.global_id

        # Build adjacency from +door facts
        # Facts are of form +door[loc1,loc2,d]
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred != "door" or not f.name.startswith("+") or len(args) != 3:
                continue
            a, b, d = args
            self._adj.setdefault(a, []).append((b, d))

        # Ensure rooms in rloc set exist in adj map even if isolated
        for room in self._rloc_facts.keys():
            self._adj.setdefault(room, [])

    # ---------- Graph distance (relaxed w.r.t. door opening) ----------
    def _edge_cost(self, state: int, door_name: str) -> int:
        """Relaxed primitive cost to traverse an arc labeled with a door.
        If door is closed: need open+move => cost>=2; else move => cost>=1.
        """
        fid = self._closed_fact.get(door_name)
        if fid is None:
            return 1
        return 2 if self._is_true(state, fid) else 1

    def _dijkstra_min_to_targets(self, state: int, start_room: str, targets: set) -> int:
        """Return min relaxed cost from start_room to any room in targets.
        If start_room is already a target, returns 0.
        """
        if start_room in targets:
            return 0
        if not targets:
            return 0

        # Small graph => simple Dijkstra with list-based priority is fine.
        # But we implement a binary heap for robustness.
        import heapq

        dist: Dict[str, int] = {start_room: 0}
        pq = [(0, start_room)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u in targets:
                return d
            for v, door_name in self._adj.get(u, []):
                nd = d + self._edge_cost(state, door_name)
                if nd < dist.get(v, 10**9):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        return 10**6  # unreachable (should not happen in benchmarks)

    def _dijkstra_distance(self, state: int, start_room: str, goal_room: str) -> int:
        if start_room == goal_room:
            return 0
        return self._dijkstra_min_to_targets(state, start_room, {goal_room})

    # ---------- Extract dynamic info from a state ----------
    def _get_robot_room(self, state: int) -> Optional[str]:
        for room, fid in self._rloc_facts.items():
            if self._is_true(state, fid):
                return room
        return None

    def _get_holding_obj(self, state: int) -> Optional[str]:
        for obj, fid in self._holding_facts.items():
            if self._is_true(state, fid):
                return obj
        return None

    def _misplaced_objects_and_rooms(self, state: int) -> Tuple[int, set]:
        """Return (k, rooms_with_misplaced_objects)."""
        k = 0
        rooms = set()

        # We rely on static goal_in mapping extracted from facts.
        for obj, goal_room in self._goal_room_of_obj.items():
            # Check if obj is currently in its goal room.
            in_goal_fid = self._in_fact.get((obj, goal_room))
            if in_goal_fid is not None and self._is_true(state, in_goal_fid):
                continue

            # If holding it, it's not in goal room (unless we are in goal, still holding).
            holding_fid = self._holding_facts.get(obj)
            if holding_fid is not None and self._is_true(state, holding_fid):
                k += 1
                # No pickup location room to include.
                continue

            # Otherwise find which room it is in (scan its possible in facts by checking all rooms).
            # This is O(#rooms) per object in worst case, but rooms<=50 and only for objects with goals.
            obj_room = None
            for (o, r), fid in self._in_fact.items():
                if o == obj and self._is_true(state, fid):
                    obj_room = r
                    break

            k += 1
            if obj_room is not None:
                rooms.add(obj_room)

        return k, rooms

    # ---------- Heuristic evaluation ----------
    def __call__(self, parent_node, node) -> int:
        state = node.state

        robot_room = self._get_robot_room(state)
        if robot_room is None:
            return 0

        holding_obj = self._get_holding_obj(state)

        # Count how many objects are not yet at their goal room.
        k, misplaced_rooms = self._misplaced_objects_and_rooms(state)
        if k == 0:
            return 0

        h = 0

        # Mandatory manipulation lower bound: pickup+putdown for each misplaced object.
        # (Even if currently holding, that object still needs a putdown; counting 2 is still admissible
        # as a lower bound for all remaining objects because the held object must have been picked
        # already in the past, but in the remaining cost-to-go it does not necessarily need pickup.
        # To stay admissible, only count 2 for objects not already held.)
        # We therefore compute:
        #   - For held object: at least 1 action (putdown) remaining (plus travel)
        #   - For each other misplaced object: at least 2 actions remaining.
        if holding_obj is not None:
            # one object is being carried -> it only needs putdown (1) remaining, not pickup.
            h += 1
            h += 2 * (k - 1)
        else:
            h += 2 * k

        # Navigation lower bound
        if holding_obj is not None:
            goal_room = self._goal_room_of_obj.get(holding_obj)
            if goal_room is not None:
                h += self._dijkstra_distance(state, robot_room, goal_room)
        else:
            # Need to reach at least one misplaced object to pick something up.
            if misplaced_rooms:
                h += self._dijkstra_min_to_targets(state, robot_room, misplaced_rooms)

        # Keep heuristic non-negative integer
        if h < 0:
            h = 0
        return int(h)
