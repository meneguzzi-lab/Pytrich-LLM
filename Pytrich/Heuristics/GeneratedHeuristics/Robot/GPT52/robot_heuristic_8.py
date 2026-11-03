from collections import deque
from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic8(Heuristic):
    """Domain-dependent heuristic for the HDDL 'robot' domain.

    Domain structure recap (grounded):
      - Facts include: rloc[room], in[obj,room], holding[obj], armempty,
        closed[door], door[room,room,door], goal_in[obj,room].
      - Primitive operators: pickup, putdown, move, open.
      - HTN tasks cycle: achieve-goals -> (pickup_abstract -> release) or move/open loops until all goals satisfied.

    Key domain insight:
      - The actual goal is conjunction of (in o_i goal_room_i).
      - To fix a misplaced package requires at least: pickup (1), putdown (1), and some travel.
      - Travel between rooms requires traversing edges (door facts). If the traversed door is closed,
        it must be opened first (open action cost 1) *from the source side*.

    Heuristic design (admissible lower bound):
      h(state, tn) = max(
          unsatisfied_goal_count,
          max_per_object_delivery_lower_bound,
          task_network_min_refinement_lower_bound
      )

    Components:
      1) unsatisfied_goal_count: each unsatisfied (in o,goal) needs at least one primitive action.
         (Very weak but cheap.)

      2) max_per_object_delivery_lower_bound: for each object not at its goal room, compute a lower bound:
            pickup+putdown (2) + min moves from robot to object + min moves object-room to goal-room
            plus mandatory opens that are *unavoidably* needed on those paths.
         We take MAX over objects (not sum) to stay admissible despite the single-arm coupling
         and because summing would overcount shared travel.

         Path computation uses 0-1 BFS on a directed graph of rooms where each door-edge has:
            move_cost = 1
            open_needed = 0 if door already open, else 1 additional action.
         So edge weight is (open_needed + 1). This is a lower bound because:
            - It ignores that after opening once the door stays open (so repeated traversals are cheaper in reality)
              => overestimation risk if we summed; but we take MAX over single delivery paths.
            - It ignores arm constraints while moving (allowed).

      3) task_network_min_refinement_lower_bound: if the current task network begins with abstract tasks,
         each abstract task must be refined at least once. We add a tiny admissible bound:
            count of abstract tasks in TN that are not primitive operators.
         (In this domain, each abstract task expands to exactly one primitive operator eventually.)

    Efficiency:
      - Precomputes room graph topology and mappings from fact ids.
      - Avoids state_explicit_repr(state) in evaluation; uses bit checks.
      - Uses per-evaluation BFS a few times; bounded by rooms (<=50 in training), cheap.

    __init__ parameter name required by prompt: robot_gpt52_heuristic_8
    """

    def __init__(self, model, robot_gpt52_heuristic_8: str = "robot_gpt52_heuristic_8"):
        super().__init__(model)
        self.name = robot_gpt52_heuristic_8

        # --- Index facts for fast access ---
        # Store global_id for key unary facts
        self._armempty_fid: Optional[int] = None

        # rloc: room -> fact id
        self._rloc_fid_by_room: Dict[str, int] = {}

        # holding: obj -> fact id
        self._holding_fid_by_obj: Dict[str, int] = {}

        # in: (obj, room) -> fact id
        self._in_fid_by_obj_room: Dict[Tuple[str, str], int] = {}

        # goal_in: obj -> goal room (static), and also its fact id (not needed for state, but for parsing)
        self._goal_room_by_obj: Dict[str, str] = {}

        # closed: door -> fact id
        self._closed_fid_by_door: Dict[str, int] = {}

        # door connectivity: (loc1, loc2) -> door_id (directed edges may share same door)
        self._door_id_by_rooms: Dict[Tuple[str, str], str] = {}

        # room adjacency list: room -> list of (neighbor_room, door_id)
        self._adj: Dict[str, List[Tuple[str, str]]] = {}

        # Collect all rooms/objects encountered for iteration (from facts)
        self._rooms: List[str] = []
        self._objects: List[str] = []

        self._build_indices()

        # Cache goal facts (bitwise int) for quick unsatisfied count
        self._goal_fact_ids: List[int] = []
        for fname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(fname)
            if pred == "in" and len(args) == 2:
                fid = self._fact_id_by_name(fname)
                if fid is not None:
                    self._goal_fact_ids.append(fid)

    # ----------------- Core Heuristic API -----------------
    def __call__(self, state: int, task_network) -> int:
        # 0) If already goal and no tasks, heuristic is 0
        # (planner checks goal separately; keep fast check)
        if self.model.goals <= state and (task_network is None or len(task_network) == 0):
            return 0

        # 1) Unsatisfied goal count (very cheap)
        unsat = 0
        for fid in self._goal_fact_ids:
            if ((state >> fid) & 1) == 0:
                unsat += 1

        # 2) Compute robot location (room)
        rroom = self._get_robot_room(state)

        # 3) For each object not at goal room, compute a per-object lower bound; take max.
        max_obj_lb = 0
        if rroom is not None:
            for obj in self._objects:
                goal_room = self._goal_room_by_obj.get(obj)
                if goal_room is None:
                    continue

                # If already in goal room in state, skip
                if self._is_true_in(state, obj, goal_room):
                    continue

                # Determine where the object currently is (or if held)
                held = self._is_holding(state, obj)
                if held:
                    # If holding, we just need to deliver to goal.
                    # Lower bound: travel from current robot room to goal room + putdown (1)
                    dist_to_goal = self._dist_room_to_room(state, rroom, goal_room)
                    if dist_to_goal is None:
                        # unreachable => no guidance
                        continue
                    obj_lb = dist_to_goal + 1
                else:
                    obj_room = self._get_object_room(state, obj)
                    if obj_room is None:
                        # If neither held nor located, ignore (shouldn't happen)
                        continue
                    # Need pickup+putdown at least (2) and travel r->obj + obj->goal
                    dist_r_to_obj = self._dist_room_to_room(state, rroom, obj_room)
                    dist_obj_to_goal = self._dist_room_to_room(state, obj_room, goal_room)
                    if dist_r_to_obj is None or dist_obj_to_goal is None:
                        continue
                    obj_lb = 2 + dist_r_to_obj + dist_obj_to_goal

                if obj_lb > max_obj_lb:
                    max_obj_lb = obj_lb

        # 4) Minimal refinement bound from task network (admissible, tiny)
        tn_lb = 0
        if task_network is not None:
            # Count non-primitive tasks: AbstractTask instances have 'decompositions' attribute.
            # Operators have 'applicable' and 'apply' but no 'decompositions'.
            for t in task_network:
                if hasattr(t, "decompositions"):
                    tn_lb += 1

        # Combine with max to keep admissible
        h = max(unsat, max_obj_lb, tn_lb)
        return int(h)

    # ----------------- Index Building -----------------
    def _build_indices(self):
        rooms_set = set()
        objs_set = set()

        for fact in self.model.facts:
            name = fact.name
            pred, args = self._parse_fact_name(name)
            if pred is None:
                continue

            if pred == "armempty" and len(args) == 0 and name.startswith("+"):
                self._armempty_fid = fact.global_id

            elif pred == "rloc" and len(args) == 1 and name.startswith("+"):
                room = args[0]
                self._rloc_fid_by_room[room] = fact.global_id
                rooms_set.add(room)

            elif pred == "holding" and len(args) == 1 and name.startswith("+"):
                obj = args[0]
                self._holding_fid_by_obj[obj] = fact.global_id
                objs_set.add(obj)

            elif pred == "in" and len(args) == 2 and name.startswith("+"):
                obj, room = args
                self._in_fid_by_obj_room[(obj, room)] = fact.global_id
                objs_set.add(obj)
                rooms_set.add(room)

            elif pred == "goal_in" and len(args) == 2 and name.startswith("+"):
                obj, room = args
                # goal_in facts are part of state representation but effectively static.
                # We'll use them to get each object's goal room.
                self._goal_room_by_obj[obj] = room
                objs_set.add(obj)
                rooms_set.add(room)

            elif pred == "closed" and len(args) == 1 and name.startswith("+"):
                door = args[0]
                self._closed_fid_by_door[door] = fact.global_id

            elif pred == "door" and len(args) == 3 and name.startswith("+"):
                loc1, loc2, door = args
                self._door_id_by_rooms[(loc1, loc2)] = door
                rooms_set.add(loc1)
                rooms_set.add(loc2)
                if loc1 not in self._adj:
                    self._adj[loc1] = []
                self._adj[loc1].append((loc2, door))

        self._rooms = sorted(rooms_set)
        self._objects = sorted(objs_set)

        # Ensure adjacency dict keys exist
        for r in self._rooms:
            self._adj.setdefault(r, [])

    # ----------------- State queries (bitwise) -----------------
    def _get_robot_room(self, state: int) -> Optional[str]:
        # rloc is exclusive; find true one by checking known rloc facts.
        for room, fid in self._rloc_fid_by_room.items():
            if ((state >> fid) & 1) == 1:
                return room
        return None

    def _is_holding(self, state: int, obj: str) -> bool:
        fid = self._holding_fid_by_obj.get(obj)
        if fid is None:
            return False
        return ((state >> fid) & 1) == 1

    def _is_true_in(self, state: int, obj: str, room: str) -> bool:
        fid = self._in_fid_by_obj_room.get((obj, room))
        if fid is None:
            return False
        return ((state >> fid) & 1) == 1

    def _get_object_room(self, state: int, obj: str) -> Optional[str]:
        # Scan rooms for object's in-facts. Rooms <= 50, objects <= 100 => OK.
        for room in self._rooms:
            fid = self._in_fid_by_obj_room.get((obj, room))
            if fid is not None and ((state >> fid) & 1) == 1:
                return room
        return None

    def _is_door_closed(self, state: int, door: str) -> bool:
        fid = self._closed_fid_by_door.get(door)
        if fid is None:
            return False
        return ((state >> fid) & 1) == 1

    # ----------------- Distances on room graph -----------------
    def _dist_room_to_room(self, state: int, start: str, goal: str) -> Optional[int]:
        if start == goal:
            return 0
        if start not in self._adj or goal not in self._adj:
            return None

        # 0-1 BFS on derived weighted edges where each directed edge costs:
        #   1 for move + (1 if door is closed else 0)
        # We can implement with deque by splitting into open edges (weight 1) and closed edges (weight 2)
        # but simplest is standard BFS on small graphs with integer weights via deque-based SPFA for weights 1/2.

        INF = 10 ** 9
        dist: Dict[str, int] = {start: 0}
        inq: Dict[str, bool] = {start: True}
        q = deque([start])

        while q:
            u = q.popleft()
            inq[u] = False
            du = dist[u]
            for v, door in self._adj.get(u, []):
                w = 1 + (1 if self._is_door_closed(state, door) else 0)
                nd = du + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    if not inq.get(v, False):
                        q.append(v)
                        inq[v] = True

        d = dist.get(goal, INF)
        return None if d >= INF else d

    # ----------------- Utilities -----------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            bracket_idx = name2.find("[")
            if bracket_idx > 0 and name2.endswith("]"):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _fact_id_by_name(self, fact_name: str) -> Optional[int]:
        # Build a small on-demand mapping once; facts are many but stable.
        # This is only used in __init__ for goals, so overhead is negligible.
        if not hasattr(self, "_fid_by_name"):
            self._fid_by_name = {f.name: f.global_id for f in self.model.facts}
        return self._fid_by_name.get(fact_name)
