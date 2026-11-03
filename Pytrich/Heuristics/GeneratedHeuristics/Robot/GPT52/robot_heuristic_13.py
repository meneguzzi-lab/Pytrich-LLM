from collections import deque
from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic13(Heuristic):
    """Domain-dependent heuristic for the HDDL 'robot' domain.

    Domain observations (from robot.hddl)
    -----------------------------------
    - Only primitive actions: move, open, pickup, putdown.
    - The HTN control essentially loops: achieve-goals keeps choosing among
      pickup/open/move until no more violations, while release keeps trying
      to putdown if holding an object at its goal room.
    - Key resource constraint: single gripper (armempty/holding).
    - Doors may be closed; moving through a door requires it to be open.
      Opening is local: requires rloc at the source room.

    Heuristic design
    ----------------
    We compute a cheap lower bound on remaining primitive actions by combining:

    (A) Unsatisfied delivery count (admissible):
        Count objects not currently in their goal room. Each such object needs
        at least one putdown into its goal room => +1 per unsatisfied object.

    (B) Current holding adjustment (admissible):
        If holding an object that is not already "at goal" (i.e., goal_in(obj, rloc)
        is false), then at least one pickup has already happened; still at least
        one putdown will be required (already counted in A if unsatisfied).
        We do not add extra for pickup to preserve admissibility.

    (C) Navigation/opening lower bound to reach *some* useful room (admissible):
        If there exists an unsatisfied object, then to make progress we must reach either:
          - a room containing an object to pick up (if armempty), or
          - the goal room of the currently held object (if holding).
        We compute a shortest-path distance in a relaxed graph where traversing an edge
        (room -> room) has cost:
            1 for move + 1 if its door is currently closed (because open will be needed).
        This is admissible because any real plan must pay at least these moves/opens
        to reach the target room.

    (D) Task-network signal (lightweight):
        If task network is empty, heuristic is 0 (goal check handled by planner).
        Otherwise we rely on state-based estimate; we do not add a per-abstract-task
        constant to keep admissible.

    Efficiency
    ----------
    - All fact queries use bitwise checks by global_id.
    - We pre-index all relevant facts (rloc, in, goal_in, holding, armempty, closed, door)
      once in __init__.
    - We pre-build a room adjacency list with door ids, then run 0-1 style Dijkstra
      with small integer weights (1 or 2) per evaluation.

    This heuristic is admissible and typically informative on large instances because
    it captures (1) number of remaining deliveries and (2) minimal navigation+opening
    required to reach the next pickup/delivery opportunity.
    """

    def __init__(self, model, robot_gpt52_heuristic_13: str = "robot_gpt52_heuristic_13"):
        super().__init__(model, robot_gpt52_heuristic_13)
        self.model = model

        # --- Parse and index facts ---
        self.facts_by_pred: Dict[str, List] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self.facts_by_pred.setdefault(pred, []).append((f, args))

        # Index specific predicate facts for fast iteration
        self.armempty_facts = [fa for fa, _ in self.facts_by_pred.get("armempty", [])]
        self.holding_facts: List[Tuple] = self.facts_by_pred.get("holding", [])
        self.rloc_facts: List[Tuple] = self.facts_by_pred.get("rloc", [])
        self.in_facts: List[Tuple] = self.facts_by_pred.get("in", [])
        self.goal_in_facts: List[Tuple] = self.facts_by_pred.get("goal_in", [])
        self.closed_facts: List[Tuple] = self.facts_by_pred.get("closed", [])
        self.door_facts: List[Tuple] = self.facts_by_pred.get("door", [])

        # Build lookups: goal room per obj, and in-facts per obj
        self.goal_room_of: Dict[str, str] = {}
        for f, args in self.goal_in_facts:
            if len(args) == 2:
                obj, room = args
                # Each object has exactly one goal in the generated instances
                self.goal_room_of[obj] = room

        self.in_fact_by_obj_room: Dict[Tuple[str, str], int] = {}
        self.in_facts_by_obj: Dict[str, List[Tuple[int, str]]] = {}
        for f, args in self.in_facts:
            if len(args) == 2:
                obj, room = args
                self.in_fact_by_obj_room[(obj, room)] = f.global_id
                self.in_facts_by_obj.setdefault(obj, []).append((f.global_id, room))

        self.holding_fact_by_obj: Dict[str, int] = {}
        for f, args in self.holding_facts:
            if len(args) == 1:
                obj = args[0]
                self.holding_fact_by_obj[obj] = f.global_id

        self.rloc_fact_by_room: Dict[str, int] = {}
        for f, args in self.rloc_facts:
            if len(args) == 1:
                room = args[0]
                self.rloc_fact_by_room[room] = f.global_id

        self.goal_in_fact_by_obj_room: Dict[Tuple[str, str], int] = {}
        for f, args in self.goal_in_facts:
            if len(args) == 2:
                obj, room = args
                self.goal_in_fact_by_obj_room[(obj, room)] = f.global_id

        self.armempty_gid: Optional[int] = self.armempty_facts[0].global_id if self.armempty_facts else None

        self.closed_gid_by_door: Dict[str, int] = {}
        for f, args in self.closed_facts:
            if len(args) == 1:
                d = args[0]
                self.closed_gid_by_door[d] = f.global_id

        # Build room graph: room -> list of (neighbor_room, door_id)
        self.graph: Dict[str, List[Tuple[str, str]]] = {}
        self.rooms: set = set()
        for f, args in self.door_facts:
            if len(args) == 3:
                r1, r2, d = args
                self.rooms.add(r1)
                self.rooms.add(r2)
                self.graph.setdefault(r1, []).append((r2, d))

        # Cache list of all objects seen in goal_in (drives the task)
        self.objects: List[str] = list(self.goal_room_of.keys())

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

    def _is_true(self, state: int, fact_gid: int) -> int:
        return (state >> fact_gid) & 1

    def _get_robot_room(self, state: int) -> Optional[str]:
        # Exactly one rloc[room] is true.
        for room, gid in self.rloc_fact_by_room.items():
            if self._is_true(state, gid):
                return room
        return None

    def _get_holding_obj(self, state: int) -> Optional[str]:
        # At most one holding[obj]
        for obj, gid in self.holding_fact_by_obj.items():
            if self._is_true(state, gid):
                return obj
        return None

    def _obj_current_room(self, state: int, obj: str) -> Optional[str]:
        # If holding, it's not in any room.
        hold_gid = self.holding_fact_by_obj.get(obj)
        if hold_gid is not None and self._is_true(state, hold_gid):
            return None
        for gid, room in self.in_facts_by_obj.get(obj, []):
            if self._is_true(state, gid):
                return room
        return None

    def _is_door_closed(self, state: int, door_id: str) -> bool:
        gid = self.closed_gid_by_door.get(door_id)
        if gid is None:
            return False
        return bool(self._is_true(state, gid))

    def _dijkstra_room_costs(self, state: int, start_room: str) -> Dict[str, int]:
        # Edge cost: 1(move) + 1(if door closed now)
        # Small integer weights => classic Dijkstra with deque is fine.
        dist: Dict[str, int] = {start_room: 0}
        dq = deque([start_room])
        inq = {start_room}

        while dq:
            r = dq.popleft()
            inq.discard(r)
            base = dist[r]
            for nxt, d in self.graph.get(r, []):
                w = 2 if self._is_door_closed(state, d) else 1
                nd = base + w
                if nxt not in dist or nd < dist[nxt]:
                    dist[nxt] = nd
                    if nxt not in inq:
                        dq.append(nxt)
                        inq.add(nxt)
        return dist

    def __call__(self, parent_node, node) -> int:
        state = node.state

        # If no tasks remain, don't bias search.
        if hasattr(node, "task_network") and len(node.task_network) == 0:
            return 0

        # (A) Count unsatisfied deliveries: objects not in their goal room
        unsat = 0
        unsat_objs: List[str] = []
        for obj in self.objects:
            goal_room = self.goal_room_of.get(obj)
            if goal_room is None:
                continue
            in_goal_gid = self.in_fact_by_obj_room.get((obj, goal_room))
            if in_goal_gid is not None and self._is_true(state, in_goal_gid):
                continue
            # If holding the object, it's certainly not in goal room yet (unless goal is "holding", which doesn't exist)
            unsat += 1
            unsat_objs.append(obj)

        if unsat == 0:
            return 0

        h = unsat  # at least one putdown per remaining object

        # (C) Minimal navigation/open effort to enable next progress
        robot_room = self._get_robot_room(state)
        if robot_room is None:
            return h

        holding_obj = self._get_holding_obj(state)
        armempty = True
        if self.armempty_gid is not None:
            armempty = bool(self._is_true(state, self.armempty_gid))

        # Compute distances once
        dist = self._dijkstra_room_costs(state, robot_room)

        nav_lb = 0
        if holding_obj is not None and not armempty:
            # Need to reach goal room of held object
            g = self.goal_room_of.get(holding_obj)
            if g is not None and g in dist:
                nav_lb = dist[g]
            else:
                # unreachable in current connectivity (should be rare); keep admissible with 0
                nav_lb = 0
        else:
            # armempty: need to reach some room containing an unsatisfied object
            best = None
            for obj in unsat_objs:
                obj_room = self._obj_current_room(state, obj)
                if obj_room is None:
                    continue
                d = dist.get(obj_room)
                if d is None:
                    continue
                if best is None or d < best:
                    best = d
            nav_lb = best if best is not None else 0

        h += nav_lb
        return int(h)
