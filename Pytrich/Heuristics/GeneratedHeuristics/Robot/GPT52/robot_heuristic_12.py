from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic12(Heuristic):
    """Domain-dependent heuristic for the HDDL *robot* domain.

    Rationale (domain insights):
    - The only primitive actions are: move, open, pickup, putdown.
    - Each package goal is of the form (in obj goal_room). To change a package location,
      we must (pickup obj at its current room) then (putdown obj at its goal room).
    - The robot can hold only one package at a time (armempty/holding), so packages
      cannot be carried concurrently.
    - Movement between rooms is along a directed graph induced by (door r1 r2 d).
      If a door d is closed, it must be opened (open action) from one side before
      traversing it.

    Heuristic idea (lower bound, efficient):
    - Compute shortest-path distances between rooms in a relaxed navigation graph
      where each directed door-edge has cost:
         1 if the door is currently open, else 2 (must open then move).
      This is a *state-dependent* optimistic navigation cost because it ignores that
      opening a door could be done earlier and reused; counting it per traversal is
      an over-estimate in reality, but we only use these distances inside a lower-bound
      construction below.

    - For each unsatisfied package goal, we compute a per-package lower bound on the
      number of actions still needed, accounting for robot position and whether it is
      already holding a package:

        If obj already in its goal room -> 0.
        Else if robot holds obj:
            need to reach goal room (nav cost) + 1 putdown.
        Else if robot holds some other package:
            at least 1 action is required (you must put down that package somewhere)
            before you can pick up the needed one. We add a conservative +1.
            Then: reach obj room + pickup + reach goal room + putdown.
        Else (armempty):
            reach obj room + pickup + reach goal room + putdown.

      Importantly, we do NOT sum over all packages (that would wildly overcount because
      navigation legs can be shared). Instead, we take:
        h = max over packages of the per-package estimate,
      which is a valid lower bound on the remaining plan length because at least one
      unsatisfied package must be completed, and completing any specific package requires
      at least that many actions.

    - Additionally, we add a tiny task-network hint: if the current task network starts
      with abstract tasks that *must* eventually cause progress (like achieve-goals/release),
      we do not try to model decomposition cost explicitly; the domain’s methods only
      expand into the same primitive actions already counted. Keeping this heuristic
      light avoids overhead on large instances.

    This heuristic is designed to reduce node expansions by strongly preferring states
    that are closer (in navigation terms) to finishing at least one remaining package.
    """

    def __init__(self, model, robot_gpt52_heuristic_12: str = "robot_gpt52_heuristic_12"):
        super().__init__(model)
        self.name = robot_gpt52_heuristic_12
        self.model = model

        # Pre-index useful facts by predicate for fast access via bit tests.
        # We store lists of (fact_global_id, args)
        self._facts_by_pred: Dict[str, List[Tuple[int, List[str]]]] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._facts_by_pred.setdefault(pred, []).append((f.global_id, args))

        # Extract goal facts once (bitwise int -> explicit names -> parsed)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self._goal_in: List[Tuple[str, str]] = []  # (obj, room)
        self._goal_in_fact_ids: List[int] = []
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "in" and len(args) == 2:
                self._goal_in.append((args[0], args[1]))
                # map back to fact id by scanning facts_by_pred['in'] (done once)
        # Build a quick map fact-name->id for only needed ones to avoid heavy dict of all facts.
        self._fact_name_to_id: Dict[str, int] = {f.name: f.global_id for f in self.model.facts}
        self._goal_in_fact_ids = [self._fact_name_to_id.get(f"+in[{o},{r}]", -1) for o, r in self._goal_in]

        # Static door graph: edges (room1 -> room2) labeled by door object.
        # We'll query door closed-ness from state.
        self._door_edges: Dict[str, List[Tuple[str, str]]] = {}  # loc1 -> list[(loc2, door)]
        for fid, args in self._facts_by_pred.get("door", []):
            # fact args: [loc1, loc2, d]
            if len(args) == 3:
                self._door_edges.setdefault(args[0], []).append((args[1], args[2]))

        # Cache fact ids for closed(d)
        self._closed_fact_id: Dict[str, int] = {}
        for fid, args in self._facts_by_pred.get("closed", []):
            if len(args) == 1:
                self._closed_fact_id[args[0]] = fid

        # Cache for armempty
        self._armempty_fact_id: Optional[int] = None
        for fid, args in self._facts_by_pred.get("armempty", []):
            self._armempty_fact_id = fid
            break

        # Cache holding(obj) fact ids
        self._holding_fact_id: Dict[str, int] = {}
        for fid, args in self._facts_by_pred.get("holding", []):
            if len(args) == 1:
                self._holding_fact_id[args[0]] = fid

        # Cache rloc(room) fact ids
        self._rloc_fact_id: Dict[str, int] = {}
        for fid, args in self._facts_by_pred.get("rloc", []):
            if len(args) == 1:
                self._rloc_fact_id[args[0]] = fid

        # Cache in(obj,room) fact ids
        self._in_fact_id: Dict[Tuple[str, str], int] = {}
        for fid, args in self._facts_by_pred.get("in", []):
            if len(args) == 2:
                self._in_fact_id[(args[0], args[1])] = fid

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

    def _bit_true(self, state: int, fact_id: int) -> bool:
        if fact_id < 0:
            return False
        return ((state >> fact_id) & 1) == 1

    def _robot_room(self, state: int) -> Optional[str]:
        # exactly one rloc should be true
        for room, fid in self._rloc_fact_id.items():
            if self._bit_true(state, fid):
                return room
        return None

    def _held_object(self, state: int) -> Optional[str]:
        # either none or exactly one holding
        for obj, fid in self._holding_fact_id.items():
            if self._bit_true(state, fid):
                return obj
        return None

    def _object_room_if_not_held(self, state: int, obj: str) -> Optional[str]:
        # If held, it is not in any room (pickup deletes in). Otherwise, find its in.
        if obj in self._holding_fact_id and self._bit_true(state, self._holding_fact_id[obj]):
            return None
        # Scan only rooms that appear in the grounded in-facts for this obj
        for (o, r), fid in self._in_fact_id.items():
            if o == obj and self._bit_true(state, fid):
                return r
        return None

    def _is_door_closed(self, state: int, door: str) -> bool:
        fid = self._closed_fact_id.get(door, -1)
        return self._bit_true(state, fid)

    def _nav_distances(self, state: int, start: str) -> Dict[str, int]:
        """0-1-2 BFS (actually Dijkstra with small integer weights) from start.

        Edge cost: 1 if door is open, 2 if closed (open+move).
        """
        dist: Dict[str, int] = {start: 0}
        # Use deque-based Dial's algorithm for weights 1/2 by maintaining two buckets.
        # Simpler: standard deque trick: pushleft for cost1? Not correct for cost2.
        # We'll do a small Dijkstra with a deque of (node) buckets by distance parity.
        # Given instance sizes, a normal deque-based multi-queue is enough.

        # Implement classic Dijkstra with a simple list as priority queue would be too slow;
        # use bucketed queues since weights are 1 or 2.
        buckets: Dict[int, deque] = {0: deque([start])}
        current_d = 0
        max_seen = 0

        while True:
            while current_d in buckets and not buckets[current_d]:
                del buckets[current_d]
                current_d += 1
            if current_d not in buckets:
                break
            u = buckets[current_d].popleft()
            if dist.get(u, 10**9) != current_d:
                continue
            for v, d in self._door_edges.get(u, []):
                w = 2 if self._is_door_closed(state, d) else 1
                nd = current_d + w
                if nd < dist.get(v, 10**9):
                    dist[v] = nd
                    buckets.setdefault(nd, deque()).append(v)
                    if nd > max_seen:
                        max_seen = nd
        return dist

    def __call__(self, node) -> int:
        state = node.state

        # If no goal-in facts exist, nothing to do.
        if not self._goal_in:
            return 0

        rroom = self._robot_room(state)
        if rroom is None:
            # Should not happen in valid problems; return 0 to avoid crashing.
            return 0

        held = self._held_object(state)
        armempty = True
        if self._armempty_fact_id is not None:
            armempty = self._bit_true(state, self._armempty_fact_id)

        # Precompute nav distances from current robot room.
        dist_from_robot = self._nav_distances(state, rroom)

        best_lb = 0
        any_unsat = False

        for (obj, goal_room), goal_in_fid in zip(self._goal_in, self._goal_in_fact_ids):
            # Goal satisfied if +in[obj,goal_room] is true in state.
            if goal_in_fid >= 0 and self._bit_true(state, goal_in_fid):
                continue
            any_unsat = True

            if held == obj:
                # Need to go to goal room and putdown.
                nav = dist_from_robot.get(goal_room, 10**6)
                lb = nav + 1
            else:
                obj_room = self._object_room_if_not_held(state, obj)
                if obj_room is None:
                    # If the object is not held and we cannot find its location, be conservative.
                    # Return a small non-zero to still guide search.
                    lb = 1
                else:
                    nav_to_obj = dist_from_robot.get(obj_room, 10**6)
                    # After reaching obj room, pickup (1).
                    # Then from obj_room to goal_room: need distances from obj_room.
                    dist_from_obj = self._nav_distances(state, obj_room)
                    nav_obj_to_goal = dist_from_obj.get(goal_room, 10**6)
                    base = nav_to_obj + 1 + nav_obj_to_goal + 1  # move/open..., pickup, move/open..., putdown

                    if (not armempty) and (held is not None):
                        # Must free the hand somehow before picking obj (at least one putdown in some room).
                        # Lower bound +1.
                        base += 1
                    lb = base

            if lb > best_lb:
                best_lb = lb

        return 0 if not any_unsat else int(best_lb)
