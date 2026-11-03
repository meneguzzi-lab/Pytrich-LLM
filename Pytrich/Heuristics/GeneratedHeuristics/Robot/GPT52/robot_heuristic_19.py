from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

# Pytrich base class (available in the framework)
from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic19(Heuristic):
    """Domain-dependent heuristic for the HDDL *robot* domain.

    Rationale / domain insights
    --------------------------
    The domain’s search difficulty is driven by:
      1) Navigation through a graph of rooms connected by doors.
      2) Doors may be closed, requiring an `open` action *once per door*.
      3) Packages must be transported one-by-one (single arm): to fix one package
         typically needs: go to package room, pickup, go to goal room, putdown.

    The HDDL hierarchy encodes this loop:
      achieve-goals -> (pickup_abstract obj) -> release -> ... -> putdown_abstract -> achieve-goals
    but at the grounded level we can lower-bound remaining work by counting *unsatisfied goals*
    and estimating minimal travel/opening cost to accomplish them.

    Heuristic design
    ----------------
    We compute a relaxed lower bound consisting of:
      h = (number of unsatisfied goal facts "in(o,goal_room)")
          + minimal navigation/opening effort for at least one remaining delivery

    Details:
      - Unsatisfied goals: for each goal fact +in[o,loc] that is not true in current state,
        we add 1. This is admissible because to make a false `in(o,loc)` true requires at least
        one action (in fact at least pickup+putdown+moves), so counting 1 per unsatisfied goal
        is a safe lower bound.

      - Navigation/opening lower bound: for one unsatisfied package, we compute a shortest-path
        estimate from the robot’s current room to (a) package location and then to its goal room.
        Each edge has cost 1 for `move` plus +1 extra if the door is currently closed (for the
        needed `open`). This remains admissible because any real plan must traverse some path and
        open any closed door it passes through at least once.

        We take the MIN over unsatisfied packages (best-case next delivery). This is still a lower
        bound for the overall remaining work, but provides useful guidance.

    Efficiency
    ----------
    Large instances have up to 50 rooms and many packages.
    We precompute adjacency lists from static `door(r1,r2,d)` facts.
    Per evaluation we:
      - Extract current robot location and closed doors with fast bit checks.
      - Run at most a few BFS/Dijkstra-like (0/1 extra cost) computations:
        For each evaluation, we compute a single-source shortest path from current room to all rooms,
        then for candidate packages we compute an additional shortest path from the package room to all
        rooms *only for the best few candidates*.

    Robustness
    ----------
    If we cannot parse some facts (unexpected grounding), we fall back to the unsatisfied goal count.
    """

    def __init__(self, model, robot_gpt52_heuristic_19: str = "robot_gpt52_heuristic_19"):
        super().__init__(model, robot_gpt52_heuristic_19)
        self.model = model

        # --- Cache fact ids by predicate ---
        self._fact_ids_by_pred: Dict[str, List[int]] = {}
        self._fact_args_by_id: Dict[int, Tuple[str, List[str]]] = {}

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._fact_args_by_id[f.global_id] = (pred, args)
            self._fact_ids_by_pred.setdefault(pred, []).append(f.global_id)

        # Goal facts (explicit names) and parsed targets
        self._goal_in_targets: List[Tuple[str, str, int]] = []  # (obj, room, fact_id)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        goal_fact_ids = self._names_to_fact_ids(goal_fact_names)
        for fid in goal_fact_ids:
            pred_args = self._fact_args_by_id.get(fid)
            if not pred_args:
                continue
            pred, args = pred_args
            if pred == "in" and len(args) == 2:
                self._goal_in_targets.append((args[0], args[1], fid))

        # Static door graph: from door(room1, room2, d)
        # Note: domain has symmetric door facts, but we just read whatever is present.
        self._adj: Dict[str, List[Tuple[str, str]]] = {}  # room -> [(neighbor_room, door_id)]
        for fid in self._fact_ids_by_pred.get("door", []):
            pred, args = self._fact_args_by_id.get(fid, (None, []))
            if pred != "door" or len(args) != 3:
                continue
            r1, r2, d = args
            self._adj.setdefault(r1, []).append((r2, d))

        # Helpful indices for quick lookup in a state
        # rloc facts: +rloc[room]
        self._rloc_fact_by_room: Dict[str, int] = {}
        for fid in self._fact_ids_by_pred.get("rloc", []):
            pred, args = self._fact_args_by_id.get(fid, (None, []))
            if pred == "rloc" and len(args) == 1:
                self._rloc_fact_by_room[args[0]] = fid

        # closed facts: +closed[door]
        self._closed_fact_by_door: Dict[str, int] = {}
        for fid in self._fact_ids_by_pred.get("closed", []):
            pred, args = self._fact_args_by_id.get(fid, (None, []))
            if pred == "closed" and len(args) == 1:
                self._closed_fact_by_door[args[0]] = fid

        # in facts: +in[obj,room]
        self._in_fact_by_obj_room: Dict[Tuple[str, str], int] = {}
        for fid in self._fact_ids_by_pred.get("in", []):
            pred, args = self._fact_args_by_id.get(fid, (None, []))
            if pred == "in" and len(args) == 2:
                self._in_fact_by_obj_room[(args[0], args[1])] = fid

    # -------- Heuristic API --------

    def __call__(self, node) -> int:
        state = node.state

        # 1) Count unsatisfied in-goals (admissible)
        unsat: List[Tuple[str, str, int]] = []
        for obj, room, fid in self._goal_in_targets:
            if not self._is_true(state, fid):
                unsat.append((obj, room, fid))
        base = len(unsat)
        if base == 0:
            return 0

        # 2) Add a small but informative admissible navigation/opening lower bound
        rroom = self._get_robot_room(state)
        if rroom is None:
            return base

        closed_doors = self._get_closed_doors_set(state)

        # From robot to all rooms
        dist_from_robot = self._dijkstra_rooms(rroom, closed_doors)
        if not dist_from_robot:
            return base

        # For each unsatisfied goal, identify current package room and estimate travel.
        # We only need the MIN over candidates to keep it admissible and cheap.
        best = None
        # Preselect candidates with smallest robot->package_room distance (cheap filter)
        candidates: List[Tuple[int, str, str]] = []  # (d1, obj, goal_room)
        for obj, goal_room, _ in unsat:
            pkg_room = self._find_object_room(state, obj)
            if pkg_room is None:
                continue
            d1 = dist_from_robot.get(pkg_room)
            if d1 is None:
                continue
            candidates.append((d1, obj, goal_room))

        if not candidates:
            return base

        candidates.sort(key=lambda x: x[0])
        # Evaluate only top-K to bound time; still admissible since we take MIN over subset,
        # but to preserve admissibility we must ensure subset contains the true minimum.
        # To be safe, we use all candidates when small, otherwise a modest cap.
        # (Even if we cap, we only *underestimate* by possibly missing the best => still admissible
        # to return base + min_over_subset >=? Actually missing best could make estimate larger,
        # which can break admissibility. Therefore we must NOT cap in a way that can increase.
        # So we compute exact min across all candidates, but we can do it with caching.)

        # Cache package-room SSSP to avoid recomputation when many objects are in same room
        sssp_cache: Dict[str, Dict[str, int]] = {}

        for d1, obj, goal_room in candidates:
            pkg_room = self._find_object_room(state, obj)
            if pkg_room is None:
                continue
            if pkg_room not in sssp_cache:
                sssp_cache[pkg_room] = self._dijkstra_rooms(pkg_room, closed_doors)
            dist_from_pkg = sssp_cache[pkg_room]
            d2 = dist_from_pkg.get(goal_room)
            if d2 is None:
                continue

            # We do NOT add pickup/putdown here because base already counts 1 per goal.
            # This term purely guides toward closer deliveries.
            cand = d1 + d2
            if best is None or cand < best:
                best = cand

        if best is None:
            return base

        # Combine: unsatisfied-goal count + minimal travel/open lower bound for at least one delivery
        return base + best

    # -------- Helpers --------

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

    def _names_to_fact_ids(self, names: List[str]) -> List[int]:
        # Build a hash map from name to id once if needed; here we do a one-off creation.
        # The fact list is not huge (facts are manageable), and this is only called in __init__.
        name_to_id = {f.name: f.global_id for f in self.model.facts}
        return [name_to_id[n] for n in names if n in name_to_id]

    def _is_true(self, state_bitwise: int, fact_id: int) -> int:
        return (state_bitwise >> fact_id) & 1

    def _get_robot_room(self, state_bitwise: int) -> Optional[str]:
        # Find which +rloc[room] is true.
        for room, fid in self._rloc_fact_by_room.items():
            if self._is_true(state_bitwise, fid):
                return room
        return None

    def _get_closed_doors_set(self, state_bitwise: int) -> Set[str]:
        closed = set()
        for door, fid in self._closed_fact_by_door.items():
            if self._is_true(state_bitwise, fid):
                closed.add(door)
        return closed

    def _find_object_room(self, state_bitwise: int, obj: str) -> Optional[str]:
        # Search all possible (obj, room) in-facts quickly by scanning known in-facts.
        # This is acceptable because #rooms <= 50 in the largest instance.
        for (o, room), fid in self._in_fact_by_obj_room.items():
            if o == obj and self._is_true(state_bitwise, fid):
                return room
        return None

    def _dijkstra_rooms(self, start_room: str, closed_doors: Set[str]) -> Dict[str, int]:
        # Edge cost: 1 for move, plus 1 if door is closed (must open at least once before moving through it).
        # All costs are positive small ints; use deque-based 0-1 BFS variant is not applicable due to 1/2,
        # but we can use a simple BFS on expanded costs (still tiny graphs) or Dijkstra with deque.
        # Here we use Dijkstra with a deque-like SPFA pattern; for <=50 nodes it is fast.

        dist: Dict[str, int] = {start_room: 0}
        in_q: Set[str] = {start_room}
        q = deque([start_room])

        while q:
            u = q.popleft()
            in_q.discard(u)
            du = dist[u]
            for v, d in self._adj.get(u, []):
                w = 1 + (1 if d in closed_doors else 0)
                nd = du + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    if v not in in_q:
                        q.append(v)
                        in_q.add(v)
        return dist
