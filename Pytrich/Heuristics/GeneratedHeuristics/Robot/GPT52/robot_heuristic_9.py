from collections import deque
from typing import Dict, List, Optional, Set, Tuple

# Pytrich import (works in the official framework layout)
from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic9(Heuristic):
    """Domain-dependent heuristic for the HDDL 'robot' domain.

    Domain recap (grounded):
      Facts include (examples):
        +rloc[c], +in[o1,r1], +holding[o1], +armempty, +closed[d01], +door[c,r1,d01], +goal_in[o1,r1]

    What makes the domain tricky:
      - The HTN essentially loops achieve-goals until all packages are in their goal rooms.
      - To fix one misplaced object you must: reach its current room, pick it up, reach its goal room,
        and put it down (putdown is only allowed *in the goal room*).
      - Movement requires doors to be open (i.e., not +closed[d]). Opening is local: you must be at loc1.
      - Only one object can be held at a time (armempty/holding).

    Heuristic idea (admissible lower bound):
      For each package not yet at its goal location, we compute a *relaxed* minimal number of primitive
      steps needed to make progress, using:
        - shortest-path distance over the room graph where traversing a directed edge (loc1->loc2 via door d)
          costs 1 move, plus an additional +1 if the door is currently closed (because we would need an open).
          This is a *lower bound* (ignores interactions between multiple doors, ignores that opening requires
          being at loc1 which the path already accounts for).
        - then estimate per-package cost as:
            if holding(obj): dist(robot_loc, goal_room) + 1 (putdown)
            else: dist(robot_loc, obj_room) + 1 (pickup) + dist(obj_room, goal_room) + 1 (putdown)
          (If arm not empty and obj not held, we still use the same formula; it remains a lower bound.)

      To combine across many packages while staying admissible, we do NOT sum (that would overcount because
      tasks are sequential and the robot can only carry one at a time, but summing can easily exceed optimal
      cost). Instead we take:
         h = max over misplaced packages of the per-package lower bound.
      This yields a safe estimate and still provides strong guidance in large instances.

    Efficiency:
      - We parse and index facts once in __init__.
      - We compute weighted shortest paths from a small set of sources using 0-1 BFS on each call.
        (edge cost is 1 for move plus 0/1 for closed door => total 1 or 2; we convert to 0/1 by splitting:
         handle as weights in {1,2} via deque-based variant; still fast for <= 50 rooms).
      - We avoid any calls to state_explicit_repr(state) during evaluation.

    Notes:
      - This heuristic is purely state-based (task-network independent). In this domain, the remaining HTN
        work correlates almost entirely with remaining unsatisfied (in obj goal_room) facts.
      - If all goals satisfied, returns 0.
    """

    def __init__(self, model, robot_gpt52_heuristic_9=None):
        super().__init__(model)
        self.model = model

        # Fact indices
        self._fact_by_name: Dict[str, int] = {}

        # Parsed predicate -> list of (args, global_id)
        self._in_facts: List[Tuple[str, str, int]] = []          # (obj, room, gid) for +in[obj,room]
        self._goal_in: Dict[str, str] = {}                       # obj -> goal_room
        self._rloc_facts: List[Tuple[str, int]] = []             # (room, gid) for +rloc[room]
        self._holding_facts: Dict[str, int] = {}                 # obj -> gid for +holding[obj]
        self._armempty_gid: Optional[int] = None                 # gid for +armempty

        # Door model: edges with associated door object
        # adjacency[src] = list of (dst, door_name)
        self._adj: Dict[str, List[Tuple[str, str]]] = {}
        self._closed_gid: Dict[str, int] = {}                    # door -> gid for +closed[door]

        # Precompute goal fact list (as names) once, then parse out goal_in from *facts* too.
        # goal_in facts are part of the state (static) but still represented as facts.
        self._build_indices()

    # ------------------------ parsing / indexing ------------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _build_indices(self):
        # Map fact name -> global_id and collect relevant predicates.
        for f in self.model.facts:
            self._fact_by_name[f.name] = f.global_id
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue

            if pred == "in" and len(args) == 2 and f.name.startswith("+"):
                obj, room = args
                self._in_facts.append((obj, room, f.global_id))
            elif pred == "goal_in" and len(args) == 2 and f.name.startswith("+"):
                obj, room = args
                self._goal_in[obj] = room
            elif pred == "rloc" and len(args) == 1 and f.name.startswith("+"):
                self._rloc_facts.append((args[0], f.global_id))
            elif pred == "holding" and len(args) == 1 and f.name.startswith("+"):
                self._holding_facts[args[0]] = f.global_id
            elif pred == "armempty" and len(args) == 0 and f.name.startswith("+"):
                self._armempty_gid = f.global_id
            elif pred == "door" and len(args) == 3 and f.name.startswith("+"):
                loc1, loc2, d = args
                self._adj.setdefault(loc1, []).append((loc2, d))
            elif pred == "closed" and len(args) == 1 and f.name.startswith("+"):
                d = args[0]
                self._closed_gid[d] = f.global_id

        # Ensure all rooms appear in adjacency keys (even if sink-only)
        # Not strictly necessary but helps robust BFS.
        for src in list(self._adj.keys()):
            for dst, _ in self._adj[src]:
                self._adj.setdefault(dst, self._adj.get(dst, []))

    # ------------------------ core helpers ------------------------
    def _state_has(self, state: int, gid: int) -> bool:
        return ((state >> gid) & 1) == 1

    def _get_robot_room(self, state: int) -> Optional[str]:
        # Exactly one rloc is true.
        for room, gid in self._rloc_facts:
            if self._state_has(state, gid):
                return room
        return None

    def _get_held_object(self, state: int) -> Optional[str]:
        for obj, gid in self._holding_facts.items():
            if self._state_has(state, gid):
                return obj
        return None

    def _get_object_room(self, state: int, obj: str) -> Optional[str]:
        # Scan in-facts for this obj. (At most one room per obj in valid states.)
        # This is O(#in_facts) which is O(#packages). With 100 packages it's fine.
        for o, room, gid in self._in_facts:
            if o == obj and self._state_has(state, gid):
                return room
        return None

    def _edge_cost(self, state: int, door: str) -> int:
        # move always costs 1; if door is closed we need open (>=1) additionally.
        closed_gid = self._closed_gid.get(door, None)
        if closed_gid is not None and self._state_has(state, closed_gid):
            return 2
        return 1

    def _dijkstra_small_int_weights(self, state: int, start: str) -> Dict[str, int]:
        """Compute shortest distances from start to all rooms.

        Edge weights are 1 or 2. For small graphs (<=50 rooms) we can do a simple
        deque-based Dial/0-1-2 BFS using two-ended queue with relaxation.
        Complexity is very small in this domain.
        """
        if start is None:
            return {}

        dist: Dict[str, int] = {start: 0}
        dq = deque([start])

        while dq:
            u = dq.popleft()
            du = dist[u]
            for v, d in self._adj.get(u, []):
                w = self._edge_cost(state, d)
                nd = du + w
                old = dist.get(v)
                if old is None or nd < old:
                    dist[v] = nd
                    # If w==1, pushleft for slightly better performance; if w==2, pushright.
                    if w == 1:
                        dq.appendleft(v)
                    else:
                        dq.append(v)
        return dist

    # ------------------------ heuristic ------------------------
    def __call__(self, node) -> int:
        state = node.state

        # If goals already reached at state-level, return 0.
        # (HTN may still have tasks, but in this domain achieve-goals can finish when nothing to do.)
        if (state & self.model.goals) == self.model.goals:
            return 0

        robot_room = self._get_robot_room(state)
        held_obj = self._get_held_object(state)

        # Precompute distances from robot location once.
        dist_from_robot = self._dijkstra_small_int_weights(state, robot_room)

        best = 0

        # Consider each object that has a goal.
        for obj, goal_room in self._goal_in.items():
            # If already satisfied, skip.
            obj_room = self._get_object_room(state, obj)
            if obj_room == goal_room:
                continue

            # If holding this object, only need to reach goal and putdown.
            if held_obj == obj:
                d_rg = dist_from_robot.get(goal_room)
                if d_rg is None:
                    # unreachable under current door-closure costs; still admissible to return large,
                    # but keep it bounded and informative.
                    per = 0
                else:
                    per = d_rg + 1  # putdown
                if per > best:
                    best = per
                continue

            # Not holding it: need reach object, pickup, reach goal, putdown.
            if obj_room is None:
                # In inconsistent intermediate states this might happen; don't crash.
                continue

            d_ro = dist_from_robot.get(obj_room)
            if d_ro is None:
                # can't even reach object's room (under current closure costs); ignore for admissibility
                # (heuristic 0 for this package).
                continue

            # Distances from object-room to goal-room.
            dist_from_obj_room = self._dijkstra_small_int_weights(state, obj_room)
            d_og = dist_from_obj_room.get(goal_room)
            if d_og is None:
                continue

            per = d_ro + 1 + d_og + 1
            if per > best:
                best = per

        # Mild extra lower bound: if holding a wrong object, at least 1 action needed (either putdown is
        # only possible at its goal, so likely move/open first; but 1 is safe).
        # Not strictly necessary, but helps some corner cases.
        if best == 0 and held_obj is not None:
            return 1

        return best
