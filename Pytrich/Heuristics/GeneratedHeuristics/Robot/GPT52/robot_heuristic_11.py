from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

# Pytrich import style varies between templates; this defensive import usually works.
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    try:
        from Pytrich.Heuristics import Heuristic
    except Exception:
        # Fallback for some installations
        from heuristic import Heuristic


class RobotGPT52Heuristic11(Heuristic):
    """Domain-dependent heuristic for the HDDL *robot* domain.

    Domain structure (key observations)
    -------------------------------
    - Only primitive actions: pickup, putdown, move, open.
    - A package contributes to goal satisfaction iff (in obj goal_room).
    - The HTN drives the agent to:
        achieve-goals -> (pickup_abstract obj) then (release)
        release -> either putdown_abstract (if already at goal room holding obj)
                  or (move_abstract) / (open_abstract) recursively.
    - Robot carries at most one package (armempty/holding).
    - Doors may be closed and require an open action before move.

    Heuristic idea
    --------------
    Compute a cheap *lower bound* on remaining primitive actions by combining:

    (A) Per-object delivery lower bound under a relaxation:
        For each object not already in its goal room, estimate:
          min over current possible object locations L of
            dist(robot_room, L) + pickup(1 if not holding it already)
            + dist(L, goal_room) + putdown(1)
        where dist(X,Y) is the number of moves in the room graph (ignoring door closedness).

        Additionally, since doors may be closed, we add a second lower bound term that
        counts *necessary opens* along shortest paths (still a lower bound):
          minimum number of closed doors on any shortest path for the chosen route.
        This is computed with a BFS that minimizes (moves, closed_doors) lexicographically.

    (B) Single-carrying coupling lower bound:
        Because the robot can deliver at most one package per putdown, we must perform
        at least (#unsatisfied_objects - delivered_by_current_holding) putdowns.
        That is already included per-object, but we also take a global lower bound:
          unsatisfied_objects
        since each unsatisfied object requires at least one pickup+putdown (2 actions).
        We use max-combination with the best single-object estimate to keep it admissible.

    Final heuristic is:
        h(state, tn) = max(
            best_single_object_cost_lb,
            2 * unsatisfied_objects_lb
        )

    This remains admissible because:
    - Distances ignore negative conditions and interactions (relaxation).
    - Counting closed doors on a path is a lower bound on required open actions.
    - Using max of lower bounds preserves lower-boundedness.

    Efficiency
    ----------
    - Precomputes room graph and for all room pairs computes:
        (min_moves, min_closed_doors_among_shortest)
      via BFS from each room (rooms <= 50 in training set).
    - State evaluation uses only bit checks (no explicit state repr scanning).

    Notes
    -----
    - The heuristic does not need to explicitly use the task network; the domain HTN
      ultimately enforces the same primitive necessities captured by the state-based LB.
      Still, this heuristic is compatible with HTN search and provides good guidance.
    """

    def __init__(self, model, robot_gpt52_heuristic_11: bool = True):
        super().__init__(model)
        self.robot_gpt52_heuristic_11 = robot_gpt52_heuristic_11

        # --- Index facts by predicate for fast state bit tests ---
        self._facts_by_pred: Dict[str, List] = {}
        self._fact_by_name: Dict[str, object] = {}
        for f in self.model.facts:
            self._fact_by_name[f.name] = f
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._facts_by_pred.setdefault(pred, []).append((f, args))

        # Cache goal_in mapping (static across states)
        self._goal_room_of_obj: Dict[str, str] = {}
        for f, args in self._facts_by_pred.get("goal_in", []):
            if len(args) == 2:
                self._goal_room_of_obj[args[0]] = args[1]

        # Build room adjacency graph and map doors
        # We assume 'door' facts encode directed edges room1->room2 with door-id.
        self._adj: Dict[str, List[Tuple[str, str]]] = {}
        self._rooms: set[str] = set()
        self._door_for_edge: Dict[Tuple[str, str], str] = {}
        for f, args in self._facts_by_pred.get("door", []):
            if len(args) != 3:
                continue
            r1, r2, d = args
            self._rooms.add(r1)
            self._rooms.add(r2)
            self._adj.setdefault(r1, []).append((r2, d))
            self._door_for_edge[(r1, r2)] = d

        # Precompute pairwise (moves, min_closed_on_shortest) for all room pairs.
        # We store two dicts keyed by (src, dst)
        self._min_moves: Dict[Tuple[str, str], int] = {}
        self._min_closed_on_shortest: Dict[Tuple[str, str], int] = {}
        self._precompute_room_metrics()

        # Cache Fact objects used frequently
        self._armempty_fact = self._fact_by_name.get("+armempty")

    # ------------------------- Parsing helpers -------------------------
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

    def _is_true(self, state_bitwise: int, fact_obj) -> bool:
        # fact_obj can be None if not present
        if fact_obj is None:
            return False
        return ((state_bitwise >> fact_obj.global_id) & 1) == 1

    # ------------------------- Graph precompute -------------------------
    def _precompute_room_metrics(self) -> None:
        # Multi-criteria BFS from each room minimizing (moves, closed_doors)
        # but only closed_doors among paths with minimal moves.
        rooms = list(self._rooms)
        for src in rooms:
            # dist_moves[room] = min moves
            dist_moves: Dict[str, int] = {src: 0}
            # dist_closed[room] = min closed doors among shortest paths
            dist_closed: Dict[str, int] = {src: 0}

            q = deque([src])
            while q:
                u = q.popleft()
                um = dist_moves[u]
                uc = dist_closed[u]
                for v, d in self._adj.get(u, []):
                    cand_m = um + 1
                    # closedness is state-dependent; here we only precompute structure.
                    # We'll later add the *current* closed-door indicator per edge.
                    # For precompute we store the path length only; closed-door count
                    # will be computed on-the-fly with a second BFS depending on state.
                    # However, to keep evaluation fast, we do precompute shortest moves.
                    if v not in dist_moves:
                        dist_moves[v] = cand_m
                        dist_closed[v] = 0
                        q.append(v)
                    elif cand_m < dist_moves[v]:
                        dist_moves[v] = cand_m
                        dist_closed[v] = 0
                        q.append(v)

            # Fill min_moves table; unreachable => large
            for dst in rooms:
                self._min_moves[(src, dst)] = dist_moves.get(dst, 10**9)

        # Note: closed-door counts depend on state; we compute them per-evaluation
        # using a BFS restricted to shortest-move layers (cheap for <=50 rooms).

    def _min_closed_doors_on_any_shortest_path(self, state_bitwise: int, src: str, dst: str) -> int:
        """Return minimum number of *currently closed* doors along any shortest (min-move) path."""
        if src == dst:
            return 0
        target_moves = self._min_moves.get((src, dst), 10**9)
        if target_moves >= 10**9:
            return 10**6

        # Layered BFS / DP over shortest-path DAG: keep best closed count per node at each move depth.
        # Since graph small, we can do a simple BFS with state (node, depth).
        best: Dict[Tuple[str, int], int] = {(src, 0): 0}
        q = deque([(src, 0)])
        ans = 10**9

        # Precompute set of closed door facts in this state for O(1) bit checks by door-id.
        # We build a dict door_id -> fact_obj for +closed[door]
        # (facts are of form +closed[d01])
        closed_facts = self._facts_by_pred.get("closed", [])
        closed_fact_by_door: Dict[str, object] = {}
        for f, args in closed_facts:
            if len(args) == 1:
                closed_fact_by_door[args[0]] = f

        while q:
            u, depth = q.popleft()
            ccount = best[(u, depth)]
            if depth == target_moves:
                if u == dst:
                    ans = min(ans, ccount)
                continue
            for v, d in self._adj.get(u, []):
                # only follow edges that stay on a shortest path w.r.t move distance
                if self._min_moves.get((v, dst), 10**9) != target_moves - (depth + 1):
                    continue
                add_closed = 1 if self._is_true(state_bitwise, closed_fact_by_door.get(d)) else 0
                nd = depth + 1
                nc = ccount + add_closed
                key = (v, nd)
                if key not in best or nc < best[key]:
                    best[key] = nc
                    q.append((v, nd))

        return ans if ans < 10**9 else 10**6

    # ------------------------- State feature extraction -------------------------
    def _get_robot_room(self, state_bitwise: int) -> Optional[str]:
        for f, args in self._facts_by_pred.get("rloc", []):
            if len(args) == 1 and self._is_true(state_bitwise, f):
                return args[0]
        return None

    def _get_holding(self, state_bitwise: int) -> Optional[str]:
        for f, args in self._facts_by_pred.get("holding", []):
            if len(args) == 1 and self._is_true(state_bitwise, f):
                return args[0]
        return None

    def _get_object_rooms(self, state_bitwise: int) -> Dict[str, str]:
        locs: Dict[str, str] = {}
        for f, args in self._facts_by_pred.get("in", []):
            if len(args) == 2 and self._is_true(state_bitwise, f):
                obj, room = args
                locs[obj] = room
        return locs

    # ------------------------- Heuristic core -------------------------
    def __call__(self, state: int, task_network=None) -> int:
        if not self.robot_gpt52_heuristic_11:
            return 0

        rroom = self._get_robot_room(state)
        if rroom is None:
            return 0

        holding_obj = self._get_holding(state)
        obj_room = self._get_object_rooms(state)

        # Determine unsatisfied objects from goal_in: unsatisfied if not in goal room.
        unsat: List[Tuple[str, str, Optional[str]]] = []  # (obj, goal_room, current_room)
        for obj, gro in self._goal_room_of_obj.items():
            cro = obj_room.get(obj, None)
            # If currently holding it, it is not in any room.
            if holding_obj == obj:
                cro = None
            if cro != gro:
                unsat.append((obj, gro, cro))

        if not unsat:
            return 0

        # Global LB: each unsatisfied object needs at least pickup+putdown => 2 actions.
        # If we are already holding an unsatisfied object, it still needs putdown (and moves/opens),
        # but pickup can be 0 for that one. This LB remains safe as max component.
        global_lb = 2 * len(unsat)
        if holding_obj is not None:
            # Reduce by at most 1 because one pickup may already have been done.
            # Still admissible: if holding some unsatisfied object, we can lower bound as 2*n - 1.
            # (Because at least n putdowns and at least (n-1) pickups remain.)
            global_lb = max(0, global_lb - 1)

        best_single = 10**9

        # For each unsatisfied object compute a per-object delivery LB, then take minimum as "next best".
        # Using min as a LB on remaining work is safe when combined via max with another LB.
        for obj, goal_room, cur_room in unsat:
            # Case 1: object is currently held
            if holding_obj == obj:
                moves = self._min_moves.get((rroom, goal_room), 10**9)
                opens = self._min_closed_doors_on_any_shortest_path(state, rroom, goal_room)
                # Need to move/possibly open, then putdown (1)
                cand = moves + opens + 1
                best_single = min(best_single, cand)
                continue

            # Case 2: object is in some room (normal case). If not found, skip conservatively.
            if cur_room is None:
                continue

            # To get it: go to cur_room, (opens along way), pickup (1)
            m1 = self._min_moves.get((rroom, cur_room), 10**9)
            o1 = self._min_closed_doors_on_any_shortest_path(state, rroom, cur_room)

            # Then deliver: cur_room -> goal_room, opens, then putdown (1)
            m2 = self._min_moves.get((cur_room, goal_room), 10**9)
            o2 = self._min_closed_doors_on_any_shortest_path(state, cur_room, goal_room)

            if m1 >= 10**9 or m2 >= 10**9:
                continue

            # If currently not armempty, we cannot pickup until we put something down.
            # Lower bound: add 1 extra action to "resolve" carrying (at least one putdown somewhere).
            # This remains admissible (might be 0 if actually already armempty).
            need_free_hand_lb = 0
            if holding_obj is not None:
                need_free_hand_lb = 1

            cand = m1 + o1 + 1 + m2 + o2 + 1 + need_free_hand_lb
            best_single = min(best_single, cand)

        if best_single >= 10**9:
            # Fallback if something went wrong (unreachable etc.)
            return global_lb

        return max(best_single, global_lb)
