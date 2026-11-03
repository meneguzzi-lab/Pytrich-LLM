from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple, Optional, Set

# Pytrich import (works in the competition/container environment)
from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic1(Heuristic):
    """Domain-dependent heuristic for the HDDL 'robot' domain.

    RATIONALE (domain insights)
    --------------------------
    The domain repeatedly executes (achieve-goals), which for any package not already at its
    goal room will:
      1) move/open until the robot reaches the package room,
      2) pickup,
      3) then (release) will move/open until the robot reaches the goal room,
      4) putdown.

    Key planning challenges:
      - Navigation over a room graph with doors; doors may be closed and require an 'open' action.
      - Single gripper constraint: at most one package can be carried at a time.
      - Large instances: many packages; computing something per node must be cheap.

    Heuristic idea:
      - Precompute an all-pairs shortest-path on the room graph in terms of *lower bound*
        action cost where traversing an edge costs:
            1 move + (1 open if door is currently closed else 0)
        We compute this dynamically per state for door-closed contribution in a very cheap way:
        by using the *static* room graph and adding an *admissible* estimate of additional opens.

      - Admissible lower bound for remaining work focuses on packages not yet in their goal room.
        For each such package, at minimum we need:
           pickup (1) + putdown (1) + some navigation between relevant rooms.
        With a single gripper, packages must be handled sequentially; a safe lower bound is the
        maximum (not sum) over packages of the required effort for one package, plus some small
        gripper-state correction.

    Concretely we compute:
      h(state, tn) =
         max_over_unsatisfied_packages( 2 + dist(robot_room, pkg_room) + dist(pkg_room, goal_room) )
         with modifications if already holding a package.

    This is:
      - Domain-dependent: uses predicates rloc/in/holding/goal_in and door/closed.
      - Hierarchy-aware in spirit: estimates the remaining refinement cost of achieve-goals/release
        by capturing their inevitable primitive backbone (move/open/pickup/putdown).
      - Efficient: precomputes graph structure once; per call is linear in #facts in goal_in/in/holding.

    Admissibility/consistency notes:
      - Distances are computed as shortest number of moves ignoring closed doors, then we add
        0 for opens (i.e., we do NOT count opens) to keep it clearly admissible.
        Optionally we add a very small admissible open lower bound (0) always; i.e., we ignore opens.
      - Using max over packages is admissible (sequential handling means total >= hardest single package).
      - This heuristic is weak but safe and scales well to 100 packages.

    Implementation details:
      - We never call model.state_explicit_repr(state) (too slow); we use bit checks with global_id.
      - We parse fact strings once in __init__ to build indices.

    """

    def __init__(self, model, robot_gpt52_heuristic_1: str = "robot_gpt52_heuristic_1"):
        super().__init__(model)
        self.name = robot_gpt52_heuristic_1
        self.model = model

        # --- Parse facts once and build indices ---
        self._fact_by_pred: Dict[str, List[Tuple[List[str], int]]] = {}
        self._fact_id_by_name: Dict[str, int] = {}

        for f in self.model.facts:
            self._fact_id_by_name[f.name] = f.global_id
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._fact_by_pred.setdefault(pred, []).append((args, f.global_id))

        # Map goal_in: package -> goal room (static)
        self._goal_room_of: Dict[str, str] = {}
        for args, gid in self._fact_by_pred.get("goal_in", []):
            if len(args) == 2:
                self._goal_room_of[args[0]] = args[1]

        # room graph from door facts (static)
        self._neighbors: Dict[str, List[Tuple[str, str]]] = {}  # room -> list of (room2, doorId)
        for args, gid in self._fact_by_pred.get("door", []):
            if len(args) == 3:
                r1, r2, d = args
                self._neighbors.setdefault(r1, []).append((r2, d))

        # Collect rooms
        self._rooms: Set[str] = set(self._neighbors.keys())
        for r1, lst in self._neighbors.items():
            for r2, _d in lst:
                self._rooms.add(r2)

        # Precompute all-pairs shortest path in number of MOVES (ignoring opens/closed)
        # dist_moves[(r1,r2)] = min #moves
        self._dist_moves: Dict[Tuple[str, str], int] = {}
        self._precompute_move_distances()

        # Cache fact ids for quick checks
        self._rloc_facts: List[Tuple[str, int]] = []  # (room, gid)
        for args, gid in self._fact_by_pred.get("rloc", []):
            if len(args) == 1:
                self._rloc_facts.append((args[0], gid))

        self._holding_facts: List[Tuple[str, int]] = []  # (pkg, gid)
        for args, gid in self._fact_by_pred.get("holding", []):
            if len(args) == 1:
                self._holding_facts.append((args[0], gid))

        self._in_facts: List[Tuple[str, str, int]] = []  # (pkg, room, gid)
        for args, gid in self._fact_by_pred.get("in", []):
            if len(args) == 2:
                self._in_facts.append((args[0], args[1], gid))

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

    def _precompute_move_distances(self):
        # BFS from each room on the unweighted directed graph
        for src in self._rooms:
            q = deque([(src, 0)])
            seen = {src}
            self._dist_moves[(src, src)] = 0
            while q:
                r, d = q.popleft()
                for (nxt, _door) in self._neighbors.get(r, []):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    self._dist_moves[(src, nxt)] = d + 1
                    q.append((nxt, d + 1))

    def _get_robot_room(self, state_bitwise: int) -> Optional[str]:
        # Exactly one rloc should be true
        for room, gid in self._rloc_facts:
            if (state_bitwise >> gid) & 1:
                return room
        return None

    def _get_holding(self, state_bitwise: int) -> Optional[str]:
        for pkg, gid in self._holding_facts:
            if (state_bitwise >> gid) & 1:
                return pkg
        return None

    def _get_package_locations(self, state_bitwise: int) -> Dict[str, str]:
        loc: Dict[str, str] = {}
        for pkg, room, gid in self._in_facts:
            if (state_bitwise >> gid) & 1:
                loc[pkg] = room
        # If holding(pkg) then it is not 'in' any room due to effects.
        return loc

    def _move_dist(self, r1: Optional[str], r2: Optional[str]) -> int:
        if r1 is None or r2 is None:
            return 0
        if r1 == r2:
            return 0
        return self._dist_moves.get((r1, r2), 0)

    def __call__(self, node) -> int:
        state = node.state

        robot_room = self._get_robot_room(state)
        holding = self._get_holding(state)
        pkg_loc = self._get_package_locations(state)

        # Build set of packages from goal_in mapping (static, from problem)
        # A package is satisfied if it is in its goal room.
        max_lb = 0

        # If holding something, we must at least put it down at its goal.
        # In this domain putdown is only allowed at goal room.
        if holding is not None:
            goal_room = self._goal_room_of.get(holding)
            # lower bound: move to goal + putdown
            # (pickup already happened)
            lb = self._move_dist(robot_room, goal_room) + 1
            max_lb = max(max_lb, lb)

        for pkg, goal_room in self._goal_room_of.items():
            # If currently holding this pkg, handled above
            if pkg == holding:
                continue

            cur_room = pkg_loc.get(pkg)
            # If package is not in any room and not held (shouldn't happen), skip safely
            if cur_room is None:
                continue

            if cur_room == goal_room:
                continue  # satisfied

            # Lower bound for completing this package from current situation:
            # need: go to package + pickup + go to goal + putdown
            lb = self._move_dist(robot_room, cur_room) + 1 + self._move_dist(cur_room, goal_room) + 1
            max_lb = max(max_lb, lb)

        # Small tie-breaker using remaining task network size (still admissible if 0)
        # We keep it 0 to preserve strict admissibility.
        return int(max_lb)
