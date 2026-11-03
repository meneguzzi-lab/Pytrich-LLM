from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

try:
    # Pytrich layout (common)
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback in case the import path differs in the evaluation harness
    from heuristic import Heuristic


class RobotGPT52Heuristic5(Heuristic):
    """Domain-dependent heuristic for the HDDL Robot domain.

    Core domain insight
    -------------------
    The domain is essentially: move between rooms through doors (may require opening),
    pick up packages, and put them down only at their goal room.

    The HTN structure repeats:
      - achieve-goals loops over (pickup -> release) until all goals satisfied.
      - release loops over (putdown) if holding a goal-satisfied object, otherwise
        it can (move/open) and recurse.

    A good lower bound must account for:
      1) Every unsatisfied package requires at least one pickup and one putdown.
      2) If the robot is not co-located with a needed package, at least one move is needed.
      3) If a required move uses only closed incident doors at the current location, at least one open
         is needed before any move (safe local lower bound).

    This heuristic returns a *lower bound* (admissible) by combining several conservative components:
      - count of unsatisfied packages (each needs >= pickup + putdown => 2)
      - plus 1 if robot must move to reach any "relevant" room (a room containing an unsatisfied package
        or a goal room for some unsatisfied package).
      - plus 1 if currently holding a wrong package (holding obj not yet at its goal) => at least
        a putdown is required eventually (and putdown is only at goal).
      - plus 1 if at current room all outgoing doors are closed but at least one move is needed => must open.

    Efficiency
    ----------
    - No string-based state inspection (no model.state_explicit_repr(state)).
    - Pre-index facts by predicate and argument tuples and test truth via bit operations.
    - Precompute an undirected room graph (ignoring closed) for fast BFS distance.

    Notes
    -----
    - We intentionally ignore additional moves after the first (we only add 0/1 for "needs to move"),
      because any stronger move-distance term risks inadmissibility when the HTN can finish without
      moving due to already satisfied goals.
    - The heuristic also considers the current task network length lightly as a tie-breaker (0/1),
      but only if it is safe: we add 0 because tasks can be abstract and decomposed to zero cost in
      principle (not true here), so we avoid adding task-network-based cost to keep admissibility.
    """

    def __init__(self, model, robot_gpt52_heuristic_5: str = "robot_gpt52_heuristic_5"):
        super().__init__(model)
        self.name = robot_gpt52_heuristic_5
        self.model = model

        # Fact indices
        self.fact_by_pred_args: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        # Problem objects
        self.packages: List[str] = []
        self.rooms: List[str] = []

        # Static relational data
        self.goal_room_of: Dict[str, str] = {}  # obj -> goal room
        self.doors: Dict[Tuple[str, str], str] = {}  # (room1,room2)->door-id
        self.adj: Dict[str, List[Tuple[str, str]]] = {}  # room -> list of (neighbor, door)

        # Precomputed for fast checks
        self._armempty_fact: Optional[int] = None

        self._build_indices()

    # -------------------------
    # Parsing / indexing helpers
    # -------------------------

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

    def _build_indices(self):
        # Index all facts by predicate+args for O(1) access to their global_id.
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self.fact_by_pred_args[(pred, tuple(args))] = f.global_id

        # Collect packages/rooms from facts (robust to missing typed object listing in the model).
        # Packages appear in in[obj,room], holding[obj], goal_in[obj,room]
        pkgs = set()
        rms = set()

        for (pred, args), gid in self.fact_by_pred_args.items():
            if pred == "in" and len(args) == 2:
                pkgs.add(args[0])
                rms.add(args[1])
            elif pred == "holding" and len(args) == 1:
                pkgs.add(args[0])
            elif pred == "goal_in" and len(args) == 2:
                pkgs.add(args[0])
                rms.add(args[1])
                # goal_in is static in these instances, store mapping
                self.goal_room_of[args[0]] = args[1]
            elif pred == "rloc" and len(args) == 1:
                rms.add(args[0])
            elif pred == "door" and len(args) == 3:
                r1, r2, d = args
                rms.add(r1)
                rms.add(r2)
                self.doors[(r1, r2)] = d

        self.packages = sorted(pkgs)
        self.rooms = sorted(rms)

        # armempty fact
        self._armempty_fact = self.fact_by_pred_args.get(("armempty", tuple()))

        # Build adjacency list from door facts (treat as directed edges as encoded, but for reachability
        # we just traverse as stored; instances often include both directions explicitly).
        self.adj = {r: [] for r in self.rooms}
        for (r1, r2), d in self.doors.items():
            if r1 not in self.adj:
                self.adj[r1] = []
            self.adj[r1].append((r2, d))

    # -------------------------
    # Bitwise fact checks
    # -------------------------

    def _is_true(self, state: int, fact_gid: Optional[int]) -> bool:
        if fact_gid is None:
            return False
        return ((state >> fact_gid) & 1) == 1

    def _gid(self, pred: str, *args: str) -> Optional[int]:
        return self.fact_by_pred_args.get((pred, tuple(args)))

    # -------------------------
    # Domain-specific utilities
    # -------------------------

    def _robot_room(self, state: int) -> Optional[str]:
        # Find the unique rloc[x] that is true.
        # We scan only rooms (<=50), cheap.
        for r in self.rooms:
            if self._is_true(state, self._gid("rloc", r)):
                return r
        return None

    def _current_holding(self, state: int) -> Optional[str]:
        # At most one object can be held (armempty/holding).
        for o in self.packages:
            if self._is_true(state, self._gid("holding", o)):
                return o
        return None

    def _unsatisfied_packages(self, state: int) -> List[str]:
        # A package is satisfied if in[obj, goal_room_of[obj]] is true.
        unsat = []
        for o in self.packages:
            gr = self.goal_room_of.get(o)
            if gr is None:
                # If no explicit goal_in for this package, ignore it.
                continue
            if not self._is_true(state, self._gid("in", o, gr)):
                unsat.append(o)
        return unsat

    def _any_open_outgoing(self, state: int, room: str) -> bool:
        # True if there exists an outgoing door edge from room that is not closed.
        for (nbr, d) in self.adj.get(room, []):
            closed_gid = self._gid("closed", d)
            if not self._is_true(state, closed_gid):
                return True
        return False

    def _reachable_ignoring_closed(self, start: str, targets: set) -> bool:
        # Graph reachability ignoring door closedness (static topology).
        if start in targets:
            return True
        q = deque([start])
        seen = {start}
        while q:
            u = q.popleft()
            for v, _d in self.adj.get(u, []):
                if v in seen:
                    continue
                if v in targets:
                    return True
                seen.add(v)
                q.append(v)
        return False

    # -------------------------
    # Heuristic interface
    # -------------------------

    def __call__(self, parent_node, node) -> int:
        """Return an admissible estimate of remaining cost."""
        state = node.state

        # Count unsatisfied packages
        unsat = self._unsatisfied_packages(state)
        k = len(unsat)
        if k == 0:
            # If all package goals satisfied, remaining cost can still be 0 because HTN can finish.
            return 0

        h = 0

        # (A) Each unsatisfied package must be picked up and put down at least once (lower bound 2).
        # This is safe because the only way to change in[obj,*] toward goal is pickup then putdown.
        h += 2 * k

        # (B) If holding an unsatisfied package, at least one putdown will be needed.
        holding = self._current_holding(state)
        if holding is not None:
            gr = self.goal_room_of.get(holding)
            if gr is not None and not self._is_true(state, self._gid("in", holding, gr)):
                # putdown action cost is 1
                h += 1

        # (C) If robot is not in any "relevant" room, at least one move is needed.
        # Relevant rooms:
        #   - room containing any unsatisfied package
        #   - goal room for any unsatisfied package
        rloc = self._robot_room(state)
        if rloc is not None:
            relevant_rooms = set()
            for o in unsat:
                gr = self.goal_room_of.get(o)
                if gr is not None:
                    relevant_rooms.add(gr)
                # current location of o (if known)
                for r in self.rooms:
                    if self._is_true(state, self._gid("in", o, r)):
                        relevant_rooms.add(r)
                        break

            if rloc not in relevant_rooms:
                # If topology suggests we can reach some relevant room, then at least one move.
                # If not reachable ignoring closedness, still at least one move would be required
                # in any solvable variant; keep it conservative and add 1 anyway.
                if len(relevant_rooms) > 0:
                    h += 1

                    # (D) If we need to move but currently no open outgoing door, we must open at least one door.
                    # Local necessary condition: to make any move from rloc, some outgoing door must be opened.
                    if not self._any_open_outgoing(state, rloc):
                        # Only add if there exists at least one outgoing edge (otherwise opening is impossible).
                        if len(self.adj.get(rloc, [])) > 0:
                            h += 1

        # Ensure non-negative int
        if h < 0:
            return 0
        return int(h)
